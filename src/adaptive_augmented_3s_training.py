"""Generic ECAPA/AAM fine-tuning with cached SpeechBrain Fbanks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from src.aam_training import (
    AAMSoftmax,
    apply_batchnorm_policy,
    build_adamw_optimizer,
)
from src.adaptive_augmented_3s_verification import (
    TRIAL_FIELDS,
    ValidationRow,
    ValidationTrial,
    sha256_file,
    validate_validation_trials,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_metrics import calculate_eer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "outputs/fbank_cache_adaptive_augmented_3s_v1"
DEFAULT_CONFIG = ROOT / "configs/adaptive_augmented_3s_training_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/ecapa_aam_adaptive_augmented_3s_v1"
TRAIN_MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_train_manifest.csv"
VALIDATION_MANIFEST = (
    ROOT / "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
)
VALIDATION_TRIALS = (
    ROOT
    / "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
)
CACHE_CONFIG_NAME = "fbank_cache_config_adaptive_augmented_3s_v1.json"
CACHE_IDENTITY_NAME = "fbank_cache_identity_adaptive_augmented_3s_v1.json"
MANIFEST_FIELDS = (
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "final_split",
)
FEATURE_SHAPE = (301, 80)
EMBEDDING_DIM = 192


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def atomic_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(to_cpu(dict(value)), temporary)
    os.replace(temporary, path)


def read_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Invalid manifest schema: {path}")
        for line, raw in enumerate(reader, start=2):
            relative = raw["relative_audio_path"].strip()
            speaker = raw["speaker_id"].strip()
            label = int(raw["speaker_label"])
            if (
                not relative
                or relative in seen
                or not speaker
                or raw["final_split"].strip() != split
            ):
                raise ValueError(f"Invalid manifest row at {path}:{line}")
            if split == "train" and label < 0:
                raise ValueError(f"Negative train label at {path}:{line}")
            if split in {"validation", "final_test"} and label != -1:
                raise ValueError(f"Evaluation label must be -1 at {path}:{line}")
            seen.add(relative)
            rows.append(
                {
                    "relative_audio_path": relative,
                    "speaker_id": speaker,
                    "speaker_label": label,
                    "final_split": split,
                }
            )
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def read_trials(path: Path) -> tuple[ValidationTrial, ...]:
    trials: list[ValidationTrial] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError(f"Invalid trial schema: {path}")
        for raw in reader:
            trials.append(
                ValidationTrial(
                    raw["trial_id"],
                    raw["left_audio_path"],
                    raw["right_audio_path"],
                    raw["left_speaker_id"],
                    raw["right_speaker_id"],
                    int(raw["target"]),
                )
            )
    if not trials:
        raise ValueError("Validation trials are empty")
    positives = sum(trial.target == 1 for trial in trials)
    negatives = sum(trial.target == 0 for trial in trials)
    if positives < 1 or negatives < 1:
        raise ValueError("Validation trials need genuine and impostor pairs")
    return tuple(trials)


class CachedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        cache_root: Path,
        split: str,
        shard_size: int,
        max_cached_shards: int,
    ) -> None:
        self.rows = list(rows)
        self.cache_root = cache_root
        self.split = split
        self.shard_size = shard_size
        self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def shard_path_for_index(self, index: int) -> str:
        return f"{self.split}/shard_{index // self.shard_size:05d}.pt"

    def _load_shard(self, shard_number: int) -> Mapping[str, Any]:
        if shard_number in self._cache:
            value = self._cache.pop(shard_number)
            self._cache[shard_number] = value
            return value
        path = self.cache_root / self.split / f"shard_{shard_number:05d}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[shard_number] = value
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard_number, offset = divmod(index, self.shard_size)
        shard = self._load_shard(shard_number)
        if shard["relative_audio_paths"][offset] != row["relative_audio_path"]:
            raise ValueError("Cache and manifest row order disagree")
        feature = shard["features"][offset]
        if tuple(feature.shape) != FEATURE_SHAPE or feature.dtype != torch.float32:
            raise ValueError("Invalid cached FBank tensor")
        return {
            "fbank": feature,
            "speaker_label": int(shard["speaker_labels"][offset]),
            "speaker_id": row["speaker_id"],
            "relative_audio_path": row["relative_audio_path"],
            "dataset_index": index,
            "final_split": self.split,
        }


def collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fbank": torch.stack([sample["fbank"] for sample in samples]),
        "speaker_label": torch.tensor(
            [sample["speaker_label"] for sample in samples], dtype=torch.long
        ),
        "speaker_id": [sample["speaker_id"] for sample in samples],
        "relative_audio_path": [
            sample["relative_audio_path"] for sample in samples
        ],
        "dataset_index": torch.tensor(
            [sample["dataset_index"] for sample in samples], dtype=torch.long
        ),
        "final_split": [sample["final_split"] for sample in samples],
    }


class ShardAwarePKSampler(Sampler[list[int]]):
    """Deterministic speaker-balanced P×K sampling with a shard window."""

    def __init__(
        self,
        dataset: CachedDataset,
        *,
        speakers_per_batch: int,
        samples_per_speaker: int,
        active_shard_window: int,
        batches_per_epoch: int,
        seed: int,
        epoch: int,
    ) -> None:
        self.dataset = dataset
        self.p = speakers_per_batch
        self.k = samples_per_speaker
        self.active_window = active_shard_window
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = epoch
        grouped: dict[str, list[int]] = defaultdict(list)
        shards: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row["speaker_id"]].append(index)
            shards[dataset.shard_path_for_index(index)].append(index)
        if len(grouped) < self.p:
            raise ValueError(f"P={self.p} exceeds {len(grouped)} train speakers")
        if any(len(indexes) < self.k for indexes in grouped.values()):
            raise ValueError("A train speaker has fewer utterances than K")
        self.speaker_indexes = {
            speaker: tuple(indexes) for speaker, indexes in grouped.items()
        }
        self.shard_indexes = {
            shard: tuple(indexes) for shard, indexes in shards.items()
        }
        self.shards = tuple(sorted(shards))
        if not 1 <= self.active_window <= len(self.shards):
            raise ValueError(
                f"active_shard_window={self.active_window} is invalid for "
                f"{len(self.shards)} train shards"
            )

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        shard_order = list(self.shards)
        rng.shuffle(shard_order)
        queues = {
            speaker: list(indexes)
            for speaker, indexes in self.speaker_indexes.items()
        }
        for queue in queues.values():
            rng.shuffle(queue)
        exposure: Counter[str] = Counter()
        cursor = 0
        for _ in range(self.batches_per_epoch):
            width = min(self.active_window, len(shard_order))
            while True:
                active = [
                    shard_order[(cursor + offset) % len(shard_order)]
                    for offset in range(width)
                ]
                preferred: dict[str, set[int]] = defaultdict(set)
                for shard in active:
                    for index in self.shard_indexes[shard]:
                        preferred[self.dataset.rows[index]["speaker_id"]].add(index)
                if len(preferred) >= self.p or width == len(shard_order):
                    break
                width = min(len(shard_order), width + self.active_window)
            tie = {speaker: rng.random() for speaker in preferred}
            selected_speakers = sorted(
                preferred,
                key=lambda speaker: (exposure[speaker], tie[speaker]),
            )[: self.p]
            if len(selected_speakers) != self.p:
                raise RuntimeError("Unable to form a complete P×K batch")
            batch: list[int] = []
            for speaker in selected_speakers:
                chosen: list[int] = []
                queue = queues[speaker]
                while len(chosen) < self.k:
                    candidates = [
                        index
                        for index in queue
                        if index in preferred[speaker] and index not in chosen
                    ]
                    if not candidates:
                        candidates = [index for index in queue if index not in chosen]
                    if not candidates:
                        queue.extend(self.speaker_indexes[speaker])
                        rng.shuffle(queue)
                        continue
                    value = candidates[0]
                    queue.remove(value)
                    chosen.append(value)
                batch.extend(chosen)
                exposure[speaker] += 1
            rng.shuffle(batch)
            if len(batch) != self.p * self.k or len(set(batch)) != len(batch):
                raise RuntimeError("Sampler created an invalid P×K batch")
            yield batch
            cursor = (cursor + self.active_window) % len(shard_order)


def round_robin(batch: Mapping[str, Any], p: int, k: int) -> dict[str, Any]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(batch["speaker_label"].tolist()):
        grouped[label].append(position)
    if len(grouped) != p or any(len(values) != k for values in grouped.values()):
        raise ValueError("Logical batch does not satisfy P×K")
    order = [grouped[label][rank] for rank in range(k) for label in sorted(grouped)]
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = (
            value[order]
            if isinstance(value, torch.Tensor)
            else [value[index] for index in order]
        )
    return result


def cosine_factor(step: int, total_steps: int, minimum: float) -> float:
    progress = min(max(step / total_steps, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def load_configuration(
    config_path: Path,
    cache_root: Path,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    trial_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError("Training config schema_version must be 2")
    if config.get("model") != {
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
    }:
        raise ValueError("Training config must use the SpeechBrain ECAPA-192 model")
    if config.get("aam", {}).get("margin") != 0.2 or config.get("aam", {}).get(
        "scale"
    ) != 30.0:
        raise ValueError("Training config must use AAM margin=0.2 and scale=30")
    if config.get("optimizer", {}).get("name") != "AdamW":
        raise ValueError("Training config optimizer must be AdamW")
    if config.get("scheduler", {}).get("name") != "cosine":
        raise ValueError("Training config scheduler must be cosine")
    if config.get("scheduler", {}).get("warmup_steps") != 0:
        raise ValueError("This training implementation expects warmup_steps=0")
    if config.get("amp", {}).get("ecapa_dtype") != "float16":
        raise ValueError("ECAPA AMP dtype must be float16")
    numeric_positive = (
        config["optimizer"]["ecapa_lr"],
        config["optimizer"]["aam_lr"],
        config["optimizer"]["weight_decay"],
        config["early_stopping"]["patience"],
        config["checkpoint_interval_updates"],
        config["log_every_updates"],
    )
    if any(float(value) <= 0.0 for value in numeric_positive):
        raise ValueError("Learning rates, decay, patience and intervals must be positive")
    sampler = config["sampler"]
    if sampler.get("name") != "HybridShardAwareSpeakerBatchSampler":
        raise ValueError("Training config sampler is invalid")
    logical_batch = (
        int(sampler["speakers_per_batch"])
        * int(sampler["samples_per_speaker"])
    )
    microbatch = int(config["microbatch_size"])
    if logical_batch % microbatch or int(sampler["speakers_per_batch"]) % microbatch:
        raise ValueError("Logical batch must be divisible by microbatch_size")
    classes = len({row["speaker_id"] for row in train_rows})
    labels = {int(row["speaker_label"]) for row in train_rows}
    if labels != set(range(classes)):
        raise ValueError("Train labels must be contiguous and match train speakers")
    steps = sampler.get("batches_per_epoch", "auto")
    if steps == "auto":
        steps = math.ceil(len(train_rows) / logical_batch)
    steps = int(steps)
    epochs = int(config["scheduler"]["max_epochs"])
    if steps < 1 or epochs < 1:
        raise ValueError("Training steps and epochs must be positive")

    cache_config_path = cache_root / CACHE_CONFIG_NAME
    cache_identity_path = cache_root / CACHE_IDENTITY_NAME
    cache_config = json.loads(cache_config_path.read_text(encoding="utf-8"))
    cache_identity = json.loads(cache_identity_path.read_text(encoding="utf-8"))
    current_manifest_hashes = {
        "train_manifest": sha256_file(TRAIN_MANIFEST),
        "validation_manifest": sha256_file(VALIDATION_MANIFEST),
    }
    cached_bindings = cache_config.get("input_bindings", {})
    if (
        cache_config.get("expected_rows")
        != {"train": len(train_rows), "validation": len(validation_rows)}
        or cache_config.get("train_class_count") != classes
        or cache_config.get("feature_shape") != list(FEATURE_SHAPE)
        or cache_identity.get("config_sha256") != sha256_file(cache_config_path)
        or any(
            cached_bindings.get(name, {}).get("sha256") != digest
            for name, digest in current_manifest_hashes.items()
        )
        or cache_identity.get("input_bindings") != cached_bindings
    ):
        raise ValueError("Cache identity/counts disagree with current manifests")
    runtime = json.loads(json.dumps(config))
    runtime["aam"]["num_classes"] = classes
    runtime["sampler"]["batches_per_epoch"] = steps
    runtime["scheduler"]["steps_per_epoch"] = steps
    runtime["scheduler"]["total_steps"] = steps * epochs
    runtime["validation"]["trial_count"] = trial_count
    runtime["cache"] = {
        "identity_sha256": cache_identity["identity_sha256"],
        "config_sha256": cache_identity["config_sha256"],
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "feature_shape": list(FEATURE_SHAPE),
    }
    binding = {
        "configuration_sha256": canonical_digest(runtime),
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "validation_manifest_sha256": sha256_file(VALIDATION_MANIFEST),
        "validation_trials_sha256": sha256_file(VALIDATION_TRIALS),
    }
    return runtime, binding


def build_models(
    config: Mapping[str, Any], device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module, AAMSoftmax, Any, Any, Any]:
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm
    embedding_model = frontend.classifier.mods.embedding_model
    del frontend
    mean_var_norm.eval().to(device)
    for parameter in mean_var_norm.parameters():
        parameter.requires_grad_(False)
    embedding_model.to(device)
    apply_batchnorm_policy(embedding_model)
    aam = AAMSoftmax(
        embedding_dim=EMBEDDING_DIM,
        num_classes=config["aam"]["num_classes"],
        margin=config["aam"]["margin"],
        scale=config["aam"]["scale"],
        seed=config["sampler"]["seed"],
    ).to(device)
    optimizer = build_adamw_optimizer(
        embedding_model,
        aam,
        embedding_lr=config["optimizer"]["ecapa_lr"],
        classifier_lr=config["optimizer"]["aam_lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    total_steps = config["scheduler"]["total_steps"]
    minimum = config["scheduler"]["minimum_factor"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_factor(step, total_steps, minimum),
    )
    amp_enabled = bool(config["amp"]["enabled"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
        init_scale=config["amp"]["grad_scaler_initial_scale"],
    )
    return mean_var_norm, embedding_model, aam, optimizer, scheduler, scaler


def evaluate_validation(
    dataset: CachedDataset,
    trials: Sequence[ValidationTrial],
    mean_var_norm: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    mean_var_norm.eval()
    embedding_model.eval()
    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            features = batch["fbank"].to(device=device, dtype=torch.float32)
            lengths = torch.ones(features.shape[0], device=device)
            normalized = mean_var_norm(features, lengths)
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda", dtype=torch.float16
            ):
                value = embedding_model(normalized, lengths).squeeze(1).float()
            value = F.normalize(value, p=2, dim=1).cpu()
            embeddings.update(zip(batch["relative_audio_path"], value))
            if batch_number % 25 == 0 or batch_number == len(loader):
                print(
                    f"VALIDATION batch={batch_number}/{len(loader)}", flush=True
                )
    scores: list[float] = []
    targets: list[int] = []
    for trial in trials:
        try:
            score = torch.dot(
                embeddings[trial.left_audio_path],
                embeddings[trial.right_audio_path],
            )
        except KeyError as error:
            raise ValueError(f"Trial path missing from validation cache: {error}") from error
        scores.append(float(score))
        targets.append(trial.target)
    result = calculate_eer(scores, targets)
    apply_batchnorm_policy(embedding_model)
    return {
        "eer": float(result.interpolated_eer),
        "threshold": float(result.empirical_threshold),
    }


def run_training(
    *,
    config_path: Path,
    cache_root: Path,
    output_dir: Path,
    device_name: str,
    resume: Path | None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    train_rows = read_manifest(TRAIN_MANIFEST, "train")
    validation_rows = read_manifest(VALIDATION_MANIFEST, "validation")
    if {row["speaker_id"] for row in train_rows} & {
        row["speaker_id"] for row in validation_rows
    }:
        raise ValueError("Speaker leakage between train and validation")
    trials = read_trials(VALIDATION_TRIALS)
    trial_rows = tuple(
        ValidationRow(row["relative_audio_path"], row["speaker_id"])
        for row in validation_rows
    )
    validate_validation_trials(
        trials,
        trial_rows,
        genuine_count=sum(trial.target == 1 for trial in trials),
        impostor_count=sum(trial.target == 0 for trial in trials),
    )
    config, binding = load_configuration(
        config_path, cache_root, train_rows, validation_rows, len(trials)
    )
    cache_config = json.loads(
        (cache_root / CACHE_CONFIG_NAME).read_text(encoding="utf-8")
    )
    shard_size = int(cache_config["shard_size"])
    max_shards = max(1, int(config["sampler"]["active_shard_window"]))
    train_dataset = CachedDataset(
        train_rows, cache_root, "train", shard_size, max_shards
    )
    validation_dataset = CachedDataset(
        validation_rows, cache_root, "validation", shard_size, 2
    )

    seed = int(config["sampler"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    mean_var_norm, embedding_model, aam, optimizer, scheduler, scaler = build_models(
        config, device
    )
    state: dict[str, Any] = {
        "next_epoch": 0,
        "next_position": 0,
        "global_step": 0,
        "best_eer": None,
        "best_epoch": None,
        "patience_counter": 0,
    }

    def checkpoint_payload(reason: str) -> dict[str, Any]:
        return {
            "schema": "generic_ecapa_aam_training_v2",
            "reason": reason,
            "binding": binding,
            "runtime_config": config,
            "embedding_model_state_dict": embedding_model.state_dict(),
            "mean_var_norm_state_dict": mean_var_norm.state_dict(),
            "aam_state_dict": aam.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "cursor": dict(state),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema") != "generic_ecapa_aam_training_v2"
            or checkpoint.get("binding") != binding
            or checkpoint.get("runtime_config") != config
        ):
            raise ValueError("Resume checkpoint belongs to another dataset/config")
        embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
        mean_var_norm.load_state_dict(checkpoint["mean_var_norm_state_dict"])
        aam.load_state_dict(checkpoint["aam_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        state.update(checkpoint["cursor"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch_state = torch.as_tensor(
            checkpoint["torch_random_state"], dtype=torch.uint8, device="cpu"
        ).contiguous()
        torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and checkpoint.get("cuda_random_states"):
            cuda_states = [
                torch.as_tensor(item, dtype=torch.uint8, device="cpu").contiguous()
                for item in checkpoint["cuda_random_states"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        print(
            f"RESUME epoch={state['next_epoch'] + 1} "
            f"position={state['next_position']} step={state['global_step']}",
            flush=True,
        )
    elif (output_dir / "last.pt").exists():
        raise FileExistsError("last.pt exists; pass --resume or use another output dir")

    sampler_config = config["sampler"]
    logical_size = (
        sampler_config["speakers_per_batch"]
        * sampler_config["samples_per_speaker"]
    )
    microbatch_size = int(config["microbatch_size"])
    batches_per_epoch = int(sampler_config["batches_per_epoch"])
    updates_this_call = 0
    for epoch in range(state["next_epoch"], config["scheduler"]["max_epochs"]):
        sampler = ShardAwarePKSampler(
            train_dataset,
            speakers_per_batch=sampler_config["speakers_per_batch"],
            samples_per_speaker=sampler_config["samples_per_speaker"],
            active_shard_window=min(
                sampler_config["active_shard_window"],
                math.ceil(len(train_dataset) / shard_size),
            ),
            batches_per_epoch=batches_per_epoch,
            seed=seed,
            epoch=epoch,
        )
        planned = list(sampler)
        start_position = (
            int(state["next_position"])
            if epoch == int(state["next_epoch"])
            else 0
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=planned[start_position:],
            num_workers=0,
            collate_fn=collate,
        )
        embedding_model.train()
        aam.train()
        apply_batchnorm_policy(embedding_model)
        for position, batch in enumerate(loader, start=start_position):
            logical = round_robin(
                batch,
                sampler_config["speakers_per_batch"],
                sampler_config["samples_per_speaker"],
            )
            overflow_retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, logical_size, microbatch_size):
                    stop = start + microbatch_size
                    features = logical["fbank"][start:stop].to(
                        device=device, dtype=torch.float32
                    )
                    labels = logical["speaker_label"][start:stop].to(device)
                    lengths = torch.ones(features.shape[0], device=device)
                    with torch.no_grad():
                        normalized = mean_var_norm(features, lengths)
                    with torch.cuda.amp.autocast(
                        enabled=device.type == "cuda", dtype=torch.float16
                    ):
                        embedding = embedding_model(normalized, lengths).squeeze(1)
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = aam(embedding.float(), labels)
                        loss = (
                            F.cross_entropy(
                                logits.float(), labels, reduction="sum"
                            )
                            / logical_size
                        )
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Non-finite ECAPA/AAM loss")
                    scaler.scale(loss).backward()
                    total_loss += float(loss.detach())
                scaler.unscale_(optimizer)
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                optimizer_updated = scale_after >= scale_before
                if optimizer_updated:
                    break
                overflow_retries += 1
                if overflow_retries > 8:
                    raise RuntimeError("AMP overflow retry limit reached")
                print(
                    f"AMP overflow retry={overflow_retries}/8 "
                    f"scale={scale_after}",
                    flush=True,
                )
            scheduler.step()
            state["global_step"] += 1
            updates_this_call += 1
            if position + 1 < batches_per_epoch:
                state["next_epoch"] = epoch
                state["next_position"] = position + 1
            else:
                state["next_epoch"] = epoch + 1
                state["next_position"] = 0
            if state["global_step"] % config["log_every_updates"] == 0:
                print(
                    f"TRAIN epoch={epoch + 1}/{config['scheduler']['max_epochs']} "
                    f"batch={position + 1}/{batches_per_epoch} "
                    f"step={state['global_step']} loss={total_loss:.6f}",
                    flush=True,
                )
            if state["global_step"] % config["checkpoint_interval_updates"] == 0:
                atomic_save(output_dir / "last.pt", checkpoint_payload("rolling"))
            if max_updates is not None and updates_this_call >= max_updates:
                atomic_save(output_dir / "last.pt", checkpoint_payload("partial"))
                return {**state, "stopped_for_max_updates": True}

        metrics = evaluate_validation(
            validation_dataset,
            trials,
            mean_var_norm,
            embedding_model,
            device,
            config["validation"]["batch_size"],
        )
        improved = state["best_eer"] is None or metrics["eer"] < (
            float(state["best_eer"])
            - config["early_stopping"]["min_improvement_eer"]
        )
        if improved:
            state["best_eer"] = metrics["eer"]
            state["best_epoch"] = epoch + 1
            state["patience_counter"] = 0
            atomic_save(
                output_dir / "best.pt", checkpoint_payload("best_validation_eer")
            )
        else:
            state["patience_counter"] += 1
        state["next_epoch"] = epoch + 1
        state["next_position"] = 0
        atomic_save(output_dir / "last.pt", checkpoint_payload("epoch_complete"))
        print(
            f"EPOCH {epoch + 1} | validation_EER={metrics['eer'] * 100:.4f}% "
            f"| threshold={metrics['threshold']:.6f} "
            f"| best={float(state['best_eer']) * 100:.4f}% "
            f"| patience={state['patience_counter']}/"
            f"{config['early_stopping']['patience']}",
            flush=True,
        )
        if state["patience_counter"] >= config["early_stopping"]["patience"]:
            break
    return dict(state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.run == args.dry_run:
        parser.error("Specify exactly one of --run or --dry-run")
    config_path = args.config.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        if args.resume is not None:
            parser.error("--dry-run does not accept --resume")
        first = run_training(
            config_path=config_path,
            cache_root=cache_root,
            output_dir=output_dir,
            device_name=args.device,
            resume=None,
            max_updates=1,
        )
        second = run_training(
            config_path=config_path,
            cache_root=cache_root,
            output_dir=output_dir,
            device_name=args.device,
            resume=output_dir / "last.pt",
            max_updates=1,
        )
        print(json.dumps({"fresh": first, "resume": second}, indent=2))
    else:
        result = run_training(
            config_path=config_path,
            cache_root=cache_root,
            output_dir=output_dir,
            device_name=args.device,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
