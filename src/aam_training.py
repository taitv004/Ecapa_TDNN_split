"""Focused AAM-Softmax, microbatch, optimizer, and checkpoint helpers."""

from __future__ import annotations

import hashlib
import math
import os
import random
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


CHECKPOINT_SCHEMA_NAME = "speaker_verification_aam_training_smoke"
CHECKPOINT_SCHEMA_VERSION = 1
PRETRAINED_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM = 192
LEGACY_SMOKE_NUM_CLASSES = 488
SEED = 20260727
APPROVED_TRAIN_CACHE_RELATIVE = Path("outputs/fbank_cache_v1")
APPROVED_TRAIN_INDEX_FILENAME = "train_feature_index_v1.csv"
APPROVED_TRAIN_SHARD_DIRECTORY = "train"


def validate_train_only_smoke_request(
    *,
    project_root: str | Path,
    cache_dir: str | Path,
    split: str,
    index_filename: str,
) -> dict[str, str]:
    """Validate an explicit train-only cache request before Dataset construction."""
    if type(split) is not str or split != "train":
        raise ValueError("AAM smoke split must be exactly 'train'")
    if type(index_filename) is not str or index_filename != APPROVED_TRAIN_INDEX_FILENAME:
        raise ValueError(
            "AAM smoke index filename must be exactly "
            f"{APPROVED_TRAIN_INDEX_FILENAME!r}"
        )
    root = Path(project_root).resolve()
    approved_cache = (root / APPROVED_TRAIN_CACHE_RELATIVE).resolve()
    requested_cache = Path(cache_dir)
    if not requested_cache.is_absolute():
        requested_cache = root / requested_cache
    requested_cache = requested_cache.resolve()
    if requested_cache != approved_cache:
        raise ValueError("AAM smoke cache directory is outside the approved train cache")
    approved_index = approved_cache / APPROVED_TRAIN_INDEX_FILENAME
    requested_index = requested_cache / index_filename
    if requested_index != approved_index:
        raise ValueError("AAM smoke index path is not the approved train index")
    return {
        "split": "train",
        "cache_dir": approved_cache.relative_to(root).as_posix(),
        "index_path": approved_index.relative_to(root).as_posix(),
        "shard_directory": (
            approved_cache / APPROVED_TRAIN_SHARD_DIRECTORY
        ).relative_to(root).as_posix(),
    }


def validate_train_only_smoke_metadata(
    dataset: Any,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Fail closed on non-train metadata before any Dataset tensor iteration."""
    root = Path(project_root).resolve()
    approved_cache = (root / APPROVED_TRAIN_CACHE_RELATIVE).resolve()
    approved_shards = (approved_cache / APPROVED_TRAIN_SHARD_DIRECTORY).resolve()
    if getattr(dataset, "split", None) != "train":
        raise ValueError("AAM smoke Dataset split must be exactly 'train'")
    if Path(getattr(dataset, "cache_dir", "")).resolve() != approved_cache:
        raise ValueError("AAM smoke Dataset cache directory is not approved")
    rows = getattr(dataset, "rows", None)
    if not isinstance(rows, tuple) or not rows:
        raise ValueError("AAM smoke Dataset rows must be a non-empty tuple")
    shard_paths: set[str] = set()
    for index, row in enumerate(rows):
        if getattr(row, "final_split", None) != "train":
            raise ValueError(f"AAM smoke row {index} is not train")
        label = getattr(row, "speaker_label", None)
        if type(label) is not int or not 0 <= label <= 487:
            raise ValueError(f"AAM smoke row {index} label must be in 0..487")
        relative = getattr(row, "feature_shard_path", None)
        if not isinstance(relative, str):
            raise ValueError(f"AAM smoke row {index} has no shard path")
        pure = PurePosixPath(relative)
        if (
            len(pure.parts) != 2
            or pure.parts[0] != APPROVED_TRAIN_SHARD_DIRECTORY
            or not pure.parts[1].startswith("shard_")
            or not pure.parts[1].endswith(".pt")
            or len(pure.parts[1]) != len("shard_00000.pt")
            or not pure.parts[1][6:11].isdigit()
        ):
            raise ValueError(
                f"AAM smoke row {index} shard path is outside the train allowlist"
            )
        resolved = (approved_cache / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(approved_shards)
        except ValueError as error:
            raise ValueError(
                f"AAM smoke row {index} shard path escapes the train directory"
            ) from error
        shard_paths.add(relative)
    return {
        "rows_validated": len(rows),
        "labels_in_0_487": True,
        "all_final_split_train": True,
        "referenced_train_shards": sorted(shard_paths),
    }


def create_train_only_smoke_dataset(
    *,
    project_root: str | Path,
    cache_dir: str | Path,
    split: str = "train",
    index_filename: str = APPROVED_TRAIN_INDEX_FILENAME,
    max_cached_shards: int = 8,
    validate_finite: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Construct the cache Dataset only after the train request is allowlisted."""
    request = validate_train_only_smoke_request(
        project_root=project_root,
        cache_dir=cache_dir,
        split=split,
        index_filename=index_filename,
    )
    from src.cached_fbank_dataset import CachedFbankDataset

    dataset = CachedFbankDataset(
        Path(project_root).resolve() / request["cache_dir"],
        "train",
        max_cached_shards=max_cached_shards,
        validate_finite=validate_finite,
    )
    metadata = validate_train_only_smoke_metadata(dataset, project_root=project_root)
    return dataset, {"request": request, "metadata": metadata}


def _require_finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


class AAMSoftmax(nn.Module):
    """Project-owned additive angular-margin classifier without a bias."""

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        num_classes: int | None = None,
        margin: float = 0.2,
        scale: float = 30.0,
        seed: int = SEED,
    ) -> None:
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be derived from the train manifest")
        if embedding_dim < 1 or num_classes < 2:
            raise ValueError("embedding_dim and num_classes must be positive")
        if not math.isfinite(margin) or not 0.0 <= margin < math.pi / 2:
            raise ValueError("margin must be finite and in [0, pi/2)")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = float(margin)
        self.scale = float(scale)
        self.seed = seed
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.weight)
        self.register_buffer("_cos_m", torch.tensor(math.cos(margin), dtype=torch.float32))
        self.register_buffer("_sin_m", torch.tensor(math.sin(margin), dtype=torch.float32))
        self.register_buffer("_threshold", torch.tensor(math.cos(math.pi - margin), dtype=torch.float32))
        self.register_buffer(
            "_monotonic_offset",
            torch.tensor(math.sin(math.pi - margin) * margin, dtype=torch.float32),
        )

    def normalized_softmax_logits(self, embeddings: Tensor) -> Tensor:
        """Return scaled cosine logits without applying a target margin."""
        self._validate_embeddings(embeddings)
        embeddings_f32 = F.normalize(embeddings.float(), p=2, dim=1)
        weights_f32 = F.normalize(self.weight.float(), p=2, dim=1)
        cosine = F.linear(embeddings_f32, weights_f32).clamp(
            -1.0 + 1e-7, 1.0 - 1e-7
        )
        logits = cosine * self.scale
        _require_finite(logits, "normalized-softmax logits")
        return logits

    def _validate_embeddings(self, embeddings: Tensor) -> None:
        if not isinstance(embeddings, Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings must have shape [B, {self.embedding_dim}], "
                f"got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] < 1:
            raise ValueError("embeddings must contain at least one sample")
        if not embeddings.is_floating_point():
            raise TypeError("embeddings must be floating point")
        _require_finite(embeddings, "embeddings")
        _require_finite(self.weight, "AAM weight")

    def _validate_labels(self, labels: Tensor, batch_size: int) -> None:
        if not isinstance(labels, Tensor):
            raise TypeError("labels must be a torch.Tensor")
        if labels.dtype != torch.long:
            raise TypeError("labels must have dtype torch.long")
        if labels.ndim != 1 or tuple(labels.shape) != (batch_size,):
            raise ValueError(f"labels must have shape [{batch_size}], got {tuple(labels.shape)}")
        if bool(((labels < 0) | (labels >= self.num_classes)).any().item()):
            raise ValueError(f"labels must be in 0..{self.num_classes - 1}")

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        self._validate_embeddings(embeddings)
        self._validate_labels(labels, embeddings.shape[0])
        embeddings_f32 = F.normalize(embeddings.float(), p=2, dim=1)
        weights_f32 = F.normalize(self.weight.float(), p=2, dim=1)
        cosine = F.linear(embeddings_f32, weights_f32).clamp(
            -1.0 + 1e-7, 1.0 - 1e-7
        )
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=0.0))
        phi = cosine * self._cos_m - sine * self._sin_m
        phi = torch.where(
            cosine > self._threshold,
            phi,
            cosine - self._monotonic_offset,
        )
        target = F.one_hot(labels, num_classes=self.num_classes).to(torch.bool)
        logits = torch.where(target, phi, cosine) * self.scale
        if tuple(logits.shape) != (embeddings.shape[0], self.num_classes):
            raise RuntimeError("AAM logits have an unexpected shape")
        _require_finite(logits, "AAM logits")
        return logits


def apply_batchnorm_policy(embedding_model: nn.Module) -> tuple[str, ...]:
    """Train the encoder while keeping only BatchNorm running buffers frozen."""
    embedding_model.train()
    batchnorm_names: list[str] = []
    for name, module in embedding_model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            batchnorm_names.append(name)
            if module.affine:
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
    for parameter in embedding_model.parameters():
        parameter.requires_grad_(True)
    if not all(parameter.requires_grad for parameter in embedding_model.parameters()):
        raise AssertionError("not all ECAPA embedding-model parameters require gradients")
    return tuple(batchnorm_names)


def batchnorm_running_state(embedding_model: nn.Module) -> dict[str, Tensor]:
    """Clone every BatchNorm running-statistics buffer."""
    state: dict[str, Tensor] = {}
    for name, module in embedding_model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, buffer_name, None)
                if value is not None:
                    state[f"{name}.{buffer_name}"] = value.detach().cpu().clone()
    return state


def assert_batchnorm_running_state_exact(
    embedding_model: nn.Module, expected: Mapping[str, Tensor]
) -> None:
    actual = batchnorm_running_state(embedding_model)
    if set(actual) != set(expected):
        raise AssertionError("BatchNorm running-buffer keys changed")
    changed = [name for name in actual if not torch.equal(actual[name], expected[name])]
    if changed:
        raise AssertionError(f"BatchNorm running buffers changed: {changed[:3]}")


def batchnorm_parameter_names(embedding_model: nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in embedding_model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            if module.weight is not None:
                names.add(prefix + "weight")
            if module.bias is not None:
                names.add(prefix + "bias")
    return names


def build_adamw_optimizer(
    embedding_model: nn.Module,
    aam_classifier: AAMSoftmax,
    *,
    embedding_lr: float = 1e-5,
    classifier_lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "embedding_model",
                "params": list(embedding_model.parameters()),
                "lr": embedding_lr,
                "weight_decay": weight_decay,
            },
            {
                "name": "aam_classifier",
                "params": list(aam_classifier.parameters()),
                "lr": classifier_lr,
                "weight_decay": weight_decay,
            },
        ]
    )
    validate_optimizer_coverage(optimizer, embedding_model, aam_classifier)
    return optimizer


def validate_optimizer_coverage(
    optimizer: torch.optim.Optimizer,
    embedding_model: nn.Module,
    aam_classifier: AAMSoftmax,
) -> None:
    if len(optimizer.param_groups) != 2:
        raise ValueError("optimizer must have exactly two parameter groups")
    expected_groups = [
        list(embedding_model.parameters()),
        list(aam_classifier.parameters()),
    ]
    seen: set[int] = set()
    for number, (group, expected) in enumerate(zip(optimizer.param_groups, expected_groups)):
        actual = list(group["params"])
        if {id(parameter) for parameter in actual} != {id(parameter) for parameter in expected}:
            raise ValueError(f"optimizer parameter group {number} has incorrect coverage")
        for parameter in actual:
            if id(parameter) in seen:
                raise ValueError("a parameter appears in more than one optimizer group")
            seen.add(id(parameter))
    intended = {id(parameter) for parameters in expected_groups for parameter in parameters}
    if seen != intended:
        raise ValueError("optimizer does not cover all intended parameters exactly once")


def round_robin_reorder(
    batch: Mapping[str, Any],
    *,
    speakers_per_batch: int = 16,
    samples_per_speaker: int = 2,
) -> dict[str, Any]:
    """Copy a logical P x K batch into utterance-rank-major speaker order."""
    required = {"fbank", "speaker_label", "speaker_id", "dataset_index"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"logical batch is missing fields: {sorted(missing)}")
    size = speakers_per_batch * samples_per_speaker
    labels = batch["speaker_label"]
    indexes = batch["dataset_index"]
    speakers = batch["speaker_id"]
    features = batch["fbank"]
    if (
        not isinstance(features, Tensor)
        or features.ndim < 1
        or features.shape[0] != size
        or not isinstance(labels, Tensor)
        or labels.dtype != torch.long
        or tuple(labels.shape) != (size,)
        or not isinstance(indexes, Tensor)
        or indexes.dtype != torch.long
        or tuple(indexes.shape) != (size,)
        or not isinstance(speakers, Sequence)
        or len(speakers) != size
    ):
        raise ValueError("logical batch has malformed aligned fields")
    if len(set(indexes.tolist())) != size:
        raise ValueError("logical batch contains duplicate Dataset indexes")

    label_to_positions: dict[int, list[int]] = {}
    label_to_speaker: dict[int, str] = {}
    for position, (label, speaker) in enumerate(zip(labels.tolist(), speakers)):
        if not isinstance(speaker, str) or not speaker:
            raise ValueError("speaker IDs must be non-empty strings")
        label_to_positions.setdefault(label, []).append(position)
        previous = label_to_speaker.setdefault(label, speaker)
        if previous != speaker:
            raise ValueError("one speaker label maps to multiple speaker IDs")
    if len(label_to_positions) != speakers_per_batch:
        raise ValueError(f"logical batch must contain exactly {speakers_per_batch} speakers")
    if any(len(positions) != samples_per_speaker for positions in label_to_positions.values()):
        raise ValueError(f"each speaker must occur exactly {samples_per_speaker} times")
    ordered_labels = sorted(label_to_positions)
    order = [
        label_to_positions[label][utterance]
        for utterance in range(samples_per_speaker)
        for label in ordered_labels
    ]
    if sorted(order) != list(range(size)):
        raise AssertionError("round-robin order dropped or duplicated a sample")

    reordered: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            if value.ndim < 1 or value.shape[0] != size:
                raise ValueError(f"tensor field {key!r} is not batch-aligned")
            reordered[key] = value[order]
        elif isinstance(value, list):
            if len(value) != size:
                raise ValueError(f"list field {key!r} is not batch-aligned")
            reordered[key] = [value[position] for position in order]
        elif isinstance(value, tuple):
            if len(value) != size:
                raise ValueError(f"tuple field {key!r} is not batch-aligned")
            reordered[key] = tuple(value[position] for position in order)
        else:
            raise TypeError(f"unsupported batch field {key!r}: {type(value).__name__}")
    if sorted(reordered["dataset_index"].tolist()) != sorted(indexes.tolist()):
        raise AssertionError("round-robin reordering changed Dataset identities")
    return reordered


def iter_microbatches(
    batch: Mapping[str, Any], microbatch_size: int
) -> Iterator[dict[str, Any]]:
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    size = int(batch["fbank"].shape[0])
    if size % microbatch_size:
        raise ValueError("logical batch size must be divisible by microbatch size")
    for start in range(0, size, microbatch_size):
        stop = start + microbatch_size
        result: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                result[key] = value[start:stop]
            elif isinstance(value, list):
                result[key] = value[start:stop]
            elif isinstance(value, tuple):
                result[key] = value[start:stop]
            else:
                raise TypeError(f"unsupported batch field {key!r}")
        if len(set(result["speaker_id"])) != microbatch_size:
            raise ValueError("a physical microbatch contains duplicate speakers")
        yield result


def scaled_cross_entropy_sum(logits: Tensor, labels: Tensor, logical_batch_size: int) -> Tensor:
    if logical_batch_size < 1:
        raise ValueError("logical_batch_size must be positive")
    return F.cross_entropy(logits.float(), labels, reduction="sum") / logical_batch_size


def batch_at_position(batch_sampler: Iterable[Sequence[int]], position: int) -> list[int]:
    if not isinstance(position, int) or position < 0:
        raise ValueError("batch position must be a non-negative integer")
    for current, batch in enumerate(batch_sampler):
        if current == position:
            result = list(batch)
            if not result or len(result) != len(set(result)):
                raise ValueError("positioned batch is empty or contains duplicate indexes")
            return result
    raise IndexError(f"batch position {position} is outside the sampler")


def aggregate_gradient_norm(
    parameters: Iterable[nn.Parameter] | Iterable[tuple[str, nn.Parameter]],
    name: str,
) -> float:
    total = 0.0
    count = 0
    for number, item in enumerate(parameters):
        if isinstance(item, tuple):
            parameter_name, parameter = item
        else:
            parameter_name, parameter = str(number), item
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(
                f"{name} parameter {parameter_name!r} has no gradient"
            )
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            nan_count = int(torch.isnan(gradient).sum().item())
            inf_count = int(torch.isinf(gradient).sum().item())
            raise ValueError(
                f"{name} parameter {parameter_name!r} gradient contains "
                f"{nan_count} NaN and {inf_count} Inf values"
            )
        total += float(gradient.float().square().sum().item())
        count += 1
    if count == 0:
        raise AssertionError(f"{name} has no trainable parameters")
    norm = math.sqrt(total)
    if not math.isfinite(norm) or norm <= 0.0:
        raise AssertionError(f"{name} aggregate gradient norm is not finite and positive")
    return norm


def cpu_clone_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def parameter_delta(
    module: nn.Module,
    before: Mapping[str, Tensor],
    *,
    exclude: set[str] | None = None,
) -> dict[str, float]:
    excluded = exclude or set()
    maximum = 0.0
    aggregate = 0.0
    compared = 0
    for name, parameter in module.named_parameters():
        if name in excluded:
            continue
        if name not in before:
            raise AssertionError(f"missing before-state parameter {name}")
        difference = parameter.detach().float().cpu() - before[name].float()
        _require_finite(difference, f"{name} parameter delta")
        maximum = max(maximum, float(difference.abs().max().item()))
        aggregate += float(difference.abs().sum().item())
        compared += 1
    if compared == 0 or maximum <= 0.0 or aggregate <= 0.0:
        raise AssertionError("parameter update did not produce a finite nonzero change")
    return {"maximum_absolute_delta": maximum, "aggregate_absolute_delta": aggregate}


def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    return value


def values_exactly_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            values_exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_exactly_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if include_cuda and torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: Mapping[str, Any], *, restore_cuda: bool = True) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("RNG state has incorrect keys")
    if not isinstance(state["torch_cpu"], Tensor) or state["torch_cpu"].dtype != torch.uint8:
        raise ValueError("malformed torch CPU RNG state")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or any(
        not isinstance(item, Tensor) or item.dtype != torch.uint8 for item in cuda_states
    ):
        raise ValueError("malformed torch CUDA RNG state")
    if restore_cuda and torch.cuda.is_available() and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("CUDA RNG state count does not match available devices")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if restore_cuda and cuda_states:
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_path(value: Any, description: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
        or ":" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f"unsafe {description}: {value!r}")
    return value


CHECKPOINT_KEYS = {
    "schema_name",
    "schema_version",
    "pretrained_model_identifier",
    "embedding_model_state_dict",
    "mean_var_norm_state_dict",
    "aam_classifier_state_dict",
    "optimizer_state_dict",
    "grad_scaler_state_dict",
    "epoch",
    "next_logical_batch_position",
    "global_optimizer_step",
    "sampler",
    "physical_microbatch_size",
    "accumulation_steps",
    "aam",
    "optimizer_groups",
    "batchnorm_policy",
    "amp_policy",
    "train_data_identity",
    "next_logical_batch_identity",
    "rng_state",
}


def validate_checkpoint_v1(checkpoint: Mapping[str, Any]) -> None:
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != CHECKPOINT_KEYS:
        raise ValueError("checkpoint has incorrect top-level keys")
    if (
        checkpoint["schema_name"] != CHECKPOINT_SCHEMA_NAME
        or checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or checkpoint["pretrained_model_identifier"] != PRETRAINED_MODEL_ID
    ):
        raise ValueError("checkpoint schema or pretrained model identifier is invalid")
    for key in (
        "embedding_model_state_dict",
        "mean_var_norm_state_dict",
        "aam_classifier_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"{key} must be a mapping")
    if (
        checkpoint["epoch"] != 0
        or checkpoint["next_logical_batch_position"] != 1
        or checkpoint["global_optimizer_step"] != 1
    ):
        raise ValueError("checkpoint counters do not describe the post-step-1 smoke state")
    expected_sampler = {
        "name": "HybridShardAwareSpeakerBatchSampler",
        "seed": SEED,
        "epoch": 0,
        "speakers_per_batch": 16,
        "samples_per_speaker": 2,
        "active_shard_window": 8,
        "logical_batch_size": 32,
    }
    if checkpoint["sampler"] != expected_sampler:
        raise ValueError("checkpoint sampler configuration is invalid")
    if checkpoint["physical_microbatch_size"] != 2 or checkpoint["accumulation_steps"] != 16:
        raise ValueError("checkpoint microbatch configuration is invalid")
    if checkpoint["aam"] != {
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": LEGACY_SMOKE_NUM_CLASSES,
        "margin_radians": 0.2,
        "scale": 30.0,
        "initialization_seed": SEED,
    }:
        raise ValueError("checkpoint AAM configuration is invalid")
    if checkpoint["optimizer_groups"] != [
        {"name": "embedding_model", "learning_rate": 1e-5, "weight_decay": 1e-4},
        {"name": "aam_classifier", "learning_rate": 1e-3, "weight_decay": 1e-4},
    ]:
        raise ValueError("checkpoint optimizer configuration is invalid")
    if checkpoint["batchnorm_policy"] != {
        "embedding_model_training": True,
        "batchnorm_modules_eval": True,
        "batchnorm_affine_trainable": True,
        "running_buffers_exactly_frozen": True,
    }:
        raise ValueError("checkpoint BatchNorm policy is invalid")
    if checkpoint["amp_policy"] != {
        "enabled": True,
        "ecapa_autocast_dtype": "float16",
        "aam_math_dtype": "float32",
        "loss_dtype": "float32",
        "grad_scaler_initial_scale": 128.0,
    }:
        raise ValueError("checkpoint AMP policy is invalid")
    identity = checkpoint["train_data_identity"]
    if not isinstance(identity, Mapping) or set(identity) != {"files"}:
        raise ValueError("checkpoint train-data identity is malformed")
    files = identity["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint train-data identity must contain files")
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError("checkpoint train-data file identity is malformed")
        _validate_relative_path(item["path"], "train-data path")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("checkpoint train-data hash is malformed")
    next_identity = checkpoint["next_logical_batch_identity"]
    if not isinstance(next_identity, Mapping) or set(next_identity) != {
        "dataset_indexes", "speaker_ids", "relative_audio_paths"
    }:
        raise ValueError("checkpoint next-batch identity is malformed")
    indexes = next_identity["dataset_indexes"]
    speakers = next_identity["speaker_ids"]
    paths = next_identity["relative_audio_paths"]
    if (
        not isinstance(indexes, list)
        or len(indexes) != 32
        or len(set(indexes)) != 32
        or any(not isinstance(value, int) or value < 0 for value in indexes)
        or not isinstance(speakers, list)
        or len(speakers) != 32
        or any(not isinstance(value, str) or not value for value in speakers)
        or not isinstance(paths, list)
        or len(paths) != 32
    ):
        raise ValueError("checkpoint next-batch identity fields are invalid")
    for path in paths:
        _validate_relative_path(path, "next-batch audio path")
    rng = checkpoint["rng_state"]
    if not isinstance(rng, Mapping) or set(rng) != {
        "python", "numpy", "torch_cpu", "torch_cuda"
    }:
        raise ValueError("checkpoint RNG state is malformed")
    if not isinstance(rng["torch_cpu"], Tensor) or rng["torch_cpu"].dtype != torch.uint8:
        raise ValueError("checkpoint torch CPU RNG state is malformed")
    if not isinstance(rng["torch_cuda"], list) or any(
        not isinstance(state, Tensor) or state.dtype != torch.uint8
        for state in rng["torch_cuda"]
    ):
        raise ValueError("checkpoint torch CUDA RNG state is malformed")


def atomic_save_checkpoint(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    validate_checkpoint_v1(checkpoint)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(dict(checkpoint), temporary)
    os.replace(temporary, target)
