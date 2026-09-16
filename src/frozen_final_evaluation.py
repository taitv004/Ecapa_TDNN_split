"""Independent final-test evaluation for frozen-handoff ECAPA checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.adaptive_augmented_3s_training import (
    EMBEDDING_DIM,
    CachedDataset,
    collate,
)
from src.frozen_handoff_cache import (
    FrozenVerificationProtocol,
    canonical_digest,
    read_cache_artifact,
    read_frozen_verification_protocol,
    sha256_file,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_metrics import calculate_eer


CHECKPOINT_SCHEMA = "frozen_handoff_ecapa_aam_training_v1"
PROTOCOL_IDENTITY_KIND = "frozen_independent_final_test_protocol"


def validate_checkpoint_payload(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            "Checkpoint is not produced by the frozen-handoff training pipeline"
        )
    required = {
        "embedding_model_state_dict",
        "mean_var_norm_state_dict",
        "runtime_config",
        "cursor",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is missing evaluation state: "
            + ", ".join(sorted(missing))
        )
    runtime = checkpoint.get("runtime_config")
    if not isinstance(runtime, Mapping):
        raise ValueError("Checkpoint runtime_config is malformed")
    model = runtime.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Checkpoint model configuration is malformed")
    if model.get("source") != SpeechBrainECAPAFrontend.SOURCE:
        raise ValueError("Checkpoint ECAPA source differs from evaluator frontend")
    if int(model.get("embedding_dim", -1)) != EMBEDDING_DIM:
        raise ValueError("Checkpoint embedding dimension is incompatible")


def load_protocol_identity(
    path: Path,
    *,
    cache_identity: Mapping[str, Any],
    protocol: FrozenVerificationProtocol,
    sample_count: int,
    speaker_count: int,
) -> Mapping[str, Any]:
    identity_path = path.expanduser().resolve(strict=True)
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    check = dict(identity)
    digest = check.pop("identity_sha256", None)
    if digest != canonical_digest(check):
        raise ValueError("Final-test protocol identity digest is invalid")
    if (
        identity.get("schema_version") != 1
        or identity.get("identity_kind") != PROTOCOL_IDENTITY_KIND
    ):
        raise ValueError("Unsupported final-test protocol identity")
    counts = identity.get("trial_counts", {})
    positives = int(protocol.targets.sum())
    negatives = int(len(protocol) - positives)
    if (
        identity.get("manifest_sha256") != cache_identity.get("manifest_sha256")
        or identity.get("trials_sha256") != protocol.sha256
        or int(identity.get("sample_count", -1)) != sample_count
        or int(identity.get("speaker_count", -1)) != speaker_count
        or int(counts.get("total", -1)) != len(protocol)
        or int(counts.get("target", -1)) != positives
        or int(counts.get("nontarget", -1)) != negatives
    ):
        raise ValueError(
            "Final-test protocol identity does not bind to this cache/trial set"
        )
    return identity


def score_trials(
    embeddings: torch.Tensor,
    protocol: FrozenVerificationProtocol,
    *,
    chunk_size: int = 100_000,
) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding tensor [N,{EMBEDDING_DIM}], got "
            f"{tuple(embeddings.shape)}"
        )
    if len(protocol) < 1:
        raise ValueError("Final-test protocol is empty")
    chunks: list[np.ndarray] = []
    for start in range(0, len(protocol), chunk_size):
        stop = min(start + chunk_size, len(protocol))
        enroll = torch.from_numpy(
            protocol.enroll_indices[start:stop].astype(np.int64, copy=False)
        )
        test = torch.from_numpy(
            protocol.test_indices[start:stop].astype(np.int64, copy=False)
        )
        values = (embeddings[enroll] * embeddings[test]).sum(dim=1)
        chunks.append(values.numpy().astype(np.float64, copy=False))
    return np.concatenate(chunks)


def evaluate(
    *,
    checkpoint_path: Path,
    test_cache_root: Path,
    test_trials_path: Path,
    protocol_identity_path: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    if checkpoint_path.name != "best.pt":
        raise ValueError("Final test must use the validation-selected best.pt")
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    artifact = read_cache_artifact(test_cache_root, "final_test")
    protocol = read_frozen_verification_protocol(
        test_trials_path,
        artifact.rows,
        expected_split="final_test",
        verify_unique_pairs=True,
    )
    speaker_count = len({row.speaker_id for row in artifact.rows})
    protocol_identity = load_protocol_identity(
        protocol_identity_path,
        cache_identity=artifact.identity,
        protocol=protocol,
        sample_count=len(artifact.rows),
        speaker_count=speaker_count,
    )

    dataset = CachedDataset(artifact, max_cached_shards=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(checkpoint)
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm
    embedding_model = frontend.classifier.mods.embedding_model
    del frontend
    mean_var_norm.load_state_dict(checkpoint["mean_var_norm_state_dict"], strict=True)
    embedding_model.load_state_dict(
        checkpoint["embedding_model_state_dict"], strict=True
    )
    mean_var_norm.eval().to(device)
    embedding_model.eval().to(device)

    embeddings = torch.empty(
        (len(dataset), EMBEDDING_DIM), dtype=torch.float32, device="cpu"
    )
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            features = batch["fbank"].to(device=device, dtype=torch.float32)
            lengths = torch.ones(features.shape[0], device=device)
            normalized = mean_var_norm(features, lengths)
            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda", dtype=torch.float16
            ):
                value = embedding_model(normalized, lengths).squeeze(1).float()
            value = F.normalize(value, p=2, dim=1).cpu()
            embeddings[batch["dataset_index"]] = value
            if number % 25 == 0 or number == len(loader):
                print(f"TEST batch={number}/{len(loader)}", flush=True)

    scores = score_trials(embeddings, protocol)
    targets = protocol.targets.astype(np.int64, copy=False)
    metric = calculate_eer(scores.tolist(), targets.tolist())

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "test_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("trial_id", "label", "score"),
            lineterminator="\n",
        )
        writer.writeheader()
        ids = protocol.trial_ids or tuple(
            f"trial-{index:08d}" for index in range(len(protocol))
        )
        for trial_id, target, score in zip(ids, targets, scores):
            writer.writerow(
                {
                    "trial_id": trial_id,
                    "label": int(target),
                    "score": f"{float(score):.12g}",
                }
            )

    result = {
        "schema_version": 1,
        "evaluation_kind": "independent_frozen_final_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_best_validation_eer": checkpoint.get("cursor", {}).get(
            "best_eer"
        ),
        "test_cache_identity_sha256": artifact.identity["identity_sha256"],
        "test_manifest_sha256": artifact.identity["manifest_sha256"],
        "test_trials_sha256": protocol.sha256,
        "test_protocol_identity_sha256": protocol_identity["identity_sha256"],
        "test_rows": len(dataset),
        "test_speakers": speaker_count,
        "trial_count": len(protocol),
        "target_trials": int(targets.sum()),
        "nontarget_trials": int(len(targets) - int(targets.sum())),
        "eer": float(metric.interpolated_eer),
        "eer_percentage": float(metric.interpolated_eer_percentage),
        "eer_interpolated_threshold": float(metric.interpolated_threshold),
        "eer_empirical_threshold_descriptive_only": float(
            metric.empirical_threshold
        ),
        "threshold_note": (
            "Final-test thresholds are descriptive only. Any operational "
            "threshold must be selected/calibrated on validation data."
        ),
        "scores_csv": str(score_path),
    }
    (output_dir / "test_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-cache-root", type=Path, required=True)
    parser.add_argument("--test-trials", type=Path, required=True)
    parser.add_argument("--protocol-identity", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    identity = args.protocol_identity
    if identity is None:
        identity = args.test_trials.with_name("test_protocol_identity.json")
    result = evaluate(
        checkpoint_path=args.checkpoint,
        test_cache_root=args.test_cache_root,
        test_trials_path=args.test_trials,
        protocol_identity_path=identity,
        output_dir=args.output_dir,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
