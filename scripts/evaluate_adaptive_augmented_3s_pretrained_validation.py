#!/usr/bin/env python3
"""Evaluate untouched pretrained ECAPA on the frozen validation trials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_training import (
    CACHE_CONFIG_NAME,
    CACHE_IDENTITY_NAME,
    CachedDataset,
    FEATURE_SHAPE,
    collate,
    read_trials,
)
from src.adaptive_augmented_3s_package import read_common_manifest
from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    sha256_file,
    validate_validation_trials,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_metrics import calculate_eer


DEFAULT_CACHE = ROOT / "outputs/fbank_cache_adaptive_augmented_3s_v1"
DEFAULT_OUTPUT = ROOT / "outputs/pretrained_ecapa_validation_baseline_adaptive_augmented_3s_v1"


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"{path} exists; pass --overwrite")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def validate_cache(
    cache_root: Path,
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> tuple[int, Mapping[str, Any]]:
    config_path = cache_root / CACHE_CONFIG_NAME
    identity_path = cache_root / CACHE_IDENTITY_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    bindings = config.get("input_bindings", {})
    if (
        config.get("expected_rows", {}).get("validation") != len(rows)
        or config.get("feature_shape") != list(FEATURE_SHAPE)
        or bindings.get("authoritative_manifest", {}).get("sha256")
        != sha256_file(manifest_path)
        or identity.get("config_sha256") != sha256_file(config_path)
        or identity.get("input_bindings") != bindings
    ):
        raise ValueError("Validation cache disagrees with current manifest")
    return int(config["shard_size"]), identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-trials", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cache_root = args.cache_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    trials_path = args.validation_trials.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    mapped, _, _ = read_common_manifest(manifest_path)
    rows = mapped["validation"]
    trials = read_trials(trials_path)
    validation_rows = tuple(
        ValidationRow(
            row["sample_id"],
            row["relative_audio_path"],
            row["speaker_id"],
            row["source_dataset"],
            row["source_recording_id"],
        )
        for row in rows
    )
    validate_validation_trials(
        trials,
        validation_rows,
        genuine_count=sum(trial.target == 1 for trial in trials),
        impostor_count=sum(trial.target == 0 for trial in trials),
    )
    shard_size, cache_identity = validate_cache(
        cache_root, rows, manifest_path
    )
    dataset = CachedDataset(rows, cache_root, "validation", shard_size, 2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    device = torch.device(args.device)
    frontend = SpeechBrainECAPAFrontend(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm.eval().to(device)
    embedding_model = frontend.classifier.mods.embedding_model.eval().to(device)
    del frontend
    for module in (mean_var_norm, embedding_model):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for number, batch in enumerate(loader, start=1):
            features = batch["fbank"].to(device=device, dtype=torch.float32)
            lengths = torch.ones(features.shape[0], device=device)
            normalized = mean_var_norm(features, lengths)
            value = embedding_model(normalized, lengths).squeeze(1).float()
            value = F.normalize(value, p=2, dim=1).cpu()
            embeddings.update(zip(batch["sample_id"], value))
            if number % 25 == 0 or number == len(loader):
                print(f"PRETRAINED VALIDATION batch={number}/{len(loader)}", flush=True)

    scores: list[float] = []
    targets: list[int] = []
    score_rows: list[dict[str, Any]] = []
    for trial in trials:
        try:
            score = float(
                torch.dot(
                    embeddings[trial.left_sample_id],
                    embeddings[trial.right_sample_id],
                )
            )
        except KeyError as error:
            raise ValueError(f"Trial sample ID is absent from cache: {error}") from error
        scores.append(score)
        targets.append(trial.target)
        score_rows.append(
            {
                "trial_id": trial.trial_id,
                "target": trial.target,
                "score": f"{score:.12g}",
            }
        )
    eer = calculate_eer(scores, targets)

    score_path = output_dir / "validation_scores.csv"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=("trial_id", "target", "score"), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(score_rows)
    score_payload = stream.getvalue().encode("utf-8")
    atomic_write(score_path, score_payload, args.overwrite)
    result = {
        "schema_version": 2,
        "identity_kind": "dynamic_pretrained_ecapa_validation_baseline",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "authoritative_manifest_sha256": sha256_file(manifest_path),
        "validation_trials_sha256": sha256_file(trials_path),
        "validation_speakers": len({row["speaker_id"] for row in rows}),
        "validation_samples": len(rows),
        "trial_count": len(trials),
        "eer": float(eer.interpolated_eer),
        "eer_percentage": float(eer.interpolated_eer_percentage),
        "eer_threshold_descriptive": float(eer.empirical_threshold),
        "score_csv_sha256": hashlib.sha256(score_payload).hexdigest(),
        "final_test_access": False,
    }
    result["identity_sha256"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    atomic_write(
        output_dir / "validation_results.json",
        canonical_json(result),
        args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
