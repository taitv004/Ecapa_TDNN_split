"""Evaluate ``best.pt`` on the independent final-test FBank cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.adaptive_augmented_3s_training import (
    CachedDataset,
    collate,
    read_manifest,
    read_trials,
)
from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    sha256_file,
    validate_validation_trials,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.verification_metrics import calculate_eer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adaptive_augmented_3s_final_evaluation_v1.json"
TEST_MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
TEST_TRIALS = (
    ROOT / "manifests/verification/adaptive_augmented_3s_v1_final_test_trials.csv"
)
TEST_CACHE_CONFIG = "fbank_cache_config_adaptive_augmented_3s_v1_final_test.json"
TEST_CACHE_IDENTITY = "fbank_cache_identity_adaptive_augmented_3s_v1_final_test.json"


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_defaults(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 2:
        raise ValueError("Final-evaluation config schema_version must be 2")
    return value


def validate_test_cache(
    cache_root: Path, row_count: int
) -> tuple[int, Mapping[str, Any]]:
    config_path = cache_root / TEST_CACHE_CONFIG
    identity_path = cache_root / TEST_CACHE_IDENTITY
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        config.get("expected_rows") != {"final_test": row_count}
        or config.get("feature_shape") != [301, 80]
        or identity.get("row_count") != row_count
        or identity.get("config_sha256") != sha256_file(config_path)
        or config.get("input_bindings", {})
        .get("final_test_manifest", {})
        .get("sha256")
        != sha256_file(TEST_MANIFEST)
        or identity.get("input_bindings") != config.get("input_bindings")
    ):
        raise ValueError("Final-test cache disagrees with the current test manifest")
    shard_size = int(config["shard_size"])
    if shard_size < 1:
        raise ValueError("Invalid final-test shard size")
    return shard_size, identity


def evaluate(
    checkpoint_path: Path,
    cache_root: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    if checkpoint_path.name != "best.pt":
        raise ValueError("Final test must use the validation-selected best.pt")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    rows = read_manifest(TEST_MANIFEST, "final_test")
    trials = read_trials(TEST_TRIALS)
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
    train_validation_speakers: set[str] = set()
    for path, split in (
        (ROOT / "manifests/adaptive_augmented_3s_v1_train_manifest.csv", "train"),
        (
            ROOT / "manifests/adaptive_augmented_3s_v1_validation_manifest.csv",
            "validation",
        ),
    ):
        train_validation_speakers.update(
            row["speaker_id"] for row in read_manifest(path, split)
        )
    overlap = train_validation_speakers & {row["speaker_id"] for row in rows}
    if overlap:
        raise ValueError(
            "Speaker leakage into final test: " + ", ".join(sorted(overlap)[:20])
        )
    shard_size, cache_identity = validate_test_cache(cache_root, len(rows))
    dataset = CachedDataset(rows, cache_root, "final_test", shard_size, 2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "generic_ecapa_aam_training_v3":
        raise ValueError("Checkpoint is not produced by the generic training pipeline")
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

    embeddings: dict[str, torch.Tensor] = {}
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
            embeddings.update(zip(batch["sample_id"], value))
            if number % 25 == 0 or number == len(loader):
                print(f"TEST batch={number}/{len(loader)}", flush=True)

    score_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    targets: list[int] = []
    for trial in trials:
        try:
            score = float(
                torch.dot(
                    embeddings[trial.left_sample_id],
                    embeddings[trial.right_sample_id],
                )
            )
        except KeyError as error:
            raise ValueError(
                f"Test trial sample ID is missing from cache: {error}"
            ) from error
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
    result = {
        "checkpoint": str(checkpoint_path),
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "test_speakers": len({row["speaker_id"] for row in rows}),
        "test_rows": len(rows),
        "trial_count": len(trials),
        "eer": float(eer.interpolated_eer),
        "eer_percentage": float(eer.interpolated_eer_percentage),
        "eer_threshold_descriptive_only": float(eer.empirical_threshold),
        "threshold_note": (
            "The final-test EER threshold is descriptive only. Select an "
            "operational threshold on validation data."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_scores.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("trial_id", "target", "score"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(score_rows)
    (output_dir / "test_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Backward-compatible flag; evaluation is the default action.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    defaults = load_defaults(args.config.expanduser().resolve(strict=True))
    checkpoint = (
        args.checkpoint.expanduser().resolve(strict=True)
        if args.checkpoint
        else resolve_from_root(defaults["checkpoint"])
    )
    cache_root = (
        args.cache_root.expanduser().resolve(strict=True)
        if args.cache_root
        else resolve_from_root(defaults["cache_root"])
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else resolve_from_root(defaults["output_dir"])
    )
    device = args.device or defaults["device"]
    batch_size = args.batch_size or int(defaults["batch_size"])
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    result = evaluate(checkpoint, cache_root, output_dir, device, batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
