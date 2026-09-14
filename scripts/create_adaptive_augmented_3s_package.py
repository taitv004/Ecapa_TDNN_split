#!/usr/bin/env python3
"""Import a frozen common manifest without changing its speaker split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_package import (
    INTERNAL_MANIFEST_FIELDS,
    read_common_manifest,
    validate_training_rows,
)
from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    generate_validation_trials,
    sha256_bytes,
    trial_recording_statistics,
    trials_csv_bytes,
)


PACKAGE_VERSION = "adaptive_augmented_3s_v1"
FULL_MANIFEST = "manifests/adaptive_augmented_3s_v1_full_manifest.csv"
DATASET_IDENTITY = "manifests/adaptive_augmented_3s_v1_dataset_identity.json"
TRAIN_MANIFEST = "manifests/adaptive_augmented_3s_v1_train_manifest.csv"
VALIDATION_MANIFEST = "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
LABEL_MAPPING = "manifests/adaptive_augmented_3s_v1_speaker_to_label.json"
SPEAKER_SPLIT = "splits/adaptive_augmented_3s_v1_speaker_split.csv"
SPLIT_IDENTITY = "splits/adaptive_augmented_3s_v1_split_identity.json"
VALIDATION_TRIALS = (
    "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
)
VALIDATION_TRIAL_IDENTITY = (
    "manifests/verification/"
    "adaptive_augmented_3s_v1_validation_trials_identity.json"
)
SPLIT_FIELDS = ("speaker_id", "source_stratum", "split")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def render_csv(
    fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def publish(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def create_package(
    manifest_path: Path,
    project_root: Path,
    *,
    dataset_root: Path | None,
    check_audio_exists: bool,
    trial_seed: int,
    genuine_trials: int,
    impostor_trials: int,
    overwrite: bool,
) -> dict[str, Any]:
    source = manifest_path.expanduser().resolve(strict=True)
    mapped, labels, summary = read_common_manifest(
        source,
        dataset_root=dataset_root,
        check_audio_exists=check_audio_exists,
    )
    validate_training_rows(mapped, labels)
    validation_rows = tuple(
        ValidationRow(
            row["sample_id"],
            row["relative_audio_path"],
            row["speaker_id"],
            row["source_dataset"],
            row["source_recording_id"],
        )
        for row in mapped["validation"]
    )
    trials = generate_validation_trials(
        validation_rows,
        seed=trial_seed,
        genuine_count=genuine_trials,
        impostor_count=impostor_trials,
    )
    trial_payload = trials_csv_bytes(trials)

    speaker_sources: dict[str, set[str]] = {}
    speaker_split: dict[str, str] = {}
    for split in ("train", "validation"):
        for row in mapped[split]:
            speaker = row["speaker_id"]
            speaker_sources.setdefault(speaker, set()).add(row["source_dataset"])
            speaker_split[speaker] = split
    split_rows = [
        {
            "speaker_id": speaker,
            "source_stratum": "+".join(sorted(speaker_sources[speaker])),
            "split": speaker_split[speaker],
        }
        for speaker in sorted(speaker_split)
    ]
    payloads = {
        FULL_MANIFEST: source.read_bytes(),
        TRAIN_MANIFEST: render_csv(
            INTERNAL_MANIFEST_FIELDS, mapped["train"]
        ),
        VALIDATION_MANIFEST: render_csv(
            INTERNAL_MANIFEST_FIELDS, mapped["validation"]
        ),
        LABEL_MAPPING: canonical_json(labels),
        SPEAKER_SPLIT: render_csv(SPLIT_FIELDS, split_rows),
        VALIDATION_TRIALS: trial_payload,
    }
    dataset_identity = {
        "schema_version": 4,
        "identity_kind": "imported_frozen_common_manifest",
        "package_version": PACKAGE_VERSION,
        "authoritative_manifest_sha256": summary["manifest_sha256"],
        "row_counts": summary["row_counts"],
        "speaker_counts": summary["speaker_counts"],
        "speaker_overlap": 0,
        "source_summary": summary["source_summary"],
        "path_normalization": summary["path_normalization"],
        "split_reused_without_regeneration": True,
    }
    payloads[DATASET_IDENTITY] = canonical_json(dataset_identity)
    split_identity = {
        "schema_version": 4,
        "identity_kind": "imported_frozen_speaker_disjoint_split",
        "authoritative_manifest_sha256": summary["manifest_sha256"],
        "speaker_counts": summary["speaker_counts"],
        "row_counts": summary["row_counts"],
        "speaker_overlap": 0,
        "speaker_split_sha256": sha256_bytes(payloads[SPEAKER_SPLIT]),
        "train_manifest_sha256": sha256_bytes(payloads[TRAIN_MANIFEST]),
        "validation_manifest_sha256": sha256_bytes(
            payloads[VALIDATION_MANIFEST]
        ),
        "speaker_to_label_sha256": sha256_bytes(payloads[LABEL_MAPPING]),
        "train_label_range": summary["train_label_range"],
    }
    payloads[SPLIT_IDENTITY] = canonical_json(split_identity)
    trial_identity = {
        "schema_version": 4,
        "identity_kind": "frozen_validation_trials",
        "authoritative_manifest_sha256": summary["manifest_sha256"],
        "seed": trial_seed,
        "speaker_count": summary["speaker_counts"]["validation"],
        "trial_counts": {
            "genuine": genuine_trials,
            "impostor": impostor_trials,
            "total": genuine_trials + impostor_trials,
        },
        "recording_pairing": trial_recording_statistics(trials, validation_rows),
        "trial_csv_sha256": sha256_bytes(trial_payload),
        "validation_audio_augmented": False,
    }
    trial_identity["identity_sha256"] = sha256_bytes(
        canonical_json(trial_identity)
    )
    payloads[VALIDATION_TRIAL_IDENTITY] = canonical_json(trial_identity)
    for relative, payload in payloads.items():
        publish(project_root / relative, payload, overwrite)
    return {
        "result": "PASS",
        "split_reused_without_regeneration": True,
        "row_counts": summary["row_counts"],
        "speaker_counts": summary["speaker_counts"],
        "speaker_overlap": 0,
        "train_classes": len(labels),
        "validation_trials": trial_identity["trial_counts"],
        "recording_pairing": trial_identity["recording_pairing"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--check-audio-exists", action="store_true")
    parser.add_argument("--trial-seed", type=int, default=2026)
    parser.add_argument("--genuine-trials", type=int, default=10_000)
    parser.add_argument("--impostor-trials", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.check_audio_exists and args.dataset_root is None:
        parser.error("--check-audio-exists requires --dataset-root")
    result = create_package(
        args.manifest,
        ROOT,
        dataset_root=args.dataset_root,
        check_audio_exists=args.check_audio_exists,
        trial_seed=args.trial_seed,
        genuine_trials=args.genuine_trials,
        impostor_trials=args.impostor_trials,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
