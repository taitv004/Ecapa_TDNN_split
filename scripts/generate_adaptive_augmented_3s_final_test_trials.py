#!/usr/bin/env python3
"""Create a manifest and fixed verification trials for a separate test set.

The test root must use ``<speaker_id>/<file>.wav``. It is never mixed into
the train/validation split. By default this script also rejects speaker IDs
that occur in the existing train/validation split CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_adaptive_augmented_3s_package import (
    PORTABLE_FIELDS,
    canonical_json,
    publish,
    render_csv,
    scan_dataset,
)
from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    generate_validation_trials,
    sha256_bytes,
    trials_csv_bytes,
)


MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
TRIALS = (
    ROOT
    / "manifests/verification/adaptive_augmented_3s_v1_final_test_trials.csv"
)
IDENTITY = (
    ROOT
    / "manifests/verification/"
    "adaptive_augmented_3s_v1_final_test_trials_identity.json"
)
TRAIN_VALIDATION_SPLIT = (
    ROOT / "splits/adaptive_augmented_3s_v1_speaker_split.csv"
)


def read_train_validation_speakers(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing train/validation split: {path}. "
            "Create train/validation manifests first."
        )
    speakers: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"speaker_id", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("Train/validation split CSV has an invalid schema")
        for line, row in enumerate(reader, start=2):
            speaker = row["speaker_id"].strip()
            split = row["split"].strip()
            if not speaker or split not in {"train", "validation"}:
                raise ValueError(f"Invalid split row at line {line}")
            if speaker in speakers:
                raise ValueError(f"Duplicate speaker in split CSV: {speaker}")
            speakers.add(speaker)
    return speakers


def create_test_package(
    dataset_root: Path,
    *,
    seed: int,
    genuine_trials: int,
    impostor_trials: int,
    overwrite: bool,
    skip_audio_contract_check: bool,
    allow_speaker_overlap: bool,
) -> dict[str, object]:
    rows = scan_dataset(dataset_root, skip_audio_contract_check)
    test_speakers = {row.speaker_id for row in rows}
    if len(test_speakers) < 2:
        raise ValueError("The test set needs at least two speakers")
    if not allow_speaker_overlap:
        overlap = test_speakers & read_train_validation_speakers(
            TRAIN_VALIDATION_SPLIT
        )
        if overlap:
            raise ValueError(
                "Speaker leakage between test and train/validation: "
                + ", ".join(sorted(overlap)[:20])
            )

    manifest_rows = [
        {
            "relative_audio_path": row.audio_path,
            "speaker_id": row.speaker_id,
            "speaker_label": -1,
            "final_split": "final_test",
        }
        for row in rows
    ]
    manifest_payload = render_csv(PORTABLE_FIELDS, manifest_rows)
    verification_rows = tuple(
        ValidationRow(row.audio_path, row.speaker_id) for row in rows
    )
    utterances_per_speaker: dict[str, int] = {}
    for row in rows:
        utterances_per_speaker[row.speaker_id] = (
            utterances_per_speaker.get(row.speaker_id, 0) + 1
        )
    if genuine_trials == 0:
        minimum_genuine_capacity = min(
            count * (count - 1) // 2
            for count in utterances_per_speaker.values()
        )
        genuine_trials = min(
            10_000, len(test_speakers) * minimum_genuine_capacity
        )
    if impostor_trials == 0:
        counts = list(utterances_per_speaker.values())
        impostor_capacity = sum(
            counts[left] * counts[right]
            for left in range(len(counts))
            for right in range(left + 1, len(counts))
        )
        # Keep headroom because endpoint-balanced speaker pairing can revisit
        # one speaker pair before exhausting the global cross-speaker capacity.
        impostor_trials = min(10_000, max(1, impostor_capacity // 2))
    if genuine_trials < 1 or impostor_trials < 1:
        raise ValueError(
            "The test set does not contain enough recordings for both trial types"
        )
    generated = generate_validation_trials(
        verification_rows,
        seed=seed,
        genuine_count=genuine_trials,
        impostor_count=impostor_trials,
    )
    trials = tuple(
        replace(
            trial,
            trial_id=f"adaptive-augmented-3s-v1-final-test-{index:05d}",
        )
        for index, trial in enumerate(generated)
    )
    trial_payload = trials_csv_bytes(trials)
    identity = {
        "schema_version": 2,
        "identity_kind": "generic_independent_final_test",
        "package_version": "adaptive_augmented_3s_v1",
        "path_base": "independent_test_dataset_root",
        "seed": seed,
        "speaker_count": len(test_speakers),
        "row_count": len(rows),
        "speaker_disjoint_from_train_validation": not allow_speaker_overlap,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "trial_counts": {
            "genuine": genuine_trials,
            "impostor": impostor_trials,
            "total": genuine_trials + impostor_trials,
        },
        "trial_csv_sha256": sha256_bytes(trial_payload),
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json(identity))
    publish(MANIFEST, manifest_payload, overwrite)
    publish(TRIALS, trial_payload, overwrite)
    publish(IDENTITY, canonical_json(identity), overwrite)
    return {
        "result": "PASS",
        "test_speakers": len(test_speakers),
        "test_rows": len(rows),
        "trials": identity["trial_counts"],
        "speaker_overlap": 0 if not allow_speaker_overlap else "not_checked",
        "manifest": str(MANIFEST),
        "trial_csv": str(TRIALS),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--genuine-trials", type=int, default=0,
        help="0 chooses the largest safe value up to 10000.",
    )
    parser.add_argument(
        "--impostor-trials", type=int, default=0,
        help="0 chooses the largest safe value up to 10000.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-audio-contract-check", action="store_true")
    parser.add_argument(
        "--allow-speaker-overlap",
        action="store_true",
        help="Disable the default train/validation speaker-overlap rejection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_test_package(
        args.dataset_root,
        seed=args.seed,
        genuine_trials=args.genuine_trials,
        impostor_trials=args.impostor_trials,
        overwrite=args.overwrite,
        skip_audio_contract_check=args.skip_audio_contract_check,
        allow_speaker_overlap=args.allow_speaker_overlap,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
