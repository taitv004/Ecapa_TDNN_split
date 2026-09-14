#!/usr/bin/env python3
"""Regenerate fixed validation trials from the current validation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_adaptive_augmented_3s_package import canonical_json, publish
from src.adaptive_augmented_3s_package import read_common_manifest
from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    generate_validation_trials,
    sha256_bytes,
    sha256_file,
    trial_recording_statistics,
    trials_csv_bytes,
)


TRIALS = (
    ROOT
    / "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
)
IDENTITY = (
    ROOT
    / "manifests/verification/"
    "adaptive_augmented_3s_v1_validation_trials_identity.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=TRIALS)
    parser.add_argument("--identity-output", type=Path, default=IDENTITY)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--genuine-trials", type=int, default=10_000)
    parser.add_argument("--impostor-trials", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    mapped, _, _ = read_common_manifest(manifest_path)
    rows = tuple(
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
        rows,
        seed=args.seed,
        genuine_count=args.genuine_trials,
        impostor_count=args.impostor_trials,
    )
    payload = trials_csv_bytes(trials)
    identity = {
        "schema_version": 3,
        "identity_kind": "frozen_validation_trials",
        "package_version": "adaptive_augmented_3s_v1",
        "seed": args.seed,
        "input_split": "validation",
        "authoritative_manifest_sha256": sha256_file(manifest_path),
        "speaker_count": len({row.speaker_id for row in rows}),
        "trial_counts": {
            "genuine": args.genuine_trials,
            "impostor": args.impostor_trials,
            "total": len(trials),
        },
        "recording_pairing": trial_recording_statistics(trials, rows),
        "trial_csv_sha256": sha256_bytes(payload),
        "validation_audio_augmented": False,
        "final_test_access": False,
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json(identity))
    output = args.output.expanduser().resolve()
    identity_output = args.identity_output.expanduser().resolve()
    publish(output, payload, args.overwrite)
    publish(identity_output, canonical_json(identity), args.overwrite)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
