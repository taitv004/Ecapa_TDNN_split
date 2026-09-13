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
from src.adaptive_augmented_3s_verification import (
    generate_validation_trials,
    read_validation_manifest,
    sha256_bytes,
    sha256_file,
    trials_csv_bytes,
)


MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--genuine-trials", type=int, default=10_000)
    parser.add_argument("--impostor-trials", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_validation_manifest(MANIFEST)
    trials = generate_validation_trials(
        rows,
        seed=args.seed,
        genuine_count=args.genuine_trials,
        impostor_count=args.impostor_trials,
    )
    payload = trials_csv_bytes(trials)
    identity = {
        "schema_version": 2,
        "identity_kind": "generic_validation_trials",
        "package_version": "adaptive_augmented_3s_v1",
        "seed": args.seed,
        "input_split": "validation",
        "validation_manifest_sha256": sha256_file(MANIFEST),
        "speaker_count": len({row.speaker_id for row in rows}),
        "trial_counts": {
            "genuine": args.genuine_trials,
            "impostor": args.impostor_trials,
            "total": len(trials),
        },
        "trial_csv_sha256": sha256_bytes(payload),
        "final_test_access": False,
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json(identity))
    publish(TRIALS, payload, args.overwrite)
    publish(IDENTITY, canonical_json(identity), args.overwrite)
    print(json.dumps(identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
