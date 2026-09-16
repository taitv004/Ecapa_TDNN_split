#!/usr/bin/env python3
"""Generate one frozen sample-ID trial parquet from a final-test manifest.

Run this once for a test set.  Later evaluations should only verify/reuse the
saved parquet and protocol identity; they must not regenerate model-specific
trials.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    generate_validation_trials,
    trial_recording_statistics,
)
from src.frozen_handoff_cache import (
    canonical_digest,
    read_manifest_split,
    sha256_file,
)


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _trial_content_digest(frame: Any) -> str:
    # Digest logical trial content independently from Parquet container bytes.
    records = [
        [
            str(row.trial_id),
            str(row.enroll_sample_id),
            str(row.test_sample_id),
            int(row.label),
            str(row.source_dataset),
            str(row.target_pair_type),
        ]
        for row in frame.itertuples(index=False)
    ]
    return canonical_digest(records)


def _require_parquet_stack():
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Final-test trial generation requires pandas + pyarrow. "
            "Install requirements-cuda.txt."
        ) from error
    return pd


def _auto_trial_counts(rows: Sequence[ValidationRow]) -> tuple[int, int, int, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.speaker_id] = counts.get(row.speaker_id, 0) + 1
    if len(counts) < 2:
        raise ValueError("Final test needs at least two speakers")
    target_capacity = sum(count * (count - 1) // 2 for count in counts.values())
    values = list(counts.values())
    nontarget_capacity = sum(
        values[left] * values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    if target_capacity < 1 or nontarget_capacity < 1:
        raise ValueError("Final test lacks target/non-target pair capacity")
    target_default = min(10_000, target_capacity)
    # Keep some headroom for the endpoint-balanced historical sampler.
    nontarget_default = min(10_000, max(1, nontarget_capacity // 2))
    return target_default, nontarget_default, target_capacity, nontarget_capacity


def generate_protocol(
    *,
    manifest_path: Path,
    output_trials: Path,
    identity_path: Path,
    seed: int = 2026,
    target_trials: int = 0,
    nontarget_trials: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    pd = _require_parquet_stack()
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    rows, labels, summary = read_manifest_split(manifest_path, "final_test")
    if labels:
        raise ValueError("Final-test manifest must not define training labels")

    verification_rows = tuple(
        ValidationRow(
            row.sample_id,
            row.relative_audio_path,
            row.speaker_id,
            row.source_dataset,
            row.source_recording_id,
        )
        for row in rows
    )
    auto_target, auto_nontarget, target_capacity, nontarget_capacity = (
        _auto_trial_counts(verification_rows)
    )
    target_count = auto_target if target_trials == 0 else int(target_trials)
    nontarget_count = (
        auto_nontarget if nontarget_trials == 0 else int(nontarget_trials)
    )
    if target_count < 1 or target_count > target_capacity:
        raise ValueError(
            f"target-trials must be in [1, {target_capacity}], got {target_count}"
        )
    if nontarget_count < 1 or nontarget_count > nontarget_capacity:
        raise ValueError(
            "nontarget-trials must be in "
            f"[1, {nontarget_capacity}], got {nontarget_count}"
        )

    output_trials = output_trials.expanduser().resolve()
    identity_path = identity_path.expanduser().resolve()
    manifest_sha = sha256_file(manifest_path)

    if output_trials.exists() and not overwrite:
        if not identity_path.is_file():
            raise FileNotFoundError(
                "Frozen test trials exist but protocol identity is missing: "
                f"{identity_path}"
            )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        check = dict(identity)
        digest = check.pop("identity_sha256", None)
        if digest != canonical_digest(check):
            raise ValueError("Existing test protocol identity digest is invalid")
        if identity.get("manifest_sha256") != manifest_sha:
            raise ValueError(
                "Existing test trials were generated from another manifest"
            )
        if identity.get("trials_sha256") != sha256_file(output_trials):
            raise ValueError("Existing test trial parquet hash is invalid")
        frame = pd.read_parquet(output_trials)
        required = {
            "trial_id",
            "enroll_sample_id",
            "test_sample_id",
            "label",
            "source_dataset",
            "target_pair_type",
        }
        if set(frame.columns) != required:
            raise ValueError("Existing test trial parquet schema is invalid")
        target_actual = int(frame["label"].sum())
        non_actual = int(len(frame) - target_actual)
        expected_counts = identity.get("trial_counts", {})
        if (
            len(frame) != int(expected_counts.get("total", -1))
            or target_actual != int(expected_counts.get("target", -1))
            or non_actual != int(expected_counts.get("nontarget", -1))
        ):
            raise ValueError("Existing test trial counts disagree with identity")
        return {
            "result": "REUSED",
            "trials": str(output_trials),
            "identity": str(identity_path),
            "manifest_sha256": manifest_sha,
            "trials_sha256": identity["trials_sha256"],
            "trial_counts": expected_counts,
        }

    if identity_path.exists() and not overwrite:
        raise FileExistsError(
            f"Protocol identity exists while trials are absent: {identity_path}"
        )

    generated = generate_validation_trials(
        verification_rows,
        seed=int(seed),
        genuine_count=target_count,
        impostor_count=nontarget_count,
    )
    generated = tuple(
        replace(trial, trial_id=f"final-test-{index:08d}")
        for index, trial in enumerate(generated)
    )
    row_by_sample = {row.sample_id: row for row in rows}
    frame = pd.DataFrame(
        [
            {
                "trial_id": trial.trial_id,
                "enroll_sample_id": trial.left_sample_id,
                "test_sample_id": trial.right_sample_id,
                "label": int(trial.target),
                "source_dataset": (
                    row_by_sample[trial.left_sample_id].source_dataset
                    if row_by_sample[trial.left_sample_id].source_dataset
                    == row_by_sample[trial.right_sample_id].source_dataset
                    else (
                        row_by_sample[trial.left_sample_id].source_dataset
                        + "|"
                        + row_by_sample[trial.right_sample_id].source_dataset
                    )
                ),
                "target_pair_type": (
                    "same_speaker" if trial.target == 1 else "different_speaker"
                ),
            }
            for trial in generated
        ],
        columns=(
            "trial_id",
            "enroll_sample_id",
            "test_sample_id",
            "label",
            "source_dataset",
            "target_pair_type",
        ),
    )

    output_trials.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_trials.with_name(output_trials.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="snappy")
    os.replace(temporary, output_trials)

    trial_sha = sha256_file(output_trials)
    stats = trial_recording_statistics(generated, verification_rows)
    identity = {
        "schema_version": 1,
        "identity_kind": "frozen_independent_final_test_protocol",
        "seed": int(seed),
        "manifest_sha256": manifest_sha,
        "trials_sha256": trial_sha,
        "trial_content_sha256": _trial_content_digest(frame),
        "sample_count": len(rows),
        "speaker_count": int(summary["speakers"]),
        "trial_counts": {
            "target": target_count,
            "nontarget": nontarget_count,
            "total": target_count + nontarget_count,
        },
        "pair_capacity": {
            "target": target_capacity,
            "nontarget": nontarget_capacity,
        },
        "recording_statistics": stats,
    }
    identity["identity_sha256"] = canonical_digest(identity)
    atomic_bytes(identity_path, canonical_json(identity))
    return {
        "result": "CREATED",
        "trials": str(output_trials),
        "identity": str(identity_path),
        "manifest_sha256": manifest_sha,
        "trials_sha256": trial_sha,
        "trial_counts": identity["trial_counts"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-trials", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--target-trials",
        type=int,
        default=0,
        help="0 chooses a deterministic safe value up to 10000.",
    )
    parser.add_argument(
        "--nontarget-trials",
        type=int,
        default=0,
        help="0 chooses a deterministic safe value up to 10000.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    identity = args.identity
    if identity is None:
        identity = args.output_trials.with_name("test_protocol_identity.json")
    result = generate_protocol(
        manifest_path=args.manifest,
        output_trials=args.output_trials,
        identity_path=identity,
        seed=args.seed,
        target_trials=args.target_trials,
        nontarget_trials=args.nontarget_trials,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
