#!/usr/bin/env python3
"""Build a resumable raw-FBank cache for the independent final-test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_adaptive_augmented_3s_fbank_cache as base
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


MANIFEST = ROOT / "manifests/adaptive_augmented_3s_v1_final_test_manifest.csv"
CONFIG_FILENAME = "fbank_cache_config_adaptive_augmented_3s_v1_final_test.json"
IDENTITY_FILENAME = "fbank_cache_identity_adaptive_augmented_3s_v1_final_test.json"
INDEX_FILENAME = "final_test_feature_index_adaptive_augmented_3s_v1.csv"


def read_manifest() -> list[base.SourceRow]:
    rows: list[base.SourceRow] = []
    seen_samples: set[str] = set()
    seen_paths: set[str] = set()
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != base.MANIFEST_FIELDS:
            raise ValueError("Invalid final-test manifest schema")
        for index, raw in enumerate(reader):
            relative = base.safe_path(raw["relative_audio_path"].strip())
            sample_id = raw["sample_id"].strip()
            source_dataset = raw["source_dataset"].strip()
            source_recording_id = raw["source_recording_id"].strip()
            speaker = raw["speaker_id"].strip()
            if (
                not sample_id
                or not source_dataset
                or not source_recording_id
                or relative in seen_paths
                or sample_id in seen_samples
                or not speaker
                or raw["speaker_label"].strip() != "-1"
                or raw["final_split"].strip() != "final_test"
            ):
                raise ValueError(f"Invalid final-test row at line {index + 2}")
            seen_samples.add(sample_id)
            seen_paths.add(relative)
            rows.append(
                base.SourceRow(
                    sample_id,
                    relative,
                    source_dataset,
                    source_recording_id,
                    speaker,
                    -1,
                    "final_test",
                    index,
                )
            )
    if not rows or len({row.speaker_id for row in rows}) < 2:
        raise ValueError("Final test needs at least two speakers")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shard-size", default=256, type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve()
    rows = read_manifest()
    binding = {
        "final_test_manifest": {
            "path": MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": base.sha256_file(MANIFEST),
        }
    }
    config = {
        "schema_version": 5,
        "cache_version": "adaptive_augmented_3s_v1_generic_final_test",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "input_bindings": binding,
        "included_splits": ["final_test"],
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": list(base.FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "shard_size": args.shard_size,
        "extraction_batch_size": args.batch_size,
        "expected_rows": {"final_test": len(rows)},
        "evaluation_label": -1,
        "index_filename": INDEX_FILENAME,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    config_path = cache_root / CONFIG_FILENAME
    config_payload = base.canonical_json(config)
    if config_path.exists() and config_path.read_bytes() != config_payload:
        raise ValueError("Existing final-test cache has another config")
    base.atomic_bytes(config_path, config_payload)
    index_payload = base.render_index(rows, "final_test", args.shard_size)
    index_path = cache_root / INDEX_FILENAME
    if index_path.exists() and index_path.read_bytes() != index_payload:
        raise ValueError("Existing final-test cache index conflicts")
    base.atomic_bytes(index_path, index_payload)

    base.SPLITS = ("final_test",)
    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    outcome = base.build_cache(
        frontend,
        dataset_root,
        cache_root,
        {"final_test": rows},
        args.shard_size,
        args.batch_size,
    )
    expected_shards = math.ceil(len(rows) / args.shard_size)
    actual_shards = len(list((cache_root / "final_test").glob("shard_*.pt")))
    if actual_shards != expected_shards:
        raise RuntimeError("Final-test shard count is incomplete")
    first = torch.load(
        cache_root / "final_test/shard_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    if tuple(first["features"].shape[1:]) != base.FEATURE_SHAPE:
        raise ValueError("Final-test FBank shape is invalid")
    identity = {
        "schema_version": 5,
        "identity_kind": "generic_final_test_fbank_cache",
        "cache_version": config["cache_version"],
        "config_path": CONFIG_FILENAME,
        "config_sha256": base.sha256_file(config_path),
        "input_bindings": binding,
        "row_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "feature_shape": list(base.FEATURE_SHAPE),
        "shard_size": args.shard_size,
        "index_sha256": base.sha256_file(index_path),
    }
    identity["identity_sha256"] = base.canonical_digest(identity)
    base.atomic_bytes(cache_root / IDENTITY_FILENAME, base.canonical_json(identity))
    print(
        json.dumps(
            {
                "result": "PASS",
                "rows": len(rows),
                "speakers": identity["speaker_count"],
                "shards": outcome,
                "identity_sha256": identity["identity_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
