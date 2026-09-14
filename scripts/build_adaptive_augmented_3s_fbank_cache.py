#!/usr/bin/env python3
"""Build a resumable SpeechBrain raw-FBank cache for train/validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import speechbrain
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.speechbrain_frontend import SpeechBrainECAPAFrontend
from src.adaptive_augmented_3s_verification import MANIFEST_FIELDS
from src.adaptive_augmented_3s_package import (
    read_common_manifest,
    validate_training_rows,
)


SPLITS = ("train", "validation")
CONFIG_FILENAME = "fbank_cache_config_adaptive_augmented_3s_v1.json"
IDENTITY_FILENAME = "fbank_cache_identity_adaptive_augmented_3s_v1.json"
INDEX_FILENAMES = {
    "train": "train_feature_index_adaptive_augmented_3s_v1.csv",
    "validation": "validation_feature_index_adaptive_augmented_3s_v1.csv",
}
INDEX_FIELDS = MANIFEST_FIELDS + ("shard_path", "within_shard_index")
FEATURE_SHAPE = (301, 80)


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    relative_audio_path: str
    source_dataset: str
    source_recording_id: str
    speaker_id: str
    speaker_label: int
    final_split: str
    manifest_row_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def safe_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in value
    ):
        raise ValueError(f"Unsafe/nonportable audio path: {value!r}")
    return value


def read_manifests(
    manifest_path: Path, dataset_root: Path
) -> tuple[dict[str, list[SourceRow]], dict[str, int], dict[str, Any]]:
    mapped, labels, summary = read_common_manifest(
        manifest_path,
        dataset_root=dataset_root,
        check_audio_exists=False,
    )
    validate_training_rows(mapped, labels)
    result: dict[str, list[SourceRow]] = {}
    for split in SPLITS:
        result[split] = [
            SourceRow(
                str(row["sample_id"]),
                safe_path(str(row["relative_audio_path"])),
                str(row["source_dataset"]),
                str(row["source_recording_id"]),
                str(row["speaker_id"]),
                int(row["speaker_label"]),
                split,
                index,
            )
            for index, row in enumerate(mapped[split])
        ]
    return result, labels, summary


def render_index(rows: Sequence[SourceRow], split: str, shard_size: int) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        shard, offset = divmod(row.manifest_row_index, shard_size)
        writer.writerow(
            {
                "sample_id": row.sample_id,
                "relative_audio_path": row.relative_audio_path,
                "source_dataset": row.source_dataset,
                "source_recording_id": row.source_recording_id,
                "speaker_id": row.speaker_id,
                "speaker_label": row.speaker_label,
                "final_split": split,
                "shard_path": f"{split}/shard_{shard:05d}.pt",
                "within_shard_index": offset,
            }
        )
    return stream.getvalue().encode("utf-8")


def load_waveform(dataset_root: Path, row: SourceRow) -> torch.Tensor:
    path = dataset_root.joinpath(*PurePosixPath(row.relative_audio_path).parts)
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    if (
        sample_rate != 16_000
        or tuple(waveform.shape) != (1, 48_000)
        or not bool(torch.isfinite(waveform).all())
    ):
        raise ValueError(f"Expected mono 16 kHz/3 s WAV: {path}")
    return waveform.squeeze(0)


def expected_payload(
    selected: Sequence[SourceRow], split: str, features: torch.Tensor
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "features": features.cpu().float().contiguous(),
        "sample_ids": [row.sample_id for row in selected],
        "speaker_labels": torch.tensor(
            [row.speaker_label for row in selected], dtype=torch.long
        ),
        "speaker_ids": [row.speaker_id for row in selected],
        "relative_audio_paths": [row.relative_audio_path for row in selected],
        "final_split": split,
    }


def validate_shard(
    payload: Any, selected: Sequence[SourceRow], split: str
) -> None:
    required = {
        "schema_version",
        "features",
        "sample_ids",
        "speaker_labels",
        "speaker_ids",
        "relative_audio_paths",
        "final_split",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Invalid cache shard schema")
    features = payload["features"]
    labels = payload["speaker_labels"]
    count = len(selected)
    if (
        payload["schema_version"] != 5
        or tuple(features.shape) != (count, *FEATURE_SHAPE)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not bool(torch.isfinite(features).all())
        or labels.dtype != torch.long
        or labels.tolist() != [row.speaker_label for row in selected]
        or payload["sample_ids"] != [row.sample_id for row in selected]
        or payload["speaker_ids"] != [row.speaker_id for row in selected]
        or payload["relative_audio_paths"]
        != [row.relative_audio_path for row in selected]
        or payload["final_split"] != split
    ):
        raise ValueError("Cache shard differs from its manifest rows")


def build_cache(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    cache_root: Path,
    rows: Mapping[str, Sequence[SourceRow]],
    shard_size: int,
    batch_size: int,
) -> dict[str, dict[str, int]]:
    outcome: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        written = reused = 0
        split_rows = rows[split]
        total = math.ceil(len(split_rows) / shard_size)
        for shard_number, start in enumerate(range(0, len(split_rows), shard_size)):
            selected = split_rows[start : start + shard_size]
            target = cache_root / split / f"shard_{shard_number:05d}.pt"
            if target.is_file():
                validate_shard(
                    torch.load(target, map_location="cpu", weights_only=False),
                    selected,
                    split,
                )
                reused += 1
                print(f"{split} shard {shard_number + 1}/{total} reused", flush=True)
                continue
            feature_batches: list[torch.Tensor] = []
            for batch_start in range(0, len(selected), batch_size):
                batch_rows = selected[batch_start : batch_start + batch_size]
                waveforms = torch.stack(
                    [load_waveform(dataset_root, row) for row in batch_rows]
                )
                with torch.inference_mode():
                    features = frontend.compute_features(waveforms)
                feature_batches.append(features.detach().cpu().float().contiguous())
                del waveforms, features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            payload = expected_payload(
                selected, split, torch.cat(feature_batches, dim=0)
            )
            validate_shard(payload, selected, split)
            atomic_torch_save(target, payload)
            written += 1
            print(f"{split} shard {shard_number + 1}/{total} complete", flush=True)
        outcome[split] = {"written": written, "reused": reused, "total": total}
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shard-size", default=256, type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve()
    rows, labels, manifest_summary = read_manifests(
        manifest_path, dataset_root
    )
    bindings = {
        "authoritative_manifest": {
            "sha256": sha256_file(manifest_path),
            "row_counts": manifest_summary["row_counts"],
            "speaker_counts": manifest_summary["speaker_counts"],
        },
        "speaker_to_label_sha256": canonical_digest(labels),
    }
    config = {
        "schema_version": 5,
        "cache_version": "adaptive_augmented_3s_v1_generic",
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "speechbrain_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "input_bindings": bindings,
        "included_splits": list(SPLITS),
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "shard_size": args.shard_size,
        "extraction_batch_size": args.batch_size,
        "expected_rows": {split: len(rows[split]) for split in SPLITS},
        "train_class_count": len(labels),
        "train_label_range": [0, len(labels) - 1],
        "validation_label": -1,
        "index_filenames": INDEX_FILENAMES,
        "index_fields": list(INDEX_FIELDS),
    }
    config_payload = canonical_json(config)
    cache_root.mkdir(parents=True, exist_ok=True)
    config_path = cache_root / CONFIG_FILENAME
    if config_path.exists() and config_path.read_bytes() != config_payload:
        raise ValueError(
            "Existing cache was built for different manifests/settings; "
            "use an empty cache directory"
        )
    atomic_bytes(config_path, config_payload)
    for split in SPLITS:
        index_payload = render_index(rows[split], split, args.shard_size)
        index_path = cache_root / INDEX_FILENAMES[split]
        if index_path.exists() and index_path.read_bytes() != index_payload:
            raise ValueError(f"Existing {split} index conflicts with current manifest")
        atomic_bytes(index_path, index_payload)

    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    outcome = build_cache(
        frontend,
        dataset_root,
        cache_root,
        rows,
        args.shard_size,
        args.batch_size,
    )
    identity = {
        "schema_version": 5,
        "identity_kind": "generic_train_validation_fbank_cache",
        "cache_version": config["cache_version"],
        "config_path": CONFIG_FILENAME,
        "config_sha256": sha256_file(config_path),
        "input_bindings": bindings,
        "included_splits": list(SPLITS),
        "row_counts": config["expected_rows"],
        "train_class_count": len(labels),
        "feature_shape": list(FEATURE_SHAPE),
        "shard_size": args.shard_size,
        "index_sha256": {
            split: sha256_file(cache_root / INDEX_FILENAMES[split])
            for split in SPLITS
        },
        "final_test_cache_absent": True,
    }
    identity["identity_sha256"] = canonical_digest(identity)
    atomic_bytes(cache_root / IDENTITY_FILENAME, canonical_json(identity))
    print(
        json.dumps(
            {
                "result": "PASS",
                "cache_identity": identity["identity_sha256"],
                "rows": identity["row_counts"],
                "train_classes": len(labels),
                "shards": outcome,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
