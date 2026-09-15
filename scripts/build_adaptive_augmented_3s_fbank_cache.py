#!/usr/bin/env python3
"""Build one split-specific SpeechBrain raw-FBank cache from a frozen handoff manifest."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import speechbrain
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.frozen_handoff_cache import (
    CACHE_CONFIG_NAME,
    CACHE_IDENTITY_NAME,
    CACHE_INDEX_FIELDS,
    CACHE_INDEX_NAME,
    CACHE_SCHEMA_VERSION,
    CACHE_SHARD_DIR,
    CACHE_VERSION,
    FEATURE_SHAPE,
    ManifestFeatureRow,
    canonical_digest,
    read_manifest_split,
    sha256_file,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend


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


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def render_index(rows: Sequence[ManifestFeatureRow], shard_size: int) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=CACHE_INDEX_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for index, row in enumerate(rows):
        shard, offset = divmod(index, shard_size)
        writer.writerow(
            {
                "sample_id": row.sample_id,
                "relative_audio_path": row.relative_audio_path,
                "source_dataset": row.source_dataset,
                "source_recording_id": row.source_recording_id,
                "speaker_id": row.speaker_id,
                "speaker_label": row.speaker_label,
                "final_split": row.final_split,
                "shard_path": f"{CACHE_SHARD_DIR}/shard_{shard:05d}.pt",
                "within_shard_index": offset,
            }
        )
    return stream.getvalue().encode("utf-8")


def load_waveform(dataset_root: Path, row: ManifestFeatureRow) -> torch.Tensor:
    audio_path = dataset_root.joinpath(
        *PurePosixPath(row.relative_audio_path).parts
    )
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Manifest WAV is missing under dataset root: {audio_path}"
        )
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
    if (
        sample_rate != 16_000
        or tuple(waveform.shape) != (1, 48_000)
        or not bool(torch.isfinite(waveform).all())
    ):
        raise ValueError(f"Expected mono 16 kHz/3 s WAV: {audio_path}")
    return waveform.squeeze(0)


def expected_payload(
    selected: Sequence[ManifestFeatureRow],
    split: str,
    features: torch.Tensor,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
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
    payload: Any,
    selected: Sequence[ManifestFeatureRow],
    split: str,
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
        payload["schema_version"] != CACHE_SCHEMA_VERSION
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
    rows: Sequence[ManifestFeatureRow],
    split: str,
    shard_size: int,
    batch_size: int,
) -> dict[str, int]:
    written = reused = 0
    total = math.ceil(len(rows) / shard_size)
    for shard_number, start in enumerate(range(0, len(rows), shard_size)):
        selected = rows[start : start + shard_size]
        target = cache_root / CACHE_SHARD_DIR / f"shard_{shard_number:05d}.pt"
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
            selected,
            split,
            torch.cat(feature_batches, dim=0),
        )
        validate_shard(payload, selected, split)
        atomic_torch_save(target, payload)
        written += 1
        print(f"{split} shard {shard_number + 1}/{total} complete", flush=True)

    return {"written": written, "reused": reused, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "validation"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--shard-size", default=256, type=int)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-speakers", type=int)

    # Optional explicit column mapping for train-only manifests.  CommonRawBase
    # works with auto-detection and normally needs none of these flags.
    parser.add_argument("--sample-id-column")
    parser.add_argument("--path-column")
    parser.add_argument("--speaker-id-column")
    parser.add_argument("--source-dataset-column")
    parser.add_argument("--source-recording-id-column")
    parser.add_argument("--split-column")
    args = parser.parse_args()

    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive")
    if args.expected_rows is not None and args.expected_rows < 1:
        raise ValueError("expected-rows must be positive")
    if args.expected_speakers is not None and args.expected_speakers < 1:
        raise ValueError("expected-speakers must be positive")

    dataset_root = args.dataset_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve()

    rows, labels, summary = read_manifest_split(
        manifest_path,
        args.split,
        sample_id_column=args.sample_id_column,
        path_column=args.path_column,
        speaker_id_column=args.speaker_id_column,
        source_dataset_column=args.source_dataset_column,
        source_recording_id_column=args.source_recording_id_column,
        split_column=args.split_column,
    )
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(
            f"Expected {args.expected_rows} {args.split} rows, found {len(rows)}"
        )
    speaker_count = len({row.speaker_id for row in rows})
    if args.expected_speakers is not None and speaker_count != args.expected_speakers:
        raise ValueError(
            f"Expected {args.expected_speakers} {args.split} speakers, "
            f"found {speaker_count}"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    index_payload = render_index(rows, args.shard_size)
    index_path = cache_root / CACHE_INDEX_NAME

    label_mapping_sha = canonical_digest(labels) if args.split == "train" else None
    config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "split": args.split,
        "model_source": SpeechBrainECAPAFrontend.SOURCE,
        "speechbrain_version": speechbrain.__version__,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_columns": summary,
        "speaker_to_label_sha256": label_mapping_sha,
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "shard_size": args.shard_size,
        "extraction_batch_size": args.batch_size,
        "row_count": len(rows),
        "speaker_count": speaker_count,
        "train_class_count": len(labels) if args.split == "train" else None,
        "validation_label": -1,
        "index_filename": CACHE_INDEX_NAME,
        "index_fields": list(CACHE_INDEX_FIELDS),
        "shard_dir": CACHE_SHARD_DIR,
    }
    config_payload = canonical_json(config)
    config_path = cache_root / CACHE_CONFIG_NAME
    if config_path.exists() and config_path.read_bytes() != config_payload:
        raise ValueError(
            "Existing cache was built for another manifest/split/settings; "
            "use an empty cache directory"
        )
    if index_path.exists() and index_path.read_bytes() != index_payload:
        raise ValueError("Existing feature index conflicts with current manifest")
    atomic_bytes(config_path, config_payload)
    atomic_bytes(index_path, index_payload)

    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    outcome = build_cache(
        frontend,
        dataset_root,
        cache_root,
        rows,
        args.split,
        args.shard_size,
        args.batch_size,
    )

    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity_kind": "frozen_handoff_split_fbank_cache",
        "cache_version": CACHE_VERSION,
        "split": args.split,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(rows),
        "speaker_count": speaker_count,
        "train_class_count": len(labels) if args.split == "train" else None,
        "feature_shape": list(FEATURE_SHAPE),
        "shard_size": args.shard_size,
        "shard_count": outcome["total"],
    }
    identity["identity_sha256"] = canonical_digest(identity)
    atomic_bytes(cache_root / CACHE_IDENTITY_NAME, canonical_json(identity))

    print(
        json.dumps(
            {
                "result": "PASS",
                "split": args.split,
                "cache_identity": identity["identity_sha256"],
                "rows": len(rows),
                "speakers": speaker_count,
                "train_classes": len(labels) if args.split == "train" else None,
                "shards": outcome,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
