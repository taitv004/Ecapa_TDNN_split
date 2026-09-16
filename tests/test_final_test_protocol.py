import csv
import json
import wave
from pathlib import Path

import torch

from scripts.create_final_test_manifest import (
    render_manifest,
    scan_final_test,
)
from src.frozen_handoff_cache import (
    CACHE_CONFIG_NAME,
    CACHE_IDENTITY_NAME,
    CACHE_INDEX_FIELDS,
    CACHE_INDEX_NAME,
    CACHE_SCHEMA_VERSION,
    CACHE_SHARD_DIR,
    CACHE_VERSION,
    FEATURE_SHAPE,
    canonical_digest,
    read_cache_artifact,
    read_manifest_split,
    sha256_file,
)


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 48_000)


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_final_test_manifest_scan_is_deterministic(tmp_path: Path):
    root = tmp_path / "test"
    _write_wav(root / "speaker_b" / "b.wav")
    _write_wav(root / "speaker_a" / "a.wav")

    first = scan_final_test(root, source_dataset="demo")
    second = scan_final_test(root, source_dataset="demo")
    assert first == second
    assert render_manifest(first) == render_manifest(second)
    assert [row["speaker_id"] for row in first] == ["speaker_a", "speaker_b"]
    assert all(row["speaker_label"] == -1 for row in first)
    assert all(row["final_split"] == "final_test" for row in first)


def test_read_manifest_split_accepts_final_test(tmp_path: Path):
    manifest = tmp_path / "test_manifest.csv"
    fields = (
        "sample_id",
        "relative_audio_path",
        "source_dataset",
        "source_recording_id",
        "speaker_id",
        "speaker_label",
        "final_split",
    )
    _write_csv(
        manifest,
        fields,
        [
            {
                "sample_id": "x2",
                "relative_audio_path": "b/b.wav",
                "source_dataset": "demo",
                "source_recording_id": "b/b.wav",
                "speaker_id": "b",
                "speaker_label": -1,
                "final_split": "final_test",
            },
            {
                "sample_id": "x1",
                "relative_audio_path": "a/a.wav",
                "source_dataset": "demo",
                "source_recording_id": "a/a.wav",
                "speaker_id": "a",
                "speaker_label": -1,
                "final_split": "final_test",
            },
        ],
    )
    rows, labels, summary = read_manifest_split(manifest, "final_test")
    assert labels == {}
    assert [row.sample_id for row in rows] == ["x1", "x2"]
    assert all(row.speaker_label == -1 for row in rows)
    assert summary["speakers"] == 2


def test_final_test_cache_artifact_accepts_evaluation_labels(tmp_path: Path):
    cache = tmp_path / "cache"
    (cache / CACHE_SHARD_DIR).mkdir(parents=True)
    index_path = cache / CACHE_INDEX_NAME
    _write_csv(
        index_path,
        CACHE_INDEX_FIELDS,
        [
            {
                "sample_id": "x1",
                "relative_audio_path": "a/a.wav",
                "source_dataset": "demo",
                "source_recording_id": "a/a.wav",
                "speaker_id": "a",
                "speaker_label": -1,
                "final_split": "final_test",
                "shard_path": f"{CACHE_SHARD_DIR}/shard_00000.pt",
                "within_shard_index": 0,
            }
        ],
    )
    config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "split": "final_test",
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "shard_size": 256,
        "row_count": 1,
        "speaker_count": 1,
        "train_class_count": None,
        "index_filename": CACHE_INDEX_NAME,
        "shard_dir": CACHE_SHARD_DIR,
    }
    (cache / CACHE_CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "split": "final_test",
        "config_sha256": sha256_file(cache / CACHE_CONFIG_NAME),
        "index_sha256": sha256_file(index_path),
    }
    identity["identity_sha256"] = canonical_digest(identity)
    (cache / CACHE_IDENTITY_NAME).write_text(json.dumps(identity), encoding="utf-8")

    artifact = read_cache_artifact(cache, "final_test")
    assert artifact.rows[0].sample_id == "x1"
    assert artifact.rows[0].speaker_label == -1
