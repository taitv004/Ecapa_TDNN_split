import csv
import json
from pathlib import Path

import pytest
import torch

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
from src.adaptive_augmented_3s_training import CachedDataset


def write_csv(path: Path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_common_manifest_split_and_train_only_manifest(tmp_path: Path):
    common = tmp_path / "manifest.csv"
    fields = (
        "sample_id",
        "canonical_wav_relpath",
        "global_speaker_id",
        "source_dataset",
        "source_recording_id",
        "split",
    )
    write_csv(
        common,
        fields,
        [
            {
                "sample_id": "t2",
                "canonical_wav_relpath": "train/spk_b/b.wav",
                "global_speaker_id": "spk_b",
                "source_dataset": "B",
                "source_recording_id": "r2",
                "split": "train",
            },
            {
                "sample_id": "t1",
                "canonical_wav_relpath": "train/spk_a/a.wav",
                "global_speaker_id": "spk_a",
                "source_dataset": "A",
                "source_recording_id": "r1",
                "split": "train",
            },
            {
                "sample_id": "v1",
                "canonical_wav_relpath": "validation/spk_v/v.wav",
                "global_speaker_id": "spk_v",
                "source_dataset": "A",
                "source_recording_id": "rv",
                "split": "validation",
            },
        ],
    )
    train, labels, summary = read_manifest_split(common, "train")
    assert [row.sample_id for row in train] == ["t1", "t2"]
    assert labels == {"spk_a": 0, "spk_b": 1}
    assert summary["rows"] == 2

    validation, labels, summary = read_manifest_split(common, "validation")
    assert labels == {}
    assert validation[0].sample_id == "v1"
    assert validation[0].speaker_label == -1
    assert summary["speakers"] == 1

    train_only = tmp_path / "train_manifest.csv"
    write_csv(
        train_only,
        ("sample_id", "output_wav_relpath", "speaker_id"),
        [
            {
                "sample_id": "x1",
                "output_wav_relpath": "train/s1/x.wav",
                "speaker_id": "s1",
            }
        ],
    )
    rows, labels, summary = read_manifest_split(train_only, "train")
    assert rows[0].sample_id == "x1"
    assert rows[0].relative_audio_path == "train/s1/x.wav"
    assert labels == {"s1": 0}
    assert summary["split_column"] is None
    with pytest.raises(ValueError, match="validation cache"):
        read_manifest_split(train_only, "validation")


def test_split_cache_can_be_loaded_without_manifest(tmp_path: Path):
    cache = tmp_path / "cache"
    (cache / CACHE_SHARD_DIR).mkdir(parents=True)
    index_path = cache / CACHE_INDEX_NAME
    rows = [
        {
            "sample_id": "s1",
            "relative_audio_path": "validation/v/a.wav",
            "source_dataset": "src",
            "source_recording_id": "r1",
            "speaker_id": "v",
            "speaker_label": -1,
            "final_split": "validation",
            "shard_path": f"{CACHE_SHARD_DIR}/shard_00000.pt",
            "within_shard_index": 0,
        }
    ]
    write_csv(index_path, CACHE_INDEX_FIELDS, rows)
    config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_version": CACHE_VERSION,
        "split": "validation",
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
        "split": "validation",
        "config_sha256": sha256_file(cache / CACHE_CONFIG_NAME),
        "index_sha256": sha256_file(index_path),
    }
    identity["identity_sha256"] = canonical_digest(identity)
    (cache / CACHE_IDENTITY_NAME).write_text(json.dumps(identity), encoding="utf-8")
    torch.save(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "features": torch.zeros((1, *FEATURE_SHAPE), dtype=torch.float32),
            "sample_ids": ["s1"],
            "speaker_labels": torch.tensor([-1], dtype=torch.long),
            "speaker_ids": ["v"],
            "relative_audio_paths": ["validation/v/a.wav"],
            "final_split": "validation",
        },
        cache / CACHE_SHARD_DIR / "shard_00000.pt",
    )

    artifact = read_cache_artifact(cache, "validation")
    dataset = CachedDataset(artifact, max_cached_shards=1)
    sample = dataset[0]
    assert sample["sample_id"] == "s1"
    assert tuple(sample["fbank"].shape) == FEATURE_SHAPE


def test_primary_frozen_protocol_prefers_label_column():
    from src.frozen_handoff_cache import _resolve_frozen_trial_target_column

    assert (
        _resolve_frozen_trial_target_column(
            [
                "trial_id",
                "enroll_sample_id",
                "test_sample_id",
                "label",
                "source_dataset",
                "target_pair_type",
            ]
        )
        == "label"
    )


def test_frozen_protocol_accepts_legacy_target_column():
    from src.frozen_handoff_cache import _resolve_frozen_trial_target_column

    assert (
        _resolve_frozen_trial_target_column(
            ["enroll_sample_id", "test_sample_id", "target"]
        )
        == "target"
    )


def test_frozen_protocol_rejects_missing_binary_label_column():
    from src.frozen_handoff_cache import _resolve_frozen_trial_target_column

    with pytest.raises(ValueError, match="must contain a binary 'label' column"):
        _resolve_frozen_trial_target_column(
            ["enroll_sample_id", "test_sample_id"]
        )
