"""Frozen handoff manifest/cache/protocol helpers for the primary thesis experiment.

This module deliberately separates three contracts:

- manifests are used only while extracting FBank features from WAV files;
- split-specific FBank caches are sufficient for training/inference afterwards;
- the frozen validation parquet specifies which validation sample IDs are scored.

The primary experiment is RAW vs RANDOM_NOISE vs ADAPTIVE_NOISE.  Validation is
always the one RAW validation partition from CommonRawBase.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


CACHE_VERSION = "frozen_handoff_split_fbank_v1"
CACHE_SCHEMA_VERSION = 1
CACHE_CONFIG_NAME = "fbank_cache_config_frozen_handoff_v1.json"
CACHE_IDENTITY_NAME = "fbank_cache_identity_frozen_handoff_v1.json"
CACHE_INDEX_NAME = "feature_index_frozen_handoff_v1.csv"
CACHE_SHARD_DIR = "shards"
FEATURE_SHAPE = (301, 80)
FROZEN_SPLITS = ("train", "validation", "final_test")
EVALUATION_SPLITS = ("validation", "final_test")
CACHE_INDEX_FIELDS = (
    "sample_id",
    "relative_audio_path",
    "source_dataset",
    "source_recording_id",
    "speaker_id",
    "speaker_label",
    "final_split",
    "shard_path",
    "within_shard_index",
)

FROZEN_VALIDATION_TRIAL_SHA256 = (
    "9f7a905376d8ed8e7f4aed6bab8a1981f11fe910697c4bfa406e29bdf4f3813c"
)
FROZEN_VALIDATION_TRIAL_COUNT = 1_064_352
FROZEN_VALIDATION_TARGET_COUNT = 532_176
FROZEN_VALIDATION_NONTARGET_COUNT = 532_176


@dataclass(frozen=True)
class ManifestFeatureRow:
    sample_id: str
    relative_audio_path: str
    source_dataset: str
    source_recording_id: str
    speaker_id: str
    speaker_label: int
    final_split: str


@dataclass(frozen=True)
class CacheFeatureRow:
    sample_id: str
    relative_audio_path: str
    source_dataset: str
    source_recording_id: str
    speaker_id: str
    speaker_label: int
    final_split: str
    shard_path: str
    within_shard_index: int


@dataclass(frozen=True)
class CacheArtifact:
    root: Path
    config: Mapping[str, Any]
    identity: Mapping[str, Any]
    rows: tuple[CacheFeatureRow, ...]


@dataclass(frozen=True)
class FrozenValidationProtocol:
    """Resolved sample-ID verification protocol.

    The historical name is retained because the training pipeline imports it,
    but the same representation is also used by the independent final-test
    evaluator. ``trial_ids`` is optional because the primary frozen validation
    parquet does not need it during training.
    """

    enroll_indices: np.ndarray
    test_indices: np.ndarray
    targets: np.ndarray
    sha256: str
    trial_ids: tuple[str, ...] | None = None

    def __len__(self) -> int:
        return int(self.targets.shape[0])


# Semantically clearer alias for new final-test code.
FrozenVerificationProtocol = FrozenValidationProtocol


_SAMPLE_ID_ALIASES = (
    "sample_id",
    "base_sample_id",
    "canonical_sample_id",
)
_PATH_ALIASES = (
    "canonical_wav_relpath",
    "relative_wav_path",
    "relative_audio_path",
    "output_wav_relpath",
    "wav_relpath",
    "output_path",
)
_SPEAKER_ID_ALIASES = (
    "global_speaker_id",
    "speaker_id",
    "canonical_speaker_id",
)
_SOURCE_DATASET_ALIASES = ("source_dataset", "dataset", "source")
_SOURCE_RECORDING_ALIASES = (
    "source_recording_id",
    "recording_id",
    "source_file",
)
_SPLIT_ALIASES = ("split", "final_split")


def _resolve_frozen_trial_target_column(fields: Sequence[str]) -> str:
    """Return the physical parquet column that stores the binary trial label.

    The primary frozen handoff uses ``label``.  ``target`` is accepted only for
    backward compatibility with historical/local protocol files.
    """
    names = set(fields)
    if "label" in names:
        return "label"
    if "target" in names:
        return "target"
    raise ValueError(
        "Frozen validation parquet must contain a binary 'label' column "
        "(or legacy 'target' column). Available columns: "
        f"{sorted(names)}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def safe_relative_path(value: str, *, field: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or ".." in pure.parts
        or (pure.parts and ":" in pure.parts[0])
    ):
        raise ValueError(f"Unsafe/nonportable {field}: {value!r}")
    return pure.as_posix()


def _pick_column(
    fields: Sequence[str],
    aliases: Sequence[str],
    *,
    explicit: str | None,
    required: bool,
    description: str,
) -> str | None:
    if explicit:
        if explicit not in fields:
            raise ValueError(
                f"Requested {description} column {explicit!r} is absent. "
                f"Available columns: {list(fields)}"
            )
        return explicit
    matches = [name for name in aliases if name in fields]
    if matches:
        return matches[0]
    if required:
        raise ValueError(
            f"Could not identify {description} column. "
            f"Accepted defaults: {list(aliases)}; available: {list(fields)}"
        )
    return None


def read_manifest_split(
    manifest_path: Path,
    split: str,
    *,
    sample_id_column: str | None = None,
    path_column: str | None = None,
    speaker_id_column: str | None = None,
    source_dataset_column: str | None = None,
    source_recording_id_column: str | None = None,
    split_column: str | None = None,
) -> tuple[tuple[ManifestFeatureRow, ...], dict[str, int], dict[str, Any]]:
    """Read one frozen split from either CommonRawBase or a train-only manifest.

    Column names are auto-detected from a small explicit alias set.  CLI callers
    can override any column name when a supplied train manifest uses another
    documented name.  No speaker identity is inferred from a filename/path.
    """
    if split not in FROZEN_SPLITS:
        raise ValueError("split must be 'train', 'validation' or 'final_test'")
    path = manifest_path.expanduser().resolve(strict=True)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if not fields:
            raise ValueError(f"Manifest has no header: {path}")

        sample_col = _pick_column(
            fields, _SAMPLE_ID_ALIASES, explicit=sample_id_column,
            required=True, description="sample ID",
        )
        wav_col = _pick_column(
            fields, _PATH_ALIASES, explicit=path_column,
            required=True, description="relative WAV path",
        )
        speaker_col = _pick_column(
            fields, _SPEAKER_ID_ALIASES, explicit=speaker_id_column,
            required=True, description="speaker ID",
        )
        source_col = _pick_column(
            fields, _SOURCE_DATASET_ALIASES, explicit=source_dataset_column,
            required=False, description="source dataset",
        )
        recording_col = _pick_column(
            fields, _SOURCE_RECORDING_ALIASES,
            explicit=source_recording_id_column,
            required=False,
            description="source recording ID",
        )
        split_col = _pick_column(
            fields, _SPLIT_ALIASES, explicit=split_column,
            required=False, description="split",
        )

        raw_rows: list[dict[str, str]] = []
        seen_samples: set[str] = set()
        seen_paths: set[str] = set()
        for line, raw in enumerate(reader, start=2):
            if split_col:
                row_split = str(raw[split_col]).strip().casefold()
                if row_split not in FROZEN_SPLITS:
                    raise ValueError(
                        f"{path}:{line}: invalid split {raw[split_col]!r}"
                    )
                if row_split != split:
                    continue
            elif split != "train":
                if split == "validation":
                    raise ValueError(
                        "A validation cache must be built from a manifest with an "
                        "explicit split/final_split column"
                    )
                raise ValueError(
                    "A final_test cache must be built from a manifest with an "
                    "explicit split/final_split column"
                )

            sample_id = str(raw[sample_col]).strip()
            relative = safe_relative_path(str(raw[wav_col]), field=wav_col)
            speaker_id = str(raw[speaker_col]).strip()
            source_dataset = str(raw[source_col]).strip() if source_col else ""
            recording_id = (
                str(raw[recording_col]).strip() if recording_col else sample_id
            )
            if not sample_id or not speaker_id:
                raise ValueError(f"{path}:{line}: blank sample/speaker ID")
            if sample_id in seen_samples:
                raise ValueError(f"{path}:{line}: duplicate sample_id {sample_id!r}")
            if relative in seen_paths:
                raise ValueError(f"{path}:{line}: duplicate WAV path {relative!r}")
            seen_samples.add(sample_id)
            seen_paths.add(relative)
            raw_rows.append(
                {
                    "sample_id": sample_id,
                    "relative_audio_path": relative,
                    "source_dataset": source_dataset,
                    "source_recording_id": recording_id,
                    "speaker_id": speaker_id,
                }
            )

    if not raw_rows:
        raise ValueError(f"No {split} rows found in manifest: {path}")

    speaker_ids = sorted({row["speaker_id"] for row in raw_rows})
    labels = (
        {speaker_id: index for index, speaker_id in enumerate(speaker_ids)}
        if split == "train"
        else {}
    )
    rows = tuple(
        ManifestFeatureRow(
            sample_id=row["sample_id"],
            relative_audio_path=row["relative_audio_path"],
            source_dataset=row["source_dataset"],
            source_recording_id=row["source_recording_id"],
            speaker_id=row["speaker_id"],
            speaker_label=labels[row["speaker_id"]] if split == "train" else -1,
            final_split=split,
        )
        for row in sorted(raw_rows, key=lambda item: (item["speaker_id"], item["sample_id"]))
    )
    summary = {
        "manifest_sha256": sha256_file(path),
        "split": split,
        "rows": len(rows),
        "speakers": len(speaker_ids),
        "sample_id_column": sample_col,
        "path_column": wav_col,
        "speaker_id_column": speaker_col,
        "source_dataset_column": source_col,
        "source_recording_id_column": recording_col,
        "split_column": split_col,
    }
    return rows, labels, summary


def read_cache_artifact(cache_root: Path, expected_split: str) -> CacheArtifact:
    if expected_split not in FROZEN_SPLITS:
        raise ValueError(
            "expected_split must be 'train', 'validation' or 'final_test'"
        )
    root = cache_root.expanduser().resolve(strict=True)
    config_path = root / CACHE_CONFIG_NAME
    identity_path = root / CACHE_IDENTITY_NAME
    index_path = root / CACHE_INDEX_NAME
    if not config_path.is_file() or not identity_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"Incomplete frozen-handoff FBank cache under {root}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Malformed cache metadata under {root}: {error}") from error

    if (
        config.get("schema_version") != CACHE_SCHEMA_VERSION
        or config.get("cache_version") != CACHE_VERSION
        or config.get("split") != expected_split
        or config.get("feature_shape") != list(FEATURE_SHAPE)
        or config.get("feature_dtype") != "float32"
        or config.get("index_filename") != CACHE_INDEX_NAME
        or config.get("shard_dir") != CACHE_SHARD_DIR
    ):
        raise ValueError(f"Cache contract mismatch under {root}")
    if (
        identity.get("schema_version") != CACHE_SCHEMA_VERSION
        or identity.get("cache_version") != CACHE_VERSION
        or identity.get("split") != expected_split
        or identity.get("config_sha256") != sha256_file(config_path)
        or identity.get("index_sha256") != sha256_file(index_path)
    ):
        raise ValueError(f"Cache identity mismatch under {root}")
    expected_identity = dict(identity)
    actual_identity_sha = expected_identity.pop("identity_sha256", None)
    if actual_identity_sha != canonical_digest(expected_identity):
        raise ValueError(f"Cache identity digest mismatch under {root}")

    rows: list[CacheFeatureRow] = []
    seen_samples: set[str] = set()
    seen_paths: set[str] = set()
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CACHE_INDEX_FIELDS:
            raise ValueError(f"Invalid cache index schema: {index_path}")
        for line, raw in enumerate(reader, start=2):
            sample_id = raw["sample_id"].strip()
            relative = safe_relative_path(
                raw["relative_audio_path"], field="relative_audio_path"
            )
            shard_path = safe_relative_path(raw["shard_path"], field="shard_path")
            speaker_id = raw["speaker_id"].strip()
            try:
                label = int(raw["speaker_label"])
                offset = int(raw["within_shard_index"])
            except ValueError as error:
                raise ValueError(f"{index_path}:{line}: invalid integer") from error
            if (
                not sample_id
                or not speaker_id
                or raw["final_split"].strip() != expected_split
                or sample_id in seen_samples
                or relative in seen_paths
                or offset < 0
                or offset >= int(config["shard_size"])
            ):
                raise ValueError(f"{index_path}:{line}: invalid cache index row")
            if expected_split == "train" and label < 0:
                raise ValueError(f"{index_path}:{line}: negative train label")
            if expected_split in EVALUATION_SPLITS and label != -1:
                raise ValueError(
                    f"{index_path}:{line}: evaluation label must be -1"
                )
            seen_samples.add(sample_id)
            seen_paths.add(relative)
            rows.append(
                CacheFeatureRow(
                    sample_id=sample_id,
                    relative_audio_path=relative,
                    source_dataset=raw["source_dataset"].strip(),
                    source_recording_id=raw["source_recording_id"].strip(),
                    speaker_id=speaker_id,
                    speaker_label=label,
                    final_split=expected_split,
                    shard_path=shard_path,
                    within_shard_index=offset,
                )
            )
    if len(rows) != int(config.get("row_count", -1)):
        raise ValueError(f"Cache row count mismatch under {root}")
    if len({row.speaker_id for row in rows}) != int(config.get("speaker_count", -1)):
        raise ValueError(f"Cache speaker count mismatch under {root}")
    if expected_split == "train":
        class_count = int(config.get("train_class_count", -1))
        if {row.speaker_label for row in rows} != set(range(class_count)):
            raise ValueError("Train cache labels are not contiguous")
    return CacheArtifact(root=root, config=config, identity=identity, rows=tuple(rows))


def read_frozen_verification_protocol(
    parquet_path: Path,
    cache_rows: Sequence[CacheFeatureRow],
    *,
    expected_split: str,
    expected_sha256: str | None = None,
    expected_trial_count: int | None = None,
    expected_target_count: int | None = None,
    expected_nontarget_count: int | None = None,
    verify_unique_pairs: bool = True,
) -> FrozenVerificationProtocol:
    """Resolve a frozen sample-ID verification parquet against one cache.

    Mapping is always ``sample_id -> cache row index``.  Trial row order and
    cache row order therefore never need to match.  The function validates that
    trial labels agree with the speaker IDs stored in the cache.
    """
    if expected_split not in EVALUATION_SPLITS:
        raise ValueError("verification protocol split must be validation/final_test")
    if any(row.final_split != expected_split for row in cache_rows):
        raise ValueError(
            f"Cache rows do not all belong to expected split {expected_split!r}"
        )

    path = parquet_path.expanduser().resolve(strict=True)
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "Frozen verification trial SHA-256 mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )

    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Reading verification parquet requires pandas + pyarrow. "
            "Install the repository requirements."
        ) from error

    try:
        import pyarrow.parquet as pq

        schema_names = pq.read_schema(path).names
        target_column = _resolve_frozen_trial_target_column(schema_names)
        required = {"enroll_sample_id", "test_sample_id", target_column}
        missing_columns = required - set(schema_names)
        if missing_columns:
            raise ValueError(
                "Verification parquet is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        columns = ["enroll_sample_id", "test_sample_id", target_column]
        has_trial_id = "trial_id" in schema_names
        if has_trial_id:
            columns.insert(0, "trial_id")
        frame = pd.read_parquet(path, columns=columns)
    except Exception as error:
        raise ValueError(
            f"Could not read frozen verification parquet {path}: {error}"
        ) from error

    if target_column != "target":
        frame = frame.rename(columns={target_column: "target"})
    if frame.empty:
        raise ValueError("Frozen verification protocol is empty")
    if frame["enroll_sample_id"].isna().any() or frame["test_sample_id"].isna().any():
        raise ValueError("Frozen verification protocol contains blank sample IDs")

    enroll_ids = frame["enroll_sample_id"].astype(str)
    test_ids = frame["test_sample_id"].astype(str)
    if (enroll_ids.str.len() == 0).any() or (test_ids.str.len() == 0).any():
        raise ValueError("Frozen verification protocol contains blank sample IDs")

    targets = frame["target"].to_numpy(dtype=np.int8, copy=True)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Frozen verification targets must be binary 0/1")

    sample_to_index = {row.sample_id: index for index, row in enumerate(cache_rows)}
    if len(sample_to_index) != len(cache_rows):
        raise ValueError(f"{expected_split} cache sample IDs are not unique")

    enroll = enroll_ids.map(sample_to_index)
    test = test_ids.map(sample_to_index)
    if enroll.isna().any() or test.isna().any():
        missing = set(enroll_ids[enroll.isna()])
        missing.update(test_ids[test.isna()])
        raise ValueError(
            "Frozen trials reference sample IDs absent from "
            f"{expected_split} cache: " + ", ".join(sorted(missing)[:10])
        )

    enroll_indices = enroll.to_numpy(dtype=np.int32, copy=True)
    test_indices = test.to_numpy(dtype=np.int32, copy=True)
    if np.any(enroll_indices == test_indices):
        raise ValueError("Frozen verification protocol contains a self-pair")

    trial_ids: tuple[str, ...] | None = None
    if has_trial_id:
        if frame["trial_id"].isna().any():
            raise ValueError("Frozen verification protocol contains blank trial IDs")
        trial_values = tuple(frame["trial_id"].astype(str).tolist())
        if any(not value for value in trial_values):
            raise ValueError("Frozen verification protocol contains blank trial IDs")
        if len(set(trial_values)) != len(trial_values):
            raise ValueError("Frozen verification protocol has duplicate trial IDs")
        trial_ids = trial_values

    if verify_unique_pairs:
        # Integer packing avoids retaining a Python set of long sample-ID strings.
        count = len(cache_rows)
        left = np.minimum(enroll_indices, test_indices).astype(np.int64)
        right = np.maximum(enroll_indices, test_indices).astype(np.int64)
        packed = left * np.int64(count) + right
        if np.unique(packed).size != packed.size:
            raise ValueError("Frozen verification protocol has duplicate sample pairs")

    speaker_ids = np.asarray([row.speaker_id for row in cache_rows], dtype=object)
    same = speaker_ids[enroll_indices] == speaker_ids[test_indices]
    if np.any((targets == 1) != same):
        raise ValueError(
            "Frozen verification target disagrees with cache speaker identity"
        )

    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if expected_trial_count is not None and targets.size != expected_trial_count:
        raise ValueError(
            f"Frozen verification trial count mismatch: "
            f"expected {expected_trial_count}, got {targets.size}"
        )
    if expected_target_count is not None and positives != expected_target_count:
        raise ValueError(
            f"Frozen verification target count mismatch: "
            f"expected {expected_target_count}, got {positives}"
        )
    if (
        expected_nontarget_count is not None
        and negatives != expected_nontarget_count
    ):
        raise ValueError(
            f"Frozen verification non-target count mismatch: "
            f"expected {expected_nontarget_count}, got {negatives}"
        )
    if positives == 0 or negatives == 0:
        raise ValueError("Both target and non-target trials are required")

    return FrozenVerificationProtocol(
        enroll_indices=enroll_indices,
        test_indices=test_indices,
        targets=targets,
        sha256=digest,
        trial_ids=trial_ids,
    )


def read_frozen_validation_protocol(
    parquet_path: Path,
    validation_rows: Sequence[CacheFeatureRow],
    *,
    enforce_primary_frozen_identity: bool = True,
) -> FrozenValidationProtocol:
    """Resolve the primary frozen validation protocol against its cache."""
    return read_frozen_verification_protocol(
        parquet_path,
        validation_rows,
        expected_split="validation",
        expected_sha256=(
            FROZEN_VALIDATION_TRIAL_SHA256
            if enforce_primary_frozen_identity
            else None
        ),
        expected_trial_count=(
            FROZEN_VALIDATION_TRIAL_COUNT
            if enforce_primary_frozen_identity
            else None
        ),
        expected_target_count=(
            FROZEN_VALIDATION_TARGET_COUNT
            if enforce_primary_frozen_identity
            else None
        ),
        expected_nontarget_count=(
            FROZEN_VALIDATION_NONTARGET_COUNT
            if enforce_primary_frozen_identity
            else None
        ),
        # The primary validation reader historically did not spend memory/time
        # rechecking pair uniqueness for >1M already-frozen trials.
        verify_unique_pairs=False,
    )
