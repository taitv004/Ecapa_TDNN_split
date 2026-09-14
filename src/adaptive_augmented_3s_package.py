"""Read the frozen, provenance-rich train/validation common manifest."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


REQUIRED_SOURCE_FIELDS = (
    "canonical_audio_id",
    "canonical_wav_relpath",
    "canonical_speaker_id",
    "split",
    "source_dataset",
    "global_speaker_id",
    "source_recording_id",
    "sample_id",
    "start_sec",
    "end_sec",
    "speech_ratio",
    "sample_rate",
    "num_samples",
    "duration_sec",
    "padded",
)
INTERNAL_MANIFEST_FIELDS = (
    "sample_id",
    "relative_audio_path",
    "source_dataset",
    "source_recording_id",
    "speaker_id",
    "speaker_label",
    "final_split",
)
ALLOWED_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class CommonManifestRow:
    sample_id: str
    relative_audio_path: str
    source_dataset: str
    source_recording_id: str
    speaker_id: str
    speaker_label: int
    final_split: str
    canonical_audio_id: str
    canonical_speaker_id: str
    start_sec: float
    end_sec: float
    speech_ratio: float
    sample_rate: int
    num_samples: int
    duration_sec: float
    padded: bool

    def training_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "relative_audio_path": self.relative_audio_path,
            "source_dataset": self.source_dataset,
            "source_recording_id": self.source_recording_id,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "final_split": self.final_split,
        }


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative_path(value: str) -> str:
    """Convert a Windows manifest path to a safe portable POSIX path."""
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or ".." in pure.parts
        or ":" in pure.parts[0]
    ):
        raise ValueError(f"Unsafe/nonportable canonical_wav_relpath: {value!r}")
    return pure.as_posix()


def _parse_bool(value: str, line: int) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"manifest line {line}: padded must be boolean")


def _parse_float(value: str, field: str, line: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"manifest line {line}: invalid {field}") from error
    if not math.isfinite(result):
        raise ValueError(f"manifest line {line}: non-finite {field}")
    return result


def _parse_int(value: str, field: str, line: int) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"manifest line {line}: invalid {field}") from error


def _validate_audio_contract(
    *,
    sample_rate: int,
    num_samples: int,
    duration_sec: float,
    line: int,
) -> None:
    if (
        sample_rate != 16_000
        or num_samples != 48_000
        or abs(duration_sec - 3.0) > 1e-6
    ):
        raise ValueError(
            f"manifest line {line}: expected 16 kHz, 48000 samples, 3 seconds"
        )


def read_common_manifest(
    manifest_path: Path,
    *,
    dataset_root: Path | None = None,
    check_audio_exists: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[str, Any]]:
    """Map the 22-column common manifest to dynamic train/validation rows.

    Existing split assignments are authoritative and are never regenerated.
    Train labels are assigned deterministically from sorted global speaker IDs.
    """
    path = manifest_path.expanduser().resolve(strict=True)
    root = (
        dataset_root.expanduser().resolve(strict=True)
        if dataset_root is not None
        else None
    )
    raw_rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_canonical_ids: set[str] = set()
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in fields]
        if missing:
            raise ValueError(
                "Common manifest is missing required columns: "
                + ", ".join(missing)
            )
        for line, raw in enumerate(reader, start=2):
            sample_id = raw["sample_id"].strip()
            canonical_id = raw["canonical_audio_id"].strip()
            speaker_id = raw["global_speaker_id"].strip()
            canonical_speaker = raw["canonical_speaker_id"].strip()
            source_dataset = raw["source_dataset"].strip()
            recording_id = raw["source_recording_id"].strip()
            split = raw["split"].strip().casefold()
            relative = normalize_relative_path(raw["canonical_wav_relpath"])
            if (
                not sample_id
                or not canonical_id
                or not speaker_id
                or not canonical_speaker
                or not source_dataset
                or not recording_id
                or split not in ALLOWED_SPLITS
                or sample_id in seen_sample_ids
                or canonical_id in seen_canonical_ids
                or relative in seen_paths
            ):
                raise ValueError(f"manifest line {line}: invalid or duplicate identity")
            start = _parse_float(raw["start_sec"], "start_sec", line)
            end = _parse_float(raw["end_sec"], "end_sec", line)
            speech_ratio = _parse_float(raw["speech_ratio"], "speech_ratio", line)
            sample_rate = _parse_int(raw["sample_rate"], "sample_rate", line)
            num_samples = _parse_int(raw["num_samples"], "num_samples", line)
            duration = _parse_float(raw["duration_sec"], "duration_sec", line)
            padded = _parse_bool(raw["padded"], line)
            if start < 0.0 or end <= start or not 0.0 <= speech_ratio <= 1.0:
                raise ValueError(f"manifest line {line}: invalid time/speech metadata")
            _validate_audio_contract(
                sample_rate=sample_rate,
                num_samples=num_samples,
                duration_sec=duration,
                line=line,
            )
            if check_audio_exists:
                if root is None:
                    raise ValueError("dataset_root is required with check_audio_exists")
                audio_path = root.joinpath(*PurePosixPath(relative).parts)
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"manifest line {line}: missing audio: {audio_path}"
                    )
            seen_sample_ids.add(sample_id)
            seen_canonical_ids.add(canonical_id)
            seen_paths.add(relative)
            raw_rows.append(
                {
                    "sample_id": sample_id,
                    "relative_audio_path": relative,
                    "source_dataset": source_dataset,
                    "source_recording_id": recording_id,
                    "speaker_id": speaker_id,
                    "canonical_audio_id": canonical_id,
                    "canonical_speaker_id": canonical_speaker,
                    "final_split": split,
                    "start_sec": start,
                    "end_sec": end,
                    "speech_ratio": speech_ratio,
                    "sample_rate": sample_rate,
                    "num_samples": num_samples,
                    "duration_sec": duration,
                    "padded": padded,
                }
            )
    if not raw_rows:
        raise ValueError("Common manifest is empty")

    speaker_splits: dict[str, set[str]] = {}
    canonical_by_global: dict[str, set[str]] = {}
    for row in raw_rows:
        speaker_splits.setdefault(row["speaker_id"], set()).add(row["final_split"])
        canonical_by_global.setdefault(row["speaker_id"], set()).add(
            row["canonical_speaker_id"]
        )
    leaking = sorted(
        speaker for speaker, splits in speaker_splits.items() if len(splits) != 1
    )
    if leaking:
        raise ValueError(
            "Speaker leakage between train and validation: "
            + ", ".join(leaking[:20])
        )
    inconsistent = sorted(
        speaker
        for speaker, names in canonical_by_global.items()
        if len(names) != 1
    )
    if inconsistent:
        raise ValueError(
            "A global speaker maps to multiple canonical IDs: "
            + ", ".join(inconsistent[:20])
        )

    train_speakers = sorted(
        (
            speaker
            for speaker, splits in speaker_splits.items()
            if splits == {"train"}
        ),
        key=natural_key,
    )
    validation_speakers = sorted(
        (
            speaker
            for speaker, splits in speaker_splits.items()
            if splits == {"validation"}
        ),
        key=natural_key,
    )
    if len(train_speakers) < 16 or len(validation_speakers) < 2:
        raise ValueError(
            "Manifest needs at least 16 train and 2 validation speakers"
        )
    labels = {speaker: index for index, speaker in enumerate(train_speakers)}
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    for raw in raw_rows:
        label = labels[raw["speaker_id"]] if raw["final_split"] == "train" else -1
        row = CommonManifestRow(
            **raw,
            speaker_label=label,
        )
        split_rows[row.final_split].append(row.training_dict())
    for split in ALLOWED_SPLITS:
        split_rows[split].sort(
            key=lambda row: (natural_key(row["speaker_id"]), row["sample_id"])
        )

    sources: dict[str, dict[str, set[str] | int]] = {}
    for raw in raw_rows:
        item = sources.setdefault(
            raw["source_dataset"],
            {"train_speakers": set(), "validation_speakers": set(), "rows": 0},
        )
        item["rows"] = int(item["rows"]) + 1
        cast_set = item[f"{raw['final_split']}_speakers"]
        assert isinstance(cast_set, set)
        cast_set.add(raw["speaker_id"])
    source_summary = {
        source: {
            "rows": int(values["rows"]),
            "train_speakers": len(values["train_speakers"]),
            "validation_speakers": len(values["validation_speakers"]),
        }
        for source, values in sorted(sources.items())
    }
    summary = {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "row_counts": {split: len(split_rows[split]) for split in ALLOWED_SPLITS},
        "speaker_counts": {
            "train": len(train_speakers),
            "validation": len(validation_speakers),
        },
        "speaker_overlap": 0,
        "source_summary": source_summary,
        "train_label_range": [0, len(labels) - 1],
        "path_normalization": "backslash_to_forward_slash",
    }
    return split_rows, labels, summary


def validate_training_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]], labels: Mapping[str, int]
) -> None:
    train = rows["train"]
    validation = rows["validation"]
    train_speakers = {str(row["speaker_id"]) for row in train}
    validation_speakers = {str(row["speaker_id"]) for row in validation}
    if train_speakers & validation_speakers:
        raise ValueError("Speaker leakage between train and validation")
    if set(labels) != train_speakers or set(labels.values()) != set(
        range(len(labels))
    ):
        raise ValueError("Train speaker labels are not contiguous")
    if any(int(row["speaker_label"]) != labels[row["speaker_id"]] for row in train):
        raise ValueError("Train row label mapping mismatch")
    if any(int(row["speaker_label"]) != -1 for row in validation):
        raise ValueError("Validation speaker labels must be -1")
