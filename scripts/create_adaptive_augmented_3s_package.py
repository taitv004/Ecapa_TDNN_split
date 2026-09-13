#!/usr/bin/env python3
"""Create generic speaker-disjoint train/validation manifests and trials.

Expected layout::

    dataset_root/
      speaker_001/utterance_001.wav
      speaker_001/utterance_002.wav
      speaker_002/utterance_001.wav

The validation ratio is applied to speakers, never individual utterances.
The independent final-test dataset is intentionally not read by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_augmented_3s_verification import (
    ValidationRow,
    generate_validation_trials,
    sha256_bytes,
    trials_csv_bytes,
)


PACKAGE_VERSION = "adaptive_augmented_3s_v1"
DEFAULT_SEED = 2026
DEFAULT_VALIDATION_RATIO = 0.10
EXPECTED_SAMPLE_RATE = 16_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLES = 48_000
EXPECTED_SAMPLE_WIDTH = 2

FULL_MANIFEST = "manifests/adaptive_augmented_3s_v1_full_manifest.csv"
DATASET_IDENTITY = "manifests/adaptive_augmented_3s_v1_dataset_identity.json"
TRAIN_MANIFEST = "manifests/adaptive_augmented_3s_v1_train_manifest.csv"
VALIDATION_MANIFEST = "manifests/adaptive_augmented_3s_v1_validation_manifest.csv"
LABEL_MAPPING = "manifests/adaptive_augmented_3s_v1_speaker_to_label.json"
SPEAKER_SPLIT = "splits/adaptive_augmented_3s_v1_speaker_split.csv"
SPLIT_IDENTITY = "splits/adaptive_augmented_3s_v1_split_identity.json"
VALIDATION_TRIALS = (
    "manifests/verification/adaptive_augmented_3s_v1_validation_trials.csv"
)
VALIDATION_TRIAL_IDENTITY = (
    "manifests/verification/"
    "adaptive_augmented_3s_v1_validation_trials_identity.json"
)

FULL_FIELDS = (
    "audio_path",
    "speaker_id",
    "sample_rate",
    "num_channels",
    "num_samples",
    "duration_sec",
)
PORTABLE_FIELDS = (
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "final_split",
)
SPLIT_FIELDS = ("speaker_id", "split", "ranking_sha256")


@dataclass(frozen=True)
class AudioRow:
    audio_path: str
    speaker_id: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_sec: str


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def render_csv(
    fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def ranking_digest(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}|{speaker_id}".encode("utf-8")).hexdigest()


def inspect_wav(path: Path, dataset_root: Path, skip_contract: bool) -> AudioRow:
    relative = path.relative_to(dataset_root).as_posix()
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or len(pure.parts) != 2
    ):
        raise ValueError(
            f"Audio must use <speaker_id>/<file>.wav layout: {relative}"
        )
    speaker_id = pure.parent.name
    try:
        with wave.open(str(path), "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            samples = stream.getnframes()
            sample_width = stream.getsampwidth()
            compression = stream.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Cannot read WAV {relative}: {error}") from error
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {relative}: {sample_rate}")
    if not skip_contract:
        actual = (sample_rate, channels, samples, sample_width, compression)
        expected = (
            EXPECTED_SAMPLE_RATE,
            EXPECTED_CHANNELS,
            EXPECTED_SAMPLES,
            EXPECTED_SAMPLE_WIDTH,
            "NONE",
        )
        if actual != expected:
            raise ValueError(
                f"Audio contract violation for {relative}: actual={actual}, "
                f"expected={expected}"
            )
    return AudioRow(
        audio_path=relative,
        speaker_id=speaker_id,
        sample_rate=sample_rate,
        num_channels=channels,
        num_samples=samples,
        duration_sec=f"{samples / sample_rate:.9f}".rstrip("0").rstrip("."),
    )


def scan_dataset(dataset_root: Path, skip_contract: bool = False) -> list[AudioRow]:
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    wavs = sorted(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".wav"
        ),
        key=lambda path: path.relative_to(dataset_root).as_posix().casefold(),
    )
    if not wavs:
        raise ValueError(f"No WAV files found under {dataset_root}")
    rows: list[AudioRow] = []
    for index, path in enumerate(wavs, start=1):
        rows.append(inspect_wav(path, dataset_root, skip_contract))
        if index % 1000 == 0 or index == len(wavs):
            print(f"SCAN {index}/{len(wavs)} WAV", flush=True)
    rows.sort(key=lambda row: (natural_key(row.speaker_id), row.audio_path.casefold()))
    paths = [row.audio_path for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative audio paths found")
    return rows


def split_speakers(
    rows: Sequence[AudioRow], validation_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    speakers = sorted({row.speaker_id for row in rows}, key=natural_key)
    if len(speakers) < 18:
        raise ValueError(
            "At least 18 speakers are required: 16 train speakers for P=16 "
            "and at least 2 validation speakers for EER"
        )
    if not math.isfinite(validation_ratio) or not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    validation_count = max(2, round(len(speakers) * validation_ratio))
    validation_count = min(validation_count, len(speakers) - 16)
    ranked = sorted(
        speakers, key=lambda speaker: (ranking_digest(seed, speaker), speaker)
    )
    validation_set = set(ranked[:validation_count])
    train_speakers = sorted(
        (speaker for speaker in speakers if speaker not in validation_set),
        key=natural_key,
    )
    validation_speakers = sorted(validation_set, key=natural_key)
    split_rows = [
        {
            "speaker_id": speaker,
            "split": "validation" if speaker in validation_set else "train",
            "ranking_sha256": ranking_digest(seed, speaker),
        }
        for speaker in speakers
    ]
    return split_rows, train_speakers, validation_speakers


def portable_manifests(
    rows: Sequence[AudioRow],
    train_speakers: Sequence[str],
    validation_speakers: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    train_set = set(train_speakers)
    validation_set = set(validation_speakers)
    if train_set & validation_set:
        raise RuntimeError("Speaker overlap between train and validation")
    labels = {speaker: index for index, speaker in enumerate(train_speakers)}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for row in rows:
        if row.speaker_id in train_set:
            train.append(
                {
                    "relative_audio_path": row.audio_path,
                    "speaker_id": row.speaker_id,
                    "speaker_label": labels[row.speaker_id],
                    "final_split": "train",
                }
            )
        elif row.speaker_id in validation_set:
            validation.append(
                {
                    "relative_audio_path": row.audio_path,
                    "speaker_id": row.speaker_id,
                    "speaker_label": -1,
                    "final_split": "validation",
                }
            )
        else:
            raise RuntimeError(f"Speaker was not assigned: {row.speaker_id}")
    if len(train) + len(validation) != len(rows):
        raise RuntimeError("Train/validation manifests dropped dataset rows")
    return train, validation, labels


def publish(path: Path, payload: bytes, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def create_package(
    dataset_root: Path,
    project_root: Path,
    seed: int = DEFAULT_SEED,
    *,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    trial_seed: int = DEFAULT_SEED,
    genuine_trials: int = 10_000,
    impostor_trials: int = 10_000,
    overwrite: bool = False,
    skip_audio_contract_check: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve(strict=True)
    project_root = project_root.expanduser().resolve(strict=True)
    rows = scan_dataset(dataset_root, skip_audio_contract_check)
    split_rows, train_speakers, validation_speakers = split_speakers(
        rows, validation_ratio, seed
    )
    train, validation, labels = portable_manifests(
        rows, train_speakers, validation_speakers
    )
    validation_rows = tuple(
        ValidationRow(row["relative_audio_path"], row["speaker_id"])
        for row in validation
    )
    trials = generate_validation_trials(
        validation_rows,
        seed=trial_seed,
        genuine_count=genuine_trials,
        impostor_count=impostor_trials,
    )
    trial_payload = trials_csv_bytes(trials)

    payloads: dict[str, bytes] = {
        FULL_MANIFEST: render_csv(FULL_FIELDS, (asdict(row) for row in rows)),
        TRAIN_MANIFEST: render_csv(PORTABLE_FIELDS, train),
        VALIDATION_MANIFEST: render_csv(PORTABLE_FIELDS, validation),
        LABEL_MAPPING: canonical_json(labels),
        SPEAKER_SPLIT: render_csv(SPLIT_FIELDS, split_rows),
        VALIDATION_TRIALS: trial_payload,
    }
    dataset_identity = {
        "schema_version": 2,
        "identity_kind": "generic_train_validation_dataset",
        "package_version": PACKAGE_VERSION,
        "path_base": "train_validation_dataset_root",
        "wav_count": len(rows),
        "speaker_count": len(train_speakers) + len(validation_speakers),
        "full_manifest_path": FULL_MANIFEST,
        "full_manifest_sha256": sha256_bytes(payloads[FULL_MANIFEST]),
        "audio_contract": {
            "sample_rate": EXPECTED_SAMPLE_RATE,
            "num_channels": EXPECTED_CHANNELS,
            "num_samples": EXPECTED_SAMPLES,
            "duration_sec": 3.0,
        },
    }
    payloads[DATASET_IDENTITY] = canonical_json(dataset_identity)
    split_identity = {
        "schema_version": 2,
        "identity_kind": "generic_train_validation_speaker_split",
        "package_version": PACKAGE_VERSION,
        "dataset_identity_sha256": sha256_bytes(payloads[DATASET_IDENTITY]),
        "split_seed": seed,
        "validation_ratio_requested": validation_ratio,
        "speaker_counts": {
            "train": len(train_speakers),
            "validation": len(validation_speakers),
        },
        "row_counts": {"train": len(train), "validation": len(validation)},
        "speaker_sets_pairwise_disjoint": True,
        "speaker_split_sha256": sha256_bytes(payloads[SPEAKER_SPLIT]),
        "manifest_sha256": {
            "train": sha256_bytes(payloads[TRAIN_MANIFEST]),
            "validation": sha256_bytes(payloads[VALIDATION_MANIFEST]),
        },
        "speaker_to_label_sha256": sha256_bytes(payloads[LABEL_MAPPING]),
        "train_label_range": [0, len(labels) - 1],
    }
    payloads[SPLIT_IDENTITY] = canonical_json(split_identity)
    trial_identity = {
        "schema_version": 2,
        "identity_kind": "generic_validation_trials",
        "package_version": PACKAGE_VERSION,
        "seed": trial_seed,
        "input_split": "validation",
        "validation_manifest_sha256": sha256_bytes(
            payloads[VALIDATION_MANIFEST]
        ),
        "speaker_count": len(validation_speakers),
        "trial_counts": {
            "genuine": genuine_trials,
            "impostor": impostor_trials,
            "total": genuine_trials + impostor_trials,
        },
        "trial_csv_sha256": sha256_bytes(trial_payload),
        "final_test_access": False,
    }
    trial_identity["identity_sha256"] = sha256_bytes(
        canonical_json(trial_identity)
    )
    payloads[VALIDATION_TRIAL_IDENTITY] = canonical_json(trial_identity)

    for relative, payload in payloads.items():
        publish(project_root / relative, payload, overwrite)
    return {
        "result": "PASS",
        "train_validation_speaker_overlap": 0,
        "speaker_counts": split_identity["speaker_counts"],
        "row_counts": split_identity["row_counts"],
        "validation_trials": trial_identity["trial_counts"],
        "num_classes": len(labels),
        "outputs": {name: sha256_bytes(value) for name, value in payloads.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trial-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--genuine-trials", type=int, default=10_000)
    parser.add_argument("--impostor-trials", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-audio-contract-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_package(
        args.dataset_root,
        PROJECT_ROOT,
        args.seed,
        validation_ratio=args.validation_ratio,
        trial_seed=args.trial_seed,
        genuine_trials=args.genuine_trials,
        impostor_trials=args.impostor_trials,
        overwrite=args.overwrite,
        skip_audio_contract_check=args.skip_audio_contract_check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
