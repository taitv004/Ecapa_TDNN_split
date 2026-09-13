#!/usr/bin/env python3
"""Create deterministic speaker-disjoint manifests for an ECAPA dataset.

Expected dataset layout:

    dataset_root/
      speaker_001/audio_001.wav
      speaker_001/audio_002.wav
      speaker_002/audio_001.wav

Unlike the repository's production-only package generator, this script does
not fix the number of speakers, WAV files, or split sizes. Split sizes are
calculated from ratios. The output manifest schema remains compatible with the
ECAPA_TDNN_split repository.
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
import uuid
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SEED = 2026
DEFAULT_TRAIN_RATIO = 0.80
DEFAULT_VALIDATION_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10

# These are defaults, not locked dataset identities. They can be changed from
# the command line. Keeping 16 kHz mono, 48,000 samples preserves the repo's
# expected 3-second FBank shape [301, 80].
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLES = 48_000
DEFAULT_SAMPLE_WIDTH = 2

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
VALID_SPLITS = ("train", "validation", "final_test")


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


def render_csv(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ranking_digest(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}|{speaker_id}".encode("utf-8")).hexdigest()


def safe_relative_audio_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"Unsafe relative audio path: {relative!r}")
    if len(pure.parts) != 2:
        raise ValueError(
            f"Audio must be directly below its speaker folder: {relative}. "
            "Expected <speaker_id>/<audio>.wav"
        )
    return relative


def inspect_wav(
    path: Path,
    root: Path,
    expected_sample_rate: int,
    expected_channels: int,
    expected_samples: int,
    expected_sample_width: int,
    skip_audio_contract_check: bool,
) -> AudioRow:
    relative = safe_relative_audio_path(path, root)
    speaker_id = PurePosixPath(relative).parent.name
    try:
        with wave.open(str(path), "rb") as stream:
            sample_rate = stream.getframerate()
            channels = stream.getnchannels()
            samples = stream.getnframes()
            sample_width = stream.getsampwidth()
            compression = stream.getcomptype()
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Cannot read WAV header {relative}: {error}") from error

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in {relative}: {sample_rate}")
    duration = samples / sample_rate
    if not skip_audio_contract_check:
        violations: list[str] = []
        if sample_rate != expected_sample_rate:
            violations.append(f"sample_rate={sample_rate}, expected={expected_sample_rate}")
        if channels != expected_channels:
            violations.append(f"channels={channels}, expected={expected_channels}")
        if samples != expected_samples:
            violations.append(f"samples={samples}, expected={expected_samples}")
        if sample_width != expected_sample_width:
            violations.append(
                f"sample_width={sample_width}, expected={expected_sample_width}"
            )
        if compression != "NONE":
            violations.append(f"compression={compression}, expected=NONE")
        if violations:
            raise ValueError(
                f"Audio contract violation for {relative}: " + "; ".join(violations)
            )

    return AudioRow(
        audio_path=relative,
        speaker_id=speaker_id,
        sample_rate=sample_rate,
        num_channels=channels,
        num_samples=samples,
        duration_sec=f"{duration:.9f}".rstrip("0").rstrip("."),
    )


def discover_dataset(
    root: Path,
    expected_sample_rate: int,
    expected_channels: int,
    expected_samples: int,
    expected_sample_width: int,
    skip_audio_contract_check: bool,
) -> list[AudioRow]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    wavs = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".wav"
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not wavs:
        raise ValueError(f"No WAV files found under {root}")

    rows: list[AudioRow] = []
    for index, path in enumerate(wavs, start=1):
        rows.append(
            inspect_wav(
                path,
                root,
                expected_sample_rate,
                expected_channels,
                expected_samples,
                expected_sample_width,
                skip_audio_contract_check,
            )
        )
        if index % 1_000 == 0 or index == len(wavs):
            print(f"SCAN {index}/{len(wavs)} WAV", flush=True)

    rows.sort(key=lambda row: (natural_key(row.speaker_id), row.audio_path.casefold()))
    paths = [row.audio_path for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate relative audio paths found")
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError("Case-colliding relative audio paths found")
    if len({row.speaker_id for row in rows}) < 3:
        raise ValueError("At least three speakers are required for train/validation/test")
    return rows


def validate_ratios(train: float, validation: float, test: float) -> None:
    ratios = {"train": train, "validation": validation, "final_test": test}
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios.values()):
        raise ValueError("All split ratios must be finite and greater than zero")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {sum(ratios.values()):.12f}"
        )


def ratio_counts(
    total: int, train: float, validation: float, test: float
) -> dict[str, int]:
    """Allocate integer speaker counts with the largest-remainder method."""
    validate_ratios(train, validation, test)
    ratios = {"train": train, "validation": validation, "final_test": test}
    exact = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        VALID_SPLITS,
        key=lambda name: (-(exact[name] - counts[name]), VALID_SPLITS.index(name)),
    )
    for name in order[:remaining]:
        counts[name] += 1
    if any(counts[name] < 1 for name in VALID_SPLITS):
        raise ValueError(
            f"Ratios produce an empty split for {total} speakers: {counts}"
        )
    if sum(counts.values()) != total:
        raise RuntimeError("Internal split allocation error")
    return counts


def read_reused_split(path: Path, speakers: set[str]) -> list[dict[str, str]]:
    with path.resolve(strict=True).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not {"speaker_id", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("Reused split CSV needs speaker_id and split columns")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for line, raw in enumerate(reader, start=2):
            speaker = raw["speaker_id"].strip()
            split = raw["split"].strip()
            if not speaker or speaker in seen:
                raise ValueError(f"Duplicate/empty speaker at reused split line {line}")
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split {split!r} at reused split line {line}")
            seen.add(speaker)
            rows.append(
                {
                    "speaker_id": speaker,
                    "split": split,
                    "ranking_sha256": raw.get("ranking_sha256", "").strip(),
                }
            )
    missing = speakers - seen
    extra = seen - speakers
    if missing or extra:
        raise ValueError(
            "Reused split speaker set differs from dataset: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    counts = Counter(row["split"] for row in rows)
    if any(counts[name] < 1 for name in VALID_SPLITS):
        raise ValueError(f"Reused split has an empty partition: {dict(counts)}")
    rows.sort(key=lambda row: natural_key(row["speaker_id"]))
    return rows


def create_split(
    speakers: set[str],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    reuse_path: Path | None,
) -> tuple[list[dict[str, str]], dict[str, int], str]:
    if reuse_path is not None:
        rows = read_reused_split(reuse_path, speakers)
        counts = Counter(row["split"] for row in rows)
        return rows, {name: counts[name] for name in VALID_SPLITS}, "reused_csv"

    counts = ratio_counts(
        len(speakers), train_ratio, validation_ratio, test_ratio
    )
    ranked = sorted(
        speakers,
        key=lambda speaker: (ranking_digest(seed, speaker), speaker),
    )
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]
    assignments: dict[str, str] = {}
    for index, speaker in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "final_test"
        assignments[speaker] = split
    rows = [
        {
            "speaker_id": speaker,
            "split": assignments[speaker],
            "ranking_sha256": ranking_digest(seed, speaker),
        }
        for speaker in sorted(speakers, key=natural_key)
    ]
    return rows, counts, "sha256_ratio_split"


def create_payloads(
    audio_rows: Sequence[AudioRow],
    split_rows: Sequence[Mapping[str, str]],
    dataset_root: Path,
    seed: int,
    ratios: Mapping[str, float],
    split_method: str,
    audio_contract: Mapping[str, Any],
) -> dict[str, bytes]:
    assignments = {row["speaker_id"]: row["split"] for row in split_rows}
    train_speakers = sorted(
        (
            speaker
            for speaker, split in assignments.items()
            if split == "train"
        ),
        key=natural_key,
    )
    labels = {speaker: index for index, speaker in enumerate(train_speakers)}

    manifests: dict[str, list[dict[str, Any]]] = {
        name: [] for name in VALID_SPLITS
    }
    for row in audio_rows:
        split = assignments[row.speaker_id]
        manifests[split].append(
            {
                "relative_audio_path": row.audio_path,
                "speaker_id": row.speaker_id,
                "speaker_label": labels[row.speaker_id] if split == "train" else -1,
                "final_split": split,
            }
        )

    speaker_counts = Counter(assignments.values())
    row_counts = {name: len(manifests[name]) for name in VALID_SPLITS}
    payloads: dict[str, bytes] = {
        "full_manifest.csv": render_csv(
            FULL_FIELDS, (asdict(row) for row in audio_rows)
        ),
        "speaker_split.csv": render_csv(SPLIT_FIELDS, split_rows),
        "train_manifest.csv": render_csv(PORTABLE_FIELDS, manifests["train"]),
        "validation_manifest.csv": render_csv(
            PORTABLE_FIELDS, manifests["validation"]
        ),
        "final_test_manifest.csv": render_csv(
            PORTABLE_FIELDS, manifests["final_test"]
        ),
        "speaker_to_label.json": canonical_json(labels),
    }
    dataset_identity = {
        "schema_version": 1,
        "identity_kind": "generic_ecapa_dataset",
        "dataset_root_at_creation": str(dataset_root.resolve()),
        "wav_content_hashing": False,
        "wav_count": len(audio_rows),
        "speaker_count": len(assignments),
        "audio_contract": dict(audio_contract),
        "full_manifest_sha256": sha256_bytes(payloads["full_manifest.csv"]),
    }
    payloads["dataset_identity.json"] = canonical_json(dataset_identity)
    split_identity = {
        "schema_version": 1,
        "identity_kind": "generic_ecapa_speaker_split",
        "dataset_identity_sha256": sha256_bytes(payloads["dataset_identity.json"]),
        "split_method": split_method,
        "split_seed": seed if split_method == "sha256_ratio_split" else None,
        "requested_ratios": dict(ratios),
        "speaker_counts": {
            name: speaker_counts[name] for name in VALID_SPLITS
        },
        "row_counts": row_counts,
        "speaker_sets_pairwise_disjoint": True,
        "all_speakers_assigned_once": True,
        "all_rows_assigned_once": sum(row_counts.values()) == len(audio_rows),
        "train_label_range": [0, len(labels) - 1],
        "evaluation_label_sentinel": -1,
        "files": {
            name: sha256_bytes(payload)
            for name, payload in payloads.items()
            if name.endswith((".csv", ".json"))
        },
    }
    payloads["split_identity.json"] = canonical_json(split_identity)
    return payloads


def publish(output_dir: Path, payloads: Mapping[str, bytes], overwrite: bool) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in payloads if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:10])
        raise FileExistsError(
            f"Output files already exist ({names}). Use --overwrite to replace them."
        )

    staged: list[tuple[Path, Path]] = []
    try:
        for name, payload in payloads.items():
            target = output_dir / name
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(payload)
            staged.append((temporary, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise


def verify_outputs(output_dir: Path) -> dict[str, Any]:
    manifests: dict[str, list[dict[str, str]]] = {}
    for split in VALID_SPLITS:
        path = output_dir / f"{split}_manifest.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != PORTABLE_FIELDS:
                raise ValueError(f"Unexpected schema: {path}")
            manifests[split] = list(reader)

    speaker_sets = {
        split: {row["speaker_id"] for row in rows}
        for split, rows in manifests.items()
    }
    for left_index, left in enumerate(VALID_SPLITS):
        for right in VALID_SPLITS[left_index + 1 :]:
            overlap = speaker_sets[left] & speaker_sets[right]
            if overlap:
                raise RuntimeError(
                    f"Speaker overlap between {left} and {right}: {sorted(overlap)[:10]}"
                )
    train_labels = sorted(
        {int(row["speaker_label"]) for row in manifests["train"]}
    )
    if train_labels != list(range(len(speaker_sets["train"]))):
        raise RuntimeError("Train speaker labels are not contiguous from zero")
    for split in ("validation", "final_test"):
        if {row["speaker_label"] for row in manifests[split]} != {"-1"}:
            raise RuntimeError(f"{split} labels must all be -1")

    return {
        "speaker_counts": {
            split: len(speaker_sets[split]) for split in VALID_SPLITS
        },
        "row_counts": {
            split: len(manifests[split]) for split in VALID_SPLITS
        },
        "speaker_overlap": 0,
        "output_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument(
        "--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-speaker-split", type=Path)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--sample-width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    parser.add_argument(
        "--skip-audio-contract-check",
        action="store_true",
        help="record actual WAV metadata without enforcing the expected 3-second contract",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate < 1 or args.channels < 1 or args.samples < 1:
        raise ValueError("Audio contract values must be positive")
    if args.sample_width < 1:
        raise ValueError("sample-width must be positive")
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)

    dataset_root = args.dataset_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if dataset_root == output_dir or dataset_root in output_dir.parents:
        raise ValueError("Output directory must be outside dataset root")

    audio_rows = discover_dataset(
        dataset_root,
        args.sample_rate,
        args.channels,
        args.samples,
        args.sample_width,
        args.skip_audio_contract_check,
    )
    speakers = {row.speaker_id for row in audio_rows}
    split_rows, speaker_counts, method = create_split(
        speakers,
        args.seed,
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
        args.reuse_speaker_split,
    )
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "final_test": args.test_ratio,
    }
    audio_contract = {
        "validation_skipped": args.skip_audio_contract_check,
        "expected_sample_rate": args.sample_rate,
        "expected_channels": args.channels,
        "expected_samples": args.samples,
        "expected_sample_width_bytes": args.sample_width,
        "expected_compression": "NONE",
    }
    payloads = create_payloads(
        audio_rows,
        split_rows,
        dataset_root,
        args.seed,
        ratios,
        method,
        audio_contract,
    )
    publish(output_dir, payloads, args.overwrite)
    verification = verify_outputs(output_dir)
    result = {
        "result": "PASS",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "split_method": method,
        "speaker_counts": speaker_counts,
        **verification,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
