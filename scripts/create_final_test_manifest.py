#!/usr/bin/env python3
"""Create one deterministic manifest for an independent final-test WAV set.

The script does not split data and does not create verification trials.  The
manifest is intended to be created once, stored persistently, and reused by all
RAW/RANDOM/ADAPTIVE evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import wave
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_verification import MANIFEST_FIELDS
from src.frozen_handoff_cache import canonical_digest, sha256_file


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


def _safe_relative(value: str) -> str:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe final-test relative path: {value!r}")
    return pure.as_posix()


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as stream:
            actual = {
                "sample_rate": stream.getframerate(),
                "channels": stream.getnchannels(),
                "frames": stream.getnframes(),
                "sample_width": stream.getsampwidth(),
                "compression": stream.getcomptype(),
            }
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError(f"Cannot read final-test WAV {path}: {error}") from error
    expected = {
        "sample_rate": 16_000,
        "channels": 1,
        "frames": 48_000,
        "sample_width": 2,
        "compression": "NONE",
    }
    if actual != expected:
        raise ValueError(
            "Expected mono PCM16 16 kHz / 3 s final-test WAV, "
            f"got {actual}: {path}"
        )


def _speaker_from_relative(relative: PurePosixPath, mode: str) -> str:
    if len(relative.parts) < 2:
        raise ValueError(
            "Final-test WAVs must live inside a speaker directory: "
            f"{relative.as_posix()}"
        )
    if mode == "first-dir":
        speaker = relative.parts[0]
    elif mode == "parent-dir":
        speaker = relative.parts[-2]
    else:  # argparse prevents this; keep the function safe for direct use.
        raise ValueError(f"Unsupported speaker mode: {mode}")
    speaker = speaker.strip()
    if not speaker:
        raise ValueError(f"Blank speaker ID for {relative.as_posix()}")
    return speaker


def scan_final_test(
    dataset_root: Path,
    *,
    source_dataset: str,
    speaker_mode: str = "first-dir",
    check_audio_contract: bool = True,
) -> list[dict[str, Any]]:
    root = dataset_root.expanduser().resolve(strict=True)
    wavs = sorted(
        (path for path in root.rglob("*.wav") if path.is_file()),
        key=lambda value: value.relative_to(root).as_posix().casefold(),
    )
    if not wavs:
        raise ValueError(f"No WAV files found under final-test root: {root}")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for path in wavs:
        relative = _safe_relative(path.relative_to(root).as_posix())
        if check_audio_contract:
            _validate_wav(path)
        speaker_id = _speaker_from_relative(PurePosixPath(relative), speaker_mode)
        sample_id = "final-test-" + hashlib.sha256(
            f"{source_dataset}\x1f{relative}".encode("utf-8")
        ).hexdigest()[:24]
        if sample_id in seen_ids or relative in seen_paths:
            raise ValueError(f"Duplicate final-test sample/path: {relative}")
        seen_ids.add(sample_id)
        seen_paths.add(relative)
        rows.append(
            {
                "sample_id": sample_id,
                "relative_audio_path": relative,
                "source_dataset": source_dataset,
                "source_recording_id": relative,
                "speaker_id": speaker_id,
                "speaker_label": -1,
                "final_split": "final_test",
            }
        )
    return rows


def render_manifest(rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_speakers_from_manifest(path: Path) -> set[str]:
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        speaker_column = next(
            (
                name
                for name in ("global_speaker_id", "speaker_id", "canonical_speaker_id")
                if name in fields
            ),
            None,
        )
        if speaker_column is None:
            raise ValueError(
                f"Cannot identify speaker column in disjoint manifest {path}"
            )
        return {
            str(row[speaker_column]).strip()
            for row in reader
            if str(row[speaker_column]).strip()
        }


def create_manifest(
    *,
    dataset_root: Path,
    output_manifest: Path,
    identity_path: Path,
    source_dataset: str,
    speaker_mode: str,
    disjoint_manifests: Iterable[Path] = (),
    check_audio_contract: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dataset = source_dataset.strip()
    if not source_dataset:
        raise ValueError("source-dataset must not be blank")

    rows = scan_final_test(
        dataset_root,
        source_dataset=source_dataset,
        speaker_mode=speaker_mode,
        check_audio_contract=check_audio_contract,
    )
    speakers = {str(row["speaker_id"]) for row in rows}
    if len(speakers) < 2:
        raise ValueError("Independent final test needs at least two speakers")

    disjoint_speakers: set[str] = set()
    for path in disjoint_manifests:
        disjoint_speakers.update(read_speakers_from_manifest(path))
    overlap = speakers & disjoint_speakers
    if overlap:
        raise ValueError(
            "Speaker leakage between final test and supplied train/validation "
            "manifest(s): " + ", ".join(sorted(overlap)[:20])
        )

    payload = render_manifest(rows)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    identity = {
        "schema_version": 1,
        "identity_kind": "frozen_independent_final_test_manifest",
        "source_dataset": source_dataset,
        "speaker_mode": speaker_mode,
        "audio_contract": (
            "mono_pcm16_16khz_48000_samples"
            if check_audio_contract
            else "not_checked"
        ),
        "row_count": len(rows),
        "speaker_count": len(speakers),
        "manifest_sha256": manifest_sha,
        "speaker_disjointness_checked_against": [
            str(Path(path).expanduser().resolve()) for path in disjoint_manifests
        ],
    }
    identity["identity_sha256"] = canonical_digest(identity)
    identity_payload = canonical_json(identity)

    output_manifest = output_manifest.expanduser().resolve()
    identity_path = identity_path.expanduser().resolve()
    if output_manifest.exists() and not overwrite:
        if output_manifest.read_bytes() != payload:
            raise FileExistsError(
                "A different final-test manifest already exists. Refusing to "
                f"overwrite frozen artifact: {output_manifest}"
            )
        if not identity_path.is_file():
            raise FileNotFoundError(
                "Final-test manifest exists but its identity file is missing: "
                f"{identity_path}"
            )
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        actual = dict(existing)
        digest = actual.pop("identity_sha256", None)
        if (
            digest != canonical_digest(actual)
            or existing.get("manifest_sha256") != sha256_file(output_manifest)
        ):
            raise ValueError("Existing final-test manifest identity is invalid")
        return {
            "result": "REUSED",
            "manifest": str(output_manifest),
            "identity": str(identity_path),
            "rows": len(rows),
            "speakers": len(speakers),
            "manifest_sha256": manifest_sha,
        }

    if identity_path.exists() and not overwrite:
        raise FileExistsError(
            f"Identity exists while manifest is absent: {identity_path}"
        )
    atomic_bytes(output_manifest, payload)
    atomic_bytes(identity_path, identity_payload)
    return {
        "result": "CREATED",
        "manifest": str(output_manifest),
        "identity": str(identity_path),
        "rows": len(rows),
        "speakers": len(speakers),
        "manifest_sha256": manifest_sha,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--source-dataset", default="independent_test")
    parser.add_argument(
        "--speaker-mode",
        choices=("first-dir", "parent-dir"),
        default="first-dir",
        help=(
            "How to derive speaker_id from each relative WAV path. "
            "first-dir fits <root>/<speaker>/.../*.wav; parent-dir uses the "
            "immediate WAV parent directory."
        ),
    )
    parser.add_argument(
        "--disjoint-manifest",
        action="append",
        default=[],
        type=Path,
        help=(
            "Optional manifest whose speaker IDs must not overlap final test. "
            "Pass multiple times when needed."
        ),
    )
    parser.add_argument("--skip-audio-contract-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    identity = args.identity
    if identity is None:
        identity = args.output_manifest.with_name("test_manifest_identity.json")
    result = create_manifest(
        dataset_root=args.dataset_root,
        output_manifest=args.output_manifest,
        identity_path=identity,
        source_dataset=args.source_dataset,
        speaker_mode=args.speaker_mode,
        disjoint_manifests=args.disjoint_manifest,
        check_audio_contract=not args.skip_audio_contract_check,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
