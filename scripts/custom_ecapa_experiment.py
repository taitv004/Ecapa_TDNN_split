#!/usr/bin/env python3
"""Custom-data ECAPA/AAM pipeline built on the cloned ECAPA_TDNN_split repo.

The production experiment in that repository is immutable and dataset-bound.
This runner creates a separate custom experiment without modifying or
pretending to reuse the production identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler


SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 48_000
MIN_SEGMENT_SAMPLES = 24_000
FEATURE_SHAPE = (301, 80)
EMBEDDING_DIM = 192
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
MANIFEST_FIELDS = (
    "id",
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "split",
)
TRIAL_FIELDS = (
    "trial_id",
    "target",
    "left_path",
    "right_path",
    "left_speaker_id",
    "right_speaker_id",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value))


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value))


def discover_sources(raw_root: Path) -> list[tuple[str, Path]]:
    raw_root = raw_root.resolve(strict=True)
    speakers = sorted((path for path in raw_root.iterdir() if path.is_dir()), key=lambda path: natural_key(path.name))
    sources: list[tuple[str, Path]] = []
    for speaker_dir in speakers:
        files = sorted(
            (
                path
                for path in speaker_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            ),
            key=lambda path: path.relative_to(raw_root).as_posix().casefold(),
        )
        sources.extend((speaker_dir.name, path) for path in files)
    if not sources:
        raise ValueError(f"No supported audio files found under {raw_root}")
    return sources


def safe_clean_directory(path: Path) -> None:
    path = path.resolve()
    if len(path.parts) < 4:
        raise ValueError(f"Refusing to delete broad path: {path}")
    if path.exists():
        shutil.rmtree(path)


def output_segment_path(
    source: Path,
    raw_root: Path,
    processed_root: Path,
    speaker_id: str,
    segment_index: int,
) -> Path:
    relative = source.relative_to(raw_root)
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    nested_parent = relative.parent.relative_to(speaker_id)
    name = f"{source.stem}--{digest}--seg{segment_index:04d}.wav"
    return processed_root / speaker_id / nested_parent / name


def valid_processed_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        info = sf.info(path)
    except RuntimeError:
        return False
    return (
        info.samplerate == SAMPLE_RATE
        and info.channels == 1
        and info.frames == SEGMENT_SAMPLES
    )


def normalize_and_segment(
    raw_root: Path,
    processed_root: Path,
    clean: bool,
) -> dict[str, Any]:
    raw_root = raw_root.resolve(strict=True)
    processed_root = processed_root.resolve()
    if raw_root == processed_root or raw_root in processed_root.parents:
        raise ValueError("processed_root must be outside raw_root")
    if clean:
        safe_clean_directory(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(raw_root)
    rows: list[dict[str, Any]] = []
    written = reused = dropped = 0
    invalid_sources: list[dict[str, str]] = []
    for number, (speaker_id, source) in enumerate(sources, start=1):
        try:
            audio, original_rate = sf.read(source, dtype="float32", always_2d=True)
            if audio.shape[0] == 0 or audio.shape[1] == 0:
                raise ValueError("empty audio")
            waveform = torch.from_numpy(audio.T.copy()).mean(dim=0, keepdim=True)
            if not bool(torch.isfinite(waveform).all()):
                raise ValueError("audio contains NaN or Inf")
            if int(original_rate) <= 0:
                raise ValueError(f"invalid sample rate: {original_rate}")
            if int(original_rate) != SAMPLE_RATE:
                waveform = AF.resample(waveform, int(original_rate), SAMPLE_RATE)
            waveform = waveform.squeeze(0).clamp(-1.0, 1.0)
        except Exception as error:
            relative_source = source.relative_to(raw_root).as_posix()
            invalid_sources.append(
                {
                    "speaker_id": speaker_id,
                    "source": relative_source,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"SKIP INVALID AUDIO {relative_source} | "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue

        segment_index = 0
        start = 0
        produced_for_source = 0
        while start < waveform.numel():
            remaining = waveform.numel() - start
            if remaining < MIN_SEGMENT_SAMPLES:
                break
            stop = min(start + SEGMENT_SAMPLES, waveform.numel())
            segment = waveform[start:stop]
            original_segment_samples = int(segment.numel())
            if segment.numel() < SEGMENT_SAMPLES:
                segment = F.pad(segment, (0, SEGMENT_SAMPLES - segment.numel()))
            target = output_segment_path(
                source, raw_root, processed_root, speaker_id, segment_index
            )
            if valid_processed_wav(target):
                reused += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    target,
                    segment.cpu().numpy(),
                    SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )
                if not valid_processed_wav(target):
                    raise RuntimeError(f"Invalid processed output: {target}")
                written += 1
            rows.append(
                {
                    "speaker_id": speaker_id,
                    "source": source.relative_to(raw_root).as_posix(),
                    "output": target.relative_to(processed_root).as_posix(),
                    "segment_index": segment_index,
                    "source_segment_samples": original_segment_samples,
                    "zero_padding_samples": SEGMENT_SAMPLES - original_segment_samples,
                }
            )
            produced_for_source += 1
            segment_index += 1
            start += SEGMENT_SAMPLES
        if produced_for_source == 0:
            dropped += 1
        if number % 500 == 0 or number == len(sources):
            print(
                f"PREPROCESS {number}/{len(sources)} | segments={len(rows)} | "
                f"written={written} | reused={reused}",
                flush=True,
            )

    report_path = processed_root / "custom_preprocess_report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = tuple(rows[0]) if rows else (
            "speaker_id", "source", "output", "segment_index",
            "source_segment_samples", "zero_padding_samples",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    invalid_report_path = processed_root / "invalid_audio_report.csv"
    with invalid_report_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ("speaker_id", "source", "error_type", "error")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(invalid_sources)
    return {
        "source_recordings": len(sources),
        "processed_segments": len(rows),
        "written_segments": written,
        "reused_segments": reused,
        "dropped_short_recordings": dropped,
        "invalid_source_recordings": len(invalid_sources),
        "report": str(report_path),
        "invalid_audio_report": str(invalid_report_path),
    }


def scan_processed(processed_root: Path) -> dict[str, list[Path]]:
    processed_root = processed_root.resolve(strict=True)
    result: dict[str, list[Path]] = {}
    for speaker_dir in sorted(
        (path for path in processed_root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    ):
        paths = sorted(speaker_dir.rglob("*.wav"), key=lambda path: path.relative_to(processed_root).as_posix())
        if paths:
            invalid = [path for path in paths if not valid_processed_wav(path)]
            if invalid:
                raise ValueError(f"Invalid normalized WAV: {invalid[0]}")
            result[speaker_dir.name] = paths
    if len(result) < 18:
        raise ValueError("At least 18 speakers are required for P=16 plus validation")
    insufficient = [speaker for speaker, paths in result.items() if len(paths) < 2]
    if insufficient:
        raise ValueError(
            "Every speaker needs at least two 3-second segments for P×K; "
            f"insufficient={insufficient[:10]}"
        )
    return result


def speaker_rank(seed: int, speaker_id: str) -> str:
    return hashlib.sha256(f"{seed}|{speaker_id}".encode("utf-8")).hexdigest()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_manifest_rows(
    speaker_paths: Mapping[str, Sequence[Path]],
    processed_root: Path,
    split: str,
    labels: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for speaker_id in sorted(speaker_paths, key=natural_key):
        for path in speaker_paths[speaker_id]:
            relative = path.relative_to(processed_root).as_posix()
            utterance_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "id": utterance_id,
                    "relative_audio_path": relative,
                    "speaker_id": speaker_id,
                    "speaker_label": labels[speaker_id] if split == "train" else -1,
                    "split": split,
                }
            )
    return rows


def generate_trials(
    validation_rows: Sequence[Mapping[str, Any]],
    genuine_count: int,
    impostor_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in validation_rows:
        grouped[str(row["speaker_id"])].append(str(row["relative_audio_path"]))
    rng = random.Random(seed)

    genuine_candidates: list[tuple[str, str, str]] = []
    for speaker, paths in grouped.items():
        for left, right in itertools.combinations(sorted(paths), 2):
            genuine_candidates.append((speaker, left, right))
    if len(genuine_candidates) < genuine_count:
        raise ValueError(
            f"Only {len(genuine_candidates)} unique genuine pairs are available; "
            f"requested {genuine_count}"
        )
    rng.shuffle(genuine_candidates)

    trials: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for index, (speaker, left, right) in enumerate(genuine_candidates[:genuine_count]):
        pair = tuple(sorted((left, right)))
        used_pairs.add(pair)
        trials.append(
            {
                "trial_id": f"genuine-{index:06d}",
                "target": 1,
                "left_path": left,
                "right_path": right,
                "left_speaker_id": speaker,
                "right_speaker_id": speaker,
            }
        )

    speakers = sorted(grouped, key=natural_key)
    attempts = 0
    while len(trials) < genuine_count + impostor_count:
        attempts += 1
        if attempts > impostor_count * 1000:
            raise RuntimeError("Could not generate enough unique impostor trials")
        left_speaker, right_speaker = rng.sample(speakers, 2)
        left = rng.choice(grouped[left_speaker])
        right = rng.choice(grouped[right_speaker])
        pair = tuple(sorted((left, right)))
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        number = len(trials) - genuine_count
        trials.append(
            {
                "trial_id": f"impostor-{number:06d}",
                "target": 0,
                "left_path": left,
                "right_path": right,
                "left_speaker_id": left_speaker,
                "right_speaker_id": right_speaker,
            }
        )
    rng.shuffle(trials)
    for index, trial in enumerate(trials):
        trial["trial_id"] = f"trial-{index:06d}"
    return trials


def create_artifacts(
    processed_root: Path,
    artifacts_dir: Path,
    validation_ratio: float,
    split_seed: int,
    trial_seed: int,
    genuine_trials: int,
    impostor_trials: int,
) -> dict[str, Any]:
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    speakers = scan_processed(processed_root)
    ranked = sorted(speakers, key=lambda speaker: (speaker_rank(split_seed, speaker), speaker))
    validation_count = max(2, int(round(len(ranked) * validation_ratio)))
    if len(ranked) - validation_count < 16:
        raise ValueError("Split leaves fewer than 16 train speakers")
    validation_speakers = set(ranked[-validation_count:])
    train_speakers = [speaker for speaker in ranked if speaker not in validation_speakers]
    train_speakers = sorted(train_speakers, key=natural_key)
    labels = {speaker: index for index, speaker in enumerate(train_speakers)}

    train_map = {speaker: speakers[speaker] for speaker in train_speakers}
    validation_map = {
        speaker: speakers[speaker]
        for speaker in sorted(validation_speakers, key=natural_key)
    }
    train_rows = make_manifest_rows(train_map, processed_root, "train", labels)
    validation_rows = make_manifest_rows(validation_map, processed_root, "validation", labels)
    trials = generate_trials(
        validation_rows, genuine_trials, impostor_trials, trial_seed
    )
    split_rows = [
        {
            "speaker_id": speaker,
            "split": "validation" if speaker in validation_speakers else "train",
            "ranking_sha256": speaker_rank(split_seed, speaker),
        }
        for speaker in sorted(speakers, key=natural_key)
    ]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_path = artifacts_dir / "train_manifest.csv"
    validation_path = artifacts_dir / "validation_manifest.csv"
    trials_path = artifacts_dir / "validation_trials.csv"
    split_path = artifacts_dir / "speaker_split.csv"
    labels_path = artifacts_dir / "speaker_to_label.json"
    write_csv(train_path, MANIFEST_FIELDS, train_rows)
    write_csv(validation_path, MANIFEST_FIELDS, validation_rows)
    write_csv(trials_path, TRIAL_FIELDS, trials)
    write_csv(split_path, ("speaker_id", "split", "ranking_sha256"), split_rows)
    atomic_json(labels_path, labels)

    identity = {
        "schema_version": 1,
        "identity_kind": "custom_ecapa_dataset_artifacts",
        "audio_contract": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "samples": SEGMENT_SAMPLES,
            "duration_seconds": 3.0,
            "short_tail_policy": "zero_pad_if_at_least_1.5_seconds_else_drop",
        },
        "split": {
            "unit": "speaker",
            "seed": split_seed,
            "validation_ratio": validation_ratio,
            "train_speakers": len(train_speakers),
            "validation_speakers": len(validation_speakers),
            "speaker_overlap": 0,
        },
        "rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "trials": {
            "seed": trial_seed,
            "genuine": genuine_trials,
            "impostor": impostor_trials,
            "total": len(trials),
        },
        "files": {
            path.name: sha256_file(path)
            for path in (train_path, validation_path, trials_path, split_path, labels_path)
        },
    }
    identity["identity_sha256"] = hashlib.sha256(canonical_json(identity)).hexdigest()
    atomic_json(artifacts_dir / "dataset_artifacts_identity.json", identity)
    return identity


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Invalid manifest schema: {path}")
        rows = []
        for raw in reader:
            row = dict(raw)
            row["speaker_label"] = int(row["speaker_label"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def read_trials(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRIAL_FIELDS:
            raise ValueError(f"Invalid trials schema: {path}")
        trials = []
        for raw in reader:
            trial = dict(raw)
            trial["target"] = int(trial["target"])
            trials.append(trial)
    if {trial["target"] for trial in trials} != {0, 1}:
        raise ValueError("Trials need both genuine and impostor pairs")
    return trials


def load_waveform(path: Path) -> torch.Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape != (SEGMENT_SAMPLES, 1):
        raise ValueError(f"Processed WAV violates 3-second mono contract: {path}")
    waveform = torch.from_numpy(audio[:, 0].copy())
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"Non-finite waveform: {path}")
    return waveform


def ensure_repo_imports(repo_root: Path):
    repo_root = repo_root.resolve(strict=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.speechbrain_frontend import SpeechBrainECAPAFrontend

    return SpeechBrainECAPAFrontend


def validate_shard(
    shard: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    required = {"features", "speaker_labels", "speaker_ids", "relative_audio_paths", "split"}
    if not isinstance(shard, Mapping) or set(shard) != required:
        raise ValueError("Malformed custom FBank shard")
    count = len(expected_rows)
    if (
        tuple(shard["features"].shape) != (count, *FEATURE_SHAPE)
        or shard["features"].dtype != torch.float32
        or shard["features"].device.type != "cpu"
        or tuple(shard["speaker_labels"].shape) != (count,)
        or shard["speaker_labels"].dtype != torch.long
        or shard["speaker_ids"] != [row["speaker_id"] for row in expected_rows]
        or shard["relative_audio_paths"] != [row["relative_audio_path"] for row in expected_rows]
        or shard["split"] != expected_rows[0]["split"]
    ):
        raise ValueError("Custom FBank shard does not match its manifest slice")


def reclaim_processed_wavs(
    processed_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Delete only processed WAVs whose completed cache shard was verified."""
    root = processed_root.resolve(strict=True)
    removed = 0
    reclaimed = 0
    for row in rows:
        path = (root / str(row["relative_audio_path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Manifest path escapes processed root: {path}") from error
        if path.is_file():
            reclaimed += path.stat().st_size
            path.unlink()
            removed += 1
    return removed, reclaimed


def build_cache(
    repo_root: Path,
    processed_root: Path,
    artifacts_dir: Path,
    cache_dir: Path,
    device: str,
    batch_size: int,
    shard_size: int,
    delete_processed_after_shard: bool,
) -> dict[str, Any]:
    if batch_size < 1 or shard_size < 1:
        raise ValueError("batch_size and shard_size must be positive")
    SpeechBrainECAPAFrontend = ensure_repo_imports(repo_root)
    manifests = {
        "train": artifacts_dir / "train_manifest.csv",
        "validation": artifacts_dir / "validation_manifest.csv",
    }
    rows_by_split = {split: read_manifest(path) for split, path in manifests.items()}
    artifact_identity = json.loads(
        (artifacts_dir / "dataset_artifacts_identity.json").read_text(encoding="utf-8")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()

    shard_counts: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        count = math.ceil(len(rows) / shard_size)
        shard_counts[split] = count
        for shard_index in range(count):
            start = shard_index * shard_size
            shard_rows = rows[start : start + shard_size]
            shard_path = split_dir / f"shard_{shard_index:05d}.pt"
            if shard_path.is_file():
                existing = torch.load(shard_path, map_location="cpu", weights_only=False)
                validate_shard(existing, shard_rows)
                if delete_processed_after_shard:
                    removed, reclaimed = reclaim_processed_wavs(processed_root, shard_rows)
                    print(
                        f"CACHE {split}: verified existing shard {shard_index + 1}/{count} | "
                        f"deleted_wavs={removed} | reclaimed_MiB={reclaimed / 2**20:.2f}",
                        flush=True,
                    )
                continue
            features: list[torch.Tensor] = []
            for batch_start in range(0, len(shard_rows), batch_size):
                batch_rows = shard_rows[batch_start : batch_start + batch_size]
                waveforms = torch.stack(
                    [
                        load_waveform(processed_root / row["relative_audio_path"])
                        for row in batch_rows
                    ]
                )
                with torch.inference_mode():
                    value = frontend.compute_features(waveforms).float().cpu()
                if value.ndim != 3 or tuple(value.shape[1:]) != FEATURE_SHAPE:
                    raise RuntimeError(f"Unexpected SpeechBrain FBank shape: {tuple(value.shape)}")
                features.append(value)
            shard = {
                "features": torch.cat(features, dim=0),
                "speaker_labels": torch.tensor(
                    [row["speaker_label"] for row in shard_rows], dtype=torch.long
                ),
                "speaker_ids": [row["speaker_id"] for row in shard_rows],
                "relative_audio_paths": [row["relative_audio_path"] for row in shard_rows],
                "split": split,
            }
            validate_shard(shard, shard_rows)
            atomic_torch_save(shard_path, shard)
            removed = reclaimed = 0
            if delete_processed_after_shard:
                # Deletion happens only after the persisted shard is loaded back
                # and checked against the exact manifest slice.
                persisted = torch.load(shard_path, map_location="cpu", weights_only=False)
                validate_shard(persisted, shard_rows)
                removed, reclaimed = reclaim_processed_wavs(processed_root, shard_rows)
            print(
                f"CACHE {split}: shard {shard_index + 1}/{count} | rows={len(shard_rows)} | "
                f"deleted_wavs={removed} | reclaimed_MiB={reclaimed / 2**20:.2f}",
                flush=True,
            )

    identity = {
        "schema_version": 1,
        "identity_kind": "custom_ecapa_fbank_cache",
        "artifact_identity_sha256": artifact_identity["identity_sha256"],
        "manifest_sha256": {split: sha256_file(path) for split, path in manifests.items()},
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "shard_size": shard_size,
        "shard_counts": shard_counts,
        "versions": {
            "torch": torch.__version__,
            "torchaudio": __import__("torchaudio").__version__,
            "speechbrain": __import__("speechbrain").__version__,
        },
    }
    identity["identity_sha256"] = hashlib.sha256(canonical_json(identity)).hexdigest()
    identity_path = cache_dir / "custom_fbank_cache_identity.json"
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("Existing cache identity conflicts with current custom experiment")
    else:
        atomic_json(identity_path, identity)
    return identity


class CustomCacheDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        cache_dir: Path,
        split: str,
        shard_size: int,
        max_cached_shards: int = 8,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        if any(row["split"] != split for row in self.rows):
            raise ValueError(f"Manifest contains rows outside split={split}")
        self.cache_dir = cache_dir
        self.split = split
        self.shard_size = shard_size
        self.max_cached_shards = max_cached_shards
        self._cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def shard_path_for_index(self, index: int) -> str:
        return f"{self.split}/shard_{index // self.shard_size:05d}.pt"

    def _load_shard(self, shard_index: int) -> Mapping[str, Any]:
        if shard_index in self._cache:
            value = self._cache.pop(shard_index)
            self._cache[shard_index] = value
            return value
        path = self.cache_dir / self.split / f"shard_{shard_index:05d}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[shard_index] = value
        while len(self._cache) > self.max_cached_shards:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        shard_index, offset = divmod(index, self.shard_size)
        shard = self._load_shard(shard_index)
        if shard["relative_audio_paths"][offset] != row["relative_audio_path"]:
            raise ValueError("Cache/manifest row alignment failed")
        return {
            "fbank": shard["features"][offset],
            "speaker_label": int(shard["speaker_labels"][offset]),
            "speaker_id": row["speaker_id"],
            "relative_audio_path": row["relative_audio_path"],
            "dataset_index": index,
        }


def collate_cache(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "fbank": torch.stack([sample["fbank"] for sample in samples]),
        "speaker_label": torch.tensor([sample["speaker_label"] for sample in samples], dtype=torch.long),
        "speaker_id": [sample["speaker_id"] for sample in samples],
        "relative_audio_path": [sample["relative_audio_path"] for sample in samples],
        "dataset_index": torch.tensor([sample["dataset_index"] for sample in samples], dtype=torch.long),
    }


class ShardAwarePKSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: CustomCacheDataset,
        speakers_per_batch: int,
        samples_per_speaker: int,
        active_shard_window: int,
        num_batches: int,
        seed: int,
        epoch: int,
    ) -> None:
        self.dataset = dataset
        self.p = speakers_per_batch
        self.k = samples_per_speaker
        self.active_window = active_shard_window
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = epoch
        grouped: dict[str, list[int]] = defaultdict(list)
        shards: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row["speaker_id"]].append(index)
            shards[dataset.shard_path_for_index(index)].append(index)
        if len(grouped) < self.p:
            raise ValueError(f"P={self.p} exceeds {len(grouped)} train speakers")
        if any(len(indexes) < self.k for indexes in grouped.values()):
            raise ValueError("A train speaker has fewer than K cached utterances")
        self.speaker_indexes = {speaker: tuple(indexes) for speaker, indexes in grouped.items()}
        self.shard_indexes = {shard: tuple(indexes) for shard, indexes in shards.items()}
        self.shards = tuple(sorted(shards))
        if not 1 <= self.active_window <= len(self.shards):
            raise ValueError("active_shard_window is outside available shard count")

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        shard_order = list(self.shards)
        rng.shuffle(shard_order)
        queues = {speaker: list(indexes) for speaker, indexes in self.speaker_indexes.items()}
        for queue in queues.values():
            rng.shuffle(queue)
        exposure: Counter[str] = Counter()
        cursor = 0
        for _ in range(self.num_batches):
            width = self.active_window
            while True:
                active = [shard_order[(cursor + offset) % len(shard_order)] for offset in range(width)]
                preferred: dict[str, set[int]] = defaultdict(set)
                for shard in active:
                    for index in self.shard_indexes[shard]:
                        preferred[self.dataset.rows[index]["speaker_id"]].add(index)
                if len(preferred) >= self.p or width == len(shard_order):
                    break
                width = min(len(shard_order), width + self.active_window)
            tie = {speaker: rng.random() for speaker in preferred}
            selected = sorted(preferred, key=lambda speaker: (exposure[speaker], tie[speaker]))[: self.p]
            batch: list[int] = []
            for speaker in selected:
                chosen: list[int] = []
                queue = queues[speaker]
                while len(chosen) < self.k:
                    candidates = [index for index in queue if index in preferred[speaker] and index not in chosen]
                    if not candidates:
                        candidates = [index for index in queue if index not in chosen]
                    if not candidates:
                        queue.extend(self.speaker_indexes[speaker])
                        rng.shuffle(queue)
                        continue
                    value = candidates[0]
                    queue.remove(value)
                    chosen.append(value)
                batch.extend(chosen)
                exposure[speaker] += 1
            rng.shuffle(batch)
            if len(batch) != self.p * self.k or len(set(batch)) != len(batch):
                raise RuntimeError("Sampler failed to create a complete unique P×K batch")
            yield batch
            cursor = (cursor + self.active_window) % len(shard_order)


def round_robin_batch(batch: Mapping[str, Any], p: int, k: int) -> dict[str, Any]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for position, label in enumerate(batch["speaker_label"].tolist()):
        grouped[label].append(position)
    if len(grouped) != p or any(len(values) != k for values in grouped.values()):
        raise ValueError("Logical batch does not satisfy P×K")
    order = [grouped[label][rank] for rank in range(k) for label in sorted(grouped)]
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value[order] if isinstance(value, torch.Tensor) else [value[index] for index in order]
    return result


def cosine_factor(step: int, total_steps: int, minimum: float) -> float:
    progress = min(max(step / total_steps, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def training_contract(args: argparse.Namespace, artifact_identity: Mapping[str, Any], cache_identity: Mapping[str, Any], classes: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_identity_sha256": artifact_identity["identity_sha256"],
        "cache_identity_sha256": cache_identity["identity_sha256"],
        "classes": classes,
        "p": args.speakers_per_batch,
        "k": args.samples_per_speaker,
        "microbatch_size": args.microbatch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "epochs": args.epochs,
        "ecapa_lr": args.ecapa_lr,
        "aam_lr": args.aam_lr,
        "weight_decay": args.weight_decay,
        "minimum_lr_factor": args.minimum_lr_factor,
        "sampler_seed": args.sampler_seed,
    }


def to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    atomic_torch_save(path, to_cpu(dict(state)))


def validation_eer(
    dataset: CustomCacheDataset,
    trials: Sequence[Mapping[str, Any]],
    mean_var_norm: torch.nn.Module,
    embedding_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    from src.verification_metrics import calculate_eer

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_cache)
    mean_var_norm.eval()
    embedding_model.eval()
    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            features = batch["fbank"].to(device)
            lengths = torch.ones(features.shape[0], device=device)
            normalized = mean_var_norm(features, lengths)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                value = embedding_model(normalized, lengths).squeeze(1).float()
            value = F.normalize(value, p=2, dim=1).cpu()
            embeddings.update(zip(batch["relative_audio_path"], value))
            if batch_number % 25 == 0:
                print(f"VALIDATION batch={batch_number}/{len(loader)}", flush=True)

    scores: list[float] = []
    targets: list[int] = []
    for trial in trials:
        score = torch.dot(embeddings[trial["left_path"]], embeddings[trial["right_path"]])
        scores.append(float(score.item()))
        targets.append(int(trial["target"]))
    result = calculate_eer(scores, targets)
    return {
        "eer": float(result.interpolated_eer),
        "threshold": float(result.empirical_threshold),
    }


def train_custom(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    repo_root = args.repo_root.resolve(strict=True)
    ensure_repo_imports(repo_root)
    from src.aam_training import AAMSoftmax, apply_batchnorm_policy, build_adamw_optimizer
    from src.speechbrain_frontend import SpeechBrainECAPAFrontend

    device = torch.device(args.device)
    artifact_identity = json.loads((args.artifacts_dir / "dataset_artifacts_identity.json").read_text(encoding="utf-8"))
    cache_identity = json.loads((args.cache_dir / "custom_fbank_cache_identity.json").read_text(encoding="utf-8"))
    if cache_identity["artifact_identity_sha256"] != artifact_identity["identity_sha256"]:
        raise ValueError("Cache and dataset artifact identities disagree")
    shard_size = int(cache_identity["shard_size"])
    train_manifest = args.artifacts_dir / "train_manifest.csv"
    validation_manifest = args.artifacts_dir / "validation_manifest.csv"
    train_dataset = CustomCacheDataset(train_manifest, args.cache_dir, "train", shard_size)
    validation_dataset = CustomCacheDataset(validation_manifest, args.cache_dir, "validation", shard_size)
    trials = read_trials(args.artifacts_dir / "validation_trials.csv")
    classes = len({row["speaker_id"] for row in train_dataset.rows})
    logical_batch = args.speakers_per_batch * args.samples_per_speaker
    if logical_batch % args.microbatch_size:
        raise ValueError("Logical batch must be divisible by microbatch size")
    if args.speakers_per_batch % args.microbatch_size:
        raise ValueError("P must be divisible by microbatch size")
    total_steps = args.epochs * args.steps_per_epoch
    contract = training_contract(args, artifact_identity, cache_identity, classes)

    # Seed before constructing the new AAM head so a fresh experiment is
    # reproducible. Resume subsequently restores the checkpoint RNG states.
    random.seed(args.sampler_seed)
    np.random.seed(args.sampler_seed)
    torch.manual_seed(args.sampler_seed)
    torch.cuda.manual_seed_all(args.sampler_seed)

    frontend = SpeechBrainECAPAFrontend(device="cpu")
    mean_var_norm = frontend.classifier.mods.mean_var_norm
    embedding_model = frontend.classifier.mods.embedding_model
    del frontend
    mean_var_norm.eval().to(device)
    for parameter in mean_var_norm.parameters():
        parameter.requires_grad_(False)
    embedding_model.to(device)
    apply_batchnorm_policy(embedding_model)
    aam = AAMSoftmax(
        embedding_dim=EMBEDDING_DIM,
        num_classes=classes,
        margin=0.2,
        scale=30.0,
        seed=args.sampler_seed,
    ).to(device)
    optimizer = build_adamw_optimizer(
        embedding_model,
        aam,
        embedding_lr=args.ecapa_lr,
        classifier_lr=args.aam_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_factor(step, total_steps, args.minimum_lr_factor),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda", init_scale=128.0)

    state = {
        "next_epoch": 0,
        "next_position": 0,
        "global_step": 0,
        "best_eer": None,
        "best_epoch": None,
        "patience_counter": 0,
    }
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("contract") != contract:
            raise ValueError("Resume checkpoint belongs to a different custom contract")
        embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
        aam.load_state_dict(checkpoint["aam_state_dict"])
        mean_var_norm.load_state_dict(checkpoint["mean_var_norm_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        state.update(checkpoint["cursor"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_states"])
        print(f"RESUME epoch={state['next_epoch'] + 1} position={state['next_position']} step={state['global_step']}", flush=True)
    else:
        if (args.output_dir / "last.pt").exists():
            raise FileExistsError("last.pt exists; pass --resume or use another output directory")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(reason: str) -> dict[str, Any]:
        cursor = dict(state)
        return {
            "schema": "custom_ecapa_aam_training",
            "version": 1,
            "reason": reason,
            "contract": contract,
            "embedding_model_state_dict": embedding_model.state_dict(),
            "aam_state_dict": aam.state_dict(),
            "mean_var_norm_state_dict": mean_var_norm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "cursor": cursor,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    for epoch in range(int(state["next_epoch"]), args.epochs):
        sampler = ShardAwarePKSampler(
            train_dataset,
            args.speakers_per_batch,
            args.samples_per_speaker,
            args.active_shard_window,
            args.steps_per_epoch,
            args.sampler_seed,
            epoch,
        )
        planned = list(sampler)
        start_position = int(state["next_position"]) if epoch == int(state["next_epoch"]) else 0
        loader = DataLoader(
            train_dataset,
            batch_sampler=planned[start_position:],
            num_workers=0,
            collate_fn=collate_cache,
        )
        embedding_model.train()
        aam.train()
        apply_batchnorm_policy(embedding_model)
        for position, batch in enumerate(loader, start=start_position):
            logical = round_robin_batch(batch, args.speakers_per_batch, args.samples_per_speaker)
            overflow_retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for start in range(0, logical_batch, args.microbatch_size):
                    stop = start + args.microbatch_size
                    features = logical["fbank"][start:stop].to(device)
                    labels = logical["speaker_label"][start:stop].to(device)
                    lengths = torch.ones(args.microbatch_size, device=device)
                    with torch.no_grad():
                        normalized = mean_var_norm(features, lengths)
                    with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.float16):
                        embedding = embedding_model(normalized, lengths).squeeze(1)
                    with torch.cuda.amp.autocast(enabled=False):
                        logits = aam(embedding.float(), labels)
                        loss = F.cross_entropy(logits.float(), labels, reduction="sum") / logical_batch
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError("Non-finite ECAPA/AAM loss")
                    scaler.scale(loss).backward()
                    total_loss += float(loss.detach().item())
                scaler.unscale_(optimizer)
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                try:
                    found_inf = scaler._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
                except (AttributeError, KeyError) as error:
                    raise RuntimeError("GradScaler did not expose overflow state") from error
                optimizer_updated = not any(
                    bool(value.detach().item()) for value in found_inf.values()
                )
                scaler.update()
                if optimizer_updated:
                    break
                overflow_retries += 1
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"AMP overflow: retry={overflow_retries}/8 scale={scaler.get_scale()}",
                    flush=True,
                )
                if overflow_retries > 8 or float(scaler.get_scale()) >= scale_before:
                    raise RuntimeError("AMP overflow retry limit reached")
            scheduler.step()
            state["global_step"] = int(state["global_step"]) + 1
            next_epoch = epoch if position + 1 < args.steps_per_epoch else epoch + 1
            state["next_epoch"] = next_epoch
            state["next_position"] = position + 1 if next_epoch == epoch else 0
            if int(state["global_step"]) % args.log_every == 0:
                print(
                    f"TRAIN epoch={epoch + 1}/{args.epochs} batch={position + 1}/{args.steps_per_epoch} "
                    f"step={state['global_step']} loss={total_loss:.6f}",
                    flush=True,
                )
            if int(state["global_step"]) % args.checkpoint_every == 0:
                save_checkpoint(args.output_dir / "last.pt", checkpoint_payload("rolling"))
            if args.max_updates is not None and int(state["global_step"]) >= args.max_updates:
                save_checkpoint(args.output_dir / "last.pt", checkpoint_payload("partial"))
                return {**state, "stopped_for_max_updates": True}

        metrics = validation_eer(
            validation_dataset,
            trials,
            mean_var_norm,
            embedding_model,
            device,
            args.validation_batch_size,
        )
        improved = state["best_eer"] is None or metrics["eer"] < float(state["best_eer"]) - args.min_improvement
        if improved:
            state["best_eer"] = metrics["eer"]
            state["best_epoch"] = epoch + 1
            state["patience_counter"] = 0
            save_checkpoint(args.output_dir / "best.pt", checkpoint_payload("best_validation_eer"))
        else:
            state["patience_counter"] = int(state["patience_counter"]) + 1
        state["next_epoch"] = epoch + 1
        state["next_position"] = 0
        save_checkpoint(args.output_dir / "last.pt", checkpoint_payload("epoch_complete"))
        print(
            f"EPOCH {epoch + 1} | validation_EER={metrics['eer'] * 100:.4f}% | "
            f"threshold={metrics['threshold']:.6f} | best={float(state['best_eer']) * 100:.4f}% | "
            f"patience={state['patience_counter']}/{args.patience}",
            flush=True,
        )
        if int(state["patience_counter"]) >= args.patience:
            break
    return dict(state)


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw-root", required=True, type=Path)
    prepare.add_argument("--processed-root", required=True, type=Path)
    prepare.add_argument("--artifacts-dir", required=True, type=Path)
    prepare.add_argument("--validation-ratio", type=float, default=0.10)
    prepare.add_argument("--split-seed", type=int, default=2026)
    prepare.add_argument("--trial-seed", type=int, default=2026)
    prepare.add_argument("--genuine-trials", type=int, default=10_000)
    prepare.add_argument("--impostor-trials", type=int, default=10_000)
    prepare.add_argument("--clean", action="store_true")

    cache = subparsers.add_parser("build-cache")
    add_shared_paths(cache)
    cache.add_argument("--cache-dir", required=True, type=Path)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--shard-size", type=int, default=512)
    cache.add_argument("--delete-processed-after-shard", action="store_true")

    train = subparsers.add_parser("train")
    add_shared_paths(train)
    train.add_argument("--cache-dir", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", type=Path)
    train.add_argument("--max-updates", type=int)
    train.add_argument("--speakers-per-batch", type=int, default=16)
    train.add_argument("--samples-per-speaker", type=int, default=2)
    train.add_argument("--microbatch-size", type=int, default=4)
    train.add_argument("--active-shard-window", type=int, default=8)
    train.add_argument("--steps-per-epoch", type=int, default=1520)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--ecapa-lr", type=float, default=1e-5)
    train.add_argument("--aam-lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--minimum-lr-factor", type=float, default=0.1)
    train.add_argument("--sampler-seed", type=int, default=20260729)
    train.add_argument("--validation-batch-size", type=int, default=64)
    train.add_argument("--patience", type=int, default=2)
    train.add_argument("--min-improvement", type=float, default=1e-4)
    train.add_argument("--checkpoint-every", type=int, default=100)
    train.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        preprocessing = normalize_and_segment(args.raw_root, args.processed_root, args.clean)
        identity = create_artifacts(
            args.processed_root,
            args.artifacts_dir,
            args.validation_ratio,
            args.split_seed,
            args.trial_seed,
            args.genuine_trials,
            args.impostor_trials,
        )
        print(json.dumps({"preprocessing": preprocessing, "artifacts": identity}, indent=2, ensure_ascii=False))
    elif args.command == "build-cache":
        identity = build_cache(
            args.repo_root,
            args.processed_root,
            args.artifacts_dir,
            args.cache_dir,
            args.device,
            args.batch_size,
            args.shard_size,
            args.delete_processed_after_shard,
        )
        print(json.dumps(identity, indent=2, ensure_ascii=False))
    elif args.command == "train":
        result = train_custom(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
