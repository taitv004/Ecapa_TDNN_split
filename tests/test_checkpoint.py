#!/usr/bin/env python3
"""Evaluate a custom ECAPA checkpoint on a local speaker-folder test set.

Expected test layout::

    TEST_DATA_ROOT/
      speaker_001/*.wav
      speaker_002/*.wav

The script follows the inference path used by the ``ex1`` repository:

    waveform -> SpeechBrain Fbank -> mean/variance normalization
             -> fine-tuned ECAPA embedding -> L2 normalization -> cosine score

The AAM-Softmax training head is intentionally not used during verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import heapq
import os
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


# ============================================================================
# USER CONFIG - edit these paths, then run: python test_checkpoint.py
# Raw strings r"..." are important for Windows paths.
# ============================================================================
REPO_ROOT = Path(r"D:\ECAPA\Ecapa_TDNN_split")
CHECKPOINT_PATH = Path(r"D:\ECAPA\checkpoints\best.pt")
TEST_DATA_ROOT = Path(r"D:\ECAPA\external_test")
OUTPUT_DIR = Path(r"D:\ECAPA\test_results")

# Optional: speaker_split.csv created during training. Leave empty to skip the
# train/validation speaker-overlap check.
KNOWN_SPEAKERS_CSV = ""

DEVICE = "auto"               # "auto", "cuda:0", or "cpu"
INFERENCE_BATCH_SIZE = 16      # reduce to 8 or 4 if GPU memory is insufficient
TRIALS_PER_CLASS = 10_000      # maximum genuine and impostor trials
TRIAL_SEED = 2026
MINDCF_P_TARGET = 0.01
MINDCF_C_MISS = 1.0
MINDCF_C_FA = 1.0


SAMPLE_RATE = 16_000
SEGMENT_SAMPLES = 48_000
MIN_SEGMENT_SAMPLES = 24_000
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}

MANIFEST_FIELDS = (
    "segment_id",
    "source_relative_path",
    "speaker_id",
    "segment_index",
    "source_segment_samples",
    "zero_padding_samples",
)
TRIAL_FIELDS = (
    "trial_id",
    "target",
    "left_segment_id",
    "right_segment_id",
    "left_speaker_id",
    "right_speaker_id",
)
SCORE_FIELDS = TRIAL_FIELDS + ("cosine_score",)


@dataclass(frozen=True)
class Segment:
    segment_id: str
    source_relative_path: str
    speaker_id: str
    segment_index: int
    source_segment_samples: int
    zero_padding_samples: int
    waveform: torch.Tensor


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


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).hexdigest()


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("Self-pairs are forbidden")
    return (left, right) if left < right else (right, left)


def balanced_quotas(
    speakers: Sequence[str], total: int, seed: int, purpose: str
) -> dict[str, int]:
    base, remainder = divmod(total, len(speakers))
    ranked = sorted(
        speakers,
        key=lambda speaker: (stable_digest(seed, purpose, speaker), speaker),
    )
    return {
        speaker: base + int(index < remainder)
        for index, speaker in enumerate(ranked)
    }


def negative_speaker_pairs(
    speakers: Sequence[str], total: int, seed: int
) -> list[tuple[str, str]]:
    remaining = balanced_quotas(
        speakers, total * 2, seed, "negative-participation"
    )
    pairs: list[tuple[str, str]] = []
    for position in range(total):
        ranked = sorted(
            (speaker for speaker, count in remaining.items() if count),
            key=lambda speaker: (
                -remaining[speaker],
                stable_digest(seed, "negative-speaker", position, speaker),
                speaker,
            ),
        )
        if len(ranked) < 2:
            raise RuntimeError("Unable to balance impostor speaker participation")
        left, right = ranked[:2]
        remaining[left] -= 1
        remaining[right] -= 1
        pairs.append((left, right))
    if any(remaining.values()):
        raise RuntimeError("Impostor speaker participation did not reconcile")
    return pairs


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("DEVICE must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def discover_audio(root: Path) -> list[tuple[str, Path]]:
    root = root.resolve(strict=True)
    speaker_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    )
    sources: list[tuple[str, Path]] = []
    for speaker_dir in speaker_dirs:
        files = sorted(
            (
                path
                for path in speaker_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            ),
            key=lambda path: path.relative_to(root).as_posix().casefold(),
        )
        sources.extend((speaker_dir.name, path) for path in files)
    if not sources:
        raise ValueError(
            f"No supported audio found below {root}. Expected TEST_DATA_ROOT/<speaker>/*.wav"
        )
    if len({speaker for speaker, _ in sources}) < 2:
        raise ValueError("EER evaluation requires at least two test speakers")
    return sources


def decode_segments(source: Path, root: Path, speaker_id: str) -> list[Segment]:
    audio, original_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[0] == 0 or audio.shape[1] == 0:
        raise ValueError("empty audio")
    if int(original_rate) <= 0:
        raise ValueError(f"invalid sample rate: {original_rate}")
    waveform = torch.from_numpy(audio.T.copy()).mean(dim=0, keepdim=True)
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("audio contains NaN or Inf")
    if int(original_rate) != SAMPLE_RATE:
        waveform = AF.resample(waveform, int(original_rate), SAMPLE_RATE)
    waveform = waveform.squeeze(0).clamp(-1.0, 1.0)

    relative = source.relative_to(root).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    segments: list[Segment] = []
    for segment_index, start in enumerate(range(0, waveform.numel(), SEGMENT_SAMPLES)):
        segment = waveform[start : start + SEGMENT_SAMPLES]
        original_samples = int(segment.numel())
        if original_samples < MIN_SEGMENT_SAMPLES:
            break
        if original_samples < SEGMENT_SAMPLES:
            segment = F.pad(segment, (0, SEGMENT_SAMPLES - original_samples))
        segment_id = f"{speaker_id}/{source.stem}--{digest}--seg{segment_index:04d}"
        segments.append(
            Segment(
                segment_id=segment_id,
                source_relative_path=relative,
                speaker_id=speaker_id,
                segment_index=segment_index,
                source_segment_samples=original_samples,
                zero_padding_samples=SEGMENT_SAMPLES - original_samples,
                waveform=segment.contiguous(),
            )
        )
    return segments


def load_known_speakers(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "speaker_id" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} has no speaker_id column")
        return {row["speaker_id"].strip() for row in reader if row["speaker_id"].strip()}


def load_model(repo_root: Path, checkpoint_path: Path, device: torch.device):
    repo_root = repo_root.resolve(strict=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.speechbrain_frontend import SpeechBrainECAPAFrontend

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"embedding_model_state_dict", "mean_var_norm_state_dict"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint is incompatible; missing keys: " + ", ".join(sorted(missing))
        )

    # The wrapper creates exactly the architecture used during training. On the
    # first run SpeechBrain may download its base configuration/checkpoint files;
    # the fine-tuned weights below then replace encoder and normalization states.
    frontend = SpeechBrainECAPAFrontend(
        cache_dir=repo_root / "pretrained_models" / "spkrec-ecapa-voxceleb",
        device=str(device),
    )
    frontend.classifier.mods.embedding_model.load_state_dict(
        checkpoint["embedding_model_state_dict"], strict=True
    )
    frontend.classifier.mods.mean_var_norm.load_state_dict(
        checkpoint["mean_var_norm_state_dict"], strict=True
    )
    frontend.eval()
    return frontend, checkpoint


def extract_embeddings(
    sources: Sequence[tuple[str, Path]],
    root: Path,
    frontend,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], list[dict[str, str]]]:
    if batch_size < 1:
        raise ValueError("INFERENCE_BATCH_SIZE must be positive")
    manifest: list[dict[str, Any]] = []
    embeddings: dict[str, torch.Tensor] = {}
    invalid: list[dict[str, str]] = []
    pending: list[Segment] = []

    def flush() -> None:
        if not pending:
            return
        waveforms = torch.stack([item.waveform for item in pending]).to(device)
        lengths = torch.ones(waveforms.shape[0], device=device)
        with torch.inference_mode():
            features = frontend.compute_features(waveforms)
            normalized = frontend.mean_var_norm(features, lengths)
            amp_context = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                values = frontend.embedding_model(normalized, lengths).squeeze(1)
            values = F.normalize(values.float(), p=2, dim=1).cpu()
        for item, vector in zip(pending, values):
            if item.segment_id in embeddings:
                raise RuntimeError(f"Duplicate segment ID: {item.segment_id}")
            embeddings[item.segment_id] = vector
            manifest.append(
                {
                    "segment_id": item.segment_id,
                    "source_relative_path": item.source_relative_path,
                    "speaker_id": item.speaker_id,
                    "segment_index": item.segment_index,
                    "source_segment_samples": item.source_segment_samples,
                    "zero_padding_samples": item.zero_padding_samples,
                }
            )
        pending.clear()

    root = root.resolve(strict=True)
    for number, (speaker_id, source) in enumerate(sources, start=1):
        try:
            segments = decode_segments(source, root, speaker_id)
            if not segments:
                raise ValueError("recording is shorter than 1.5 seconds after resampling")
        except Exception as error:
            invalid.append(
                {
                    "speaker_id": speaker_id,
                    "source_relative_path": source.relative_to(root).as_posix(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print(
                f"SKIP INVALID {source.relative_to(root).as_posix()} | "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            continue
        for segment in segments:
            pending.append(segment)
            if len(pending) >= batch_size:
                flush()
        if number % 100 == 0 or number == len(sources):
            print(
                f"EMBED source={number}/{len(sources)} segments={len(embeddings)}",
                flush=True,
            )
    flush()
    if not embeddings:
        raise RuntimeError("No valid test segments were embedded")
    return manifest, embeddings, invalid


def create_trials(
    manifest: Sequence[dict[str, Any]],
    maximum_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    if maximum_per_class < 1:
        raise ValueError("TRIALS_PER_CLASS must be positive")
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_speaker[str(row["speaker_id"])].append(row)
    if len(by_speaker) < 2:
        raise ValueError("At least two valid test speakers are required")

    # This mirrors the friend repository's protocol: deterministic hash-ranked
    # genuine quotas per speaker and balanced impostor endpoint participation.
    # We additionally reject genuine pairs cut from the same source recording.
    speakers = sorted(by_speaker, key=natural_key)
    genuine_quotas = balanced_quotas(
        speakers, maximum_per_class, seed, "genuine-quota"
    )
    raw_trials: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    row_by_id = {
        str(row["segment_id"]): row
        for rows in by_speaker.values()
        for row in rows
    }
    for speaker in speakers:
        rows = by_speaker[speaker]
        candidates = (
            (
                stable_digest(
                    seed,
                    "genuine",
                    speaker,
                    *canonical_pair(
                        str(left["segment_id"]), str(right["segment_id"])
                    ),
                ),
                *canonical_pair(
                    str(left["segment_id"]), str(right["segment_id"])
                ),
            )
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
            if left["source_relative_path"] != right["source_relative_path"]
        )
        ranked = heapq.nsmallest(genuine_quotas[speaker], candidates)
        if len(ranked) != genuine_quotas[speaker]:
            raise ValueError(
                f"Speaker {speaker!r} cannot supply its genuine quota "
                f"({genuine_quotas[speaker]} cross-recording pairs). "
                "Add more recordings or reduce --trials-per-class."
            )
        for _, left_id, right_id in ranked:
            pair = canonical_pair(left_id, right_id)
            if pair in seen_pairs:
                raise RuntimeError("Duplicate genuine pair")
            seen_pairs.add(pair)
            raw_trials.append(
                {
                    "target": 1,
                    "left_segment_id": left_id,
                    "right_segment_id": right_id,
                    "left_speaker_id": speaker,
                    "right_speaker_id": speaker,
                }
            )

    utterance_use: dict[str, int] = defaultdict(int)
    for position, (speaker_a, speaker_b) in enumerate(
        negative_speaker_pairs(speakers, maximum_per_class, seed)
    ):
        paths_a = sorted(
            (str(row["segment_id"]) for row in by_speaker[speaker_a]),
            key=lambda path: (
                utterance_use[path],
                stable_digest(seed, "impostor", position, speaker_a, path),
                path,
            ),
        )
        paths_b = sorted(
            (str(row["segment_id"]) for row in by_speaker[speaker_b]),
            key=lambda path: (
                utterance_use[path],
                stable_digest(seed, "impostor", position, speaker_b, path),
                path,
            ),
        )
        chosen: tuple[str, str] | None = None
        for left_id in paths_a:
            for right_id in paths_b:
                pair = canonical_pair(left_id, right_id)
                if pair not in seen_pairs:
                    chosen = pair
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("Impostor utterance-pair capacity exhausted")
        left_id, right_id = chosen
        seen_pairs.add(chosen)
        left_speaker = str(row_by_id[left_id]["speaker_id"])
        right_speaker = str(row_by_id[right_id]["speaker_id"])
        raw_trials.append(
            {
                "target": 0,
                "left_segment_id": left_id,
                "right_segment_id": right_id,
                "left_speaker_id": left_speaker,
                "right_speaker_id": right_speaker,
            }
        )
        utterance_use[left_id] += 1
        utterance_use[right_id] += 1

    for index, trial in enumerate(raw_trials):
        trial["trial_id"] = f"test-{index:06d}"
    return raw_trials


def calculate_mindcf(
    p_target: float,
    c_miss: float,
    c_fa: float,
    operating_points,
) -> dict[str, float]:
    if not 0.0 < p_target < 1.0:
        raise ValueError("MINDCF_P_TARGET must be between 0 and 1")
    if c_miss <= 0.0 or c_fa <= 0.0:
        raise ValueError("minDCF costs must be positive")
    normalization = min(c_miss * p_target, c_fa * (1.0 - p_target))
    best = min(
        operating_points,
        key=lambda point: (
            c_miss * p_target * point.frr
            + c_fa * (1.0 - p_target) * point.far,
            -point.threshold,
        ),
    )
    raw = c_miss * p_target * best.frr + c_fa * (1.0 - p_target) * best.far
    return {
        "p_target": p_target,
        "c_miss": c_miss,
        "c_fa": c_fa,
        "raw": raw,
        "normalized": raw / normalization,
        "threshold": float(best.threshold),
        "far": float(best.far),
        "frr": float(best.frr),
    }


def evaluate(
    repo_root: Path,
    checkpoint_path: Path,
    test_root: Path,
    output_dir: Path,
    known_speakers_csv: Path | None,
    device_name: str,
    batch_size: int,
    trials_per_class: int,
    trial_seed: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    repo_root = repo_root.resolve(strict=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
    test_root = test_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_audio(test_root)
    test_speakers = {speaker for speaker, _ in sources}
    if known_speakers_csv is not None:
        known = load_known_speakers(known_speakers_csv.resolve(strict=True))
        overlap = sorted(test_speakers & known, key=natural_key)
        if overlap:
            raise RuntimeError(
                "Speaker leakage: test speakers occur in train/validation artifacts: "
                + ", ".join(overlap[:20])
            )

    print(f"Device: {device}", flush=True)
    print(
        f"Test root: {test_root} | speakers={len(test_speakers)} | "
        f"source recordings={len(sources)}",
        flush=True,
    )
    frontend, checkpoint = load_model(repo_root, checkpoint_path, device)
    manifest, embeddings, invalid = extract_embeddings(
        sources, test_root, frontend, device, batch_size
    )
    trials = create_trials(manifest, trials_per_class, trial_seed)

    scores: list[float] = []
    targets: list[int] = []
    score_rows: list[dict[str, Any]] = []
    for trial in trials:
        score = float(
            torch.dot(
                embeddings[str(trial["left_segment_id"])],
                embeddings[str(trial["right_segment_id"])],
            ).item()
        )
        target = int(trial["target"])
        scores.append(score)
        targets.append(target)
        score_rows.append({**trial, "cosine_score": f"{score:.10f}"})

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.verification_metrics import calculate_eer, verification_operating_points

    eer = calculate_eer(scores, targets)
    points = verification_operating_points(scores, targets)
    mindcf = calculate_mindcf(
        MINDCF_P_TARGET,
        MINDCF_C_MISS,
        MINDCF_C_FA,
        points,
    )

    manifest_path = output_dir / "test_manifest.csv"
    trials_path = output_dir / "test_trials.csv"
    scores_path = output_dir / "test_scores.csv"
    invalid_path = output_dir / "invalid_audio_report.csv"
    results_path = output_dir / "test_results.json"
    write_csv(manifest_path, MANIFEST_FIELDS, manifest)
    write_csv(trials_path, TRIAL_FIELDS, trials)
    write_csv(scores_path, SCORE_FIELDS, score_rows)
    write_csv(
        invalid_path,
        ("speaker_id", "source_relative_path", "error_type", "error"),
        invalid,
    )

    results = {
        "schema": "custom_ecapa_external_test",
        "version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "schema": checkpoint.get("schema"),
            "reason": checkpoint.get("reason"),
            "training_cursor": checkpoint.get("cursor"),
        },
        "test_data": {
            "root": str(test_root),
            "speaker_count": len({row["speaker_id"] for row in manifest}),
            "source_recordings_discovered": len(sources),
            "valid_segments": len(manifest),
            "invalid_source_recordings": len(invalid),
            "sample_rate": SAMPLE_RATE,
            "segment_samples": SEGMENT_SAMPLES,
        },
        "protocol": {
            "kind": "utterance_pair_cosine_verification",
            "trial_seed": trial_seed,
            "genuine_trials": sum(targets),
            "impostor_trials": len(targets) - sum(targets),
            "same_source_genuine_pairs_allowed": False,
        },
        "metrics": {
            "eer": float(eer.interpolated_eer),
            "eer_percent": float(eer.interpolated_eer_percentage),
            "eer_interpolated_threshold": float(eer.interpolated_threshold),
            "eer_empirical_threshold_descriptive_only": float(eer.empirical_threshold),
            "eer_empirical_far": float(eer.empirical_far),
            "eer_empirical_frr": float(eer.empirical_frr),
            "min_dcf": mindcf,
        },
        "threshold_policy": (
            "Thresholds calculated on external test are descriptive only. "
            "Choose a deployment threshold on validation data, not test data."
        ),
        "outputs": {
            "manifest": str(manifest_path),
            "trials": str(trials_path),
            "scores": str(scores_path),
            "invalid_audio_report": str(invalid_path),
        },
    }
    atomic_json(results_path, results)
    print(
        f"TEST COMPLETE | EER={results['metrics']['eer_percent']:.4f}% | "
        f"minDCF={mindcf['normalized']:.6f}",
        flush=True,
    )
    print(f"Results: {results_path}", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--test-root", type=Path, default=TEST_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--known-speakers-csv",
        type=Path,
        default=Path(KNOWN_SPEAKERS_CSV) if KNOWN_SPEAKERS_CSV.strip() else None,
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    parser.add_argument("--trials-per-class", type=int, default=TRIALS_PER_CLASS)
    parser.add_argument("--trial-seed", type=int, default=TRIAL_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate(
        repo_root=args.repo_root,
        checkpoint_path=args.checkpoint,
        test_root=args.test_root,
        output_dir=args.output_dir,
        known_speakers_csv=args.known_speakers_csv,
        device_name=args.device,
        batch_size=args.batch_size,
        trials_per_class=args.trials_per_class,
        trial_seed=args.trial_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
