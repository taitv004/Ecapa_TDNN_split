"""Deterministic sample-ID verification trials for validation and test."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


MANIFEST_FIELDS = (
    "sample_id",
    "relative_audio_path",
    "source_dataset",
    "source_recording_id",
    "speaker_id",
    "speaker_label",
    "final_split",
)
TRIAL_FIELDS = (
    "trial_id",
    "left_sample_id",
    "right_sample_id",
    "left_speaker_id",
    "right_speaker_id",
    "target",
)


@dataclass(frozen=True)
class ValidationRow:
    sample_id: str
    audio_path: str
    speaker_id: str
    source_dataset: str
    source_recording_id: str


@dataclass(frozen=True)
class ValidationTrial:
    trial_id: str
    left_sample_id: str
    right_sample_id: str
    left_speaker_id: str
    right_speaker_id: str
    target: int

    @property
    def left_relative_audio_path(self) -> str:
        return self.left_sample_id

    @property
    def right_relative_audio_path(self) -> str:
        return self.right_sample_id


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


def _safe_relative_path(value: str) -> str:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ValueError(f"Unsafe/nonportable audio path: {value!r}")
    return value


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("self-pairs are forbidden")
    return (left, right) if left < right else (right, left)


def read_validation_manifest(path: Path) -> tuple[ValidationRow, ...]:
    rows: list[ValidationRow] = []
    seen_samples: set[str] = set()
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Invalid validation manifest schema: {path}")
        for line, raw in enumerate(reader, start=2):
            sample_id = raw["sample_id"].strip()
            audio_path = _safe_relative_path(raw["relative_audio_path"].strip())
            speaker_id = raw["speaker_id"].strip()
            recording_id = raw["source_recording_id"].strip()
            if (
                not sample_id
                or not speaker_id
                or not raw["source_dataset"].strip()
                or not recording_id
                or raw["speaker_label"].strip() != "-1"
                or raw["final_split"].strip() != "validation"
                or sample_id in seen_samples
                or audio_path in seen_paths
            ):
                raise ValueError(f"{path}:{line}: invalid validation manifest row")
            seen_samples.add(sample_id)
            seen_paths.add(audio_path)
            rows.append(
                ValidationRow(
                    sample_id,
                    audio_path,
                    speaker_id,
                    raw["source_dataset"].strip(),
                    recording_id,
                )
            )
    if not rows:
        raise ValueError("validation manifest is empty")
    return tuple(rows)


def _allocate_balanced(capacities: dict[str, int], total: int, seed: int) -> dict[str, int]:
    if total > sum(capacities.values()):
        raise ValueError(
            f"Requested {total} genuine trials but capacity is "
            f"{sum(capacities.values())}"
        )
    quotas = {speaker: 0 for speaker in capacities}
    heap = [
        (0, stable_digest(seed, "genuine-quota", speaker, 0), speaker)
        for speaker, capacity in capacities.items()
        if capacity
    ]
    heapq.heapify(heap)
    for _ in range(total):
        if not heap:
            raise RuntimeError("Genuine-trial capacity exhausted")
        count, _, speaker = heapq.heappop(heap)
        quotas[speaker] += 1
        if quotas[speaker] < capacities[speaker]:
            next_count = count + 1
            heapq.heappush(
                heap,
                (
                    next_count,
                    stable_digest(seed, "genuine-quota", speaker, next_count),
                    speaker,
                ),
            )
    return quotas


def _balanced_quotas(
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


def _negative_speaker_pairs(
    speakers: Sequence[str], total: int, seed: int
) -> list[tuple[str, str]]:
    remaining = _balanced_quotas(
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


def generate_validation_trials(
    rows: Sequence[ValidationRow],
    *,
    seed: int = 2026,
    genuine_count: int = 10_000,
    impostor_count: int = 10_000,
) -> tuple[ValidationTrial, ...]:
    """Build fixed trials, preferring cross-recording genuine pairs."""
    if genuine_count < 1 or impostor_count < 1:
        raise ValueError("trial counts must be positive")
    by_speaker: dict[str, list[ValidationRow]] = defaultdict(list)
    sample_owners: dict[str, str] = {}
    for row in rows:
        if row.sample_id in sample_owners:
            raise ValueError(f"Duplicate sample_id: {row.sample_id}")
        sample_owners[row.sample_id] = row.speaker_id
        by_speaker[row.speaker_id].append(row)
    speakers = tuple(sorted(by_speaker))
    if len(speakers) < 2:
        raise ValueError("at least two validation speakers are required")
    for values in by_speaker.values():
        values.sort(key=lambda row: row.sample_id)

    genuine_candidates: dict[
        str, list[tuple[int, str, str, str]]
    ] = {}
    for speaker, speaker_rows in by_speaker.items():
        candidates = []
        for offset, left in enumerate(speaker_rows):
            for right in speaker_rows[offset + 1 :]:
                sample_left, sample_right = canonical_pair(
                    left.sample_id, right.sample_id
                )
                same_recording = int(
                    (
                        left.source_dataset,
                        left.source_recording_id,
                    )
                    == (
                        right.source_dataset,
                        right.source_recording_id,
                    )
                )
                candidates.append(
                    (
                        same_recording,
                        stable_digest(
                            seed,
                            "genuine",
                            speaker,
                            sample_left,
                            sample_right,
                        ),
                        sample_left,
                        sample_right,
                    )
                )
        candidates.sort()
        genuine_candidates[speaker] = candidates
    quotas = _allocate_balanced(
        {speaker: len(values) for speaker, values in genuine_candidates.items()},
        genuine_count,
        seed,
    )

    selected: list[tuple[str, str, str, str, int]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for speaker in speakers:
        for _, _, left, right in genuine_candidates[speaker][: quotas[speaker]]:
            pair = canonical_pair(left, right)
            if pair in seen_pairs:
                raise RuntimeError("duplicate genuine pair")
            seen_pairs.add(pair)
            selected.append((left, right, speaker, speaker, 1))

    sample_use: Counter[str] = Counter()
    for position, (speaker_a, speaker_b) in enumerate(
        _negative_speaker_pairs(speakers, impostor_count, seed)
    ):
        samples_a = sorted(
            (row.sample_id for row in by_speaker[speaker_a]),
            key=lambda value: (
                sample_use[value],
                stable_digest(seed, "impostor", position, speaker_a, value),
                value,
            ),
        )
        samples_b = sorted(
            (row.sample_id for row in by_speaker[speaker_b]),
            key=lambda value: (
                sample_use[value],
                stable_digest(seed, "impostor", position, speaker_b, value),
                value,
            ),
        )
        chosen: tuple[str, str] | None = None
        for left in samples_a:
            for right in samples_b:
                pair = canonical_pair(left, right)
                if pair not in seen_pairs:
                    chosen = pair
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("impostor sample-pair capacity exhausted")
        left, right = chosen
        seen_pairs.add(chosen)
        selected.append(
            (left, right, sample_owners[left], sample_owners[right], 0)
        )
        sample_use[left] += 1
        sample_use[right] += 1

    trials = tuple(
        ValidationTrial(
            f"adaptive-augmented-3s-v1-validation-{index:05d}",
            left,
            right,
            left_speaker,
            right_speaker,
            target,
        )
        for index, (
            left,
            right,
            left_speaker,
            right_speaker,
            target,
        ) in enumerate(selected)
    )
    validate_validation_trials(
        trials,
        rows,
        genuine_count=genuine_count,
        impostor_count=impostor_count,
    )
    return trials


def validate_validation_trials(
    trials: Sequence[ValidationTrial],
    rows: Sequence[ValidationRow],
    *,
    genuine_count: int | None = None,
    impostor_count: int | None = None,
) -> None:
    ownership = {row.sample_id: row.speaker_id for row in rows}
    if len(ownership) != len(rows):
        raise ValueError("validation sample IDs are not unique")
    trial_ids = [trial.trial_id for trial in trials]
    pair_keys = [
        canonical_pair(trial.left_sample_id, trial.right_sample_id)
        for trial in trials
    ]
    if len(trial_ids) != len(set(trial_ids)) or len(pair_keys) != len(
        set(pair_keys)
    ):
        raise ValueError("trial IDs or unordered sample pairs are not unique")
    genuine = [trial for trial in trials if trial.target == 1]
    impostor = [trial for trial in trials if trial.target == 0]
    if genuine_count is not None and len(genuine) != genuine_count:
        raise ValueError("genuine trial count is invalid")
    if impostor_count is not None and len(impostor) != impostor_count:
        raise ValueError("impostor trial count is invalid")
    if not genuine or not impostor:
        raise ValueError("both genuine and impostor trials are required")
    participation: Counter[str] = Counter()
    for trial in trials:
        if (
            trial.target not in (0, 1)
            or ownership.get(trial.left_sample_id) != trial.left_speaker_id
            or ownership.get(trial.right_sample_id) != trial.right_speaker_id
            or (trial.target == 1)
            != (trial.left_speaker_id == trial.right_speaker_id)
        ):
            raise ValueError("trial ownership or target is invalid")
        participation[trial.left_speaker_id] += 1
        participation[trial.right_speaker_id] += 1
    if set(participation) != set(row.speaker_id for row in rows):
        raise ValueError("not all validation speakers are represented")


def trial_recording_statistics(
    trials: Sequence[ValidationTrial], rows: Sequence[ValidationRow]
) -> dict[str, int]:
    recording = {
        row.sample_id: (row.source_dataset, row.source_recording_id)
        for row in rows
    }
    genuine = [trial for trial in trials if trial.target == 1]
    cross = sum(
        recording[trial.left_sample_id] != recording[trial.right_sample_id]
        for trial in genuine
    )
    return {
        "genuine_cross_recording": cross,
        "genuine_same_recording_fallback": len(genuine) - cross,
    }


def trials_csv_bytes(trials: Sequence[ValidationTrial]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    for trial in trials:
        writer.writerow(asdict(trial))
    return stream.getvalue().encode("utf-8")
