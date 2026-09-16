"""O(N log N) grouped verification operating points and EER metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    far: float
    frr: float


@dataclass(frozen=True)
class MinDCFResult:
    p_target: float
    c_miss: float
    c_fa: float
    normalized_min_dcf: float
    unnormalized_min_dcf: float
    threshold: float
    far: float
    frr: float
    tar: float


@dataclass(frozen=True)
class TARAtFARResult:
    requested_max_far: float
    achieved_far: float
    tar: float
    frr: float
    threshold: float


@dataclass(frozen=True)
class EERResult:
    interpolated_eer: float
    interpolated_eer_percentage: float
    interpolated_threshold: float
    interpolated_far: float
    interpolated_frr: float
    empirical_threshold: float
    empirical_far: float
    empirical_frr: float
    empirical_far_frr_gap: float
    empirical_average_error: float
    empirical_threshold_semantics: str
    eer_kind: str
    eer_threshold_kind: str
    eer: float
    eer_percentage: float
    eer_threshold: float
    far: float
    frr: float

    @property
    def threshold(self) -> float:
        """Backward-compatible alias for the non-empirical interpolated threshold."""
        return self.interpolated_threshold


def verification_operating_points(
    scores: Sequence[float], targets: Sequence[int]
) -> tuple[OperatingPoint, ...]:
    if len(scores) != len(targets) or not scores:
        raise ValueError("scores and targets must have equal non-zero length")
    numeric_scores = [float(score) for score in scores]
    if any(not math.isfinite(score) for score in numeric_scores):
        raise ValueError("scores contain NaN or Inf")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must be 0 or 1")
    positives, negatives = sum(targets), len(targets) - sum(targets)
    if positives == 0 or negatives == 0:
        raise ValueError("both positive and negative trials are required")
    if min(numeric_scores) < -1.000001 or max(numeric_scores) > 1.000001:
        raise ValueError("scores exceed cosine range")

    ordered = sorted(zip(numeric_scores, targets), key=lambda item: item[0], reverse=True)
    points = [OperatingPoint(math.nextafter(ordered[0][0], math.inf), 0.0, 1.0)]
    accepted_positive = accepted_negative = 0
    position = 0
    while position < len(ordered):
        threshold = ordered[position][0]
        while position < len(ordered) and ordered[position][0] == threshold:
            if ordered[position][1] == 1:
                accepted_positive += 1
            else:
                accepted_negative += 1
            position += 1
        points.append(OperatingPoint(
            threshold, accepted_negative / negatives, (positives - accepted_positive) / positives
        ))
    return tuple(points)


def _empirical_errors(
    scores: Sequence[float], targets: Sequence[int], threshold: float
) -> tuple[float, float]:
    positives, negatives = sum(targets), len(targets) - sum(targets)
    far = sum(target == 0 and score >= threshold for score, target in zip(scores, targets)) / negatives
    frr = sum(target == 1 and score < threshold for score, target in zip(scores, targets)) / positives
    return far, frr


def calculate_eer(scores: Sequence[float], targets: Sequence[int]) -> EERResult:
    points = verification_operating_points(scores, targets)
    crossing: tuple[float, float, float] | None = None
    for point in points:
        if point.far == point.frr:
            crossing = (point.threshold, point.far, point.frr)
            break
    if crossing is None:
        for first, second in zip(points, points[1:]):
            d1, d2 = first.far - first.frr, second.far - second.frr
            if d1 < 0 < d2:
                weight = -d1 / (d2 - d1)
                crossing = (
                    first.threshold + weight * (second.threshold - first.threshold),
                    first.far + weight * (second.far - first.far),
                    first.frr + weight * (second.frr - first.frr),
                )
                break
    if crossing is None:
        raise RuntimeError("FAR/FRR crossing not found")
    interpolated_threshold, interpolated_far, interpolated_frr = crossing
    interpolated_eer = (interpolated_far + interpolated_frr) / 2.0

    # Boundary point is retained for ROC completeness but an executable threshold
    # is selected from actual score thresholds, then ties prefer the higher value.
    empirical = min(
        points[1:],
        key=lambda point: (
            abs(point.far - point.frr),
            (point.far + point.frr) / 2.0,
            -point.threshold,
        ),
    )
    recomputed_far, recomputed_frr = _empirical_errors(scores, targets, empirical.threshold)
    if not (
        math.isclose(recomputed_far, empirical.far, abs_tol=1e-15)
        and math.isclose(recomputed_frr, empirical.frr, abs_tol=1e-15)
    ):
        raise RuntimeError("empirical FAR/FRR recomputation mismatch")
    semantics = "accept same speaker when score >= threshold"
    return EERResult(
        interpolated_eer=interpolated_eer,
        interpolated_eer_percentage=interpolated_eer * 100.0,
        interpolated_threshold=interpolated_threshold,
        interpolated_far=interpolated_far,
        interpolated_frr=interpolated_frr,
        empirical_threshold=empirical.threshold,
        empirical_far=recomputed_far,
        empirical_frr=recomputed_frr,
        empirical_far_frr_gap=abs(recomputed_far - recomputed_frr),
        empirical_average_error=(recomputed_far + recomputed_frr) / 2.0,
        empirical_threshold_semantics=semantics,
        eer_kind="linearly_interpolated_roc_crossing",
        eer_threshold_kind="interpolated_non_empirical",
        eer=interpolated_eer,
        eer_percentage=interpolated_eer * 100.0,
        eer_threshold=interpolated_threshold,
        far=interpolated_far,
        frr=interpolated_frr,
    )


def calculate_min_dcf(
    scores: Sequence[float],
    targets: Sequence[int],
    *,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> MinDCFResult:
    """Return minimum normalized DCF over all empirical operating points.

    Normalization follows:
        DCF / min(C_miss * P_target, C_fa * (1 - P_target))
    """
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must lie strictly between 0 and 1")
    if c_miss <= 0.0 or c_fa <= 0.0:
        raise ValueError("c_miss and c_fa must be positive")

    points = verification_operating_points(scores, targets)
    normalization = min(
        c_miss * p_target,
        c_fa * (1.0 - p_target),
    )
    if normalization <= 0.0:
        raise ValueError("DCF normalization must be positive")

    best = None
    for point in points:
        raw_dcf = (
            c_miss * point.frr * p_target
            + c_fa * point.far * (1.0 - p_target)
        )
        normalized = raw_dcf / normalization

        # Deterministic tie-breaking:
        # lower normalized DCF, then lower raw DCF, then lower FAR,
        # then lower FRR, then higher threshold.
        candidate = (
            normalized,
            raw_dcf,
            point.far,
            point.frr,
            -point.threshold,
            point,
        )
        if best is None or candidate[:-1] < best[:-1]:
            best = candidate

    if best is None:
        raise RuntimeError("No minDCF operating point was produced")

    point = best[-1]
    return MinDCFResult(
        p_target=float(p_target),
        c_miss=float(c_miss),
        c_fa=float(c_fa),
        normalized_min_dcf=float(best[0]),
        unnormalized_min_dcf=float(best[1]),
        threshold=float(point.threshold),
        far=float(point.far),
        frr=float(point.frr),
        tar=float(1.0 - point.frr),
    )


def calculate_tar_at_far(
    scores: Sequence[float],
    targets: Sequence[int],
    *,
    maximum_far: float = 0.001,
) -> TARAtFARResult:
    """Return the highest empirical TAR whose FAR does not exceed maximum_far."""
    if not 0.0 <= maximum_far <= 1.0:
        raise ValueError("maximum_far must be within [0, 1]")

    eligible = [
        point
        for point in verification_operating_points(scores, targets)
        if point.far <= maximum_far + 1e-15
    ]
    if not eligible:
        raise RuntimeError("No operating point satisfies the requested FAR")

    # Maximize TAR. Ties prefer the operating point closest to the FAR budget,
    # then the higher threshold for deterministic reporting.
    best = max(
        eligible,
        key=lambda point: (
            1.0 - point.frr,
            point.far,
            point.threshold,
        ),
    )

    return TARAtFARResult(
        requested_max_far=float(maximum_far),
        achieved_far=float(best.far),
        tar=float(1.0 - best.frr),
        frr=float(best.frr),
        threshold=float(best.threshold),
    )

