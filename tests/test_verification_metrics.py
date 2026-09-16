from __future__ import annotations

import dataclasses
import math
import random
import unittest

from src.verification_metrics import (
    calculate_eer,
    calculate_min_dcf,
    calculate_tar_at_far,
    verification_operating_points,
)


def brute_force_points(scores, targets):
    thresholds = [math.nextafter(max(scores), math.inf)] + sorted(set(scores), reverse=True)
    positives, negatives = sum(targets), len(targets) - sum(targets)
    return [
        (
            threshold,
            sum(t == 0 and s >= threshold for s, t in zip(scores, targets)) / negatives,
            sum(t == 1 and s < threshold for s, t in zip(scores, targets)) / positives,
        )
        for threshold in thresholds
    ]


def empirical_errors(scores, targets, threshold):
    positives, negatives = sum(targets), len(targets) - sum(targets)
    return (
        sum(t == 0 and s >= threshold for s, t in zip(scores, targets)) / negatives,
        sum(t == 1 and s < threshold for s, t in zip(scores, targets)) / positives,
    )


class MetricTests(unittest.TestCase):
    def assert_empirical_recomputed(self, scores, targets, result):
        far, frr = empirical_errors(scores, targets, result.empirical_threshold)
        self.assertAlmostEqual(far, result.empirical_far, places=15)
        self.assertAlmostEqual(frr, result.empirical_frr, places=15)

    def test_perfect_separation(self):
        scores, targets = [.9, .8, .2, .1], [1, 1, 0, 0]
        result = calculate_eer(scores, targets)
        self.assertEqual(result.interpolated_eer, 0.0)
        self.assertEqual((result.empirical_far, result.empirical_frr), (0.0, 0.0))
        self.assert_empirical_recomputed(scores, targets, result)

    def test_exact_empirical_equality_and_order_independence(self):
        scores, targets = [.9, .4, .6, .1], [1, 1, 0, 0]
        a = calculate_eer(scores, targets)
        b = calculate_eer(list(reversed(scores)), list(reversed(targets)))
        self.assertEqual(a, b)
        self.assertEqual(a.interpolated_eer, .5)
        self.assertEqual(a.empirical_far, a.empirical_frr)

    def test_interpolated_crossing_is_not_claimed_empirical(self):
        scores, targets = [.9, .6, .8, .7, .1], [1, 1, 0, 0, 0]
        result = calculate_eer(scores, targets)
        self.assertAlmostEqual(result.interpolated_eer, .5)
        self.assertAlmostEqual(result.interpolated_threshold, .75)
        direct = empirical_errors(scores, targets, result.interpolated_threshold)
        self.assertNotEqual(direct, (result.interpolated_far, result.interpolated_frr))
        self.assertEqual(result.eer_threshold_kind, "interpolated_non_empirical")
        self.assert_empirical_recomputed(scores, targets, result)

    def test_ties_all_equal_and_tie_order(self):
        a = calculate_eer([.5, .5, .5, .5], [1, 0, 1, 0])
        b = calculate_eer([.5, .5, .5, .5], [0, 1, 0, 1])
        self.assertEqual(a, b)
        self.assertEqual(a.interpolated_eer, .5)
        self.assert_empirical_recomputed([.5] * 4, [1, 0, 1, 0], a)

    def test_invalid_inputs(self):
        fixtures = (
            ([.1], [1]), ([.1], [0]), ([math.nan, .1], [1, 0]),
            ([math.inf, .1], [1, 0]), ([.1, .2], [1, 2]),
            ([1.01, .1], [1, 0]), ([-1.01, .1], [1, 0]),
        )
        for scores, targets in fixtures:
            with self.subTest(scores=scores, targets=targets), self.assertRaises(ValueError):
                calculate_eer(scores, targets)

    def test_threshold_semantics_range_and_serialization(self):
        result = calculate_eer([.8, .2], [1, 0])
        self.assertEqual(result.empirical_threshold_semantics, "accept same speaker when score >= threshold")
        self.assertTrue(0 <= result.interpolated_eer <= 1)
        self.assertTrue(-1.000001 <= result.empirical_threshold <= 1.000001)
        payload = dataclasses.asdict(result)
        self.assertEqual(payload["eer"], payload["interpolated_eer"])
        self.assertEqual(payload["eer_threshold"], payload["interpolated_threshold"])
        self.assertIn("empirical_average_error", payload)

    def test_optimized_points_match_brute_force_randomized(self):
        rng = random.Random(20260727)
        values = [-.8, -.2, 0., .2, .8]
        for size in range(4, 20):
            for _ in range(10):
                targets = [0, 1] + [rng.randrange(2) for _ in range(size - 2)]
                rng.shuffle(targets)
                scores = [values[rng.randrange(len(values))] for _ in range(size)]
                expected = brute_force_points(scores, targets)
                actual = [(p.threshold, p.far, p.frr) for p in verification_operating_points(scores, targets)]
                self.assertEqual(actual, expected)


    def test_min_dcf_perfect_separation_is_zero(self):
        scores, targets = [.9, .8, .2, .1], [1, 1, 0, 0]
        result = calculate_min_dcf(
            scores,
            targets,
            p_target=0.01,
            c_miss=1.0,
            c_fa=1.0,
        )
        self.assertEqual(result.normalized_min_dcf, 0.0)
        self.assertEqual(result.far, 0.0)
        self.assertEqual(result.frr, 0.0)
        self.assertEqual(result.tar, 1.0)

    def test_min_dcf_is_bounded_by_reject_all_baseline(self):
        scores, targets = [.9, .6, .8, .7, .1], [1, 1, 0, 0, 0]
        result = calculate_min_dcf(scores, targets, p_target=0.01)
        self.assertGreaterEqual(result.normalized_min_dcf, 0.0)
        self.assertLessEqual(result.normalized_min_dcf, 1.0)

    def test_tar_at_far_perfect_separation(self):
        scores, targets = [.9, .8, .2, .1], [1, 1, 0, 0]
        result = calculate_tar_at_far(
            scores,
            targets,
            maximum_far=0.001,
        )
        self.assertEqual(result.tar, 1.0)
        self.assertLessEqual(result.achieved_far, 0.001)

    def test_tar_at_far_respects_far_budget(self):
        scores = [.95, .70, .65, .90, .60, .10]
        targets = [1, 1, 1, 0, 0, 0]
        result = calculate_tar_at_far(
            scores,
            targets,
            maximum_far=0.001,
        )
        self.assertGreaterEqual(result.tar, 0.0)
        self.assertLessEqual(result.tar, 1.0)
        self.assertLessEqual(result.achieved_far, 0.001 + 1e-15)

    def test_new_metric_argument_validation(self):
        scores, targets = [.8, .2], [1, 0]
        with self.assertRaises(ValueError):
            calculate_min_dcf(scores, targets, p_target=0.0)
        with self.assertRaises(ValueError):
            calculate_min_dcf(scores, targets, c_miss=0.0)
        with self.assertRaises(ValueError):
            calculate_tar_at_far(scores, targets, maximum_far=-0.1)


if __name__ == "__main__":
    unittest.main()
