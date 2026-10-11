import math
import unittest
from statistics import NormalDist

from elitecheck.stats import cluster_bootstrap_share, half_tail, sign_test, wilson


class StatsTest(unittest.TestCase):
    def test_wilson_known_values(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)
        self.assertTrue(all(math.isnan(x) for x in wilson(0, 0)))

    def test_half_tail_exact(self):
        self.assertAlmostEqual(half_tail(8, 10), 56 / 1024)
        self.assertEqual(half_tail(0, 10), 1.0)
        self.assertEqual(half_tail(11, 10), 0.0)

    def test_no_overflow_on_large_balanced_counts(self):
        self.assertEqual(sign_test(600, 600), 1.0)
        self.assertAlmostEqual(sign_test(600, 620), 0.5865, places=3)
        # Independent check against the continuity-corrected normal approximation.
        approx = NormalDist().cdf(-(1099.5 - 1000) / 500**0.5)
        self.assertLess(abs(half_tail(1100, 2000) / approx - 1), 0.05)

    def test_sign_test(self):
        self.assertAlmostEqual(sign_test(9, 1), 2 * 11 / 1024)
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertEqual(sign_test(5, 5), 1.0)

    def test_bootstrap_deterministic_and_brackets_estimate(self):
        clusters = [(3, 1), (0, 4), (2, 2), (4, 0), (1, 3)]
        ci1 = cluster_bootstrap_share(clusters, reps=500, seed=7)
        ci2 = cluster_bootstrap_share(clusters, reps=500, seed=7)
        self.assertEqual(ci1, ci2)
        self.assertLessEqual(ci1[0], 0.5)
        self.assertGreaterEqual(ci1[1], 0.5)
        self.assertTrue(all(math.isnan(x) for x in cluster_bootstrap_share([(0, 0)])))


if __name__ == "__main__":
    unittest.main()
