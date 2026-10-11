import math
import unittest

from elitecheck.aa import aa_report, winshare_report
from elitecheck.sim import Noise, aa_spurious_rate
from elitecheck.verdicts import Verdict

from ._synth import pairs


class AATest(unittest.TestCase):
    def test_matches_model_implied_rate(self):
        # Data generated with exec_sd=1.5, 6 judgments per pair; the report should recover the model rate.
        vs = pairs(3000, 6, a="W", b="W", exec_sd=1.5, seed=3)
        expected = aa_spurious_rate(1.5, 6, "unanimous", draws=20000)
        rep = aa_report(vs, rule="unanimous", looks=10)
        lo, hi = rep["spurious_promotion_per_look_ci95"]
        self.assertLess(lo - 0.01, expected)
        self.assertGreater(hi + 0.01, expected)
        self.assertAlmostEqual(rep["p_at_least_one_spurious_if_independent"], 1 - (1 - rep["spurious_promotion_per_look"]) ** 10)
        self.assertEqual(rep["judgments_per_pair"], [6])

    def test_single_rule_is_half_the_decisive_rate(self):
        vs = pairs(2000, 1, a="W", b="W", exec_sd=0.3, tie_prob=0.2, seed=4)
        rep = aa_report(vs, rule="single")
        self.assertAlmostEqual(rep["spurious_promotion_per_look"], rep["decisive_rate"] / 2)

    def test_position_bias_detected(self):
        vs = pairs(400, 2, a="W", b="W", exec_sd=0.0, position_bias=3.0, seed=5)
        rep = aa_report(vs)
        self.assertGreater(rep["first_shown_win_rate"], 0.9)
        self.assertGreater(rep["first_shown_win_rate_ci95"][0], 0.85)

    def test_model_rate_includes_position_bias(self):
        noise = Noise(position_bias=1.5)
        vs = pairs(4000, 6, a="W", b="W", exec_sd=1.0, position_bias=1.5, seed=8)
        lo, hi = aa_report(vs, rule="unanimous")["spurious_promotion_per_look_ci95"]
        self.assertTrue(lo - 0.005 < aa_spurious_rate(1.0, 6, "unanimous", noise, draws=20000) < hi + 0.005)

    def test_swapped_orientation_is_one_cluster(self):
        # 40 A/A execution pairs, each judged 6 times at 5-1, logged with sides swapped half the time.
        vs = []
        for i in range(40):
            for j in range(6):
                w = "a" if j < 5 else "b"
                if j % 2:
                    vs.append(Verdict("t", "W", "W", {"a": "b", "b": "a"}[w], f"y{i}", f"x{i}"))
                else:
                    vs.append(Verdict("t", "W", "W", w, f"x{i}", f"y{i}"))
        rep = aa_report(vs, rule="unanimous")
        self.assertEqual((rep["aa_pairs"], rep["judgments_per_pair"], rep["spurious_promotion_per_look"]), (40, [6], 0.0))

    def test_requires_aa_records(self):
        with self.assertRaisesRegex(ValueError, "no A/A"):
            aa_report(pairs(5, 2))
        with self.assertRaises(ValueError):
            aa_report(pairs(5, 2, a="W", b="W"), rule="best")
        with self.assertRaises(ValueError):
            aa_report(pairs(5, 2, a="W", b="W"), looks=0)


class WinshareTest(unittest.TestCase):
    def test_orientation(self):
        vs = [Verdict("t", "X", "Y", "a", "1", "1"), Verdict("t", "Y", "X", "b", "2", "2"),
              Verdict("t", "Y", "X", "a", "3", "3"), Verdict("t", "X", "Z", "a", "4", "4")]
        rep = winshare_report(vs, "X", "Y")
        self.assertEqual((rep["wins"], rep["losses"], rep["clusters"]), (2, 1, 3))
        self.assertEqual(rep["records_without_run_ids"], 0)
        with self.assertRaises(ValueError):
            winshare_report(vs, "X", "X")
        with self.assertRaises(ValueError):
            winshare_report(vs, "Y", "Q")

    def test_swapped_records_share_cluster_and_degenerate_design_effect(self):
        vs = [Verdict("t", "A", "B", "a", "r1", "s1"), Verdict("t", "B", "A", "b", "s1", "r1"),
              Verdict("t", "A", "B", "tie", "r2", "s2")]
        rep = winshare_report(vs, "A", "B")
        self.assertEqual((rep["wins"], rep["clusters"]), (2, 1))
        self.assertTrue(math.isnan(rep["design_effect"]))

    def test_negative_control_coverage(self):
        # Equal candidates, strong execution noise, six judgments per execution pair (the
        # "six judgments are not six executions" case). Judgment-level intervals undercover;
        # cluster-level intervals stay near nominal. With 150 datasets the coverage SE at 0.95 is
        # about 0.018, so 0.88 is nominal minus ~4 SE; 0.85 sits well above the design-effect
        # prediction for the naive interval (about 0.71 at a design effect near 3.5).
        datasets, naive_hits, cluster_hits, deffs = 150, 0, 0, []
        for s in range(datasets):
            rep = winshare_report(pairs(30, 6, exec_sd=2.0, seed=100 + s), "A", "B", reps=400, seed=s)
            naive_hits += rep["naive_ci95"][0] <= 0.5 <= rep["naive_ci95"][1]
            cluster_hits += rep["cluster_ci95"][0] <= 0.5 <= rep["cluster_ci95"][1]
            deffs.append(rep["design_effect"])
        self.assertLess(naive_hits / datasets, 0.85)
        self.assertGreater(cluster_hits / datasets, 0.88)
        self.assertGreater(sorted(deffs)[len(deffs) // 2], 2.0)

    def test_positive_control_detects_real_difference(self):
        rep = winshare_report(pairs(60, 4, delta=1.5, exec_sd=1.0, seed=9), "A", "B")
        self.assertGreater(rep["cluster_ci95"][0], 0.5)
        self.assertLess(rep["cluster_sign_test_p"], 0.01)


if __name__ == "__main__":
    unittest.main()
