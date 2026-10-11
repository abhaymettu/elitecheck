import unittest

from elitecheck.gate import Design
from elitecheck.sim import POLICIES, Noise, aa_spurious_rate, calibrate_exec_sd, simulate


class SimTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(simulate("elite1", searches=30, seed=2), simulate("elite1", searches=30, seed=2))

    def test_policies_share_proposals_and_oracle_is_ceiling(self):
        rows = {p: simulate(p, searches=150, judgments=3) for p in POLICIES}
        offered = {r["improvements_offered"] for r in rows.values()}
        self.assertEqual(len(offered), 1)
        self.assertEqual(rows["oracle"]["false_promotions"], 0)
        for p in ("elite1", "gate"):
            self.assertGreaterEqual(rows["oracle"]["final_gain"], rows[p]["final_gain"])

    def test_gate_false_promotion_rate_per_non_improvement(self):
        # Negative control at the proposal level: gate promotes a non-improving proposal at most ~alpha.
        it = 20
        d = Design()
        g = simulate("gate", iterations=it, searches=300, judgments=1, design=d, noise=Noise(exec_sd=1.0))
        e = simulate("elite1", iterations=it, searches=300, judgments=1, noise=Noise(exec_sd=1.0))
        non_improving = it - g["improvements_offered"]
        self.assertLess(g["false_promotions"] / non_improving, d.alpha + 0.01)
        self.assertGreater(e["false_promotions"] / non_improving, 0.2)

    def test_mostly_harmful_proposals_elite1_drifts_down_gate_does_not(self):
        kw = dict(searches=150, step_mean=-0.5, step_sd=0.5, judgments=3, noise=Noise(exec_sd=0.5))
        self.assertLess(simulate("elite1", iterations=100, **kw)["final_gain"], -1.0)
        self.assertGreater(simulate("gate", iterations=12, **kw)["final_gain"], 0.0)

    def test_aa_rate_monotone_and_calibration_inverts(self):
        rates = [aa_spurious_rate(sd / 4, 6, "unanimous") for sd in range(12)]
        self.assertEqual(rates, sorted(rates))
        target = aa_spurious_rate(1.3, 6, "unanimous")
        self.assertAlmostEqual(calibrate_exec_sd(target, 6, "unanimous"), 1.3, places=3)

    def test_calibration_refuses_unidentifiable_settings(self):
        with self.assertRaises(ValueError):
            calibrate_exec_sd(0.3, 1, "unanimous")
        with self.assertRaisesRegex(ValueError, "does not depend"):
            calibrate_exec_sd(0.5, 3, "majority")
        with self.assertRaisesRegex(ValueError, "outside"):
            calibrate_exec_sd(0.9, 6, "unanimous")

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            simulate("best")
        with self.assertRaises(ValueError):
            Noise(exec_sd=-1)


if __name__ == "__main__":
    unittest.main()
