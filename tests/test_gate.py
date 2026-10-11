import math
import random
import unittest

from elitecheck.gate import CONTINUE, KEEP, PROMOTE, Design, Gate, challenge, operating_characteristics

def _valid(**kw):
    try:
        return Design(**kw)
    except ValueError:
        return None


DESIGNS = [d for d in (_valid(p1=p1, max_n=n, alpha=a) for p1 in (0.6, 0.65, 0.7, 0.75, 0.8, 0.9)
                       for n in (6, 12, 20, 40, 80) for a in (0.01, 0.05, 0.1)) if d]


class OperatingCharacteristicsTest(unittest.TestCase):
    def test_false_promotion_never_exceeds_alpha(self):
        # Negative control: an equally good challenger (p = 0.5) or a worse one.
        for d in DESIGNS:
            for p in (0.5, 0.45, 0.3):
                pr, _ = operating_characteristics(d, p)
                self.assertLessEqual(pr, d.alpha, msg=f"{d} p={p}")

    def test_every_shipped_design_can_promote(self):
        self.assertGreater(len(DESIGNS), 60)
        for d in DESIGNS:
            self.assertGreater(operating_characteristics(d, 1.0)[0], 0.99, msg=str(d))

    def test_p_must_be_a_probability(self):
        with self.assertRaises(ValueError):
            operating_characteristics(Design(), 1.5)

    def test_promotion_probability_increases_with_p(self):
        d = Design()
        probs = [operating_characteristics(d, p / 20)[0] for p in range(1, 20)]
        self.assertEqual(probs, sorted(probs))

    def test_bounded_and_deterministic(self):
        d = Design(p1=0.75, max_n=15)
        pr, en = operating_characteristics(d, 0.62)
        self.assertTrue(0 < pr < 1 and 0 < en <= 15)
        self.assertEqual((pr, en), operating_characteristics(Design(p1=0.75, max_n=15), 0.62))

    def test_positive_control_power(self):
        d = Design()
        self.assertGreater(operating_characteristics(d, 0.8)[0], 0.95)
        self.assertGreater(operating_characteristics(d, 0.7)[0], 0.65)

    def test_exact_matches_monte_carlo(self):
        d = Design(p1=0.7, max_n=30)
        p = 0.62
        rng = random.Random(1)
        sims, hits = 20000, 0
        for _ in range(sims):
            g = Gate(d)
            while g.decision == CONTINUE:
                g.record_round(1, 0) if rng.random() < p else g.record_round(0, 1)
            hits += g.decision == PROMOTE
        exact, _ = operating_characteristics(d, p)
        self.assertLess(abs(hits / sims - exact), 4 * math.sqrt(exact * (1 - exact) / sims))


class GateTest(unittest.TestCase):
    def test_design_validation(self):
        for bad in (dict(p1=0.5), dict(p1=1.0), dict(alpha=0.0), dict(alpha=0.6, beta=0.5), dict(max_n=0),
                    dict(p1=0.9, alpha=0.45, beta=0.2), dict(p1=0.6, max_n=6)):
            with self.assertRaises(ValueError):
                Design(**bad)

    def test_record_after_decision_raises(self):
        g = Gate(Design(p1=0.9, max_n=5))
        while g.decision == CONTINUE:
            g.record_round(0, 1)
        self.assertEqual(g.decision, KEEP)
        with self.assertRaises(RuntimeError):
            g.record_round(1, 0)

    def test_split_rounds_are_not_evidence(self):
        g = Gate()
        for _ in range(5):
            g.record_round(1, 1)
        self.assertEqual((g.wins, g.losses, g.undecided_rounds, g.decision), (0, 0, 5, CONTINUE))


def counter():
    n = [0]

    def run():
        n[0] += 1
        return n[0]
    return run, n


class ChallengeTest(unittest.TestCase):
    def test_dominant_challenger_promoted_in_minimum_rounds(self):
        d = Design()
        res = challenge(lambda: "inc", lambda: "ch", lambda f, s: "first" if f == "ch" else "second", design=d)
        self.assertEqual(res.decision, PROMOTE)
        self.assertEqual(res.rounds, math.ceil(d.upper / d.step_win))

    def test_position_biased_judge_never_promotes(self):
        # Negative control: a judge that always prefers whatever it sees first, on identical outputs.
        for k in (1, 2, 3):
            res = challenge(lambda: "same", lambda: "same", lambda f, s: "first", judgments_per_round=k)
            self.assertEqual(res.decision, KEEP, msg=f"k={k}")

    def test_order_alternates_and_fresh_executions_each_round(self):
        inc, n_inc = counter()
        ch, n_ch = counter()
        res = challenge(inc, ch, lambda f, s: "tie", design=Design(max_n=10), judgments_per_round=2, max_rounds=4)
        self.assertEqual(res.decision, KEEP)
        self.assertEqual((res.rounds, n_inc[0], n_ch[0]), (4, 4, 4))
        flags = [j["challenger_first"] for j in res.judgments]
        self.assertEqual(flags, [True, False] * 4)

    def test_bad_judge_output(self):
        with self.assertRaises(ValueError):
            challenge(lambda: 1, lambda: 2, lambda f, s: "left")


if __name__ == "__main__":
    unittest.main()
