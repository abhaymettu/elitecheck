import contextlib
import io
import json
import os
import unittest

from elitecheck.cli import main

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "examples", "verdicts.jsonl")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_aa_json(self):
        code, out, _ = run("--json", "aa", FIXTURE, "--rule", "unanimous")
        self.assertEqual(code, 0)
        rep = json.loads(out)
        self.assertEqual(rep["aa_pairs"], 45)
        self.assertTrue(0 < rep["spurious_promotion_per_look"] < 1)

    def test_aa_text(self):
        code, out, _ = run("aa", FIXTURE)
        self.assertEqual(code, 0)
        self.assertIn("independent fresh comparisons", out)

    def test_winshare_json(self):
        code, out, _ = run("--json", "winshare", FIXTURE, "--a", "v2", "--b", "baseline", "--cluster", "task")
        rep = json.loads(out)
        self.assertEqual((code, rep["clusters"], rep["wins"] + rep["losses"]), (0, 3, 180))

    def test_plan_table(self):
        code, out, _ = run("plan", "--p1", "0.8", "--max-n", "20")
        self.assertEqual(code, 0)
        self.assertIn("exact, not simulated", out)
        code, out, _ = run("--json", "plan")
        row = [r for r in json.loads(out)["oc"] if r["p"] == 0.5][0]
        self.assertLessEqual(row["p_promote"], 0.05)

    def test_simulate_with_calibration(self):
        code, out, _ = run("--json", "simulate", "--searches", "20", "--judgments", "1", "--aa-judgments", "6",
                           "--calibrate-rule", "unanimous", "--calibrate-rate", "0.167", "--policies", "elite1", "gate")
        rep = json.loads(out)
        self.assertEqual(code, 0)
        self.assertTrue(rep["calibrated"])
        self.assertEqual([r["policy"] for r in rep["results"]], ["elite1", "gate"])
        self.assertAlmostEqual(rep["exec_sd"], 1.38, delta=0.05)  # the fixture was generated with 1.5

    def test_json_after_subcommand_and_strict(self):
        code, out, _ = run("plan", "--json")
        self.assertEqual(code, 0)
        json.loads(out)
        code, out, _ = run("--json", "winshare", FIXTURE, "--a", "v2", "--b", "baseline")
        self.assertNotIn("NaN", out)

    def test_errors_exit_2(self):
        code, _, err = run("aa", "does-not-exist.jsonl")
        self.assertEqual(code, 2)
        self.assertIn("elitecheck: error", err)
        code, _, err = run("winshare", FIXTURE, "--a", "v2", "--b", "nope")
        self.assertEqual(code, 2)
        code, _, err = run("simulate", "--calibrate-rate", "0.2")
        self.assertEqual(code, 2)
        self.assertIn("--aa-judgments", err)
        code, _, err = run("plan", "--p1", "0.6", "--max-n", "6")
        self.assertIn("never promote", err)


if __name__ == "__main__":
    unittest.main()
