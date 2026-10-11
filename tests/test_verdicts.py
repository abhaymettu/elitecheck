import json
import os
import tempfile
import unittest

from elitecheck.verdicts import VerdictError, from_dict, group_pairs, load_jsonl


class VerdictsTest(unittest.TestCase):
    def test_required_fields_and_values(self):
        with self.assertRaisesRegex(VerdictError, "winner"):
            from_dict({"a": "x", "b": "y"})
        with self.assertRaisesRegex(VerdictError, "one of"):
            from_dict({"a": "x", "b": "y", "winner": "left"})
        with self.assertRaisesRegex(VerdictError, "first"):
            from_dict({"a": "x", "b": "y", "winner": "a", "first": "c"})

    def test_aa_needs_two_executions(self):
        with self.assertRaisesRegex(VerdictError, "two separate executions"):
            from_dict({"a": "x", "b": "x", "winner": "a"})
        v = from_dict({"a": "x", "b": "x", "a_run": "1", "b_run": "2", "winner": "TIE"})
        self.assertTrue(v.is_aa)
        self.assertEqual(v.winner, "tie")

    def test_load_reports_line_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v.jsonl")
            with open(path, "w") as fh:
                fh.write(json.dumps({"a": "x", "b": "y", "winner": "a"}) + "\n\n")
                fh.write("{not json}\n")
            with self.assertRaisesRegex(VerdictError, r"v\.jsonl:3"):
                load_jsonl(path)

    def test_group_pairs(self):
        vs = [from_dict({"a": "x", "b": "y", "a_run": "1", "b_run": "1", "winner": w}) for w in "aab"]
        vs.append(from_dict({"a": "x", "b": "y", "a_run": "2", "b_run": "2", "winner": "b"}))
        groups = group_pairs(vs)
        self.assertEqual(sorted(len(g) for g in groups.values()), [1, 3])


if __name__ == "__main__":
    unittest.main()
