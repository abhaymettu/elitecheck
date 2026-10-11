"""Pairwise judge verdict records and JSONL loading.

One record is one judgment of one pair of outputs:

    {"task": "t1", "a": "wf-3", "a_run": "r1", "b": "wf-7", "b_run": "r4",
     "winner": "a", "judge": "gpt-x/seed42", "first": "b"}

``winner`` is "a", "b" or "tie". ``first`` (optional) says which output the judge saw first.
``a_run`` and ``b_run`` identify executions. Records that share task, candidates and runs (in
either orientation) are repeated judgments of the same outputs and are treated as one cluster.
Without run ids, all judgments of the same two candidates on a task collapse into one cluster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

WINNERS = ("a", "b", "tie")


class VerdictError(ValueError):
    pass


@dataclass(frozen=True)
class Verdict:
    task: str
    a: str
    b: str
    winner: str
    a_run: str = ""
    b_run: str = ""
    judge: str = ""
    first: Optional[str] = None

    @property
    def pair(self) -> Tuple[str, str, str, str, str]:
        """Execution-pair key, the same whichever side was logged as a."""
        lo, hi = sorted([(self.a, self.a_run), (self.b, self.b_run)])
        return (self.task,) + lo + hi

    def canonical(self) -> "Verdict":
        """This record oriented so that (a, a_run) sorts before (b, b_run)."""
        if (self.a, self.a_run) <= (self.b, self.b_run):
            return self
        flip = {"a": "b", "b": "a", "tie": "tie", None: None}
        return Verdict(self.task, self.b, self.a, flip[self.winner], self.b_run, self.a_run, self.judge, flip[self.first])

    @property
    def is_aa(self) -> bool:
        return self.a == self.b


def from_dict(d: dict, where: str = "") -> Verdict:
    for key in ("a", "b", "winner"):
        if key not in d:
            raise VerdictError(f"{where}missing required field {key!r}")
    winner = str(d["winner"]).lower()
    if winner not in WINNERS:
        raise VerdictError(f"{where}winner must be one of {WINNERS}, got {d['winner']!r}")
    first = d.get("first")
    if first is not None:
        first = str(first).lower()
        if first not in ("a", "b"):
            raise VerdictError(f"{where}first must be 'a' or 'b', got {d['first']!r}")
    v = Verdict(
        task=str(d.get("task", "")),
        a=str(d["a"]),
        b=str(d["b"]),
        winner=winner,
        a_run=str(d.get("a_run", "")),
        b_run=str(d.get("b_run", "")),
        judge=str(d.get("judge", "")),
        first=first,
    )
    if v.is_aa and v.a_run == v.b_run:
        raise VerdictError(
            f"{where}A/A record compares run {v.a_run!r} of {v.a!r} with itself; "
            "an A/A control needs two separate executions"
        )
    return v


def load_jsonl(path: str) -> List[Verdict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise VerdictError(f"{path}:{lineno}: invalid JSON ({e.msg})") from None
            if not isinstance(d, dict):
                raise VerdictError(f"{path}:{lineno}: each line must be a JSON object")
            out.append(from_dict(d, where=f"{path}:{lineno}: "))
    return out


def group_pairs(verdicts: Iterable[Verdict]) -> Dict[Tuple[str, str, str, str, str], List[Verdict]]:
    """Group judgments of the same two executions, each oriented canonically."""
    groups: Dict[Tuple[str, str, str, str, str], List[Verdict]] = {}
    for v in verdicts:
        groups.setdefault(v.pair, []).append(v.canonical())
    return groups
