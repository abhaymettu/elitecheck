"""A promotion gate with known error rates for challenger-vs-incumbent decisions.

The unit of evidence is a round: one fresh execution of the incumbent and one of the
challenger, judged ``k`` times with alternating presentation order. The round goes to the
side that wins the majority of its judgments; tied or split rounds are not decisive.
Rounds are independent executions, so a truncated Wald SPRT on decisive rounds applies.

H0: P(challenger wins a decisive round) = p0 (0.5, no better).
H1: P(challenger wins a decisive round) = p1 (the smallest improvement worth promoting).
Reaching the round cap keeps the incumbent, which can only lower the false-promotion rate.
``operating_characteristics`` computes the exact P(promote) for any true p.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

PROMOTE, KEEP, CONTINUE = "promote", "keep", "continue"


@dataclass(frozen=True)
class Design:
    p1: float = 0.7
    alpha: float = 0.05
    beta: float = 0.2
    max_n: int = 40
    p0: float = 0.5

    def __post_init__(self):
        if not 0 < self.p0 < self.p1 < 1:
            raise ValueError("need 0 < p0 < p1 < 1")
        if not (0 < self.alpha < 1 and 0 < self.beta < 1 and self.alpha + self.beta < 1):
            raise ValueError("need 0 < alpha, beta and alpha + beta < 1")
        if self.max_n < 1:
            raise ValueError("max_n must be at least 1")
        if self.step_win >= self.upper:
            # One decisive win would promote, so a judge that always prefers the first-shown
            # output could promote on identical outputs in round 0.
            raise ValueError("design promotes after a single win; lower alpha or p1")
        if math.ceil(self.upper / self.step_win) > self.max_n:
            raise ValueError(
                f"design can never promote: needs {math.ceil(self.upper / self.step_win)} straight wins "
                f"but max_n is {self.max_n}"
            )

    @property
    def step_win(self) -> float:
        return math.log(self.p1 / self.p0)

    @property
    def step_loss(self) -> float:
        return math.log((1 - self.p1) / (1 - self.p0))

    @property
    def upper(self) -> float:
        return math.log((1 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1 - self.alpha))

    def decide(self, wins: int, losses: int) -> str:
        llr = wins * self.step_win + losses * self.step_loss
        if llr >= self.upper - 1e-12:
            return PROMOTE
        if llr <= self.lower + 1e-12 or wins + losses >= self.max_n:
            return KEEP
        return CONTINUE


def operating_characteristics(design: Design, p: float) -> Tuple[float, float]:
    """Exact (P(promote), expected decisive rounds) when the challenger wins each decisive round with prob p."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    states: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
    promote = 0.0
    expected_n = 0.0
    while states:
        nxt: Dict[Tuple[int, int], float] = {}
        for (w, l), prob in states.items():
            for (w2, l2), q in (((w + 1, l), p), ((w, l + 1), 1 - p)):
                mass = prob * q
                if mass == 0.0:
                    continue
                d = design.decide(w2, l2)
                if d == CONTINUE:
                    nxt[(w2, l2)] = nxt.get((w2, l2), 0.0) + mass
                else:
                    expected_n += mass * (w2 + l2)
                    if d == PROMOTE:
                        promote += mass
        states = nxt
    return promote, expected_n


def oc_table(design: Design, ps=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> List[Dict[str, float]]:
    rows = []
    for p in ps:
        pr, en = operating_characteristics(design, p)
        # A rule that promotes on one decisive fresh comparison promotes with probability p.
        rows.append({"p": p, "p_promote": pr, "expected_rounds": en, "single_comparison_p_promote": p})
    return rows


@dataclass
class Gate:
    """Incremental gate. Feed it rounds; read ``decision``."""

    design: Design = field(default_factory=Design)
    wins: int = 0
    losses: int = 0
    undecided_rounds: int = 0

    @property
    def decision(self) -> str:
        if self.wins == 0 and self.losses == 0:
            return CONTINUE
        return self.design.decide(self.wins, self.losses)

    def record_round(self, challenger_votes: int, incumbent_votes: int) -> str:
        """Record one round's judgment counts and return the current decision."""
        if self.decision != CONTINUE:
            raise RuntimeError(f"gate already decided: {self.decision}")
        if challenger_votes > incumbent_votes:
            self.wins += 1
        elif incumbent_votes > challenger_votes:
            self.losses += 1
        else:
            self.undecided_rounds += 1
        return self.decision


@dataclass
class ChallengeResult:
    decision: str
    wins: int
    losses: int
    undecided_rounds: int
    rounds: int
    judgments: List[Dict[str, Any]]


def challenge(
    run_incumbent: Callable[[], Any],
    run_challenger: Callable[[], Any],
    judge: Callable[[Any, Any], str],
    design: Optional[Design] = None,
    judgments_per_round: int = 1,
    max_rounds: Optional[int] = None,
) -> ChallengeResult:
    """Run the gate end to end.

    ``run_*`` each produce one fresh output per call. ``judge(first, second)`` returns
    "first", "second" or "tie". Presentation order alternates across judgments and rounds.
    ``max_rounds`` (default 3 * design.max_n) caps cost when many rounds are undecided;
    hitting it keeps the incumbent.
    """
    design = design or Design()
    if judgments_per_round < 1:
        raise ValueError("judgments_per_round must be at least 1")
    max_rounds = 3 * design.max_n if max_rounds is None else max_rounds
    gate = Gate(design)
    log: List[Dict[str, Any]] = []
    rounds = 0
    slot = 0
    while gate.decision == CONTINUE and rounds < max_rounds:
        inc_out, ch_out = run_incumbent(), run_challenger()
        votes = {"challenger": 0, "incumbent": 0}
        for _ in range(judgments_per_round):
            ch_first = slot % 2 == 0
            slot += 1
            first, second = (ch_out, inc_out) if ch_first else (inc_out, ch_out)
            verdict = judge(first, second)
            if verdict not in ("first", "second", "tie"):
                raise ValueError(f"judge must return 'first', 'second' or 'tie', got {verdict!r}")
            if verdict == "tie":
                who = "tie"
            else:
                who = "challenger" if (verdict == "first") == ch_first else "incumbent"
                votes[who] += 1
            log.append({"round": rounds, "challenger_first": ch_first, "winner": who})
        gate.record_round(votes["challenger"], votes["incumbent"])
        rounds += 1
    decision = gate.decision if gate.decision != CONTINUE else KEEP
    return ChallengeResult(decision, gate.wins, gate.losses, gate.undecided_rounds, rounds, log)
