"""Monte Carlo comparison of promotion rules in a noisy LLM-judged search.

Model (all on the judge's logit scale):
- Each proposal changes the incumbent's true quality by delta ~ Normal(step_mean, step_sd).
- One execution of a candidate with true quality mu yields an output of quality mu + Normal(0, exec_sd).
- A judgment of outputs (x_c, x_i) prefers the challenger with probability
  sigmoid((x_c - x_i) / judge_scale +/- position_bias), and is a tie with probability tie_prob.
  Repeated judgments of the same pair share the same outputs, so they are correlated.

Policies:
- ``elite1``: execute the challenger once, compare with the incumbent's STORED output (as in
  MASS Algorithm 2), promote on a strict majority of ``judgments`` verdicts.
- ``gate``: the sequential gate in ``elitecheck.gate`` on fresh executions of both sides.
- ``oracle``: promote exactly when delta > 0 (a ceiling, not a usable rule).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .aa import RULES
from .gate import CONTINUE, PROMOTE, Design, Gate
from .stats import sigmoid

POLICIES = ("elite1", "gate", "oracle")


@dataclass(frozen=True)
class Noise:
    exec_sd: float = 1.0
    judge_scale: float = 1.0
    tie_prob: float = 0.0
    position_bias: float = 0.0

    def __post_init__(self):
        if self.exec_sd < 0 or self.judge_scale <= 0 or not 0 <= self.tie_prob < 1:
            raise ValueError("need exec_sd >= 0, judge_scale > 0, 0 <= tie_prob < 1")


def _votes(rng: random.Random, x_c: float, x_i: float, k: int, noise: Noise, slot: int):
    """Return (challenger_votes, incumbent_votes, next_slot) for k alternating-order judgments."""
    c = i = 0
    for _ in range(k):
        if rng.random() < noise.tie_prob:
            slot += 1
            continue
        bias = noise.position_bias if slot % 2 == 0 else -noise.position_bias
        slot += 1
        if rng.random() < sigmoid((x_c - x_i) / noise.judge_scale + bias):
            c += 1
        else:
            i += 1
    return c, i, slot


def simulate(
    policy: str,
    iterations: int = 20,
    searches: int = 500,
    step_mean: float = -0.2,
    step_sd: float = 0.6,
    noise: Optional[Noise] = None,
    judgments: int = 1,
    design: Optional[Design] = None,
    max_rounds: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Run ``searches`` independent searches of ``iterations`` proposals each; return averages.

    Proposal deltas are drawn from a stream seeded only by (seed, search), so every policy
    sees the same proposals.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if judgments < 1 or iterations < 1 or searches < 1:
        raise ValueError("judgments, iterations and searches must be at least 1")
    noise = noise or Noise()
    design = design or Design()
    max_rounds = 3 * design.max_n if max_rounds is None else max_rounds
    tot = dict(promotions=0, false_promotions=0, improvements_offered=0, improvements_missed=0,
               final_gain=0.0, peak_gain=0.0, executions=0, judgments=0)
    for s in range(searches):
        deltas_rng = random.Random(f"{seed}:{s}:deltas")
        rng = random.Random(f"{seed}:{s}:{policy}")
        mu = 0.0
        peak = 0.0
        stored = mu + rng.gauss(0, noise.exec_sd)  # elite1's stored incumbent output
        tot["executions"] += policy == "elite1"
        for _ in range(iterations):
            delta = deltas_rng.gauss(step_mean, step_sd)
            if delta > 0:
                tot["improvements_offered"] += 1
            if policy == "oracle":
                promote = delta > 0
            elif policy == "elite1":
                x_c = mu + delta + rng.gauss(0, noise.exec_sd)
                tot["executions"] += 1
                c, i, _ = _votes(rng, x_c, stored, judgments, noise, rng.randrange(2))
                tot["judgments"] += judgments
                promote = c > i
                if promote:
                    stored = x_c
            else:
                gate = Gate(design)
                rounds = 0
                slot = rng.randrange(2)
                while gate.decision == CONTINUE and rounds < max_rounds:
                    x_i = mu + rng.gauss(0, noise.exec_sd)
                    x_c = mu + delta + rng.gauss(0, noise.exec_sd)
                    c, i, slot = _votes(rng, x_c, x_i, judgments, noise, slot)
                    gate.record_round(c, i)
                    rounds += 1
                tot["executions"] += 2 * rounds
                tot["judgments"] += judgments * rounds
                promote = gate.decision == PROMOTE
            if promote:
                tot["promotions"] += 1
                tot["false_promotions"] += delta <= 0
                mu += delta
            elif delta > 0:
                tot["improvements_missed"] += 1
            peak = max(peak, mu)
        tot["final_gain"] += mu
        tot["peak_gain"] += peak
    out = {k: v / searches for k, v in tot.items()}
    out["false_promotion_share"] = (
        tot["false_promotions"] / tot["promotions"] if tot["promotions"] else 0.0
    )
    out["missed_improvement_share"] = (
        tot["improvements_missed"] / tot["improvements_offered"] if tot["improvements_offered"] else 0.0
    )
    out["policy"] = policy
    return out


def aa_spurious_rate(exec_sd: float, judgments: int, rule: str = "majority", noise: Optional[Noise] = None,
                     draws: int = 4000, seed: int = 0) -> float:
    """Model-implied A/A spurious-promotion rate per look for a rule over ``judgments`` verdicts.

    Matches ``aa_report``'s statistic (symmetrised over which side is the challenger), with
    presentation order alternating across the judgments of a pair and position bias applied
    accordingly. Uses common random numbers, so for a fixed seed the result is monotone in
    ``exec_sd`` whenever the rule's per-draw probability is monotone in the output gap.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    base = noise or Noise()
    rng = random.Random(seed)
    zs = [rng.gauss(0, 1) for _ in range(draws)]
    total = 0.0
    for z in zs:
        d = z * exec_sd * math.sqrt(2) / base.judge_scale
        for offset in (0, 1):  # which side is shown first in the pair's first judgment
            biases = [base.position_bias if (offset + j) % 2 == 0 else -base.position_bias for j in range(judgments)]
            pb = [(1 - base.tie_prob) * sigmoid(d + b) for b in biases]
            pa = [(1 - base.tie_prob) * sigmoid(-d - b) for b in biases]
            total += (_side_promoted(pb, pa, base.tie_prob, rule) + _side_promoted(pa, pb, base.tie_prob, rule)) / 4
    return total / draws


def _side_promoted(win: List[float], lose: List[float], tie: float, rule: str) -> float:
    """P(one side is promoted) given per-judgment win/lose probabilities (ties fill the rest)."""
    if rule == "single":
        return win[0]
    if rule == "unanimous":
        return math.prod(win)
    margin = {0: 1.0}  # distribution of (wins - losses) over the judgments so far
    for w, l in zip(win, lose):
        nxt: Dict[int, float] = {}
        for m, pr in margin.items():
            for step, q in ((1, w), (-1, l), (0, tie)):
                nxt[m + step] = nxt.get(m + step, 0.0) + pr * q
        margin = nxt
    return sum(pr for m, pr in margin.items() if m > 0)


def calibrate_exec_sd(target_rate: float, judgments: int, rule: str = "majority",
                      noise: Optional[Noise] = None, seed: int = 0) -> float:
    """Find exec_sd whose model-implied A/A spurious rate matches a measured one (bisection).

    Identifies execution noise relative to judge noise (judge_scale). With a single judgment
    per pair the rate does not depend on exec_sd, so calibration needs judgments >= 2.
    """
    if judgments < 2 or rule == "single":
        raise ValueError("calibration needs at least 2 judgments per pair and rule majority/unanimous")
    def f(sd: float) -> float:
        return aa_spurious_rate(sd, judgments, rule, noise, seed=seed)

    lo, hi = 0.0, 50.0
    if f(hi) - f(lo) < 1e-6:
        raise ValueError(f"rule {rule!r} with {judgments} judgments does not depend on exec_sd; use 'unanimous'")
    if not f(lo) <= target_rate <= f(hi):
        raise ValueError(
            f"target {target_rate:.3f} outside the achievable range [{f(lo):.3f}, {f(hi):.3f}] for this rule"
        )
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
