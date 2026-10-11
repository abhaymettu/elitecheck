"""A/A negative controls and cluster-aware win shares from verdict logs."""

from __future__ import annotations

import math
import random
from typing import Dict, Sequence

from .stats import cluster_bootstrap_share, sign_test, wilson
from .verdicts import Verdict, group_pairs

RULES = ("single", "majority", "unanimous")


def _promotes(winners: Sequence[str], side: str, rule: str) -> bool:
    """Would ``side`` be promoted from this pair's judgments under ``rule``?"""
    other = "a" if side == "b" else "b"
    if rule == "single":
        return winners[0] == side
    w, l = winners.count(side), winners.count(other)
    if rule == "majority":
        return w > l
    if rule == "unanimous":
        return w == len(winners)
    raise ValueError(f"rule must be one of {RULES}")


def aa_report(
    verdicts: Sequence[Verdict],
    rule: str = "majority",
    looks: int = 20,
    reps: int = 2000,
    seed: int = 0,
) -> Dict:
    """Summarise A/A records (same candidate on both sides, separate executions).

    Under the null every comparison here is between equally good candidates, so any
    promotion the rule would grant is spurious. The report gives the per-look spurious
    promotion rate for ``rule`` (symmetrised over which side plays challenger) and position bias.

    The ``*_if_independent`` projections assume every look is a fresh comparison of two new
    executions. A search that keeps comparing against one stored incumbent output (as MASS
    Algorithm 2 does) has correlated looks: it promotes less often, but each promotion then
    raises the bar to a lucky draw. Use ``elitecheck.sim`` to model that case.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if looks < 1:
        raise ValueError("looks must be at least 1")
    aa = [v for v in verdicts if v.is_aa]
    pairs = group_pairs(aa)
    if not pairs:
        raise ValueError("no A/A records (records whose a and b name the same candidate)")
    per_pair = []
    split = 0
    for group in pairs.values():
        winners = [v.winner for v in group]
        per_pair.append((_promotes(winners, "a", rule) + _promotes(winners, "b", rule)) / 2)
        if "a" in winners and "b" in winners:
            split += 1
    m = len(per_pair)
    p_spur = sum(per_pair) / m
    rng = random.Random(seed)
    boots = sorted(sum(per_pair[rng.randrange(m)] for _ in range(m)) / m for _ in range(reps))
    ci = (boots[int(0.025 * reps)], boots[min(reps - 1, int(math.ceil(0.975 * reps)) - 1)])

    decisive = [v for v in aa if v.winner != "tie"]
    pos = [v for v in decisive if v.first is not None]
    first_wins = sum(v.winner == v.first for v in pos)
    pos_clusters: Dict = {}
    for v in pos:
        c = pos_clusters.setdefault(v.pair, [0, 0])
        c[0 if v.winner == v.first else 1] += 1
    k_counts = sorted({len(g) for g in pairs.values()})
    return {
        "rule": rule,
        "aa_pairs": m,
        "aa_judgments": len(aa),
        "judgments_per_pair": k_counts,
        "decisive_rate": len(decisive) / len(aa),
        "split_pair_rate": split / m,
        "spurious_promotion_per_look": p_spur,
        "spurious_promotion_per_look_ci95": ci,
        "looks": looks,
        "expected_spurious_promotions_if_independent": looks * p_spur,
        "p_at_least_one_spurious_if_independent": 1 - (1 - p_spur) ** looks,
        "position_judgments": len(pos),
        "first_shown_win_rate": first_wins / len(pos) if pos else math.nan,
        "first_shown_win_rate_ci95": cluster_bootstrap_share(
            [tuple(c) for c in pos_clusters.values()], reps=reps, seed=seed
        )
        if pos
        else (math.nan, math.nan),
    }


def winshare_report(
    verdicts: Sequence[Verdict],
    a: str,
    b: str,
    cluster: str = "pair",
    reps: int = 2000,
    seed: int = 0,
) -> Dict:
    """Win share of candidate ``a`` over ``b`` with naive and cluster-level uncertainty.

    ``cluster`` is "pair" (each pair of executions is one unit) or "task".
    """
    if cluster not in ("pair", "task"):
        raise ValueError("cluster must be 'pair' or 'task'")
    if a == b:
        raise ValueError("a and b must differ; use aa_report for A/A data")
    clusters: Dict = {}
    wins = losses = ties = missing_runs = 0
    for v in verdicts:
        if {v.a, v.b} != {a, b}:
            continue
        missing_runs += not (v.a_run and v.b_run)
        winner = v.winner if v.a == a else {"a": "b", "b": "a", "tie": "tie"}[v.winner]
        key = v.pair if cluster == "pair" else v.task
        c = clusters.setdefault(key, [0, 0])
        if winner == "a":
            wins += 1
            c[0] += 1
        elif winner == "b":
            losses += 1
            c[1] += 1
        else:
            ties += 1
    n = wins + losses
    if n == 0:
        raise ValueError(f"no decisive judgments between {a!r} and {b!r}")
    naive = wilson(wins, n)
    decisive = [tuple(c) for c in clusters.values() if c[0] + c[1] > 0]
    clus = cluster_bootstrap_share(decisive, reps=reps, seed=seed)
    cl_w = sum(1 for c in clusters.values() if c[0] > c[1])
    cl_l = sum(1 for c in clusters.values() if c[1] > c[0])
    naive_w, clus_w = naive[1] - naive[0], clus[1] - clus[0]
    return {
        "a": a,
        "b": b,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_share": wins / n,
        "naive_ci95": naive,
        "cluster": cluster,
        "clusters": len(decisive),
        "records_without_run_ids": missing_runs,
        "cluster_ci95": clus,
        # nan when the bootstrap cannot vary (fewer than 2 clusters, or every cluster identical)
        "design_effect": (clus_w / naive_w) ** 2 if naive_w > 0 and clus_w > 0 and len(decisive) > 1 else math.nan,
        "cluster_majority_wins": cl_w,
        "cluster_majority_losses": cl_l,
        "cluster_sign_test_p": sign_test(cl_w, cl_l),
    }
