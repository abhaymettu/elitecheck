"""Small exact and resampling statistics used across elitecheck (stdlib only)."""

from __future__ import annotations

import math
import random
from typing import Sequence, Tuple


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials. Returns (nan, nan) when n == 0."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def half_tail(k: int, n: int) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, 1/2), in integer arithmetic (no overflow for large n)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value for wins vs losses (ties already removed)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = half_tail(max(wins, losses), n)
    return min(1.0, 2 * tail)


def cluster_bootstrap_share(
    clusters: Sequence[Tuple[int, int]],
    reps: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile CI for pooled share sum(w)/sum(w+l), resampling whole clusters.

    Each cluster is (wins, losses) for one independent unit, such as one pair of executions
    judged several times. Resampling clusters, not judgments, keeps repeated judgments of the
    same outputs from counting as independent evidence.
    """
    clusters = [c for c in clusters if c[0] + c[1] > 0]
    if not clusters:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    m = len(clusters)
    shares = []
    for _ in range(reps):
        w = n = 0
        for _ in range(m):
            cw, cl = clusters[rng.randrange(m)]
            w += cw
            n += cw + cl
        shares.append(w / n)
    shares.sort()
    lo_i = int(math.floor((1 - level) / 2 * reps))
    hi_i = min(reps - 1, int(math.ceil((1 + level) / 2 * reps)) - 1)
    return (shares[lo_i], shares[hi_i])


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)
