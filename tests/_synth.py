"""Synthetic verdict generator for tests: execution noise plus judge noise, nothing real."""

import random

from elitecheck.stats import sigmoid
from elitecheck.verdicts import Verdict


def pairs(n_pairs, k, a="A", b="B", delta=0.0, exec_sd=1.0, position_bias=0.0, tie_prob=0.0, seed=0, task="t"):
    """n_pairs execution pairs of a (true quality delta) vs b (0), each judged k times."""
    rng = random.Random(seed)
    out = []
    for i in range(n_pairs):
        xa = delta + rng.gauss(0, exec_sd)
        xb = rng.gauss(0, exec_sd)
        for j in range(k):
            first = "a" if (i + j) % 2 == 0 else "b"
            if rng.random() < tie_prob:
                w = "tie"
            else:
                bias = position_bias if first == "a" else -position_bias
                w = "a" if rng.random() < sigmoid(xa - xb + bias) else "b"
            out.append(Verdict(task=task, a=a, b=b, winner=w, a_run=f"a{i}", b_run=f"b{i}", judge=f"j{j}", first=first))
    return out
