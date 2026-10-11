"""Write a synthetic verdict log (no real model, task or person data).

    python examples/make_fixture.py > examples/verdicts.jsonl

Two candidates: "baseline" and "v2" (truly better by 0.4 on the judge's logit scale).
Execution noise (sd 1.5) is larger than the true difference, and every execution pair is
judged six times with alternating order, as with two judges x three seeds.
"""

import json
import math
import random

rng = random.Random(2026)


def judge(xa, xb):
    p = 1 / (1 + math.exp(-(xa - xb)))
    return "a" if rng.random() < p else "b"


def emit(task, a, mu_a, b, mu_b, n_pairs, tag):
    for i in range(n_pairs):
        xa, xb = mu_a + rng.gauss(0, 1.5), mu_b + rng.gauss(0, 1.5)
        for j in range(6):
            first = "a" if (i + j) % 2 == 0 else "b"
            print(json.dumps({"task": task, "a": a, "a_run": f"{tag}{i}a", "b": b, "b_run": f"{tag}{i}b",
                              "winner": judge(xa, xb), "judge": f"judge{j // 3}/seed{j % 3}", "first": first}))


for t in ("finance", "robotics", "chemistry"):
    emit(t, "baseline", 0.0, "baseline", 0.0, 15, f"{t}-aa-")  # A/A control: same candidate twice
    emit(t, "v2", 0.4, "baseline", 0.0, 10, f"{t}-ab-")
