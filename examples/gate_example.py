"""Wire the gate into your own loop. The "agent" and "judge" here are toy stand-ins.

Replace run_incumbent / run_challenger with one fresh execution of each candidate, and
judge with your LLM judge. The judge sees (first, second) and returns "first", "second" or "tie".
"""

import random

from elitecheck import Design, challenge, operating_characteristics

rng = random.Random(0)


def make_agent(quality):
    return lambda: quality + rng.gauss(0, 1.5)  # one noisy execution -> an "output"


def judge(first, second):
    return "first" if first + rng.gauss(0, 1) > second else "second"


design = Design(p1=0.7, alpha=0.05, beta=0.2, max_n=40)

# One decision, with its evidence:
res = challenge(make_agent(0.0), make_agent(1.5), judge, design=design, judgments_per_round=2)
print(f"one challenge: {res.decision} after {res.rounds} rounds ({res.wins} won, {res.losses} lost, "
      f"{res.undecided_rounds} split)\n")

# Any single decision can be unlucky; the guarantee is about rates. Repeat each case 200 times:
for name, q in (("no better", 0.0), ("slightly better", 0.3), ("clearly better", 1.5)):
    runs = [challenge(make_agent(0.0), make_agent(q), judge, design=design, judgments_per_round=2)
            for _ in range(200)]
    rate = sum(r.decision == "promote" for r in runs) / len(runs)
    cost = sum(r.rounds for r in runs) / len(runs)
    print(f"{name:>15}: promoted in {rate:.0%} of 200 challenges, {cost:.0f} rounds on average")

pr, n = operating_characteristics(design, 0.5)
print(f"\nExact: an equally good challenger is promoted with probability {pr:.3f} (mean {n:.1f} decisive rounds).")
