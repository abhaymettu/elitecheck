# elitecheck

Check whether an LLM-judged optimizer's new best is real.

Many prompt, workflow and agent-harness optimizers keep one best-so-far candidate and replace it
when an LLM judge prefers a challenger's output in a single comparison. When one execution of a
candidate can come out very differently from the next, that comparison is partly a coin flip, and
the "best so far" drifts with luck. elitecheck measures how often that happens with your own judge
and executor, and gives you a promotion rule with exact, known error rates.

- `elitecheck aa`: an **A/A negative control**. Judge two separate executions of the *same*
  candidate against each other. Any promotion your rule would grant there is spurious. Reports
  the spurious-promotion rate per comparison, a projection over a search, and position bias.
- `elitecheck winshare`: win share of one candidate over another with judgment-level **and**
  cluster-level intervals. Six judgments of one pair of outputs are not six executions.
- `elitecheck plan`: **exact** operating characteristics of a sequential promotion gate
  (probability of promotion and expected cost for any true win rate), computed, not simulated.
- `elitecheck simulate`: compare "won once" (elite of one) against the gate in a noisy search,
  optionally calibrated to the A/A rate you measured.
- `elitecheck.challenge(...)`: the gate as a Python function you call inside your own loop.

Pure Python standard library. No model calls, no network, no GPU. Python 3.10 or newer.

## Why this exists

The motivating case is the workflow search in MASS (Lee et al., 2026, arXiv
[2610.12176](https://arxiv.org/abs/2610.12176)). Its search keeps a single best workflow per task,
executes each proposal once, and replaces the incumbent when the model's own pairwise judgment
prefers the new output. The paper's appendix documents what execution noise does to this:

- Two executions of byte-identical workflow and prompt text, with the same recorded sampling seed,
  went 6-0 and then 0-6 against the same reference across six external judgments (Appendix E.3, Table 11).
- Between successive iterates, verdict flips from at least 5/6 wins to at most 1/6 (or the reverse)
  occurred at rates of 0.22 to 0.40 in every populated edit-size bin, with little correlation between
  edit size and verdict change (Appendix E, Table 10). Because each proposal is executed once, the
  paper notes, these flips combine prompt changes with execution variability.

The authors state these limits themselves; elitecheck is a tool for measuring and controlling them
in any loop of this shape, not a reanalysis of their results.

## Install

```bash
pip install git+https://github.com/abhaymettu/elitecheck
# or, from a clone:
pip install .
```

## Quick start

The repository ships a synthetic log (`examples/verdicts.jsonl`, made by `examples/make_fixture.py`;
no real model, task or person data). It has A/A pairs of a `baseline` candidate and A/B pairs where
`v2` is truly better by 0.4 on the judge's logit scale, with execution noise of 1.5 and six judgments
per execution pair.

```console
$ elitecheck aa examples/verdicts.jsonl --rule unanimous
rule                                         unanimous
aa_pairs                                     45
aa_judgments                                 270
judgments_per_pair                           [6]
decisive_rate                                1.000
split_pair_rate                              0.667
spurious_promotion_per_look                  0.167
spurious_promotion_per_look_ci95             [0.100, 0.233]
looks                                        20
expected_spurious_promotions_if_independent  3.333
p_at_least_one_spurious_if_independent       0.974
position_judgments                           270
first_shown_win_rate                         0.456
first_shown_win_rate_ci95                    [0.404, 0.507]

The 'unanimous' rule promotes an equally good candidate in 16.7% of fresh comparisons on this judge and executor. If all 20 looks were independent fresh comparisons, that is a 97.4% chance of at least one spurious promotion. Searches that reuse a stored incumbent output have correlated looks; model them with `elitecheck simulate --calibrate-rate`.
```

Even a unanimous six-judgment win is not evidence of improvement when the six judgments look at the
same two noisy executions. The "if independent" projection fits a loop that reruns both sides at
every look. A loop that keeps one stored incumbent output (as MASS does) promotes less often but
then compares against a lucky draw; `simulate` models that case:

```console
$ elitecheck simulate --judgments 1 --aa-judgments 6 --calibrate-rule unanimous --calibrate-rate 0.167 --policies elite1 gate
...
policy  final_gain  promotions  false_promotions  false_promotion_share  missed_improvement_share  executions  judgments
elite1      -0.171       5.686             3.030                  0.533                     0.639      21.000     20.000
  gate       0.884       1.386             0.130                  0.094                     0.829     608.724    304.362
```

```console
$ elitecheck winshare examples/verdicts.jsonl --a v2 --b baseline
...
win_share                0.456
naive_ci95               [0.384, 0.528]
cluster_ci95             [0.339, 0.572]
design_effect            2.626
cluster_sign_test_p      0.701
```

`v2` is better by construction, yet 30 execution pairs cannot show it, and the judgment-level
interval is too narrow by a design effect of about 2.6.

```console
$ elitecheck plan
Truncated SPRT, p0=0.5 vs p1=0.7, alpha=0.05, beta=0.2, max_n=40
p = true probability the challenger wins a decisive round (exact, not simulated)

    p  p_promote  expected_rounds  single_comparison_p_promote
0.300      0.000            7.278                        0.300
0.400      0.003           10.556                        0.400
0.500      0.038           16.754                        0.500
0.600      0.243           23.726                        0.600
0.700      0.688           23.497                        0.700
0.800      0.966           16.878                        0.800
0.900      0.999           11.709                        0.900
```

A challenger that is no better is promoted 3.8% of the time by the gate and 50% of the time by a
rule that promotes on one decisive fresh comparison (the last column, which is just `p`). `plan`
rejects designs that could never promote within `max_n`, or that would promote on a single win.

## Using the gate in your loop

```python
from elitecheck import Design, challenge

result = challenge(
    run_incumbent=lambda: run_agent(incumbent_workflow),    # one fresh execution per call
    run_challenger=lambda: run_agent(challenger_workflow),
    judge=lambda first, second: my_judge(first, second),    # returns "first", "second" or "tie"
    design=Design(p1=0.7, alpha=0.05, beta=0.2, max_n=40),
    judgments_per_round=2,
)
if result.decision == "promote":
    incumbent_workflow = challenger_workflow
```

Each round executes both candidates fresh and judges them `judgments_per_round` times, alternating
which output is shown first. The round goes to the majority; split rounds are not evidence. The
decision is a truncated sequential probability ratio test on decisive rounds, and reaching the cap
keeps the incumbent. `examples/gate_example.py` runs it against a toy agent and judge.

## What the simulation shows, including where the gate loses

`elitecheck simulate` models each proposal as a true change `delta ~ Normal(step_mean, step_sd)`,
each execution as true quality plus `Normal(0, exec_sd)`, and each judgment as a logistic choice on
the two outputs. Elite of one compares the challenger with the incumbent's **stored** output, as
MASS's Algorithm 2 does. The numbers below are model outputs under stated settings, not
measurements of any real system. Gains are on the judge's logit scale, averaged over 500 searches,
three judgments per comparison.

| Setting | Policy | Proposals | Executions | Final gain | Share of promotions that were false |
|---|---|---|---|---|---|
| Proposals mixed (mean -0.2, sd 0.6), exec sd 1.0 | elite of one | 20 | 21 | 0.26 | 0.48 |
| same | gate | 20 | 581 | 1.38 | 0.04 |
| same | elite of one | 300 | 301 | 4.02 | 0.45 |
| Mostly harmful (mean -0.5, sd 0.5), exec sd 0.5 | elite of one | 100 | 101 | -6.91 | 0.72 |
| same | gate | 12 | 251 | 0.24 | 0.07 |

Two readings, both supported:

1. **At a comparable budget, elite of one can win.** When most proposals are neutral or better and
   executions are cheap, many noisy steps beat a few careful ones: 300 proposals reach 4.02 against the
   gate's 1.38, with about the same number of judgments (900 against 871) and half the executions (301 against 581). This is consistent with runtime analysis
   showing that a (1+1) evolutionary algorithm that never re-evaluates its incumbent tolerates
   constant noise rates on a classic benchmark ("Evolutionary Algorithms Are Significantly More Robust
   to Noise When They Ignore It", [arXiv 2409.00306](https://arxiv.org/abs/2409.00306)).
2. **When most proposals make things worse, elite of one walks downhill**, and nearly half or more of
   its promotions are false in every setting above. The gate's per-proposal false-promotion rate stays
   at or below alpha whatever the proposal distribution.

So use the gate where a wrong promotion is expensive: before you claim one workflow beats another,
before you deploy or distill the winner, or when edits to a mature candidate are mostly harmful.
Use `aa` first to learn how noisy your executor and judge are, then `simulate --calibrate-rate RATE
--aa-judgments K --calibrate-rule RULE` with the rate, judgments per pair and rule from that `aa` run
to see how your search behaves at that noise level.

## Verdict log format

JSON Lines, one judgment per line:

```json
{"task": "t1", "a": "wf-3", "a_run": "r1", "b": "wf-7", "b_run": "r4", "winner": "a", "judge": "j1/seed42", "first": "b"}
```

| Field | Required | Meaning |
|---|---|---|
| `a`, `b` | yes | candidate ids; `a == b` marks an A/A control record |
| `winner` | yes | `"a"`, `"b"` or `"tie"` |
| `a_run`, `b_run` | for A/A; strongly advised | execution ids; records sharing task, candidates and runs (either orientation) are judgments of the same outputs. Without them, all judgments of two candidates on a task form one cluster, and `winshare` reports how many records lacked them |
| `task` | no | task id, used for `--cluster task` |
| `judge` | no | free text, for your own bookkeeping |
| `first` | no | `"a"` or `"b"`, which output the judge saw first; enables the position-bias estimate |

The `single` rule in `aa` uses the first record of each pair in file order.

## Assumptions and limits

- Rounds must be independent executions. Caching, shared random seeds or a stateful environment
  that leaks between runs break the gate's error guarantee.
- The gate controls the probability of promoting a candidate the *judge* does not prefer. It cannot
  catch a judge that rewards the wrong thing. MASS reports a revision that changed the physics of a
  test task and still received five of six favorable judgments (Appendix G.3).
- `alpha` bounds the false-promotion probability per challenge, not the share of promotions that are
  false. When true improvements are rare, that share can be high even with a small alpha
  (base-rate effect); `simulate` reports both.
- The truncated test's exact power is below the nominal `1 - beta` (0.69 at p = 0.7 for the default
  cap of 40). Trust `elitecheck plan`, not the nominal values.
- Calibration identifies execution noise relative to judge noise, needs at least two judgments per
  A/A pair, and works best with the `unanimous` rule. It accounts for position bias if you pass
  `--position-bias`. With an odd number of judgments and no ties,
  the majority rule's A/A rate is always 0.5 and carries no information.
- The simulator is a teaching and planning model with Gaussian execution noise and a logistic judge.
  Real loops have heavier tails (crashes, loop halts) and drifting judges.

## Related tools

elitecheck covers A/A controls and promotion gating. For other parts of judge-based evaluation:

- Judge bias probes (position, verbosity, self-preference): [judgekit](https://pypi.org/project/judgekit/),
  [juryrig](https://pypi.org/project/juryrig/), [truescore](https://pypi.org/project/truescore/).
- Bradley-Terry and Elo leaderboards: [evalica](https://pypi.org/project/evalica/),
  [arena-rank](https://pypi.org/project/arena-rank/), [choix](https://pypi.org/project/choix/).
- Clustered standard errors for evals: [errorbars](https://pypi.org/project/errorbars/),
  [inspect-ai](https://pypi.org/project/inspect-ai/); background in Miller, "Adding Error Bars to
  Evals" ([arXiv 2411.00640](https://arxiv.org/abs/2411.00640)).
- Racing for algorithm configuration: irace (R).

## Development

```bash
python -m unittest -v      # 47 tests, about 10 seconds, no network
```

The suite includes negative controls (A/A data and a position-biased judge never promoted above
alpha; judgment-level intervals shown to undercover under clustered noise) and positive controls
(real improvements promoted with the stated power).

## License

MIT. See [LICENSE](LICENSE).
