"""Command line: elitecheck {aa,winshare,plan,simulate}."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import List, Optional

from .aa import RULES, aa_report, winshare_report
from .gate import Design, oc_table
from .sim import POLICIES, Noise, calibrate_exec_sd, simulate
from .verdicts import VerdictError, load_jsonl


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.3f}"
    if isinstance(x, (tuple, list)) and all(isinstance(v, (int, float)) for v in x):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    return str(x)


def _print_kv(d: dict) -> None:
    width = max(len(k) for k in d)
    for k, v in d.items():
        print(f"{k:<{width}}  {_fmt(v)}")


def _print_table(rows: List[dict], cols: List[str]) -> None:
    cells = [[_fmt(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for row in cells:
        print("  ".join(v.rjust(w) for v, w in zip(row, widths)))


def _design(args) -> Design:
    return Design(p1=args.p1, alpha=args.alpha, beta=args.beta, max_n=args.max_n)


def _add_design(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("gate design")
    g.add_argument("--p1", type=float, default=0.7, help="smallest round win rate worth promoting (default 0.7)")
    g.add_argument("--alpha", type=float, default=0.05, help="nominal false-promotion rate at p=0.5 (default 0.05)")
    g.add_argument("--beta", type=float, default=0.2, help="nominal miss rate at p=p1 (default 0.2)")
    g.add_argument("--max-n", type=int, default=40, help="cap on decisive rounds; cap keeps incumbent (default 40)")


def cmd_aa(args) -> dict:
    rep = aa_report(load_jsonl(args.file), rule=args.rule, looks=args.looks, seed=args.seed)
    if not args.json:
        _print_kv(rep)
        print(
            f"\nThe {args.rule!r} rule promotes an equally good candidate in {rep['spurious_promotion_per_look']:.1%} "
            f"of fresh comparisons on this judge and executor. If all {args.looks} looks were independent fresh "
            f"comparisons, that is a {rep['p_at_least_one_spurious_if_independent']:.1%} chance of at least one "
            "spurious promotion. Searches that reuse a stored incumbent output have correlated looks; "
            "model them with `elitecheck simulate --calibrate-rate`."
        )
    return rep


def cmd_winshare(args) -> dict:
    rep = winshare_report(load_jsonl(args.file), args.a, args.b, cluster=args.cluster, seed=args.seed)
    if not args.json:
        _print_kv(rep)
    return rep


def cmd_plan(args) -> dict:
    d = _design(args)
    rows = oc_table(d)
    rep = {"design": {"p1": d.p1, "alpha": d.alpha, "beta": d.beta, "max_n": d.max_n}, "oc": rows}
    if not args.json:
        print(f"Truncated SPRT, p0=0.5 vs p1={d.p1}, alpha={d.alpha}, beta={d.beta}, max_n={d.max_n}")
        print("p = true probability the challenger wins a decisive round (exact, not simulated)\n")
        _print_table(rows, ["p", "p_promote", "expected_rounds", "single_comparison_p_promote"])
    return rep


def cmd_simulate(args) -> dict:
    noise = Noise(args.exec_sd, args.judge_scale, args.tie_prob, args.position_bias)
    calibrated = None
    if args.calibrate_rate is not None:
        if args.aa_judgments is None:
            raise ValueError("--calibrate-rate needs --aa-judgments (judgments per A/A pair in the data it came from)")
        calibrated = calibrate_exec_sd(args.calibrate_rate, args.aa_judgments, args.calibrate_rule, noise, seed=args.seed)
        noise = Noise(calibrated, args.judge_scale, args.tie_prob, args.position_bias)
    rows = [
        simulate(p, iterations=args.iterations, searches=args.searches, step_mean=args.step_mean,
                 step_sd=args.step_sd, noise=noise, judgments=args.judgments, design=_design(args), seed=args.seed)
        for p in args.policies
    ]
    rep = {"exec_sd": noise.exec_sd, "calibrated": calibrated is not None, "results": rows}
    if not args.json:
        print(f"exec_sd={noise.exec_sd:.3f}{' (calibrated from A/A rate)' if calibrated is not None else ''}, "
              f"{args.searches} searches x {args.iterations} proposals, {args.judgments} judgment(s) per comparison")
        print("Averages per search. Gains are on the judge's logit scale.\n")
        _print_table(rows, ["policy", "final_gain", "promotions", "false_promotions", "false_promotion_share",
                            "missed_improvement_share", "executions", "judgments"])
    return rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="elitecheck", description="Check whether an LLM-judged optimizer's new best is real.")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print machine-readable JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("aa", parents=[common], help="A/A negative control: spurious-promotion rate and position bias")
    a.add_argument("file", help="JSONL verdicts; A/A records have a == b with different runs")
    a.add_argument("--rule", choices=RULES, default="majority", help="promotion rule to audit (default majority)")
    a.add_argument("--looks", type=int, default=20, help="search iterations to project over (default 20)")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_aa)

    w = sub.add_parser("winshare", parents=[common], help="win share of a over b with naive and cluster-level intervals")
    w.add_argument("file")
    w.add_argument("--a", required=True)
    w.add_argument("--b", required=True)
    w.add_argument("--cluster", choices=("pair", "task"), default="pair")
    w.add_argument("--seed", type=int, default=0)
    w.set_defaults(fn=cmd_winshare)

    pl = sub.add_parser("plan", parents=[common], help="exact operating characteristics of a gate design")
    _add_design(pl)
    pl.set_defaults(fn=cmd_plan)

    s = sub.add_parser("simulate", parents=[common], help="compare promotion rules in a simulated noisy search")
    s.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    s.add_argument("--iterations", type=int, default=20)
    s.add_argument("--searches", type=int, default=500)
    s.add_argument("--step-mean", type=float, default=-0.2, help="mean true change per proposal (default -0.2)")
    s.add_argument("--step-sd", type=float, default=0.6)
    s.add_argument("--exec-sd", type=float, default=1.0, help="execution noise relative to judge noise")
    s.add_argument("--judge-scale", type=float, default=1.0)
    s.add_argument("--tie-prob", type=float, default=0.0)
    s.add_argument("--position-bias", type=float, default=0.0)
    s.add_argument("--judgments", type=int, default=1, help="judgments per comparison or round")
    s.add_argument("--calibrate-rate", type=float, help="measured A/A spurious-promotion rate (from `elitecheck aa`)")
    s.add_argument("--aa-judgments", type=int, help="judgments per pair in the A/A data behind --calibrate-rate")
    s.add_argument("--calibrate-rule", choices=("majority", "unanimous"), default="majority",
                   help="rule used by `elitecheck aa` for that rate (default majority, as in `aa`)")
    s.add_argument("--seed", type=int, default=0)
    _add_design(s)
    s.set_defaults(fn=cmd_simulate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rep = args.fn(args)
    except (VerdictError, ValueError, OSError) as e:
        print(f"elitecheck: error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(_json_safe(rep), indent=2, allow_nan=False))
    return 0


def _json_safe(x):
    """Replace NaN with null so the output is strict JSON."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x
