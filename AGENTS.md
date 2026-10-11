# elitecheck: agent entry point

Read README.md before using or changing this package. This is a Python 3.10+ toolkit for A/A controls, execution-aware win-share intervals, and repeated-comparison promotion decisions. It is not a MASS reproduction or a model-training framework.

## Start here

- Install from the checkout: `python3 -m pip install .`.
- Run checks: `python3 -m unittest -v`.
- Offline examples: `python3 -m elitecheck aa examples/verdicts.jsonl`, `python3 -m elitecheck winshare examples/verdicts.jsonl --a v2 --b baseline`, and `python3 -m elitecheck plan`.
- Read `examples/gate_example.py` before integrating `challenge()`.

## Contracts

- Keep runtime dependencies empty. The package uses the Python standard library; hatchling is its build dependency.
- One evidence round means two fresh executions. Repeated judgments of the same output pair are not independent executions.
- Judge callbacks return `first`, `second`, or `tie`. Presentation order alternates. Ties and split rounds add no decisive evidence.
- At either cap, keep the incumbent unless the gate has already decided to promote.
- Log execution IDs and presentation order. Use synthetic data in examples and tests; do not commit private prompts, artifacts, credentials, or traces.

## Before claiming improvement

Run an A/A control with the user's executor and judge. Check operating characteristics with `plan`. Treat simulation output as simulation, not real-model validation. Exact rates require the stated independent-round, stable win-probability model. The gate cannot fix a judge that prefers incorrect output, and a per-challenge error probability is not a guarantee over an entire adaptive search.

The library itself makes no model calls. Callbacks supplied to `challenge()` may spend quota or money and run many times: up to `max_rounds` (default three times `design.max_n`), with two executions and `judgments_per_round` judgments per round. Obtain the owner's approval for the experiment and its cost before connecting live callbacks.

## Release checks

Run the full test suite and build/install smoke checks after changes. Keep AGENTS.md in source distributions. Review README install URLs, license attribution, private-data scrubbing, and any claimed validation results before publishing. Do not publish or change visibility without owner approval.
