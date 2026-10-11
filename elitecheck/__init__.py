"""elitecheck: check whether an LLM-judged optimizer's new best is real."""

from .aa import aa_report, winshare_report
from .gate import Design, Gate, challenge, oc_table, operating_characteristics
from .sim import Noise, calibrate_exec_sd, simulate
from .verdicts import Verdict, VerdictError, load_jsonl

__version__ = "0.1.0"
