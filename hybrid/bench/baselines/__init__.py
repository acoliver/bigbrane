"""Baseline registry. Each baseline is staged as a candidate directory: its module file(s) plus
a two-line candidate.py shim, so baselines take the same execution path as any candidate.

claimed_exponent is the asymptotic time exponent of predict_proba in context length that the
scaling probe compares against (the n-gram reads only the last three tokens). state is the host's own
knowledge of how much each baseline carries while reading a sequence: "fixed" (independent of context
length), "growing" (the transformer's KV cache) or "window" (the n-gram reads a bounded suffix and has
no memory to put pairs in). The fingerprint's capacity comparison uses it to pick the fixed-state baselines,
which sets the bar for no_capacity_advantage; the n-gram is left out of that bar because its recall
accuracy above chance comes from token statistics, not from stored pairs, and its 3-float "state" would
set a bar no storing mechanism could meet. It is never taken from an entrant's describe()."""
from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

BASELINES = {
    "ngram": {"files": ["bb_ngram.py"], "shim": "from bb_ngram import Candidate\n", "claimed_exponent": 0.0,
              "state": "window"},
    "gru": {"files": ["bb_neural.py"], "shim": "from bb_neural import GRUCandidate as Candidate\n",
            "claimed_exponent": 1.0, "state": "fixed"},
    "transformer": {"files": ["bb_neural.py"], "shim": "from bb_neural import TransformerCandidate as Candidate\n",
                    "claimed_exponent": 2.0, "state": "growing"},
    "ssm": {"files": ["bb_neural.py"], "shim": "from bb_neural import SSMCandidate as Candidate\n",
            "claimed_exponent": 1.0, "state": "fixed"},
    "linear_attention": {"files": ["bb_neural.py"],
                         "shim": "from bb_neural import LinearAttentionCandidate as Candidate\n",
                         "claimed_exponent": 1.0, "state": "fixed"},
}
BASELINE_NAMES = tuple(BASELINES)


def stage_baseline(name, destination):
    """Write the baseline's candidate directory into `destination` (must not exist)."""
    entry = BASELINES[name]
    destination = Path(destination)
    destination.mkdir(parents=True)
    for file in entry["files"]:
        shutil.copyfile(HERE / file, destination / file)
    (destination / "candidate.py").write_text(entry["shim"])
    return destination


def source_files():
    return sorted(HERE.glob("bb_*.py"))
