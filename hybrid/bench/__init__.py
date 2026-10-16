"""Bigbrane trusted evaluation bench (RESEARCH-SYSTEM-SPEC.md section 4). See README.md.

This file stays import-free: the evaluator job stages only metrics.py, scaling.py and
evaluator.py, and importing the package there must not pull in host-only modules.
Use `from hybrid.bench.evaluate import evaluate_candidate`.
"""


def __getattr__(name):
    if name == "evaluate_candidate":
        from .evaluate import evaluate_candidate
        return evaluate_candidate
    raise AttributeError(name)
