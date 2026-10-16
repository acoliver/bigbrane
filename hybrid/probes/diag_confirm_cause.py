"""Diagnostic: which operation makes the trusted fixture's HMM KL differ between macOS and the container?

Runs a copy of the fixture source with one operation textually replaced by a platform-independent
elementwise equivalent. Usage: diag_confirm_cause.py <fixture path> <variant>
variants: as_is | gemv_elementwise (A @ v in the backward pass as (A * v).sum)"""
import json
import pathlib
import sys

import numpy as np

source = pathlib.Path(sys.argv[1]).read_text()
variant = sys.argv[2]
GEMV = "back = A @ (B[:, xs[t]] * back)"
if variant == "gemv_elementwise":
    assert source.count(GEMV) == 1
    source = source.replace(GEMV, "back = (A * (B[:, xs[t]] * back)[None, :]).sum(axis=1)")
else:
    assert variant == "as_is"
namespace = {"__name__": "diagnostic_fixture"}
exec(compile(source, "diagnostic_fixture", "exec"), namespace)
record = namespace["evaluate"](namespace["CANDIDATES"][0])
print(json.dumps({"variant": variant, "numpy": np.__version__, "hmm_kl_hex": record["metrics"]["hmm_kl"].hex(),
                  "hmm_kl": record["metrics"]["hmm_kl"], "rnn_kl_hex": record["metrics"]["rnn_kl"].hex()}, indent=1))
