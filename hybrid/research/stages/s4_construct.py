"""S4 construct: the builder writes candidate.py; static checks; executor run with a causality test.

Static checks (host, AST only, the code is never imported on the host):
- imports limited to ALLOWED_MODULES;
- no calls to open, exec, eval, compile, __import__, input, breakpoint, globals, locals, getattr on
  modules by string;
- for a card whose total_sequence complexity is below quadratic, no attention building blocks
  (nn.MultiheadAttention, nn.Transformer*, scaled_dot_product_attention).
The executor check fits the candidate on a small random corpus, validates predict_proba rows,
runs hybrid.bench.api.check_causality, describe_problems and contract_problems (describe() must report integer
"hyper" and "train_max_len", which card falsifier tasks check), and counts parameters and state.
"""
from __future__ import annotations

import ast
import json
import uuid
from pathlib import Path

from hybrid.executor.job import Caps
from hybrid.research.stages.common import StageContext, card_for_prompt, claimed_exponent, load_section
from hybrid.research.stages.contracts import BUILDER

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
ALLOWED_MODULES = frozenset({"numpy", "torch", "math", "cmath", "itertools", "functools", "collections",
                             "dataclasses", "typing", "__future__", "heapq", "bisect", "random", "operator",
                             "fractions", "statistics", "abc", "enum", "copy"})
FORBIDDEN_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                             "globals", "locals", "vars", "setattr", "delattr"})
ATTENTION_NAMES = frozenset({"MultiheadAttention", "Transformer", "TransformerEncoder", "TransformerDecoder",
                             "TransformerEncoderLayer", "TransformerDecoderLayer", "scaled_dot_product_attention"})

CHECK_SCRIPT = r'''
import json, sys, traceback
sys.path.insert(0, "/in")
out = {"ok": False, "problems": []}
try:
    import numpy as np
    from api import check_causality, contract_problems, describe_problems
    from candidate import Candidate
    rng = np.random.default_rng(0)
    V = 7
    train = [[int(t) for t in rng.integers(0, V, size=int(rng.integers(12, 40)))] for _ in range(24)]
    model = Candidate(V, 0, {"max_steps": 30, "max_epochs": 3, "batch_size": 8, "lr": 0.003})
    model.fit(train)
    prefixes = [[int(t) for t in rng.integers(0, V, size=L)] for L in (1, 2, 5, 13, 31)]
    pred = np.asarray(model.predict_proba(prefixes), dtype=np.float64)
    if pred.shape != (len(prefixes), V):
        out["problems"].append("predict_proba shape %s, expected %s" % (pred.shape, (len(prefixes), V)))
    else:
        if not np.all(np.isfinite(pred)):
            out["problems"].append("predict_proba has non-finite values")
        if np.any(pred < 0):
            out["problems"].append("predict_proba has negative values")
        sums = pred.sum(axis=1)
        if np.any(np.abs(sums - 1) > 1e-3):
            out["problems"].append("rows do not sum to 1: %s" % sums.tolist())
        causality = check_causality(model, prefixes)
        out["causality"] = causality
        if not causality["ok"]:
            out["problems"].append("causality check failed: %s" % causality["max_dev"])
    desc = model.describe()
    out["describe"] = desc
    out["problems"] += describe_problems(desc)
    out["problems"] += contract_problems(desc)
    measured = 0
    torch_params = 0
    try:
        import torch
        for value in vars(model).values():
            if isinstance(value, torch.nn.Module):
                torch_params += sum(p.numel() for p in value.parameters())
            elif isinstance(value, torch.Tensor):
                torch_params += value.numel()
    except ImportError:
        pass
    numpy_params = sum(v.size for v in vars(model).values() if isinstance(v, np.ndarray))
    measured = torch_params + numpy_params
    out["measured_params"] = {"torch": int(torch_params), "numpy_arrays": int(numpy_params), "total": int(measured)}
    if isinstance(desc, dict) and isinstance(desc.get("params"), int) and measured > 0:
        claimed = desc["params"]
        if claimed * 4 < measured or claimed > 4 * measured + 1000:
            out["problems"].append("describe() params %d is far from the %d floats found in model attributes"
                                   % (claimed, measured))
except Exception:
    out["problems"].append(traceback.format_exc()[-3000:])
out["ok"] = not out["problems"]
print(json.dumps(out, default=str))
'''


def attention_allowed(card: dict) -> bool:
    exponent = claimed_exponent(card["claims"]["complexity"]["total_sequence"])
    return exponent is not None and exponent >= 2


def static_problems(code: str, allow_attention: bool) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems = []
    has_candidate = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    problems.append(f"forbidden import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                problems.append(f"forbidden import from {node.module} (line {node.lineno})")
            if not allow_attention:
                for alias in node.names:
                    if alias.name in ATTENTION_NAMES:
                        problems.append(f"attention building block {alias.name} in a non-quadratic claim")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"forbidden call {node.func.id}() (line {node.lineno})")
        elif isinstance(node, ast.Attribute):
            if not allow_attention and node.attr in ATTENTION_NAMES:
                problems.append(f"attention building block {node.attr} (line {node.lineno}) in a non-quadratic claim")
            if node.attr.startswith("__") and node.attr not in ("__init__", "__name__", "__class__"):
                problems.append(f"dunder attribute access {node.attr} (line {node.lineno})")
        elif isinstance(node, ast.ClassDef) and node.name == "Candidate":
            has_candidate = True
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            missing = {"__init__", "fit", "predict_proba", "describe"} - methods
            if missing:
                problems.append(f"Candidate is missing {sorted(missing)}")
    if not has_candidate:
        problems.append("no class named Candidate")
    return problems


def builder_prompt(card: dict, feedback: list[str] | None, previous: str | None) -> str:
    parts = [
        "CARD:\n" + card_for_prompt(card),
        "CANDIDATE API (hybrid/bench/api.py):\n" + (BENCH_DIR / "api.py").read_text(encoding="utf-8"),
        "TEMPLATE (hybrid/bench/candidate_template.py):\n"
        + (BENCH_DIR / "candidate_template.py").read_text(encoding="utf-8"),
        "ALLOWED IMPORTS: " + ", ".join(sorted(ALLOWED_MODULES)),
        load_section("builder_contract"),
    ]
    if not attention_allowed(card):
        parts.append("The card claims sub-quadratic total cost, so attention modules "
                     "(nn.MultiheadAttention, nn.Transformer*, scaled_dot_product_attention) are forbidden.")
    if feedback:
        parts.append("YOUR PREVIOUS candidate.py:\n" + (previous or ""))
        parts.append("PROBLEMS FOUND (fix all of them):\n" + "\n".join(f"- {p}" for p in feedback))
    return "\n\n".join(parts)


async def run(ctx: StageContext, card_id: str) -> str | None:
    card = ctx.card(card_id)
    ex = ctx.cfg["executor"]
    caps = Caps(wall_s=ex["construct_wall_s"], memory_mb=ex["construct_memory_mb"], pids=ex["pids"],
                tmp_mb=ex["tmp_mb"])
    api_source = (BENCH_DIR / "api.py").read_text(encoding="utf-8")
    allow = attention_allowed(card)
    calls: list[str] = []
    feedback: list[str] | None = None
    previous: str | None = None
    attempts = 1 + ctx.cfg["s4"]["repair_rounds"]
    tag = uuid.uuid4().hex[:6]   # a resumed S4 writes new files instead of colliding with old ones
    for attempt in range(attempts):
        built = await ctx.ask("builder", "builder", builder_prompt(card, feedback, previous), BUILDER)
        calls.append(built.call_id)
        code = built.parsed["candidate_py"]
        previous = code
        candidate_dir = ctx.run_dir / "candidates" / card_id / f"{tag}-attempt-{attempt}"
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "candidate.py").write_text(code, encoding="utf-8")
        artifact = ctx.store.put_artifact("candidate", f"{card_id}-{tag}-attempt-{attempt}.py", code.encode("utf-8"),
                                          card_id=card_id)
        problems = static_problems(code, allow)
        report: dict = {"attempt": attempt, "static_problems": problems}
        if not problems:
            result = await ctx.runner.run(card_id, "construct_check",
                                          {"candidate.py": code, "api.py": api_source, "main.py": CHECK_SCRIPT},
                                          caps=caps)
            calls.append(result.job_id)
            printed = result.printed_json() if result.ok else None
            report["check"] = printed
            if printed is None:
                problems = [f"check job {result.status} {result.detail}; stderr tail: {result.stderr[-1500:]}; "
                            f"stdout tail: {result.stdout[-500:]}"]
            else:
                problems = list(printed["problems"])
        ctx.put_json_artifact("construct_check", f"{card_id}-{tag}-attempt-{attempt}.json", report, card_id)
        if not problems:
            ctx.emit("candidate_built", {"card_id": card_id, "attempt": attempt,
                                         "candidate_dir": candidate_dir.relative_to(ctx.run_dir).as_posix(),
                                         "sha256": artifact["sha256"], "describe": report["check"].get("describe"),
                                         "measured_params": report["check"].get("measured_params")})
            ctx.verdict(card_id, "S4", "pass",
                        [f"attempt {attempt}: static checks and executor check passed",
                         f"describe: {json.dumps(report['check'].get('describe'))[:400]}",
                         f"measured params: {report['check'].get('measured_params')}"], calls)
            return "S5"
        feedback = problems
    ctx.verdict(card_id, "S4", "fail",
                [f"{attempts} attempt(s) failed; last problems: " + "; ".join(feedback or [])[:1500]], calls)
    return None
