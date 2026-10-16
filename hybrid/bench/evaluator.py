"""Evaluator job: trusted bench code + entrant outputs + oracle data -> bench_result.json.

    python3 -c "<entry>" <in_dir> <out_dir>      (see evaluate.EVAL_ENTRY)

Imports only hybrid.bench.metrics, scaling, fingerprint and falsifier_eval (staged under <in_dir>/code)
and NumPy. Every number in the result is computed here from the raw prediction arrays; nothing an
entrant reports is used as a score. Entrant files are external input: each file's sha256 must
match the executor receipt of the job that produced it, arrays load with allow_pickle=False,
and malformed outputs mark that entrant or file invalid instead of stopping the evaluation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hybrid.bench import falsifier_eval, fingerprint, metrics, scaling

SCHEMA = "bigbrane.bench_result/1"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_outputs(out_dir, receipt_outputs):
    """Files present must equal the receipt's output list, hash for hash."""
    present = {p.relative_to(out_dir).as_posix(): p for p in out_dir.rglob("*") if p.is_file()}
    if set(present) != set(receipt_outputs):
        raise ValueError("output files differ from receipt: %s" % sorted(set(present) ^ set(receipt_outputs)))
    for rel, path in present.items():
        if _sha256(path) != receipt_outputs[rel]["sha256"]:
            raise ValueError("sha256 mismatch for %s" % rel)


def _load_oracle(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _task_block(task, oracle, pred):
    extras = {k[len("extra_"):]: v for k, v in oracle.items() if k.startswith("extra_")}
    return metrics.task_metrics(task["oracle_kind"], oracle["oracle"], pred, oracle["group"], task["group_labels"],
                                extras=extras, meta=task["meta"])


def _load_length_oracles(in_dir, spec):
    out = {}
    for name, task in spec["tasks"].items():
        for factor in task.get("length_probes", {}):
            out[(name, factor)] = _load_oracle(in_dir / "oracle" / ("%s__x%s.npz" % (name, factor)))["oracle"]
    return out


def _length_probe_block(task, name, out_dir, n, length_oracles):
    block = {}
    for factor, info in task["length_probes"].items():
        oracle = length_oracles[(name, factor)]
        try:
            pred = np.load(out_dir / "pred" / name / ("n%d_x%s.npy" % (n, factor)), allow_pickle=False)
        except (OSError, ValueError) as error:
            block[factor] = {"valid": False, "problems": ["cannot load: %s" % error]}
            continue
        problems = metrics.validate(pred, len(oracle), task["vocab_size"])
        if problems:
            block[factor] = {"valid": False, "problems": problems}
            continue
        m = metrics.task_metrics(task["oracle_kind"], oracle, pred.astype(np.float64), np.zeros(len(oracle), dtype=np.int64),
                                 ["all"])
        block[factor] = {"valid": True, "length": info["length"], "min_prefix": info["min_prefix"], **m}
    return block


def evaluate_entrant(spec, oracles, length_oracles, entrant_dir, entrant, kept):
    """kept: filled with {task: float64 predictions at the largest train size} for the fingerprint."""
    receipt = json.loads((entrant_dir / "receipt.json").read_text())
    entry = {"role": entrant["role"], "job_status": receipt["status"], "job_detail": receipt.get("detail", ""),
             "job_id": receipt["job_id"], "claimed_exponent": entrant.get("claimed_exponent")}
    if receipt["status"] != "ok":
        entry["status"] = "job_failed"
        return entry
    out_dir = entrant_dir / "out"
    verify_outputs(out_dir, receipt["outputs"])   # a mismatch is a host-side integrity failure: stop
    entry["status"] = "ok"
    try:
        entry["describe"] = json.loads((out_dir / "describe.json").read_text())
    except (OSError, ValueError) as error:
        entry["describe"], entry["status"] = None, "invalid_describe: %s" % error
    entry["tasks"] = {}
    for name, task in spec["tasks"].items():
        sizes, block = task["train_sizes"], {"sizes": {}}
        for n in sizes:
            path = out_dir / "pred" / name / ("n%d.npy" % n)
            try:
                pred = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as error:
                block["sizes"][str(n)] = {"valid": False, "problems": ["cannot load: %s" % error]}
                continue
            problems = metrics.validate(pred, len(oracles[name]["group"]), task["vocab_size"])
            if problems:
                block["sizes"][str(n)] = {"valid": False, "problems": problems}
                continue
            block["sizes"][str(n)] = {"valid": True, "train_tokens": task["train_tokens"][str(n)],
                                      **_task_block(task, oracles[name], pred.astype(np.float64))}
            if n == sizes[-1]:
                kept[name] = pred.astype(np.float64)
        if task.get("length_probes"):
            block["length_probes"] = _length_probe_block(task, name, out_dir, sizes[-1], length_oracles)
        target = task.get("target")
        if target:
            values = []
            for n in sizes:
                m = block["sizes"][str(n)]
                values.append(metrics.metric_value(m, target["metric"], target.get("group")) if m["valid"] else None)
            block["tokens_to_target"] = {"target": target, **metrics.tokens_to_target(
                [task["train_tokens"][str(n)] for n in sizes], values, target["op"], target["value"])}
        entry["tasks"][name] = block
    try:
        raw = json.loads((out_dir / "scaling.json").read_text())
        entry["scaling"] = scaling.summarize(raw, entrant.get("claimed_exponent"), spec["scaling"]["tolerance"],
                                             spec["scaling"]["tail_points"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        entry["scaling"] = {"error": "%s: %s" % (type(error).__name__, error)}
    try:
        entry["runner_reported"] = json.loads((out_dir / "runner.json").read_text())
    except (OSError, ValueError):
        entry["runner_reported"] = None
    return entry


def compare(candidate, baseline, spec):
    """Candidate minus baseline at each train size (negative kl delta = candidate better)."""
    out = {}
    for name, task in spec["tasks"].items():
        rows = {}
        for n in task["train_sizes"]:
            c = candidate["tasks"][name]["sizes"][str(n)]
            b = baseline["tasks"][name]["sizes"][str(n)]
            if c["valid"] and b["valid"]:
                rows[str(n)] = {"kl_delta": c["kl"] - b["kl"], "acc_delta": c["acc"] - b["acc"]}
        block = {"sizes": rows}
        ct, bt = candidate["tasks"][name].get("tokens_to_target"), baseline["tasks"][name].get("tokens_to_target")
        if ct and bt and ct["reached"] and bt["reached"]:
            block["tokens_to_target_ratio"] = ct["interp_tokens"] / bt["interp_tokens"]
        out[name] = block
    return out


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    spec = json.loads((in_dir / "eval_spec.json").read_text())
    oracles = {name: _load_oracle(in_dir / "oracle" / ("%s.npz" % name)) for name in spec["tasks"]}
    for name, task in spec["tasks"].items():
        if len(oracles[name]["group"]) != task["n_prefixes"]:
            raise ValueError("oracle for %s does not match the spec" % name)
    length_oracles = _load_length_oracles(in_dir, spec)
    kept = {name: {} for name in spec["entrants"]}
    entrants = {name: evaluate_entrant(spec, oracles, length_oracles, in_dir / "entrants" / name, entrant, kept[name])
                for name, entrant in spec["entrants"].items()}
    fingerprint_block = fingerprint.attach(spec, entrants, kept, oracles)
    comparison = {}
    candidates = [n for n, e in spec["entrants"].items() if e["role"] == "candidate"]
    for cand in candidates:
        if entrants[cand]["status"] != "ok":
            continue
        comparison[cand] = {b: compare(entrants[cand], entrants[b], spec) for b, e in spec["entrants"].items()
                            if e["role"] == "baseline" and entrants[b]["status"] == "ok"}
    result = {"schema": SCHEMA, "task_config_hash": spec["task_config_hash"], "config_hash": spec["config_hash"],
              "tasks": {name: {k: task[k] for k in ("vocab_size", "train_sizes", "train_tokens", "oracle_kind",
                                                     "group_name", "group_labels", "n_prefixes", "target",
                                                     "length_probes")}
                        for name, task in spec["tasks"].items()},
              "metric_conventions": {"floor": metrics.FLOOR, "sum_tol": metrics.SUM_TOL, "tie": metrics.TIE},
              "entrants": entrants, "comparison": comparison, "fingerprint": fingerprint_block}
    if spec.get("falsifiers"):
        result["falsifiers"] = falsifier_eval.evaluate(spec, in_dir, verify_outputs)
    (out_dir / "bench_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
