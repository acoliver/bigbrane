"""Candidate job entry point. Standalone: standard library and NumPy only, no `hybrid` imports.

    python3 runner.py <in_dir> <out_dir>

<in_dir> holds spec.json, candidate/ (with candidate.py defining class Candidate) and data/
with train pools and test prefixes (and, for hmm, vom and compose, longer prefix sets for the
length-generalization fingerprint). No oracle data is ever placed here. For each task and
train size the runner builds a fresh Candidate, fits it on the first n training sequences,
predicts the test prefixes and saves pred/<task>/n<size>.npy (at the largest size it also predicts
the longer prefix sets into n<size>_x<factor>.npy). It then runs the scaling probe
on one fitted model. Outputs: pred/, describe.json, scaling.json, runner.json.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROC_STATUS = "/proc/self/status"


def _status_kb(field):
    with open(PROC_STATUS) as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError("%s missing from %s" % (field, PROC_STATUS))


def _linux_reset_peak():
    # Writing 5 to clear_refs resets VmHWM to the current RSS (Linux >= 4.0).
    with open("/proc/self/clear_refs", "w") as handle:
        handle.write("5")


def _measure(predict, prefixes, repeats, warmup, linux):
    if linux:
        _linux_reset_peak()
        base = _status_kb("VmRSS") * 1024
    else:
        base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    for _ in range(warmup):
        predict(prefixes)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = predict(prefixes)
        times.append(time.perf_counter() - start)
        if np.shape(out)[0] != len(prefixes):
            raise ValueError("predict_proba returned %s rows for %d prefixes" % (np.shape(out)[0], len(prefixes)))
    peak = _status_kb("VmHWM") * 1024 if linux else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"times_s": times, "peak_rss_delta_bytes": int(peak - base)}


def _measure_in_child(predict, prefixes, repeats, warmup):
    """Fork so each length gets its own peak-RSS window and a crash or OOM kill costs only that length."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        code = 0
        try:
            payload = _measure(predict, prefixes, repeats, warmup, True)
        except BaseException as error:  # reported to the parent, which records it
            payload, code = {"error": "%s: %s" % (type(error).__name__, error)}, 1
        with os.fdopen(write_fd, "w") as pipe:
            json.dump(payload, pipe)
        os._exit(code)
    os.close(write_fd)
    with os.fdopen(read_fd) as pipe:
        text = pipe.read()
    _, status = os.waitpid(pid, 0)
    if not text:
        return {"error": "probe child died with wait status %d" % status}
    return json.loads(text)


def scaling_probe(model, vocab, lengths, batch, repeats, warmup, seed, stop_after_s):
    linux = platform.system() == "Linux" and os.path.exists("/proc/self/clear_refs")
    result = {"batch": batch, "repeats": repeats, "warmup": warmup, "lengths": [], "stopped": None,
              "rss_method": "linux_fork_vmhwm" if linux else "in_process_ru_maxrss_monotone"}
    for n in lengths:
        r = np.random.default_rng([seed, n])
        prefixes = r.integers(0, vocab, size=(batch, n)).tolist()
        if linux:
            row = _measure_in_child(model.predict_proba, prefixes, repeats, warmup)
        else:
            row = _measure(model.predict_proba, prefixes, repeats, warmup, False)
        row["n"] = n
        result["lengths"].append(row)
        if "error" in row:
            result["stopped"] = "error at n=%d" % n
            break
        if statistics.median(row["times_s"]) > stop_after_s:
            result["stopped"] = "median %.3gs > stop_after_s at n=%d" % (statistics.median(row["times_s"]), n)
            break
    return result


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    spec = _load_json(in_dir / "spec.json")
    sys.path.insert(0, str(in_dir / "candidate"))
    Candidate = importlib.import_module("candidate").Candidate
    scaling_cfg = spec["scaling"]
    describe, log, probe_model = {}, {"tasks": {}}, None
    began = time.perf_counter()
    for task in spec["tasks"]:
        name, vocab = task["name"], task["vocab_size"]
        pool = _load_json(in_dir / task["train_file"])["sequences"]
        prefixes = _load_json(in_dir / task["prefix_file"])["prefixes"]
        (out_dir / "pred" / name).mkdir(parents=True, exist_ok=True)
        describe[name], log["tasks"][name] = {}, {}
        for n in task["train_sizes"]:
            model = Candidate(vocab, spec["seed"], dict(spec["budget"]))
            t0 = time.perf_counter()
            model.fit([list(seq) for seq in pool[:n]])
            t1 = time.perf_counter()
            probs = np.asarray(model.predict_proba([list(p) for p in prefixes]), dtype=np.float64)
            t2 = time.perf_counter()
            if probs.shape != (len(prefixes), vocab):
                raise ValueError("%s n=%d: predict_proba shape %s, expected %s"
                                 % (name, n, probs.shape, (len(prefixes), vocab)))
            np.save(out_dir / "pred" / name / ("n%d.npy" % n), probs, allow_pickle=False)
            describe[name][str(n)] = model.describe()
            log["tasks"][name][str(n)] = {"fit_s": t1 - t0, "predict_s": t2 - t1}
            if n == max(task["train_sizes"]):
                for probe in task.get("length_probes", []):
                    longer = _load_json(in_dir / probe["prefix_file"])["prefixes"]
                    t3 = time.perf_counter()
                    long_probs = np.asarray(model.predict_proba([list(p) for p in longer]), dtype=np.float64)
                    if long_probs.shape != (len(longer), vocab):
                        raise ValueError("%s n=%d x%d: predict_proba shape %s, expected %s"
                                         % (name, n, probe["factor"], long_probs.shape, (len(longer), vocab)))
                    np.save(out_dir / "pred" / name / ("n%d_x%d.npy" % (n, probe["factor"])), long_probs,
                            allow_pickle=False)
                    log["tasks"][name][str(n)]["predict_x%d_s" % probe["factor"]] = time.perf_counter() - t3
            if name == scaling_cfg["task"] and n == scaling_cfg["train_size"]:
                probe_model = model
    log["fit_predict_total_s"] = time.perf_counter() - began
    (out_dir / "describe.json").write_text(json.dumps(describe, indent=2, sort_keys=True))
    if probe_model is None:
        raise ValueError("scaling task/train_size not among the evaluated tasks")
    vocab = next(t["vocab_size"] for t in spec["tasks"] if t["name"] == scaling_cfg["task"])
    t0 = time.perf_counter()
    scaling = scaling_probe(probe_model, vocab, scaling_cfg["lengths"], scaling_cfg["batch"],
                            scaling_cfg["repeats"], scaling_cfg["warmup"], scaling_cfg["seed"],
                            scaling_cfg["stop_after_s"])
    log["scaling_s"] = time.perf_counter() - t0
    (out_dir / "scaling.json").write_text(json.dumps(scaling, indent=2, sort_keys=True))
    log["python"] = sys.version
    log["platform"] = platform.platform()
    (out_dir / "runner.json").write_text(json.dumps(log, indent=2, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: runner.py <in_dir> <out_dir>")
    main(sys.argv[1], sys.argv[2])
