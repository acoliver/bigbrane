"""Ladder job entry point (test ladder rung 1). Standalone: NumPy, torch and the standard library only.

    python3 ladder_runner.py <in_dir> <out_dir>

spec["train"]["device"] selects where training runs: "cpu" (the default, and the only device in the executor
container) or "mps" (the host's Apple GPU, used only by the ladder's host_mps mode for the trusted baselines). The
model is always built and initialized on CPU first, so a seed gives the same initial weights on every device. On
MPS the runner refuses to start when PYTORCH_ENABLE_MPS_FALLBACK is set: an op without an MPS kernel must fail
the job, not run silently on CPU.

<in_dir> holds spec.json, code/ (bb_recall.py, ladder_instrument.py and the task modules) and data/ (evaluation
sets: tokens and the positions to report, never the answers). With spec["instrument"]["enabled"] the job also runs
ladder_instrument's logging-only recorder and writes instrument.jsonl (see that module); without it the job trains
exactly as it did before the recorder existed. Two modes:

- "train": build one LadderNet, train it with answer-only loss on batches the task module generates from the
  train seed (a fresh batch every step), keep the parameters with the best accuracy on a validation stream drawn
  from the training distribution, then write the argmax prediction at every listed position of every evaluation
  set to pred/<set>.npy (int16, row-major order of the positions). Scores are computed on the host, which keeps
  the answers. Outputs: pred/, train.json (curve, timings, parameters, state floats).
- "microbench": time training steps and inference forwards for a list of configurations at the given thread
  counts. Output: microbench.json.

Evaluation reads whole sequences in one causal forward pass and takes the logits at the listed positions. That
equals predicting each prefix on its own for these baselines, whose causality the tests check (future tokens do not
change past logits). An untrusted candidate must be evaluated on prefixes instead.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

IGNORE = -100
EVAL_TOKENS_PER_CHUNK = 32768
CLIP_NORM = 1.0


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def stream(seed, part):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((int(seed), 7, int(part)))))


TRAIN_PART, VAL_PART = 1, 2


def _generate(task_mod, rng, n, params):
    return task_mod.generate(rng, n, **params)


def _peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if platform.system() == "Linux" else peak / (1024.0 * 1024.0)


def _lr_at(step, total, peak, warmup):
    if step <= warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _optimizer(torch, net, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        (decay if param.ndim >= 2 and not name.startswith("emb.") else no_decay).append(param)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.98))


def _accuracy(torch, net, tokens, targets, device="cpu"):
    net.eval()
    with torch.no_grad():
        logits = net(torch.as_tensor(tokens).to(device))
    mask = torch.as_tensor(targets != IGNORE)
    pred = logits.argmax(-1).cpu()
    return float((pred[mask] == torch.as_tensor(targets)[mask]).double().mean())


def predict_positions(torch, net, tokens, positions, device="cpu"):
    """Argmax at every True entry of positions (rows, T), in row-major order, evaluated in row chunks."""
    net.eval()
    rows = max(1, EVAL_TOKENS_PER_CHUNK // tokens.shape[1])
    out = []
    with torch.no_grad():
        for start in range(0, tokens.shape[0], rows):
            x = torch.as_tensor(tokens[start:start + rows].astype(np.int64)).to(device)
            pred = net(x).argmax(-1).cpu().numpy()
            out.append(pred[positions[start:start + rows]])
    return np.concatenate(out).astype(np.int16)


def _device(torch, name):
    """The torch device for spec["train"]["device"]; MPS only when it is present and CPU fallback is off."""
    if name == "cpu":
        return torch.device("cpu")
    if name != "mps":
        raise ValueError("unknown device %r (cpu or mps)" % (name,))
    if not torch.backends.mps.is_available():
        raise RuntimeError("device mps requested but torch.backends.mps.is_available() is False")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK is set; ladder MPS jobs must not fall back to CPU")
    return torch.device("mps")


def train(spec, in_dir, out_dir):
    import torch
    import torch.nn.functional as F
    import bb_recall

    tr = spec["train"]
    device = _device(torch, tr.get("device", "cpu"))
    torch.set_num_threads(int(tr.get("threads", 1)))
    torch.manual_seed(int(tr["model_seed"]))
    task_mod = importlib.import_module(spec["task"]["module"])
    params = spec["task"]["train_params"]
    vocab = int(spec["task"]["vocab"])
    net = bb_recall.LadderNet(vocab, spec["family"], int(spec["d"]), layers=int(spec.get("layers", 2)),
                              hyper=spec.get("hyper", {})).to(device)
    steps, batch, peak = int(tr["steps"]), int(tr["batch"]), float(tr["lr"])
    warmup = int(round(float(tr.get("warmup_frac", 0.05)) * steps))
    opt = _optimizer(torch, net, peak, float(tr.get("weight_decay", 0.1)))
    rng = stream(tr["data_seed"], TRAIN_PART)
    val = _generate(task_mod, stream(tr["data_seed"], VAL_PART), int(tr.get("val_rows", 256)), params)
    inst = None
    if spec.get("instrument", {}).get("enabled"):
        import ladder_instrument
        inst = ladder_instrument.Recorder(spec, net, opt, device, task_mod, params, rng, val, out_dir, CLIP_NORM)
    eval_every = int(tr.get("eval_every", max(1, steps // 20)))
    time_budget = float(tr.get("time_budget_s", 1e12))
    stop_at = float(tr.get("stop_val_acc", 1.01))
    best = {"acc": -1.0, "step": 0, "state": None}
    curve, losses, perfect_checks, stopped = [], [], 0, "steps"
    began = time.perf_counter()
    gen_s = 0.0
    step = 0
    for step in range(1, steps + 1):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, steps, peak, warmup)
        t0 = time.perf_counter()
        data = _generate(task_mod, rng, batch, params)
        gen_s += time.perf_counter() - t0
        if inst is not None:
            inst.on_batch(step, data)
        x = torch.as_tensor(data["tokens"]).to(device)
        y = torch.as_tensor(data["targets"]).to(device)
        net.train()
        logits = net(x)
        mask = y != IGNORE
        loss = F.cross_entropy(logits[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        if inst is not None:
            inst.before_clip(step)
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_NORM)
        opt.step()
        losses.append(loss.detach())
        if inst is not None:
            inst.after_step(step, losses[-1], grad_norm)
        if step % eval_every == 0 or step == steps:
            acc = _accuracy(torch, net, val["tokens"], val["targets"], device)
            elapsed = time.perf_counter() - began
            mean_loss = float(np.mean(torch.stack(losses).cpu().tolist()))
            curve.append({"step": step, "train_loss": mean_loss, "val_acc": acc, "elapsed_s": elapsed})
            print(json.dumps(curve[-1]), flush=True)
            losses = []
            if acc >= best["acc"]:
                best = {"acc": acc, "step": step, "state": {k: v.detach().clone() for k, v in net.state_dict().items()}}
            if inst is not None:
                inst.on_eval(step, acc, best["step"])
            perfect_checks = perfect_checks + 1 if acc >= stop_at else 0
            if perfect_checks >= 2:
                stopped = "val_acc >= %g at two checks" % stop_at
                break
            if elapsed > time_budget:
                stopped = "time budget %.0fs" % time_budget
                break
    train_s = time.perf_counter() - began
    instrument = inst.finish(step, best) if inst is not None else None
    net.load_state_dict(best["state"])
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(parents=True)
    t0 = time.perf_counter()
    for entry in spec["eval_sets"]:
        arrays = np.load(in_dir / entry["file"], allow_pickle=False)
        pred = predict_positions(torch, net, arrays["tokens"], arrays["positions"], device)
        np.save(pred_dir / ("%s.npy" % entry["name"]), pred, allow_pickle=False)
    eval_s = time.perf_counter() - t0
    report = {"steps_run": step, "best_step": best["step"], "best_val_acc": best["acc"], "stopped": stopped,
              "curve": curve, "train_s": train_s, "generate_s": gen_s, "eval_s": eval_s,
              "s_per_step": train_s / max(1, step), "params": net.params(),
              "state_floats_at": {str(n): net.state_floats(n) for n in spec.get("state_lengths", [])},
              "peak_rss_mb": _peak_rss_mb(), "threads": torch.get_num_threads(), "torch": torch.__version__,
              "python": sys.version, "platform": platform.platform(), "device": device.type,
              "dtype": str(next(net.parameters()).dtype).replace("torch.", "")}
    if instrument is not None:
        report["instrument"] = instrument
    (out_dir / "train.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def microbench(spec, out_dir):
    import torch
    import torch.nn.functional as F
    import bb_recall

    rows = []
    for threads in spec["threads"]:
        torch.set_num_threads(int(threads))
        for cfg in spec["configs"]:
            torch.manual_seed(0)
            vocab, n, b = int(cfg["vocab"]), int(cfg["seq_len"]), int(cfg["batch"])
            net = bb_recall.LadderNet(vocab, cfg["family"], int(cfg["d"]), hyper=cfg.get("hyper", {}))
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
            x = torch.randint(0, vocab, (b, n))
            y = torch.randint(0, vocab, (b, n))
            times = []
            for i in range(int(spec["warmup"]) + int(spec["steps"])):
                t0 = time.perf_counter()
                loss = F.cross_entropy(net(x).reshape(-1, vocab), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                if i >= int(spec["warmup"]):
                    times.append(time.perf_counter() - t0)
            infer = []
            net.eval()
            for length in spec.get("infer_lengths", []):
                xi = torch.randint(0, vocab, (1, int(length)))
                with torch.no_grad():
                    net(xi)
                    reps = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        net(xi)
                        reps.append(time.perf_counter() - t0)
                infer.append({"length": int(length), "median_s": float(np.median(reps))})
            rows.append({**cfg, "threads": int(threads), "train_step_s": times,
                         "median_step_s": float(np.median(times)), "tokens_per_s": b * n / float(np.median(times)),
                         "params": net.params(), "infer": infer})
            print(json.dumps({k: rows[-1][k] for k in ("family", "d", "seq_len", "batch", "threads",
                                                         "median_step_s")}), flush=True)
    report = {"rows": rows, "torch": torch.__version__, "platform": platform.platform(),
              "peak_rss_mb": _peak_rss_mb()}
    (out_dir / "microbench.json").write_text(json.dumps(report, indent=2, sort_keys=True))


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    spec = _load_json(in_dir / "spec.json")
    sys.path.insert(0, str(in_dir / "code"))
    if spec["mode"] == "train":
        train(spec, in_dir, out_dir)
    elif spec["mode"] == "microbench":
        microbench(spec, out_dir)
    else:
        raise ValueError("unknown mode %r" % (spec["mode"],))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: ladder_runner.py <in_dir> <out_dir>")
    main(sys.argv[1], sys.argv[2])
