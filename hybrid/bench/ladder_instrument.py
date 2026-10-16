"""Training instrumentation for ladder jobs, opt-in per slice (`instrument = { enabled = true, ... }` in
configs/ladder.toml). Standalone: NumPy, torch and the standard library, staged into the job's code/ directory
next to bb_recall, so it runs in the executor container and on the host GPU alike.

Logging only. The recorder never draws from the training or validation streams (its probe batch and per-K sets
come from their own SeedSequence parts), never touches the torch RNG, never writes parameters, gradients or
optimizer state, and restores the model's train/eval mode after every probe. ladder_runner calls it only when
spec["instrument"]["enabled"] is true; with instrumentation off the runner trains exactly as before.

Output: <out>/instrument.jsonl, one JSON object per line, `kind` one of
- header: seeds and stream entropy, digests of the initial weights, the train stream state, the validation set,
  the probe batch and the per-K sets, initial parameter norms, optimizer groups and the software environment.
- batch: SHA-256 of tokens and targets of each of the first `hash_batches` training batches.
- probe (steps 0, `early_steps`, every `probe_every` and the last): per-step losses and pre-clip global gradient
  norms since the previous probe, clipping fraction, per-parameter pre-clip gradient norms and update-to-weight
  ratios at this step, learning rate, per-K accuracy with the context-permutation null, and for (Gated) DeltaNet
  mixers per-layer quantiles of the decay exp(g), beta and the output gate, value norms, and recurrent-state,
  delta-residual (v - S^T k) and delta-update (beta times that residual) norms on a fixed probe batch.
- checkpoint (step 0, every `checkpoint_every` and the last): digests of weights, optimizer state, the train
  stream state, the torch CPU RNG and the running digest of every training batch so far; with `save_every` the
  weights and optimizer state are also saved to ckpt/step-<n>.pt.
- eval: the runner's validation accuracy at each of its evaluations, with per-K accuracy and null excess.
- end: steps run, selected step and its weight digest, instrumentation overhead in seconds.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import bb_recall

IGNORE = -100
SCHEMA = 1
PROBE_PART, PERK_PART = 11, 12
QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
STATE_PROFILE_EVERY = 32


def stream_entropy(seed, *parts):
    return [int(seed), 7, *(int(p) for p in parts)]


def stream(seed, *parts):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(tuple(stream_entropy(seed, *parts)))))


def tensors_digest(tensors):
    """SHA-256 over name, dtype, shape and bytes of every tensor of a name -> tensor mapping, in name order."""
    digest = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous()
        digest.update(("%s\0%s\0%s\0" % (name, t.dtype, tuple(t.shape))).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def optimizer_digest(opt):
    state = opt.state_dict()
    flat = {"groups": torch.tensor(np.frombuffer(json.dumps(state["param_groups"], sort_keys=True).encode(),
                                                 dtype=np.uint8).copy())}
    for index, entry in state["state"].items():
        for name, value in entry.items():
            flat["%s.%s" % (index, name)] = value if torch.is_tensor(value) else torch.tensor(value)
    return tensors_digest(flat)


def arrays_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(("%s\0%s\0" % (array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def rng_digest(rng):
    return hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest()


def quantiles(values):
    """Quantiles of a tensor or array, computed in float64 on the host."""
    if torch.is_tensor(values):
        values = values.detach().cpu().double().numpy()
    x = np.asarray(values, np.float64).ravel()
    return {"q%g" % q: float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))} | {"mean": float(x.mean())}


def context_null(pred, target):
    """Expected accuracy of fixed predictions when each row's targets are permuted across its query positions:
    the fraction of predictions equal to some target of the same row, divided by the row's K. pred, target
    (rows, K); MQAR values are distinct within a row."""
    return float(np.mean(pred[:, :, None] == target[:, None, :]))


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(v) for v in value)
    return value


def delta_recurrence(k, v, g, beta):
    """The gated delta rule step by step, as bb_recall.recurrent_delta, keeping per-step norms. k, v (B, T, H, dh);
    g, beta (B, T, H). Returns the Frobenius norm of the state S_t, the residual v_t - S^T k_t (before the update,
    after decay) and the update beta_t times that residual, each (B, T, H)."""
    b, t, h, dh = k.shape
    s = k.new_zeros(b, h, dh, v.shape[-1])
    state, residual, update = [], [], []
    for i in range(t):
        s = torch.exp(g[:, i])[..., None, None] * s
        r = v[:, i] - torch.einsum("bhd,bhde->bhe", k[:, i], s)
        u = beta[:, i, :, None] * r
        s = s + k[:, i, :, :, None] * u[:, :, None, :]
        state.append(s.flatten(2).norm(dim=-1))
        residual.append(r.norm(dim=-1))
        update.append(u.norm(dim=-1))
    return torch.stack(state, 1), torch.stack(residual, 1), torch.stack(update, 1)


class Recorder:
    """Called by ladder_runner.train at fixed points of the loop; see the module docstring for what it writes."""

    def __init__(self, spec, net, opt, device, task_mod, train_params, rng, val, out_dir, clip_norm):
        began = time.perf_counter()
        cfg, tr = spec["instrument"], spec["train"]
        self.net, self.opt, self.device, self.rng, self.clip_norm = net, opt, device, rng, float(clip_norm)
        self.steps = int(tr["steps"])
        self.probe_every, self.early = int(cfg["probe_every"]), {int(s) for s in cfg["early_steps"]}
        self.checkpoint_every, self.save_every = int(cfg["checkpoint_every"]), int(cfg["save_every"])
        self.hash_batches = int(cfg["hash_batches"])
        self.out_dir = Path(out_dir)
        self.file = open(self.out_dir / "instrument.jsonl", "w")
        self.batches = hashlib.sha256()
        self.norms, self.losses, self.clipped_total = [], [], 0
        self.grads, self.before = None, None
        self.perk_cache = (None, None)
        seed = int(tr["data_seed"])
        self.probe = task_mod.generate(stream(seed, PROBE_PART), int(cfg["probe_rows"]), **train_params)
        self.perk = {}
        perk_entropy = {}
        if spec["task"]["module"] == "mqar":
            for index, k in enumerate(train_params["k_values"]):
                kwargs = dict(train_params, k_values=[k])
                self.perk["K%d" % k] = task_mod.generate(stream(seed, PERK_PART, index), int(cfg["perk_rows"]),
                                                         **kwargs)
                perk_entropy["K%d" % k] = stream_entropy(seed, PERK_PART, index)
        self.names = [name for name, _ in net.named_parameters()]
        init = {k: v for k, v in net.state_dict().items()}
        if cfg.get("save_init", True):
            np.savez(self.out_dir / "init_weights.npz", **{k: v.detach().cpu().numpy() for k, v in init.items()})
        self._write({"kind": "header", "schema": SCHEMA, "step": 0,
                     "seeds": {"model_seed": int(tr["model_seed"]), "data_seed": seed,
                               "torch_initial_seed": int(torch.initial_seed()),
                               "streams": {"train": stream_entropy(seed, 1), "val": stream_entropy(seed, 2),
                                           "probe": stream_entropy(seed, PROBE_PART), "perk": perk_entropy}},
                     "torch_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                     "train_stream_sha256": rng_digest(rng),
                     "val_sha256": arrays_digest(val["tokens"], val["targets"]),
                     "probe_sha256": arrays_digest(self.probe["tokens"], self.probe["targets"]),
                     "perk_sha256": {n: arrays_digest(d["tokens"], d["targets"]) for n, d in self.perk.items()},
                     "init_weights_sha256": tensors_digest(init),
                     "init_param_norms": {n: float(p.detach().cpu().double().norm())
                                          for n, p in net.named_parameters()},
                     "optimizer_groups": [{k: v for k, v in g.items() if k != "params"} | {"n_params": len(g["params"])}
                                          for g in opt.param_groups],
                     "clip_norm": self.clip_norm, "instrument": cfg,
                     "model": {"family": spec["family"], "d": spec["d"], "layers": spec.get("layers", 2),
                               "hyper": spec.get("hyper", {}), "params": net.params()},
                     "train": tr, "task": spec["task"],
                     "env": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
                             "platform": platform.platform(), "device": device.type,
                             "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                             "mps_available": torch.backends.mps.is_available()}})
        self.overhead_s = time.perf_counter() - began
        self._probe(0, window=False)
        self._checkpoint(0)

    # ------------------------------------------------------------------------------------------- hooks

    def on_batch(self, step, data):
        began = time.perf_counter()
        self.batches.update(arrays_digest(data["tokens"], data["targets"]).encode())
        if step <= self.hash_batches:
            self._write({"kind": "batch", "step": step, "tokens_sha256": arrays_digest(data["tokens"]),
                         "targets_sha256": arrays_digest(data["targets"])})
        self.overhead_s += time.perf_counter() - began

    def before_clip(self, step):
        """After backward, before clipping: on probe steps keep pre-clip gradient norms and the weights."""
        if not self._probing(step):
            return
        began = time.perf_counter()
        params = list(self.net.parameters())
        self.grads = torch.stack([p.grad.detach().norm() for p in params])
        self.before = [p.detach().clone() for p in params]
        self.overhead_s += time.perf_counter() - began

    def after_step(self, step, loss, grad_norm):
        """After the optimizer step. loss is the detached step loss, grad_norm clip_grad_norm_'s pre-clip total."""
        began = time.perf_counter()
        self.losses.append(loss)
        self.norms.append(grad_norm.detach())
        self.overhead_s += time.perf_counter() - began
        if self._probing(step):
            self._probe(step)
        if step % self.checkpoint_every == 0 or step == self.steps:
            self._checkpoint(step)

    def on_eval(self, step, val_acc, best_step):
        began = time.perf_counter()
        record = {"kind": "eval", "step": step, "val_acc": val_acc, "best_step": best_step,
                  "perk": self._perk(step)}
        self._write(record)
        self.overhead_s += time.perf_counter() - began

    def finish(self, step, best):
        """After the loop, which may have stopped early: flush the last window and checkpoint, then the end record."""
        if self.norms:
            self._probe(step)
        if step % self.checkpoint_every and step != self.steps:
            self._checkpoint(step)
        self._write({"kind": "end", "steps_run": step, "best_step": best["step"], "best_val_acc": best["acc"],
                     "best_weights_sha256": tensors_digest(best["state"]), "clipped_steps": self.clipped_total,
                     "overhead_s": self.overhead_s})
        self.file.close()
        return {"file": "instrument.jsonl", "overhead_s": self.overhead_s, "clipped_steps": self.clipped_total}

    # ------------------------------------------------------------------------------------------- records

    def _probing(self, step):
        return step in self.early or step % self.probe_every == 0 or step == self.steps

    def _write(self, record):
        self.file.write(json.dumps(record, sort_keys=True) + "\n")
        self.file.flush()

    def _probe(self, step, window=True):
        began = time.perf_counter()
        record = {"kind": "probe", "step": step, "lr": self.opt.param_groups[0]["lr"]}
        if window and self.norms:
            norms = torch.stack(self.norms).cpu().double().numpy()
            clipped = int((norms > self.clip_norm).sum())
            self.clipped_total += clipped
            record["window"] = {"steps": len(norms), "losses": torch.stack(self.losses).cpu().tolist(),
                                "grad_norm": quantiles(norms), "grad_norm_last": float(norms[-1]),
                                "clipped": clipped, "clip_fraction": clipped / len(norms),
                                "clipped_total": self.clipped_total}
            self.norms, self.losses = [], []
        if self.grads is not None:
            grads = self.grads.cpu().double().numpy()
            after = [p.detach() for p in self.net.parameters()]
            ratios = torch.stack([(a - b).norm() / b.norm().clamp_min(1e-30) for a, b in zip(after, self.before)])
            record["grad_norms"] = dict(zip(self.names, grads.tolist()))
            record["grad_norm_global"] = float(np.sqrt(np.square(grads).sum()))
            record["update_ratios"] = dict(zip(self.names, ratios.cpu().double().tolist()))
            self.grads, self.before = None, None
        was_training = self.net.training
        self.net.eval()
        try:
            record["perk"] = self._perk(step)
            record.update(self._gates())
        finally:
            self.net.train(was_training)
        self._write(record)
        self.overhead_s += time.perf_counter() - began

    def _perk(self, step):
        if self.perk_cache[0] == step:
            return self.perk_cache[1]
        was_training = self.net.training
        self.net.eval()
        out = {}
        try:
            with torch.no_grad():
                for name, data in self.perk.items():
                    targets = data["targets"]
                    mask = targets != IGNORE
                    logits = self.net(torch.as_tensor(data["tokens"]).to(self.device))
                    mask_t = torch.as_tensor(mask).to(self.device)
                    loss = F.cross_entropy(logits[mask_t], torch.as_tensor(targets).to(self.device)[mask_t])
                    pred = logits.argmax(-1).cpu().numpy()[mask].reshape(len(targets), -1)
                    target = targets[mask].reshape(len(targets), -1)
                    acc = float((pred == target).mean())
                    null = context_null(pred, target)
                    out[name] = {"acc": acc, "null": null, "excess": acc - null, "loss": float(loss)}
        finally:
            self.net.train(was_training)
        self.perk_cache = (step, out)
        return out

    def _gates(self):
        """Probe-batch loss and accuracy, and gate/state statistics of every (Gated) DeltaNet mixer."""
        captured = []
        handles = [mix.register_forward_pre_hook(lambda module, args, i=i: captured.append((i, args[0].detach())))
                   for i, mix in enumerate(self.net.mixers)]
        try:
            with torch.no_grad():
                logits = self.net(torch.as_tensor(self.probe["tokens"]).to(self.device))
        finally:
            for handle in handles:
                handle.remove()
        targets = self.probe["targets"]
        mask = targets != IGNORE
        mask_t = torch.as_tensor(mask).to(self.device)
        with torch.no_grad():
            loss = F.cross_entropy(logits[mask_t], torch.as_tensor(targets).to(self.device)[mask_t])
        pred = logits.argmax(-1).cpu().numpy()[mask]
        out = {"probe": {"loss": float(loss), "acc": float((pred == targets[mask]).mean())}}
        layers = []
        with torch.no_grad():
            for i, xn in captured:
                mix = self.net.mixers[i]
                if not isinstance(mix, bb_recall.DeltaNet):
                    continue
                q, k, v, g, beta = mix._prepare(xn, mix.conv(mix.qkv(xn)))
                decay = torch.exp(g)
                gate = F.silu(mix.g_proj(xn))
                s_norm, r_norm, u_norm = delta_recurrence(k, v, g, beta)
                profile = s_norm.mean(dim=(0, 2))
                positions = list(range(STATE_PROFILE_EVERY - 1, profile.shape[0], STATE_PROFILE_EVERY))
                entry = {"layer": i, "decay": quantiles(decay), "beta": quantiles(beta), "out_gate": quantiles(gate),
                         "decay_above_0.999": float((decay > 0.999).float().mean()),
                         "decay_below_0.1": float((decay < 0.1).float().mean()),
                         "beta_below_0.05": float((beta < 0.05).float().mean()),
                         "beta_above_0.95": float((beta > 0.95).float().mean()),
                         "out_gate_abs_below_0.01": float((gate.abs() < 0.01).float().mean()),
                         "v_norm": quantiles(v.norm(dim=-1)), "state_norm": quantiles(s_norm),
                         "state_norm_profile": {str(p): float(profile[p]) for p in positions},
                         "residual_norm": quantiles(r_norm), "update_norm": quantiles(u_norm)}
                if mix.gated:
                    entry["A"] = torch.exp(mix.A_log).cpu().double().tolist()
                    entry["dt_bias"] = mix.dt_bias.cpu().double().tolist()
                layers.append(entry)
        if layers:
            out["gdn"] = layers
        return out

    def _checkpoint(self, step):
        began = time.perf_counter()
        state = self.net.state_dict()
        record = {"kind": "checkpoint", "step": step, "weights_sha256": tensors_digest(state),
                  "optimizer_sha256": optimizer_digest(self.opt), "train_stream_sha256": rng_digest(self.rng),
                  "torch_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                  "batches_sha256": self.batches.hexdigest()}
        if self.save_every and step % self.save_every == 0 and step:
            path = self.out_dir / "ckpt" / ("step-%06d.pt" % step)
            path.parent.mkdir(exist_ok=True)
            torch.save({"step": step, "model": to_cpu(state), "optimizer": to_cpu(self.opt.state_dict()),
                        "train_stream_state": self.rng.bit_generator.state}, path)
            record["saved"] = "ckpt/%s" % path.name
        self._write(record)
        self.overhead_s += time.perf_counter() - began
