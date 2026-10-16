"""Rung 1 baselines (IMPROVEMENT-PLAN.md C.4): chunked forms equal their recurrences, the parallel model equals
its token-by-token step() form, the host state formula equals the state step() carries, causality, and a
finite training step for every family."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from hybrid.bench import api
from hybrid.bench.baselines import HERE as BASELINE_DIR

sys.path.insert(0, str(BASELINE_DIR))
import bb_recall as R  # noqa: E402

CONFIGS = {
    "transformer": ("transformer", 32, {}),
    "transformer_hard_alibi": ("transformer", 64, {"heads": 4, "pos": "hard_alibi", "hard_alibi_heads": 2}),
    "transformer_nope": ("transformer", 32, {"pos": "nope"}),
    "sliding_window": ("sliding_window", 32, {"window": 7}),
    "linear_attention": ("linear_attention", 32, {}),
    "deltanet": ("deltanet", 32, {"heads": 2}),
    "gated_deltanet": ("gated_deltanet", 32, {}),
    "mamba2": ("mamba2", 32, {"d_state": 16}),
}


def _inputs(t, b=2, h=2, dk=8, dv=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, h, t, dk, generator=g, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, t, dv, generator=g, dtype=torch.float64)
    decay = -torch.rand(b, h, t, generator=g, dtype=torch.float64)
    beta = torch.rand(b, h, t, generator=g, dtype=torch.float64)
    return q, k, v, decay, beta


@pytest.mark.parametrize("t", [1, 63, 64, 100, 128, 130])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_gla_equals_its_recurrence(t, chunk):
    q, k, v, g, _ = _inputs(t)
    ref, s_ref = R.recurrent_gla(q, k, v, g)
    out, s = R.chunk_gla(q, k, v, g, chunk=chunk)
    assert (out - ref).abs().max() < 1e-10 and (s - s_ref).abs().max() < 1e-10


@pytest.mark.parametrize("t", [1, 63, 64, 100, 128, 130])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_delta_rule_equals_its_recurrence(t, chunk):
    q, k, v, g, beta = _inputs(t)
    ref, s_ref = R.recurrent_delta(q, k, v, g, beta)
    out, s = R.chunk_delta(q, k, v, g, beta, chunk=chunk)
    assert (out - ref).abs().max() < 1e-10 and (s - s_ref).abs().max() < 1e-10


def test_delta_recurrence_matches_the_paper_equation():
    """S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T (Gated DeltaNet Eq. 10, written for S = d_k x d_v)."""
    q, k, v, g, beta = _inputs(20, b=1, h=1)
    s = torch.zeros(8, 5, dtype=torch.float64)
    outs = []
    for t in range(20):
        kt = k[0, 0, t][:, None]
        s = torch.exp(g[0, 0, t]) * (torch.eye(8, dtype=torch.float64) - beta[0, 0, t] * kt @ kt.T) @ s \
            + beta[0, 0, t] * kt @ v[0, 0, t][None]
        outs.append(q[0, 0, t] @ s)
    ref, _ = R.recurrent_delta(q, k, v, g, beta)
    assert (torch.stack(outs) - ref[0, 0]).abs().max() < 1e-12


def test_gate_one_gives_deltanet_and_beta_zero_leaves_the_state_unchanged():
    q, k, v, g, beta = _inputs(70)
    zero = torch.zeros_like(g)
    gated, _ = R.chunk_delta(q, k, v, zero, beta)
    plain, _ = R.recurrent_delta(q, k, v, zero, beta)
    assert (gated - plain).abs().max() < 1e-10
    out, s = R.chunk_delta(q, k, v, g, torch.zeros_like(beta))
    assert out.abs().max() == 0 and s.abs().max() == 0


def test_sliding_window_at_least_the_sequence_length_equals_full_attention():
    torch.manual_seed(0)
    full = R.LadderNet(40, "transformer", 32).double()
    window = R.LadderNet(40, "sliding_window", 32, hyper={"window": 50}).double()
    window.load_state_dict(full.state_dict())
    x = torch.randint(0, 40, (3, 50))
    assert (full(x) - window(x)).abs().max() < 1e-12
    short = R.LadderNet(40, "sliding_window", 32, hyper={"window": 8}).double()
    short.load_state_dict(full.state_dict())
    assert (full(x)[:, :9] - short(x)[:, :9]).abs().max() < 1e-12      # identical while t <= w
    assert (full(x)[:, 20:] - short(x)[:, 20:]).abs().max() > 1e-6     # differs once keys leave the window


@pytest.mark.parametrize("name", list(CONFIGS))
def test_parallel_form_equals_step_form_and_state_formula_is_exact(name):
    family, d, hyper = CONFIGS[name]
    torch.manual_seed(1)
    net = R.LadderNet(30, family, d, hyper=hyper).double()
    for t in (1, 9, 70, 130):
        x = torch.randint(0, 30, (2, t))
        logits, states = net.stream(x)
        assert (net(x) - logits).abs().max() < 1e-9, t
        carried = R.count_state_tensors(states)
        assert carried == net.state_floats(t) == R.state_formula(family, d, t, hyper=hyper), (t, carried)


def test_state_is_flat_for_fixed_state_families_and_grows_for_attention():
    for name, (family, d, hyper) in CONFIGS.items():
        small, large = (R.state_formula(family, d, n, hyper=hyper) for n in (256, 4096))
        if R.STATE_CLASS[family] == "growing":
            assert large > 10 * small, name
        else:
            assert large == small, name


def test_state_formula_matches_the_plan_table():
    """IMPROVEMENT-PLAN.md B.3 (2 layers, convolution buffers left out)."""
    n, d = 512, 64
    assert R.state_formula("transformer", d, n, include_conv=False) == 2 * 2 * d * n
    assert R.state_formula("sliding_window", d, n, hyper={"window": 32}, include_conv=False) == 2 * 2 * d * 32
    assert R.state_formula("linear_attention", d, n, include_conv=False) == 2 * (d * d + d)
    assert R.state_formula("gated_deltanet", d, n, hyper={"heads": 2}, include_conv=False) == 2 * 2 * 32 * 32
    assert R.state_formula("mamba2", d, n, hyper={"d_state": 16}, include_conv=False) == 2 * (2 * d) * 16


@pytest.mark.parametrize("name", list(CONFIGS))
def test_future_tokens_do_not_change_past_logits(name):
    family, d, hyper = CONFIGS[name]
    torch.manual_seed(2)
    net = R.LadderNet(30, family, d, hyper=hyper).double().eval()
    x = torch.randint(0, 30, (1, 150))
    base = net(x)
    for t in (0, 40, 63, 64, 100):
        changed = x.clone()
        changed[0, t + 1:] = torch.randint(0, 30, (149 - t,))
        assert (net(changed)[0, :t + 1] - base[0, :t + 1]).abs().max() < 1e-10, t


@pytest.mark.parametrize("name", list(CONFIGS))
def test_bench_api_batch_independence(name):
    family, d, hyper = CONFIGS[name]
    cls = R.candidate_class(family, d, hyper)
    model = cls(24, 0, {"max_steps": 3, "batch_size": 4, "lr": 1e-3})
    rng = np.random.default_rng(0)
    model.fit([rng.integers(0, 24, int(rng.integers(5, 40))).tolist() for _ in range(16)])
    prefixes = [rng.integers(0, 24, int(rng.integers(1, 90))).tolist() for _ in range(9)]
    report = api.check_causality(model, prefixes)
    assert report["ok"], report
    assert not api.describe_problems(model.describe())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_one_training_step_gives_finite_gradients(name):
    family, d, hyper = CONFIGS[name]
    torch.manual_seed(3)
    net = R.LadderNet(50, family, d, hyper=hyper)
    x = torch.randint(0, 50, (4, 80))
    loss = torch.nn.functional.cross_entropy(net(x[:, :-1]).reshape(-1, 50), x[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_parameter_counts_are_comparable_across_families():
    """Mixer and MLP parameters (embedding and head excluded) at each width stay within a factor 1.6 of the
    transformer's; the measured counts are listed in BASELINES.md."""
    for d in (32, 64, 128):
        counts = {}
        for family, hyper in [("transformer", {}), ("sliding_window", {"window": 32}), ("linear_attention", {}),
                              ("deltanet", {}), ("gated_deltanet", {}), ("mamba2", {"d_state": 32})]:
            net = R.LadderNet(10, family, d, hyper=hyper)
            counts[family] = net.params() - 2 * 10 * d
        base = counts["transformer"]
        assert all(base / 1.6 <= c <= base * 1.6 for c in counts.values()), (d, counts)


# ------------------------------------------------------------------------------- DeltaNet / GDN recipe (FLA)
# Recipe change 01+02 (results/phase10-20261013/recipe-probe, pilot in gdn-pilot/): bias-free q/k/v convolution and
# a/b projections, A ~ U[0, 16], and std 0.02 Linear/Conv1d weights for the delta-rule families only.

DELTA_FAMILIES = ("deltanet", "gated_deltanet")


def _weights(net):
    return [(name, m) for name, m in net.named_modules() if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d))]


@pytest.mark.parametrize("family", DELTA_FAMILIES)
def test_delta_rule_models_have_no_convolution_or_projection_biases(family):
    net = R.LadderNet(50, family, 64, hyper={"heads": 2})
    for mixer in net.mixers:
        assert mixer.conv.conv.bias is None and mixer.b_proj.bias is None
        assert (mixer.a_proj.bias is None) if family == "gated_deltanet" else not hasattr(mixer, "a_proj")
    assert all(m.bias is None for _, m in _weights(net))


def test_gdn_samples_A_on_0_to_16():
    torch.manual_seed(0)
    A = torch.exp(R.DeltaNet(512, heads=512, gated=True).A_log.detach())
    assert float(A.min()) >= 0 and float(A.max()) <= 16
    assert int((A < 1).sum()) > 0                                         # the old [1, 16] floor is gone
    assert abs(float(A.mean()) - 8) < 1


@pytest.mark.parametrize("family", DELTA_FAMILIES)
def test_delta_rule_models_initialize_linear_and_conv_weights_at_std_0_02(family):
    torch.manual_seed(0)
    net = R.LadderNet(512, family, 64)
    for name, m in _weights(net):
        w = m.weight.detach()
        tol = 0.1 if w.numel() >= 1000 else 0.5                          # tiny b_proj/a_proj rows are noisy
        assert abs(float(w.std()) - 0.02) <= tol * 0.02, (name, float(w.std()))
        assert abs(float(w.mean())) < 5 * 0.02 / w.numel() ** 0.5, name     # zero-mean within 5 standard errors
    assert abs(float(net.emb.weight.detach().std()) - 0.02) <= 0.002


OTHER_FAMILIES = {name: CONFIGS[name] for name in ("transformer", "transformer_hard_alibi", "transformer_nope",
                                                   "sliding_window", "linear_attention", "mamba2")}


@pytest.mark.parametrize("name", list(OTHER_FAMILIES))
def test_other_families_keep_pytorch_default_initialization_and_their_biases(name):
    """The recipe change is scoped to the delta-rule families: other families keep PyTorch's default Linear
    scale (std 1/sqrt(3 fan_in)) and their convolution biases. A bitwise comparison with the pre-change source is
    in results/phase10-diag2-20261013T1334Z/recipe-check/."""
    family, d, hyper = OTHER_FAMILIES[name]
    torch.manual_seed(0)
    net = R.LadderNet(512, family, 64 if d == 32 else d, hyper=hyper)
    checked = 0
    for mod_name, m in _weights(net):
        w = m.weight.detach()
        if isinstance(m, torch.nn.Linear) and w.numel() >= 1000:
            expected = 1 / (3 * m.in_features) ** 0.5
            assert abs(float(w.std()) - expected) <= 0.1 * expected, (mod_name, float(w.std()), expected)
            checked += 1
    assert checked >= 4
    for mixer in net.mixers:
        if hasattr(mixer, "conv"):
            assert mixer.conv.conv.bias is not None
        if family == "mamba2":
            assert float(torch.exp(mixer.A_log.detach()).min()) >= 1


def test_parameter_counts_match_the_recipe():
    """Other families keep their pre-change counts. Delta-rule models lose layers * (3d + 2 heads) gated, or
    layers * (3d + heads) ungated, bias parameters."""
    expected = {("transformer", 64, ()): 166208, ("transformer", 32, ()): 58016,
                ("sliding_window", 64, (("window", 32),)): 166208, ("linear_attention", 32, ()): 58976,
                ("linear_attention", 64, ()): 168128, ("mamba2", 32, (("d_state", 16),)): 65388,
                ("mamba2", 64, (("d_state", 32),)): 193496,
                ("gated_deltanet", 32, ()): 61224 - 2 * (3 * 32 + 2),
                ("gated_deltanet", 64, ()): 176712 - 2 * (3 * 64 + 2),
                ("deltanet", 64, ()): 176578 - 2 * (3 * 64 + 1)}
    for (family, d, hyper), count in expected.items():
        assert R.LadderNet(512, family, d, hyper=dict(hyper)).params() == count, (family, d)


def test_flops_order_attention_above_fixed_state_at_long_context():
    for family, hyper in [("linear_attention", {}), ("gated_deltanet", {}), ("mamba2", {"d_state": 64}),
                          ("sliding_window", {"window": 64})]:
        assert R.forward_flops_per_token("transformer", 64, 4096, 512) > \
            R.forward_flops_per_token(family, 64, 4096, 512, hyper=hyper)
    assert R.forward_flops_per_token("gated_deltanet", 64, 4096, 512) == \
        R.forward_flops_per_token("gated_deltanet", 64, 256, 512)
