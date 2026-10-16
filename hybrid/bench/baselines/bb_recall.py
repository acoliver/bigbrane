"""Recall baselines for the test ladder, rung 1 (IMPROVEMENT-PLAN.md C.2 and C.3), in pure PyTorch on CPU, or on
the host's Apple GPU (MPS) when the ladder's host_mps mode calibrates these trusted baselines.

Standalone (torch and the standard library) so it runs inside the executor job. Every family shares one
two-layer backbone: embedding, then per layer [RMSNorm -> sequence mixer -> residual, RMSNorm -> SwiGLU MLP ->
residual], final RMSNorm, linear head. Only the sequence mixer differs:

| family | mixer | training form | reference form (tests) |
| --- | --- | --- | --- |
| transformer | causal softmax attention, RoPE (or NoPE or Hard-ALiBi heads) | SDPA with a mask | step() with a KV cache |
| sliding_window | softmax attention over the previous w positions plus the current one | SDPA, banded mask | step() with a ring of w past positions |
| linear_attention | elu+1 feature map, normalized (Katharopoulos et al. 2020) | chunked, `chunk_gla` | `recurrent_gla` and step() |
| deltanet / gated_deltanet | delta rule S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T (2412.06464 Eq. 10, transposed) | chunked WY form, `chunk_delta` | `recurrent_delta` and step() |
| mamba2 | SSD: h_t = a_t h_{t-1} + dt_t x_t B_t^T, y_t = h_t C_t + D x_t, scalar a_t per head (2405.21060) | chunked, `chunk_gla` | `recurrent_gla` and step() |

The fixed-state mixers get a short causal depthwise convolution (kernel 4) on their inputs, the Based recipe the
Zoology comparisons give every architecture (IMPROVEMENT-PLAN.md B.4).

State accounting is exact: `state_floats(n)` is the number of floats the step() form carries after reading n
tokens, and the tests check it against the tensors step() actually holds. It includes the convolution buffers.
`mixer_state_floats(n)` leaves the convolution buffers out, which is the quantity in the plan's B.3 table.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV_K = 4
CHUNK = 64
FAMILIES = ("transformer", "sliding_window", "linear_attention", "deltanet", "gated_deltanet", "mamba2")
STATE_CLASS = {"transformer": "growing", "sliding_window": "window", "linear_attention": "fixed",
               "deltanet": "fixed", "gated_deltanet": "fixed", "mamba2": "fixed"}


# ----------------------------------------------------------------------------------------------- kernels

def _causal_decay(cum):
    """exp(cum_i - cum_j) for i >= j, else 0. cum: (..., C). Masked before the exp so it never overflows."""
    diff = cum[..., :, None] - cum[..., None, :]
    c = cum.shape[-1]
    lower = torch.ones(c, c, dtype=torch.bool, device=cum.device).tril()
    return torch.exp(diff.masked_fill(~lower, float("-inf")))


def _pad_time(x, multiple, dim):
    t = x.shape[dim]
    extra = (-t) % multiple
    if not extra:
        return x
    shape = list(x.shape)
    shape[dim] = extra
    return torch.cat([x, x.new_zeros(shape)], dim=dim)


def recurrent_gla(q, k, v, g):
    """Reference: S_t = exp(g_t) S_{t-1} + k_t v_t^T, o_t = q_t S_t. q, k (B,H,T,Dk), v (B,H,T,Dv), g (B,H,T)."""
    b, h, t, dk = q.shape
    s = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for i in range(t):
        s = torch.exp(g[:, :, i])[..., None, None] * s + k[:, :, i, :, None] * v[:, :, i, None, :]
        out.append(torch.einsum("bhd,bhde->bhe", q[:, :, i], s))
    return torch.stack(out, dim=2), s


def chunk_gla(q, k, v, g, chunk=CHUNK):
    """Chunked form of recurrent_gla: exact decayed attention inside a chunk, the running state across chunks."""
    t = q.shape[2]
    q, k, v = (_pad_time(x, chunk, 2) for x in (q, k, v))
    g = _pad_time(g, chunk, 2)
    b, h, tp, dk = q.shape
    n = tp // chunk
    q, k, v = (x.reshape(b, h, n, chunk, x.shape[-1]) for x in (q, k, v))
    cum = g.reshape(b, h, n, chunk).cumsum(-1)
    gamma = _causal_decay(cum)                                         # (b,h,n,C,C)
    s = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for c in range(n):
        qc, kc, vc, cc = q[:, :, c], k[:, :, c], v[:, :, c], cum[:, :, c]
        intra = (qc @ kc.transpose(-1, -2)) * gamma[:, :, c]
        out.append((qc * torch.exp(cc)[..., None]) @ s + intra @ vc)
        last = cc[..., -1:]
        s = torch.exp(last)[..., None] * s + (kc * torch.exp(last - cc)[..., None]).transpose(-1, -2) @ vc
    return torch.cat(out, dim=2)[:, :, :t], s


def recurrent_delta(q, k, v, g, beta):
    """Reference gated delta rule: S <- exp(g_t) S; u = b_t (v_t - S^T k_t); S <- S + k_t u^T; o_t = q_t S.

    Equivalent to S_t = a_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T with a_t = exp(g_t); g = 0 gives DeltaNet."""
    b, h, t, dk = q.shape
    s = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for i in range(t):
        s = torch.exp(g[:, :, i])[..., None, None] * s
        u = beta[:, :, i, None] * (v[:, :, i] - torch.einsum("bhd,bhde->bhe", k[:, :, i], s))
        s = s + k[:, :, i, :, None] * u[:, :, None, :]
        out.append(torch.einsum("bhd,bhde->bhe", q[:, :, i], s))
    return torch.stack(out, dim=2), s


def chunk_delta(q, k, v, g, beta, chunk=CHUNK):
    """Chunked WY form of recurrent_delta (Yang et al. 2406.06484, extended with the gate as in 2412.06464).

    Inside a chunk, with cumulative log-decay c and G_ij = exp(c_i - c_j) for i >= j:
      A = strict_lower((diag(b) K K^T) * G), U = (I + A)^-1 diag(b) V, W = (I + A)^-1 diag(b exp(c)) K,
      V_new = U - W S, O = (Q * exp(c)) S + ((Q K^T) * G) V_new, S <- exp(c_last) S + (K * exp(c_last - c))^T V_new.
    The triangular systems are solved with torch.linalg.solve_triangular."""
    t = q.shape[2]
    q, k, v = (_pad_time(x, chunk, 2) for x in (q, k, v))
    g, beta = _pad_time(g, chunk, 2), _pad_time(beta, chunk, 2)
    b, h, tp, dk = q.shape
    n = tp // chunk
    q, k, v = (x.reshape(b, h, n, chunk, x.shape[-1]) for x in (q, k, v))
    beta = beta.reshape(b, h, n, chunk)
    cum = g.reshape(b, h, n, chunk).cumsum(-1)
    gamma = _causal_decay(cum)
    kb = k * beta[..., None]
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    a = ((kb @ k.transpose(-1, -2)) * gamma).tril(-1)
    lhs = eye + a
    u = torch.linalg.solve_triangular(lhs, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(lhs, kb * torch.exp(cum)[..., None], upper=False, unitriangular=True)
    s = q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for c in range(n):
        qc, kc, cc = q[:, :, c], k[:, :, c], cum[:, :, c]
        v_new = u[:, :, c] - w[:, :, c] @ s
        intra = (qc @ kc.transpose(-1, -2)) * gamma[:, :, c]
        out.append((qc * torch.exp(cc)[..., None]) @ s + intra @ v_new)
        last = cc[..., -1:]
        s = torch.exp(last)[..., None] * s + (kc * torch.exp(last - cc)[..., None]).transpose(-1, -2) @ v_new
    return torch.cat(out, dim=2)[:, :, :t], s


# ----------------------------------------------------------------------------------------------- pieces

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = 8 * math.ceil(8 * d / 3 / 8)
        self.gate, self.up, self.down = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False), \
            nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ShortConv(nn.Module):
    """Depthwise causal convolution, kernel CONV_K, followed by SiLU. step() keeps the last CONV_K - 1 inputs."""

    def __init__(self, channels, bias=True):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(channels, channels, CONV_K, groups=channels, padding=0, bias=bias)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (CONV_K - 1, 0)))
        return F.silu(y.transpose(1, 2))

    def init_state(self, batch, like):
        return like.new_zeros(batch, CONV_K - 1, self.channels)

    def step(self, x, buf):
        window = torch.cat([buf, x[:, None]], dim=1)                     # (B, K, C)
        y = (window * self.conv.weight[:, 0].T[None]).sum(1)
        if self.conv.bias is not None:
            y = y + self.conv.bias
        return F.silu(y), window[:, 1:]

    def state_floats(self):
        return (CONV_K - 1) * self.channels


def rope_tables(positions, dim, base=10000.0, dtype=torch.float32, device=None):
    """cos, sin tables (T, dim/2) for CPU positions, computed in float64 on CPU and then moved to `device`, so
    every device gets the same tables (Apple's MPS backend has no float64)."""
    freq = base ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    ang = positions.to(torch.float64)[:, None] * freq[None]
    return torch.cos(ang).to(device=device, dtype=dtype), torch.sin(ang).to(device=device, dtype=dtype)


def apply_rope(x, cos, sin):
    """x (..., T, dim) with cos, sin (T, dim/2): rotate consecutive pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ----------------------------------------------------------------------------------------------- mixers

class Attention(nn.Module):
    """Causal softmax attention. pos: "rope", "nope" or "hard_alibi". window: each query sees itself and the
    previous `window` positions (None for full attention). hard_alibi_heads M: head h < M sees itself and the
    previous h positions (Jelassi et al. 2024, appendix A: head 1 attends to the current token, head 2 to the
    current and preceding one, and so on); the other heads are NoPE heads."""

    def __init__(self, d, heads, pos="rope", window=None, hard_alibi_heads=0):
        super().__init__()
        if d % heads:
            raise ValueError("d %d is not divisible by heads %d" % (d, heads))
        if pos not in ("rope", "nope", "hard_alibi"):
            raise ValueError("unknown pos %r" % (pos,))
        if (pos == "hard_alibi") != (hard_alibi_heads > 0):
            raise ValueError("hard_alibi_heads must be positive exactly when pos is hard_alibi")
        if hard_alibi_heads >= heads:
            raise ValueError("hard_alibi_heads must leave at least one NoPE head")
        self.d, self.heads, self.dh, self.pos, self.window = d, heads, d // heads, pos, window
        self.hard = hard_alibi_heads
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)

    def head_windows(self):
        """Past positions each head can see (None for unbounded)."""
        wins = []
        for h in range(self.heads):
            w = self.window
            if h < self.hard:
                w = h if w is None else min(w, h)
            wins.append(w)
        return wins

    def mask(self, t, device):
        i = torch.arange(t, device=device)
        dist = i[:, None] - i[None, :]
        rows = []
        for w in self.head_windows():
            m = dist >= 0
            if w is not None:
                m = m & (dist <= w)
            rows.append(m)
        return torch.stack(rows)[None]                                    # (1, H, T, T)

    def _split(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self._split(x)
        if self.pos == "rope":
            cos, sin = rope_tables(torch.arange(t), self.dh, dtype=x.dtype, device=x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.window is None and self.hard == 0:
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(t, x.device))
        return self.out(att.transpose(1, 2).reshape(b, t, self.d))

    def init_state(self, batch, like):
        empty = like.new_zeros(batch, 0, self.dh)
        return {"n": 0, "k": [empty] * self.heads, "v": [empty] * self.heads}

    def step(self, x, state):
        q, k, v = self._split(x[:, None])                                 # (B, H, 1, dh)
        n = state["n"]
        if self.pos == "rope":
            cos, sin = rope_tables(torch.tensor([n]), self.dh, dtype=x.dtype, device=x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        outs, new_k, new_v = [], [], []
        for h, w in enumerate(self.head_windows()):
            keys = torch.cat([state["k"][h], k[:, h]], dim=1)
            vals = torch.cat([state["v"][h], v[:, h]], dim=1)
            scores = (q[:, h] @ keys.transpose(1, 2)) / math.sqrt(self.dh)
            outs.append(torch.softmax(scores, -1) @ vals)
            start = 0 if w is None else max(0, keys.shape[1] - w)        # keep the w past positions the next step needs
            new_k.append(keys[:, start:])
            new_v.append(vals[:, start:])
        y = self.out(torch.cat(outs, dim=-1)[:, 0])
        return y, {"n": n + 1, "k": new_k, "v": new_v}

    def state_floats(self, n):
        return sum(2 * self.dh * (n if w is None else min(w, n)) for w in self.head_windows())

    def mixer_state_floats(self, n):
        return self.state_floats(n)


class LinearAttention(nn.Module):
    """Single-head causal linear attention, phi(x) = elu(x) + 1, out_t = phi(q_t) S_t / (phi(q_t) z_t + eps).
    The same math as bb_neural.CausalLinearAttention, with a short convolution on q, k, v."""

    EPS = 1e-6

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.conv = ShortConv(3 * d)

    def _features(self, x_conv):
        q, k, v = x_conv.chunk(3, dim=-1)
        return F.elu(q) + 1, F.elu(k) + 1, v

    def forward(self, x):
        q, k, v = self._features(self.conv(self.qkv(x)))
        ones = v.new_ones(*v.shape[:-1], 1)
        o, _ = chunk_gla(q[:, None], k[:, None], torch.cat([v, ones], -1)[:, None], q.new_zeros(q.shape[:2])[:, None])
        o = o[:, 0]
        return self.out(o[..., :-1] / (o[..., -1:] + self.EPS))

    def init_state(self, batch, like):
        return {"conv": self.conv.init_state(batch, like), "s": like.new_zeros(batch, self.d, self.d),
                "z": like.new_zeros(batch, self.d)}

    def step(self, x, state):
        xc, buf = self.conv.step(self.qkv(x), state["conv"])
        q, k, v = self._features(xc)
        s = state["s"] + k[:, :, None] * v[:, None, :]
        z = state["z"] + k
        y = (q[:, None] @ s)[:, 0] / ((q * z).sum(-1, keepdim=True) + self.EPS)
        return self.out(y), {"conv": buf, "s": s, "z": z}

    def mixer_state_floats(self, n):
        return self.d * self.d + self.d

    def state_floats(self, n):
        return self.mixer_state_floats(n) + self.conv.state_floats()


class DeltaNet(nn.Module):
    """DeltaNet (gated=False) or Gated DeltaNet (gated=True) mixer. q, k pass the short convolution and SiLU and
    are L2-normalized per head; q is scaled by dk^-1/2; b_t = sigmoid(W_b x_t) per head. The gate (Gated DeltaNet)
    is a_t = exp(-exp(A_log) softplus(W_a x_t + dt_bias)), the Mamba-2 parameterization the paper uses. The
    output is RMS-normalized per head and gated by SiLU(W_g x_t)."""

    def __init__(self, d, heads=1, gated=True):
        super().__init__()
        if d % heads:
            raise ValueError("d %d is not divisible by heads %d" % (d, heads))
        self.d, self.heads, self.dh, self.gated = d, heads, d // heads, gated
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = ShortConv(3 * d, bias=False)
        self.b_proj = nn.Linear(d, heads, bias=False)
        self.g_proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(self.dh)
        self.out = nn.Linear(d, d, bias=False)
        if gated:
            self.a_proj = nn.Linear(d, heads, bias=False)
            self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
            dt = torch.exp(torch.empty(heads).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def _prepare(self, x, xc):
        """x (..., d) the block input, xc (..., 3d) the convolved projections -> per-head q, k, v, g, beta."""
        shape = x.shape[:-1]
        q, k, v = xc.chunk(3, dim=-1)
        q = F.normalize(q.reshape(*shape, self.heads, self.dh), dim=-1) * self.dh ** -0.5
        k = F.normalize(k.reshape(*shape, self.heads, self.dh), dim=-1)
        v = v.reshape(*shape, self.heads, self.dh)
        beta = torch.sigmoid(self.b_proj(x))
        if self.gated:
            g = -torch.exp(self.A_log) * F.softplus(self.a_proj(x) + self.dt_bias)
        else:
            g = torch.zeros_like(beta)
        return q, k, v, g, beta

    def _finish(self, x, o):
        shape = x.shape[:-1]
        o = self.norm(o).reshape(*shape, self.d) * F.silu(self.g_proj(x))
        return self.out(o)

    def forward(self, x):
        q, k, v, g, beta = self._prepare(x, self.conv(self.qkv(x)))
        heads_first = [t.transpose(1, 2) for t in (q, k, v, g, beta)]       # (B, H, T, ...)
        o, _ = chunk_delta(*heads_first)
        return self._finish(x, o.transpose(1, 2))

    def init_state(self, batch, like):
        return {"conv": self.conv.init_state(batch, like),
                "s": like.new_zeros(batch, self.heads, self.dh, self.dh)}

    def step(self, x, state):
        xc, buf = self.conv.step(self.qkv(x), state["conv"])
        q, k, v, g, beta = self._prepare(x, xc)                            # (B, H, dh), (B, H)
        s = torch.exp(g)[..., None, None] * state["s"]
        u = beta[..., None] * (v - torch.einsum("bhd,bhde->bhe", k, s))
        s = s + k[..., :, None] * u[..., None, :]
        o = torch.einsum("bhd,bhde->bhe", q, s)
        return self._finish(x, o), {"conv": buf, "s": s}

    def mixer_state_floats(self, n):
        return self.heads * self.dh * self.dh

    def state_floats(self, n):
        return self.mixer_state_floats(n) + self.conv.state_floats()


class Mamba2(nn.Module):
    """Mamba-2 SSD mixer (Dao and Gu 2024, arXiv 2405.21060): expand 2, one group of B and C shared by every
    head, head dimension 32, scalar a_t = exp(dt_t A) per head, D skip, SiLU(z) gate, then RMSNorm and the output
    projection. The scan is chunk_gla with q = C, k = B, v = dt x and g = dt A."""

    def __init__(self, d, d_state=64, headdim=32, expand=2):
        super().__init__()
        self.d, self.d_state, self.headdim = d, d_state, headdim
        self.d_inner = expand * d
        if self.d_inner % headdim:
            raise ValueError("d_inner %d is not divisible by headdim %d" % (self.d_inner, headdim))
        self.heads = self.d_inner // headdim
        self.in_proj = nn.Linear(d, 2 * self.d_inner + 2 * d_state + self.heads, bias=False)
        self.conv = ShortConv(self.d_inner + 2 * d_state)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.heads).uniform_(1, 16)))
        dt = torch.exp(torch.empty(self.heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = RMSNorm(self.d_inner)
        self.out = nn.Linear(self.d_inner, d, bias=False)

    def _split(self, proj):
        di, ds = self.d_inner, self.d_state
        z, xbc, dt = proj[..., :di], proj[..., di:2 * di + 2 * ds], proj[..., 2 * di + 2 * ds:]
        return z, xbc, dt

    def _ssm_inputs(self, xbc, dt_raw):
        di, ds = self.d_inner, self.d_state
        xs, bm, cm = xbc[..., :di], xbc[..., di:di + ds], xbc[..., di + ds:]
        dt = F.softplus(dt_raw + self.dt_bias)                              # (..., H)
        g = dt * -torch.exp(self.A_log)
        xh = xs.reshape(*xs.shape[:-1], self.heads, self.headdim)
        return xh, bm, cm, dt, g

    def _finish(self, y, xh, z):
        y = y + self.D[:, None] * xh
        y = y.reshape(*z.shape[:-1], self.d_inner) * F.silu(z)
        return self.out(self.norm(y))

    def forward(self, x):
        z, xbc, dt_raw = self._split(self.in_proj(x))
        xh, bm, cm, dt, g = self._ssm_inputs(self.conv(xbc), dt_raw)
        b, t = x.shape[:2]
        q = cm[:, None].expand(b, self.heads, t, self.d_state)
        k = bm[:, None].expand(b, self.heads, t, self.d_state)
        v = (xh * dt[..., None]).transpose(1, 2)                            # (B, H, T, P)
        y, _ = chunk_gla(q, k, v, g.transpose(1, 2))
        return self._finish(y.transpose(1, 2), xh, z)

    def init_state(self, batch, like):
        return {"conv": self.conv.init_state(batch, like),
                "h": like.new_zeros(batch, self.heads, self.d_state, self.headdim)}

    def step(self, x, state):
        z, xbc, dt_raw = self._split(self.in_proj(x))
        xbc, buf = self.conv.step(xbc, state["conv"])
        xh, bm, cm, dt, g = self._ssm_inputs(xbc, dt_raw)                 # xh (B, H, P), bm, cm (B, N)
        h = torch.exp(g)[..., None, None] * state["h"] + bm[:, None, :, None] * (xh * dt[..., None])[:, :, None, :]
        y = torch.einsum("bn,bhnp->bhp", cm, h)
        return self._finish(y, xh, z), {"conv": buf, "h": h}

    def mixer_state_floats(self, n):
        return self.d_inner * self.d_state

    def state_floats(self, n):
        return self.mixer_state_floats(n) + self.conv.state_floats()


# ----------------------------------------------------------------------------------------------- model

def make_mixer(family, d, hyper):
    if family == "transformer":
        return Attention(d, hyper.get("heads", max(1, d // 32)), pos=hyper.get("pos", "rope"),
                         hard_alibi_heads=hyper.get("hard_alibi_heads", 0))
    if family == "sliding_window":
        return Attention(d, hyper.get("heads", max(1, d // 32)), pos=hyper.get("pos", "rope"),
                         window=int(hyper["window"]))
    if family == "linear_attention":
        return LinearAttention(d)
    if family in ("deltanet", "gated_deltanet"):
        return DeltaNet(d, heads=hyper.get("heads", 1), gated=family == "gated_deltanet")
    if family == "mamba2":
        return Mamba2(d, d_state=int(hyper.get("d_state", 64)), headdim=hyper.get("headdim", 32))
    raise ValueError("unknown family %r" % (family,))


KNOWN_HYPER = {"transformer": {"heads", "pos", "hard_alibi_heads"}, "sliding_window": {"heads", "pos", "window"},
               "linear_attention": set(), "deltanet": {"heads"}, "gated_deltanet": {"heads"},
               "mamba2": {"d_state", "headdim"}}


class LadderNet(nn.Module):
    def __init__(self, vocab, family, d, layers=2, hyper=None):
        super().__init__()
        hyper = dict(hyper or {})
        unknown = set(hyper) - KNOWN_HYPER[family]
        if unknown:
            raise ValueError("%s: unknown hyperparameters %s" % (family, sorted(unknown)))
        self.family, self.d, self.layers_n, self.hyper = family, d, layers, hyper
        self.emb = nn.Embedding(vocab, d)
        self.norms1 = nn.ModuleList(RMSNorm(d) for _ in range(layers))
        self.mixers = nn.ModuleList(make_mixer(family, d, hyper) for _ in range(layers))
        self.norms2 = nn.ModuleList(RMSNorm(d) for _ in range(layers))
        self.mlps = nn.ModuleList(SwiGLU(d) for _ in range(layers))
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)
        if family in ("deltanet", "gated_deltanet"):
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        h = self.emb(x)
        for n1, mix, n2, mlp in zip(self.norms1, self.mixers, self.norms2, self.mlps):
            h = h + mix(n1(h))
            h = h + mlp(n2(h))
        return self.head(self.norm_f(h))

    def init_state(self, batch):
        like = self.emb.weight
        return [mix.init_state(batch, like) for mix in self.mixers]

    def step(self, tokens, states):
        """One position for a batch of tokens (B,). Returns logits (B, V) and the new per-layer states."""
        h = self.emb(tokens)
        new = []
        for n1, mix, n2, mlp, st in zip(self.norms1, self.mixers, self.norms2, self.mlps, states):
            y, st = mix.step(n1(h), st)
            h = h + y
            h = h + mlp(n2(h))
            new.append(st)
        return self.head(self.norm_f(h)), new

    def stream(self, x):
        """Reference evaluation token by token with step(). Returns logits (B, T, V) and the final states."""
        states = self.init_state(x.shape[0])
        out = []
        for t in range(x.shape[1]):
            logits, states = self.step(x[:, t], states)
            out.append(logits)
        return torch.stack(out, dim=1), states

    def state_floats(self, n):
        return int(sum(m.state_floats(n) for m in self.mixers))

    def mixer_state_floats(self, n):
        return int(sum(m.mixer_state_floats(n) for m in self.mixers))

    def params(self):
        return int(sum(p.numel() for p in self.parameters()))


def count_state_tensors(states):
    """Floats held by the states step() carries (the 'n' counters are integers, not state)."""
    total = 0
    for st in states:
        for key, value in st.items():
            if key == "n":
                continue
            for tensor in (value if isinstance(value, list) else [value]):
                total += tensor[0].numel() if tensor.shape[0] else 0
    return int(total)


def state_formula(family, d, n, layers=2, hyper=None, include_conv=True):
    """Host-side state floats per sequence after n tokens, from the architecture alone (never from describe())."""
    hyper = dict(hyper or {})
    conv = 0
    if family in ("transformer", "sliding_window"):
        heads = hyper.get("heads", max(1, d // 32))
        dh = d // heads
        window = hyper.get("window") if family == "sliding_window" else None
        hard = hyper.get("hard_alibi_heads", 0)
        per = 0
        for h in range(heads):
            w = window
            if h < hard:
                w = h if w is None else min(w, h)
            per += 2 * dh * (n if w is None else min(w, n))
    elif family == "linear_attention":
        per, conv = d * d + d, (CONV_K - 1) * 3 * d
    elif family in ("deltanet", "gated_deltanet"):
        heads = hyper.get("heads", 1)
        per, conv = heads * (d // heads) ** 2, (CONV_K - 1) * 3 * d
    elif family == "mamba2":
        ds = int(hyper.get("d_state", 64))
        per, conv = 2 * d * ds, (CONV_K - 1) * (2 * d + 2 * ds)
    else:
        raise ValueError("unknown family %r" % (family,))
    return int(layers * (per + (conv if include_conv else 0)))


def forward_flops_per_token(family, d, n, vocab, layers=2, hyper=None):
    """Analytic forward multiply-adds x 2 per token at context length n (projections, mixer, MLP, head).

    Attention counts 4 d n per layer for the score and value products (averaged over a length-n sequence it is
    2 d n; this is the cost of the token at position n). The fixed-state mixers count their state update and
    read, which does not grow with n."""
    hyper = dict(hyper or {})
    hidden = 8 * math.ceil(8 * d / 3 / 8)
    mlp = 2 * 3 * d * hidden
    if family in ("transformer", "sliding_window"):
        span = n if family == "transformer" else min(int(hyper["window"]), n)
        mixer = 2 * 4 * d * d + 4 * d * span
    elif family == "linear_attention":
        mixer = 2 * 4 * d * d + 2 * 3 * d * CONV_K + 6 * d * d
    elif family in ("deltanet", "gated_deltanet"):
        heads = hyper.get("heads", 1)
        dh = d // heads
        mixer = 2 * 5 * d * d + 2 * 3 * d * CONV_K + heads * 8 * dh * dh
    elif family == "mamba2":
        ds, di = int(hyper.get("d_state", 64)), 2 * d
        mixer = 2 * d * (2 * di + 2 * ds + di // 32) + 2 * di * d + 2 * (di + 2 * ds) * CONV_K + 6 * di * ds
    else:
        raise ValueError("unknown family %r" % (family,))
    return int(layers * (mixer + mlp) + 2 * d * vocab)


# ----------------------------------------------------------------------------------------------- candidate

class LadderCandidate:
    """The bench Candidate API (hybrid/bench/api.py) around LadderNet, so check_causality and the old runner can
    drive these baselines. budget: max_steps, batch_size, lr. Training is next-token loss on every position;
    the ladder runner trains with answer-only loss instead."""

    FAMILY, D, HYPER = "transformer", 64, {}

    def __init__(self, vocab_size, seed, budget):
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        self.V, self.seed, self.budget = int(vocab_size), int(seed), dict(budget)
        self.net = LadderNet(self.V, self.FAMILY, self.D, hyper=self.HYPER)

    def fit(self, train):
        train = [s for s in train if len(s) >= 2]
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.budget["lr"]))
        bs = int(self.budget["batch_size"])
        self.net.train()
        for _ in range(int(self.budget["max_steps"])):
            batch = [train[i] for i in rng.choice(len(train), size=min(bs, len(train)), replace=False)]
            width = max(len(s) for s in batch)
            x = torch.zeros(len(batch), width, dtype=torch.long)
            mask = torch.zeros(len(batch), width - 1, dtype=torch.bool)
            for i, s in enumerate(batch):
                x[i, :len(s)] = torch.as_tensor(s)
                mask[i, :len(s) - 1] = True
            loss = F.cross_entropy(self.net(x[:, :-1])[mask], x[:, 1:][mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()

    @torch.no_grad()
    def logits_all(self, seqs):
        self.net.eval()
        width = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.as_tensor(s)
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, prefixes):
        logits = self.logits_all(prefixes)
        last = torch.tensor([len(p) - 1 for p in prefixes])
        return torch.softmax(logits[torch.arange(len(prefixes)), last].double(), -1).numpy()

    def describe(self):
        return {"params": self.net.params(), "state_floats_per_sequence": self.net.state_floats(256),
                "notes": "%s d %d %s (state at 256 tokens)" % (self.FAMILY, self.D, self.HYPER)}


def candidate_class(family, d, hyper=None):
    return type("Candidate_%s_%d" % (family, d), (LadderCandidate,),
                {"FAMILY": family, "D": d, "HYPER": dict(hyper or {})})
