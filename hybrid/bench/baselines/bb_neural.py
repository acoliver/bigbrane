"""Neural baselines sharing one training loop: GRU, tiny causal transformer, diagonal linear SSM, linear attention.

Standalone (NumPy and CPU torch). All three use width 64, the same optimizer, the same
budget dict and a fixed seed, with torch on one thread. Training is next-token cross-entropy
over every position of the training sequences, with early stopping on a held-out eighth of the
training sequences (see _TorchCandidate.fit). Prediction right-pads a batch and reads the
logits at each prefix's last real position, so padding never reaches a prediction.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 64
PREDICT_TOKENS_PER_CHUNK = 16384
LA_CHUNK = 64     # positions per chunk in the chunked causal linear attention
VAL_MIN = 16      # early stopping needs at least this many training sequences
VAL_MAX = 64
EVAL_POINTS = 10


def _pad(seqs, length):
    x = torch.zeros(len(seqs), length, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
    return x


class GRUNet(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.rnn = nn.GRU(D_MODEL, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h)


def sinusoid(length, d):
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(self.ln2(x))


class TransformerNet(nn.Module):
    def __init__(self, vocab, layers=2, heads=4, ffn=256):
        super().__init__()
        self.layers_n = layers
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList(Block(D_MODEL, heads, ffn) for _ in range(layers))
        self.ln = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, vocab)

    def forward(self, x):
        h = self.emb(x) * math.sqrt(D_MODEL) + sinusoid(x.shape[1], D_MODEL)
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln(h))


class LRULayer(nn.Module):
    """Diagonal complex linear recurrence h_t = lam * h_{t-1} + B u_t, y_t = Re(C h_t) + D u_t.

    LRU-style parameterization (Orvieto et al. 2023): |lam| = exp(-exp(nu)) initialized in
    [r_min, r_max], phase exp(theta), input normalization gamma = sqrt(1 - |lam|^2). The
    recurrence is time-invariant, so it is computed as a causal convolution with kernel
    lam^k through an FFT of size 2L (linear, not circular, convolution)."""

    def __init__(self, d, n_state, r_min=0.9, r_max=0.999, max_phase=math.pi / 10):
        super().__init__()
        u1, u2 = torch.rand(n_state), torch.rand(n_state)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2 + 1e-4))
        self.B_re = nn.Parameter(torch.randn(n_state, d) / math.sqrt(2 * d))
        self.B_im = nn.Parameter(torch.randn(n_state, d) / math.sqrt(2 * d))
        self.C_re = nn.Parameter(torch.randn(d, n_state) / math.sqrt(n_state))
        self.C_im = nn.Parameter(torch.randn(d, n_state) / math.sqrt(n_state))
        self.D = nn.Parameter(torch.randn(d))

    def forward(self, u):
        b, n, d = u.shape
        log_lam = torch.complex(-torch.exp(self.nu_log), torch.exp(self.theta_log))
        gamma = torch.sqrt(1 - torch.exp(2 * log_lam.real))
        B = torch.complex(self.B_re, self.B_im) * gamma[:, None]
        Bu = u.to(torch.complex64) @ B.T                                     # (b, n, N)
        k = torch.arange(n, dtype=torch.float32)
        kernel = torch.exp(k[:, None] * log_lam[None, :])                    # (n, N)
        size = 2 * n
        h = torch.fft.ifft(torch.fft.fft(Bu, n=size, dim=1) * torch.fft.fft(kernel, n=size, dim=0)[None], dim=1)[:, :n]
        C = torch.complex(self.C_re, self.C_im)
        return (h @ C.T).real + self.D * u


class SSMNet(nn.Module):
    def __init__(self, vocab, layers=2, n_state=64):
        super().__init__()
        self.layers_n, self.n_state = layers, n_state
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.norms = nn.ModuleList(nn.LayerNorm(D_MODEL) for _ in range(layers))
        self.ssms = nn.ModuleList(LRULayer(D_MODEL, n_state) for _ in range(layers))
        self.mix = nn.ModuleList(nn.Linear(D_MODEL, 2 * D_MODEL) for _ in range(layers))
        self.ln = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, vocab)

    def forward(self, x):
        h = self.emb(x)
        for norm, ssm, mix in zip(self.norms, self.ssms, self.mix):
            h = h + F.glu(mix(F.gelu(ssm(norm(h)))), dim=-1)
        return self.out(self.ln(h))


class CausalLinearAttention(nn.Module):
    """Single-head causal linear attention with the feature map phi(x) = elu(x) + 1 (Katharopoulos et al. 2020).

    out_t = phi(q_t) S_t / (phi(q_t) z_t + eps), S_t = sum_{i<=t} phi(k_i) v_i^T (a d x d matrix) and
    z_t = sum_{i<=t} phi(k_i). The state (S, z) has a fixed size whatever the context length. Computed in
    chunks of LA_CHUNK positions: exact masked attention inside a chunk, and the running (S, z) for
    everything before it, so the cost is O(n d^2 + n LA_CHUNK d) per sequence and padding to the right
    never reaches an earlier position."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def state_floats(self):
        return self.d * self.d + self.d

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        state = x.new_zeros(b, d, d)
        norm = x.new_zeros(b, d)
        mask = torch.tril(torch.ones(LA_CHUNK, LA_CHUNK, dtype=torch.bool, device=x.device))
        outs = []
        for start in range(0, n, LA_CHUNK):
            qc, kc, vc = q[:, start:start + LA_CHUNK], k[:, start:start + LA_CHUNK], v[:, start:start + LA_CHUNK]
            c = qc.shape[1]
            scores = (qc @ kc.transpose(1, 2)).masked_fill(~mask[:c, :c], 0.0)
            num = qc @ state + scores @ vc
            den = (qc * norm[:, None, :]).sum(-1) + scores.sum(-1)
            outs.append(num / (den[..., None] + 1e-6))
            state = state + kc.transpose(1, 2) @ vc
            norm = norm + kc.sum(1)
        return self.proj(torch.cat(outs, dim=1))


class LinearAttentionBlock(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = CausalLinearAttention(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        return x + self.ff(self.ln2(x))


class LinearAttentionNet(nn.Module):
    """Embedding, two linear-attention blocks (no positional encoding, as the causal sums carry order), head."""

    def __init__(self, vocab, layers=2, ffn=256):
        super().__init__()
        self.layers_n = layers
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList(LinearAttentionBlock(D_MODEL, ffn) for _ in range(layers))
        self.ln = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, vocab)

    def state_floats(self):
        return sum(block.att.state_floats() for block in self.blocks)

    def forward(self, x):
        h = self.emb(x)
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln(h))


class _TorchCandidate:
    def __init__(self, vocab_size, seed, budget):
        torch.set_num_threads(1)
        self.V, self.seed, self.budget = int(vocab_size), int(seed), dict(budget)
        torch.manual_seed(self.seed)
        self.net = self.build(self.V)
        self.steps_run = self.best_step = 0

    def build(self, vocab):
        raise NotImplementedError

    def _loss(self, batch):
        lengths = torch.tensor([len(s) for s in batch])
        x = _pad(batch, int(lengths.max()))
        mask = torch.arange(x.shape[1] - 1)[None, :] < (lengths[:, None] - 1)
        return F.cross_entropy(self.net(x[:, :-1])[mask], x[:, 1:][mask])

    def fit(self, train):
        """Adam on next-token loss for min(max_steps, max_epochs * batches per epoch) steps.

        Early stopping: when there are at least VAL_MIN sequences, 1/8 of them (at most
        VAL_MAX) are held out, validation loss is checked EVAL_POINTS times, and the
        parameters with the lowest validation loss are kept."""
        train = [s for s in train if len(s) >= 2]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(train))
        n_val = min(VAL_MAX, len(train) // 8) if len(train) >= VAL_MIN else 0
        val = [train[i] for i in order[:n_val]]
        fit_set = [train[i] for i in order[n_val:]]
        bs = int(self.budget["batch_size"])
        steps = min(int(self.budget["max_steps"]), int(self.budget["max_epochs"]) * math.ceil(len(fit_set) / bs))
        check = max(1, steps // EVAL_POINTS)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.budget["lr"]))
        best_loss, best_state, best_step = math.inf, None, steps
        for step in range(1, steps + 1):
            self.net.train()
            batch = [fit_set[i] for i in rng.choice(len(fit_set), size=min(bs, len(fit_set)), replace=False)]
            loss = self._loss(batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            if val and (step % check == 0 or step == steps):
                self.net.eval()
                with torch.no_grad():
                    val_loss = float(self._loss(val))
                if val_loss < best_loss:
                    best_loss, best_step = val_loss, step
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.steps_run, self.best_step = steps, best_step

    @torch.no_grad()
    def logits_all(self, seqs):
        """Logits at every position of a right-padded batch (used by the causality test)."""
        self.net.eval()
        return self.net(_pad(seqs, max(len(s) for s in seqs)))

    @torch.no_grad()
    def predict_proba(self, prefixes):
        self.net.eval()
        order = sorted(range(len(prefixes)), key=lambda i: -len(prefixes[i]))
        out = np.empty((len(prefixes), self.V), dtype=np.float64)
        start = 0
        while start < len(order):
            longest = len(prefixes[order[start]])   # descending order: the chunk's first prefix is its longest
            idx = order[start:start + max(1, PREDICT_TOKENS_PER_CHUNK // longest)]
            x = _pad([prefixes[i] for i in idx], longest)
            last = torch.tensor([len(prefixes[i]) - 1 for i in idx])
            logits = self.net(x)[torch.arange(len(idx)), last]
            out[idx] = torch.softmax(logits.double(), -1).numpy()
            start += len(idx)
        return out

    def describe(self):
        return {"params": int(sum(p.numel() for p in self.net.parameters())),
                "state_floats_per_sequence": int(self.state_floats()), "notes": self.notes(),
                "train_steps": int(self.steps_run), "best_step": int(self.best_step)}


class GRUCandidate(_TorchCandidate):
    def build(self, vocab):
        return GRUNet(vocab)

    def state_floats(self):
        return self.net.hidden

    def notes(self):
        return "GRU, embedding 64, hidden 128, one layer"


class TransformerCandidate(_TorchCandidate):
    def build(self, vocab):
        return TransformerNet(vocab)

    def state_floats(self):
        return 2 * self.net.layers_n * D_MODEL * 256

    def notes(self):
        return ("causal transformer, d 64, 2 layers, 4 heads, ffn 256, sinusoidal positions; state is the KV "
                "cache, which grows linearly with context (value given at 256 tokens)")


class SSMCandidate(_TorchCandidate):
    def build(self, vocab):
        return SSMNet(vocab)

    def state_floats(self):
        return 2 * self.net.layers_n * self.net.n_state

    def notes(self):
        return "diagonal complex linear SSM (LRU-style), d 64, state 64 complex per layer, 2 layers, FFT convolution"


class LinearAttentionCandidate(_TorchCandidate):
    def build(self, vocab):
        return LinearAttentionNet(vocab)

    def state_floats(self):
        return self.net.state_floats()

    def notes(self):
        return ("causal linear attention, elu+1 feature map, d 64, one head, 2 layers, chunked O(n) evaluation; "
                "state per layer is a 64 x 64 matrix plus a 64-vector normalizer, independent of context length")
