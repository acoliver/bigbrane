"""CPU float32 models implementing the fixed matrix equations."""
import math
import numpy as np
import torch
from torch import nn


def seed_word(seed_tuple):
    return int(np.random.SeedSequence(seed_tuple).generate_state(1, dtype=np.uint64)[0])


def build(family, seed):
    torch.manual_seed(seed)
    if family == 'rnn':
        return RNN()
    if family == 'attention':
        return Attention()
    raise ValueError(f'unknown neural family {family}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def shifted(tokens):
    return np.concatenate((np.full((len(tokens), 1), 8, dtype=np.int64), tokens[:, :-1]), axis=1)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(9, 8)
        self.We = nn.Parameter(torch.empty(8, 8)); self.Wh = nn.Parameter(torch.empty(8, 8))
        self.bh = nn.Parameter(torch.zeros(8)); self.Wy = nn.Parameter(torch.empty(8, 8))
        self.by = nn.Parameter(torch.zeros(8))
        nn.init.normal_(self.E.weight, 0, .1)
        nn.init.orthogonal_(self.Wh)
        nn.init.xavier_uniform_(self.We); nn.init.xavier_uniform_(self.Wy)

    def forward(self, inputs):
        h = self.E.weight.new_zeros((inputs.shape[0], 8)); outputs = []
        for t in range(inputs.shape[1]):
            h = torch.tanh(self.E(inputs[:, t]) @ self.We.T+h @ self.Wh.T+self.bh)
            outputs.append(h @ self.Wy.T+self.by)
        return torch.stack(outputs, 1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Embedding(9, 8)
        self.q = nn.Linear(8, 8, bias=False); self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 8, bias=False); self.o = nn.Linear(8, 8, bias=False)
        self.n1 = nn.LayerNorm(8, eps=1e-5); self.n2 = nn.LayerNorm(8, eps=1e-5)
        self.w1 = nn.Linear(8, 16); self.w2 = nn.Linear(16, 8); self.wy = nn.Linear(8, 8)
        nn.init.normal_(self.E.weight, 0, .1)
        for layer in (self.q, self.k, self.v, self.o, self.w1, self.w2, self.wy):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def positions(self, length):
        pos = torch.arange(length, device=self.E.weight.device, dtype=self.E.weight.dtype)[:, None]
        div = 10000.**(torch.arange(0, 8, 2, device=pos.device, dtype=pos.dtype)/8)
        pe = self.E.weight.new_empty((length, 8))
        pe[:, ::2] = torch.sin(pos/div); pe[:, 1::2] = torch.cos(pos/div)
        return pe

    def forward(self, inputs):
        v = self.E(inputs)+self.positions(inputs.shape[1])
        a = self.n1(v)
        scores = self.q(a) @ self.k(a).transpose(1, 2)/math.sqrt(8)
        mask = torch.triu(torch.ones(inputs.shape[1], inputs.shape[1], device=inputs.device, dtype=torch.bool), 1)
        c = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
        r = v+self.o(c @ self.v(a))
        h = r+self.w2(torch.relu(self.w1(self.n2(r))))
        return self.wy(h)


def log_predict(model, tokens, batch_size=32):
    model.eval(); result = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            inputs = torch.from_numpy(shifted(tokens[start:start+batch_size]))
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise ArithmeticError('nonfinite neural logits')
            result.append(torch.log_softmax(logits.to(torch.float64), dim=-1).numpy())
    return np.concatenate(result)


def evaluate(model, tokens):
    return np.exp(log_predict(model, tokens))
