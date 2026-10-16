"""Trusted, deliberately small development fixture; no confirmation inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

CANDIDATES = [
    {'attempt_id': 'base', 'parents': [], 'margin': -0.05, 'valid': True},
    {'attempt_id': 'strong', 'parents': ['base'], 'margin': 1.0, 'valid': True},
    {'attempt_id': 'invalid', 'parents': ['base'], 'margin': 0.01, 'valid': False},
    {'attempt_id': 'resumed', 'parents': ['strong'], 'margin': -0.05, 'valid': True},
]
EQUATIONS = {
    'hmm': 'p_t=q_t B; q_(t+1)=(q_t * B[:,x_t]) A / p_t[x_t]',
    'rnn': 'h_(t+1)=tanh(0.8 h_t+0.7(2 x_t-1)); p_t(1)=sigmoid(w h_t+b)',
    'metric': 'mean_t sum_v oracle_t(v) log(oracle_t(v)/learner_t(v))',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def normalize(a):
    return a / a.sum(axis=-1, keepdims=True)


def sample(rng, A, B, count=16, length=16):
    seqs = []
    for _ in range(count):
        z = int(rng.integers(2))
        row = []
        for _ in range(length):
            row.append(int(rng.choice(2, p=B[z])))
            z = int(rng.choice(2, p=A[z]))
        seqs.append(row)
    return np.array(seqs)


def fit_hmm(seqs):
    rng = np.random.default_rng(20261007)
    A = normalize(rng.uniform(0.2, 1, (2, 2)))
    B = normalize(rng.uniform(0.2, 1, (2, 2)))
    pi = np.array([0.5, 0.5])
    for _ in range(8):
        ac, bc, pc = np.full((2, 2), 0.01), np.full((2, 2), 0.01), np.full(2, 0.01)
        for xs in seqs:
            f = []
            prior = pi.copy()
            for x in xs:
                post = prior * B[:, x]
                post /= post.sum()
                f.append(post)
                prior = post @ A
            back = np.ones(2)
            for t in range(len(xs) - 1, -1, -1):
                gamma = f[t] * back
                gamma /= gamma.sum()
                bc[:, xs[t]] += gamma
                if t == 0:
                    pc += gamma
                else:
                    xi = f[t-1][:, None] * A * (B[:, xs[t]] * back)[None, :]
                    xi /= xi.sum()
                    ac += xi
                back = A @ (B[:, xs[t]] * back)
                back /= back.sum()
        A, B, pi = normalize(ac), normalize(bc), pc / pc.sum()
    return A, B, pi


def filter_predictions(seqs, A, B, pi):
    out = []
    for xs in seqs:
        q = pi.copy()
        for x in xs:
            p = q @ B
            out.append(p.tolist())
            q = (q * B[:, x] / p[x]) @ A
    return out


def rnn_features(seqs):
    out = []
    for xs in seqs:
        h = 0.0
        for x in xs:
            out.append([h, 1.0])
            h = np.tanh(0.8 * h + 0.7 * (2 * x - 1))
    return np.array(out)


def kl(oracle, prediction):
    o, p = np.array(oracle), np.array(prediction)
    return float(np.mean(np.sum(o * np.log(o / p), axis=1)))


def evaluate(candidate):
    if not candidate['valid']:
        raise ValueError('invalid candidate: deliberate schema failure')
    A = np.array([[0.92, 0.08], [0.08, 0.92]])
    B = np.array([[0.85, 0.15], [0.15, 0.85]])
    train = sample(np.random.default_rng(8675309), A, B, count=8)
    dev = sample(np.random.default_rng(8675310), A, B)
    learned = fit_hmm(train)
    hmm = filter_predictions(dev, *learned)
    oracle = filter_predictions(dev, A, B, np.array([0.5, 0.5]))
    X, y = rnn_features(train), train.ravel()
    weights = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ weights))
        weights -= 0.5 * (X.T @ (p - y)) / len(y)
    p = 1 / (1 + np.exp(-rnn_features(dev) @ weights))
    rnn = np.column_stack([1 - p, p]).tolist()
    hk, rk = kl(oracle, hmm), kl(oracle, rnn)
    metrics = {'hmm_kl': hk, 'rnn_kl': rk, 'delta_rnn_minus_hmm': rk-hk, 'targets': len(p), 'training_tokens': int(train.size)}
    predictions = {'hmm': hmm, 'rnn': rnn, 'development_oracle': oracle}
    return {
        **candidate, 'scope': 'synthetic-development-only',
        'claim': f"HMM predictive KL is lower than this restricted RNN by at least {candidate['margin']} nats/target",
        'equations': EQUATIONS, 'falsifier': 'delta_rnn_minus_hmm < margin',
        'metrics': metrics, 'predictions': predictions, 'failures': [],
        'status': 'supported' if rk-hk >= candidate['margin'] else 'scientific_negative',
        'data_sha256': digest({'train': train.tolist(), 'development': dev.tolist()}),
        'predictions_sha256': digest(predictions),
        'learner_limits': 'Two-state HMM, eight EM iterations; scalar fixed recurrent weights with learned logistic output. Not Experiment 1 models.',
    }


def invalid_record(candidate, error):
    return {**candidate, 'scope': 'synthetic-development-only', 'status': 'invalid',
            'claim': 'Deliberately malformed candidate', 'equations': EQUATIONS,
            'falsifier': 'No scientific decision from an invalid run', 'metrics': None,
            'predictions': None, 'failures': [str(error)]}


def verify_record(record):
    if record['status'] == 'invalid':
        assert record['metrics'] is None and record['failures']
        return
    p = record['predictions']
    assert digest(p) == record['predictions_sha256']
    for values in p.values():
        assert np.allclose(np.array(values).sum(axis=1), 1)
    for name in ('hmm', 'rnn'):
        assert abs(kl(p['development_oracle'], p[name]) - record['metrics'][name+'_kl']) < 1e-14
    delta = record['metrics']['rnn_kl'] - record['metrics']['hmm_kl']
    assert abs(delta - record['metrics']['delta_rnn_minus_hmm']) < 1e-14
    assert (record['status'] == 'supported') == (delta >= record['margin'])


def persist(folder, record):
    verify_record(record)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (record['attempt_id'] + '.json')
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == text
    else:
        temporary = path.with_suffix('.partial')
        temporary.write_text(text)
        temporary.replace(path)
    return record
