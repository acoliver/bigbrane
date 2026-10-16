"""Evaluator-only scoring of checkpoints frozen by validation selection."""
from pathlib import Path
import time
import numpy as np
from .artifacts import hash_file, read_json, save_array, verify_inventory, verify_provenance, write_json
from .core import batch_forward_backward, hmm_filter, stress_filter
from .fitting import load_checkpoint, verify_candidate
from .neural import build, log_predict, seed_word
from .protocol import FitSpec


def summaries(oracle, log_model, tokens):
    if oracle.shape != log_model.shape or oracle.shape[:2] != tokens.shape:
        raise ValueError('evaluation alignment mismatch')
    if not np.isfinite(log_model).all() or np.max(np.abs(np.exp(log_model).sum(-1)-1)) > 1e-10:
        raise ArithmeticError('invalid evaluated model probabilities')
    if not np.isfinite(oracle).all() or (oracle <= 0).any() or np.max(np.abs(oracle.sum(-1)-1)) > 1e-10:
        raise ArithmeticError('invalid oracle probabilities')
    error = (oracle*(np.log(oracle)-log_model)).sum(-1)
    if error.min() < -1e-10:
        raise ArithmeticError('negative conditional KL beyond roundoff')
    model_nll = -np.take_along_axis(log_model, tokens[..., None], -1)[..., 0]
    oracle_nll = -np.take_along_axis(np.log(oracle), tokens[..., None], -1)[..., 0]
    entropy = -(oracle*np.log(oracle)).sum(-1)
    def mean_values(sl):
        nll = float(model_nll[:, sl].mean()); reference = float(oracle_nll[:, sl].mean())
        return {'kl': float(error[:, sl].mean()), 'test_nll': nll, 'oracle_nll': reference,
                'paired_nll_difference': nll-reference, 'test_bits_per_token': nll/np.log(2),
                'oracle_bits_per_token': reference/np.log(2), 'oracle_conditional_entropy': float(entropy[:, sl].mean())}
    return {'all': mean_values(slice(None)), 'initial': mean_values(slice(0, 1)),
            'noninitial': mean_values(slice(1, None)),
            'blocks': [mean_values(slice(t, t+16)) for t in range(0, tokens.shape[1], 16)],
            'per_sequence': [{'kl': float(error[i].mean()), 'test_nll': float(model_nll[i].mean()),
                              'oracle_nll': float(oracle_nll[i].mean()), 'paired_nll_difference': float((model_nll[i]-oracle_nll[i]).mean()),
                              'initial_kl': float(error[i, 0]), 'noninitial_kl': float(error[i, 1:].mean()),
                              'block_kl': [float(error[i, t:t+16].mean()) for t in range(0, tokens.shape[1], 16)],
                              'block_nll': [float(model_nll[i, t:t+16].mean()) for t in range(0, tokens.shape[1], 16)]}
                             for i in range(len(tokens))]}


def benchmark(predict, tokens, batch_size):
    for _ in range(2):
        predict(tokens, batch_size)
    timings = []
    for _ in range(10):
        start = time.perf_counter(); predict(tokens, batch_size)
        timings.append(time.perf_counter()-start)
    median = float(np.median(timings))
    return {'batch_size': batch_size, 'warmup_passes': 2, 'timed_passes': 10,
            'elapsed_s': timings, 'median_elapsed_s': median, 'tokens_per_second': tokens.size/median,
            'sequences': len(tokens), 'length': tokens.shape[1], 'scored_tokens': tokens.size,
            'includes': 'all targets, prediction head and probability normalization; excludes file I/O and model construction'}


def predictor(spec, checkpoint):
    state = load_checkpoint(checkpoint)
    if spec.family == 'hmm':
        a, b, p = (state[name].numpy() for name in ('A', 'B', 'pi'))
        def predict(tokens, batch_size=32):
            return np.concatenate([np.log(hmm_filter(tokens[start:start+batch_size], a, b, p))
                                   for start in range(0, len(tokens), batch_size)])
        return predict, (a, b, p)
    model = build(spec.family, seed_word(spec.seed(0)))
    model.load_state_dict(state['model']); model.eval()
    return lambda tokens, batch_size=32: log_predict(model, tokens, batch_size), None


def evaluate_selected(out, selection, scaling=False, *, manifest=None):
    out = Path(out)
    manifest = read_json(out/'manifest.json') if manifest is None else manifest
    verify_provenance(out, manifest)
    spec = FitSpec(**selection['spec'])
    stage_selection = manifest['selections'][str(spec.stage)]
    if selection not in read_json(out/stage_selection['path']):
        raise ValueError('evaluation input is not a frozen selection')
    record = verify_candidate(out/'candidates'/spec.key, spec,
                              completion_hash=manifest['candidate_commits'][spec.key])
    if (selection['validation_nll'], selection['checkpoint_index'], selection['checkpoint_hash']) != (
            record['validation_nll'], record['checkpoint'], record['artifact_hashes'][record['checkpoint_file']]):
        raise ValueError('evaluation selection disagrees with committed candidate')
    directory = Path(out)/'evaluations'/spec.key
    checkpoint = Path(out)/selection['checkpoint_path']
    if hash_file(checkpoint) != selection['checkpoint_hash']:
        raise ValueError('selected checkpoint changed after freeze')
    if (directory/'evaluation-digest.json').exists():
        verify_inventory(directory, read_json(directory/'evaluation-digest.json'))
        record = read_json(directory/'evaluation.json')
        if record['selection'] != selection:
            raise ValueError('evaluation selection changed')
        return record
    predict, fitted = predictor(spec, checkpoint)
    base = Path(out)/'data'/f's{spec.stage}-r{spec.regime}-g{spec.generator}'
    with np.load(base/'evaluator'/'generator.npz', allow_pickle=False) as matrices:
        a, b, p = (matrices[name] for name in ('A', 'B', 'pi'))
    tokens = np.load(base/'evaluator'/'test.npy', allow_pickle=False)
    verify_provenance(out, manifest)
    oracle = stress_filter(tokens, a, b, p) if spec.stage else hmm_filter(tokens, a, b, p)
    model_log = predict(tokens)
    record = {'selection': selection, 'test_file_hash': hash_file(base/'evaluator'/'test.npy'),
              'metrics': summaries(oracle, model_log, tokens),
              'benchmarks': [benchmark(predict, tokens, batch) for batch in (1, 32)],
              'scaling': []}
    verify_provenance(out, manifest)
    train = np.load(base/'learner'/'train.npy', allow_pickle=False)[:(8, 32, 128, 256)[spec.size_index]]
    verify_provenance(out, manifest)
    counts = np.bincount(train.reshape(-1), minlength=8)
    unigram = (counts+1)/(train.size+8)
    record['unigram'] = summaries(oracle, np.broadcast_to(np.log(unigram), oracle.shape), tokens)
    record['oracle_self_kl'] = summaries(oracle, np.log(oracle), tokens)['all']['kl']
    if fitted is not None:
        _, g, _ = batch_forward_backward(train, *fitted)
        record['expected_training_occupancy'] = g.sum((0, 1)).tolist()
    if scaling:
        for length in (16, 32, 64, 128, 256):
            seed = (spec.root, 4, spec.regime, spec.generator, length)
            scaling_path = Path(out)/'scaling'/f'r{spec.regime}-g{spec.generator}-l{length}.npy'
            # Scaling data are frozen by the runner before any checkpoint evaluation.
            verify_provenance(out, manifest)
            scaled = np.load(scaling_path, allow_pickle=False)
            verify_provenance(out, manifest)
            scaled_oracle = hmm_filter(scaled, a, b, p)
            record['scaling'].append({'length': length, 'seed': seed, 'data_hash': hash_file(scaling_path),
                                      'kl': summaries(scaled_oracle, predict(scaled), scaled)['all']['kl'],
                                      'benchmarks': [benchmark(predict, scaled, batch) for batch in (1, 32)]})
    verify_provenance(out, manifest)
    directory.mkdir(parents=True, exist_ok=True)
    save_array(directory/'log_predictions.npy', model_log)
    save_array(directory/'oracle_predictions.npy', oracle)
    write_json(directory/'evaluation.json', record)
    write_json(directory/'evaluation-digest.json', {name: hash_file(directory/name) for name in
                                                   ('evaluation.json', 'log_predictions.npy', 'oracle_predictions.npy')})
    return record


ANALYTICAL_ESTIMATES = {
    'label': 'Analytical estimates, not measured memory or runtime',
    'hmm': {'work': 'O(L(K^2+KV))', 'changing_state_bytes': 4*8, 'state': 'O(K) plus returned predictions',
            'stored_parameter_bytes': 52*8, 'em_work': 'O(NK^2) plus emissions/counts per iteration'},
    'rnn': {'work': 'O(L(h^2+dh+hV))', 'changing_state_bytes': 8*4, 'state': 'O(h); training activations grow with L'},
    'attention': {'work': 'O(Ld^2+L^2d) plus output head', 'workspace': 'O(L^2+Ld); optional KV cache O(Ld) is not benchmarked'},
}
