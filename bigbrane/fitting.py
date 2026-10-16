"""Candidate worker. Its only data port contains training and validation tokens."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import time
import numpy as np
import torch
from .artifacts import (ROOT, Ownership, hash_file, peak_rss_bytes, read_json,
                        verify_inventory, verify_sources, write_json)
from .core import L, SIZES, em_step, hmm_filter, nll, rng, validate_hmm
from .neural import build, count_parameters, log_predict, seed_word, shifted
from .protocol import FitSpec, LEARNING_RATES


@dataclass(frozen=True)
class LearnerData:
    train: np.ndarray
    validation: np.ndarray
    hashes: dict

    @classmethod
    def load(cls, directory, size_index):
        directory = Path(directory)
        train = np.load(directory/'train.npy', allow_pickle=False)[:SIZES[size_index]]
        validation = np.load(directory/'validation.npy', allow_pickle=False)
        if train.shape != (SIZES[size_index], L) or validation.shape != (32, L):
            raise ValueError('learner split counts disagree with protocol')
        for tokens in (train, validation):
            if tokens.dtype != np.int64 or (tokens < 0).any() or (tokens >= 8).any():
                raise ValueError('invalid observed tokens')
            tokens.setflags(write=False)
        return cls(train, validation, {name: hash_file(directory/name) for name in ('train.npy', 'validation.npy')})


def tensor_bytes(state):
    if torch.is_tensor(state):
        return state.numel()*state.element_size()
    if isinstance(state, dict):
        return sum(tensor_bytes(v) for v in state.values())
    if isinstance(state, (tuple, list)):
        return sum(tensor_bytes(v) for v in state)
    return 0


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def normalized(value):
    import json
    return json.loads(json.dumps(value))


def verify_resume(directory):
    resume = read_json(directory/'resume.json')
    verify_inventory(directory, resume['inventory'])
    verify_inventory(directory, {'identity.json': resume['identity_hash']})
    state = load_checkpoint(directory/resume['checkpoint'])
    expected = {f'checkpoint-{i:03d}.pt' for i in range(state['index']+1)}
    if set(resume['inventory']) != expected or resume['checkpoint'] != f"checkpoint-{state['index']:03d}.pt":
        raise ValueError('committed checkpoint inventory disagrees with state')
    trace = state['trace']
    if len(trace) != state['index']+1 or [entry['iteration_or_epoch'] for entry in trace] != list(range(len(trace))):
        raise ValueError('committed trace indexes disagree')
    best = min(trace, key=lambda entry: (entry['validation_nll'], entry['iteration_or_epoch']))
    if (state['best_index'], state['best_score']) != (best['iteration_or_epoch'], best['validation_nll']):
        raise ValueError('committed selection disagrees with trace')
    selected = load_checkpoint(directory/f"checkpoint-{state['best_index']:03d}.pt")
    if selected['index'] != state['best_index'] or selected['trace'] != trace[:state['best_index']+1]:
        raise ValueError('selected checkpoint disagrees with committed trace')
    state['checkpoint_io_s'] = resume['checkpoint_io_s']
    return resume, state


def verify_candidate(directory, spec, data_hashes=None, completion_hash=None):
    directory = Path(directory)
    if completion_hash is not None:
        verify_inventory(directory, {'completion.json': completion_hash})
    completion = read_json(directory/'completion.json')
    verify_inventory(directory, completion['hashes'])
    identity = read_json(directory/'identity.json')
    if identity['spec'] != normalized(spec.record()):
        raise ValueError('candidate spec identity changed')
    if data_hashes is not None and identity['data_hashes'] != data_hashes:
        raise ValueError('candidate data identity changed')
    resume, state = verify_resume(directory)
    record = read_json(directory/'record.json')
    if any(record[key] != value for key, value in normalized(spec.record()).items()):
        raise ValueError('completed candidate spec changed')
    if record['data_hashes'] != identity['data_hashes']:
        raise ValueError('completed candidate data changed')
    if (record['checkpoint'], record['validation_nll'], record['checkpoint_file']) != (
            state['best_index'], state['best_score'], f"checkpoint-{state['best_index']:03d}.pt"):
        raise ValueError('completed selection disagrees with committed state')
    if read_json(directory/'trace.json') != state['trace']:
        raise ValueError('completed trace disagrees with state')
    expected = {**resume['inventory'], 'trace.json': completion['hashes']['trace.json']}
    if record['artifact_hashes'] != expected:
        raise ValueError('completed inventory changed')
    verify_inventory(directory, record['artifact_hashes'])
    return record


def persist(directory, state, index):
    start = time.perf_counter()
    inventory = {}
    if (directory/'resume.json').exists():
        previous, _ = verify_resume(directory)
        inventory = previous['inventory']
    checkpoint = directory/f'checkpoint-{index:03d}.pt'
    if checkpoint.name in inventory:
        raise ValueError('cannot replace a committed checkpoint')
    temporary = checkpoint.with_suffix('.pt.tmp')
    torch.save(state, temporary); temporary.replace(checkpoint)
    inventory[checkpoint.name] = hash_file(checkpoint)
    state['checkpoint_io_s'] += time.perf_counter()-start
    write_json(directory/'resume.json', {'checkpoint': checkpoint.name, 'inventory': inventory,
                                        'identity_hash': hash_file(directory/'identity.json'),
                                        'checkpoint_io_s': state['checkpoint_io_s']})
    return checkpoint


def fit_candidate(spec: FitSpec, data: LearnerData, directory: Path, *, cap=None, interrupt_after=None):
    with Ownership(Path(directory)/'.candidate.lock'):
        return _fit_candidate(spec, data, Path(directory), cap=cap, interrupt_after=interrupt_after)


def _fit_candidate(spec: FitSpec, data: LearnerData, directory: Path, *, cap=None, interrupt_after=None):
    """cap and interrupt_after are test seams; production workers never supply them."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = {'spec': spec.record(), 'data_hashes': data.hashes}
    identity = normalized(identity)
    if (directory/'identity.json').exists():
        if read_json(directory/'identity.json') != identity:
            raise ValueError('candidate identity changed')
    else:
        write_json(directory/'identity.json', identity)
    if (directory/'completion.json').exists():
        return verify_candidate(directory, spec, data.hashes)
    # A record without the completion pointer is an uncommitted final write.
    library_overhead = peak_rss_bytes()
    model = None; opt = None
    if spec.family != 'hmm':
        model = build(spec.family, seed_word(spec.seed(0)))
        expected = 280 if spec.family == 'rnn' else 712
        if count_parameters(model) != expected:
            raise ArithmeticError('parameter count mismatch')
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATES[spec.candidate],
                               betas=(.9, .999), eps=1e-8, weight_decay=0)
    state = {'index': 0, 'best_score': float('inf'), 'best_index': 0, 'progress_score': float('inf'),
             'stale': 0, 'trace': [], 'training_elapsed_s': 0., 'validation_elapsed_s': 0.,
             'checkpoint_io_s': 0., 'token_exposures': 0, 'optimizer_steps': 0,
             'peak_process_rss_bytes': library_overhead, 'library_overhead_rss_bytes': library_overhead}
    if (directory/'resume.json').exists():
        resume, state = verify_resume(directory)
        if model is not None:
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
    elif spec.family == 'hmm':
        r = rng(spec.seed(0))
        state['pi'] = torch.from_numpy(r.dirichlet(np.ones(4)))
        state['A'] = torch.from_numpy(r.dirichlet(np.ones(4), size=4))
        state['B'] = torch.from_numpy(r.dirichlet(np.ones(8), size=4))

    def arrays():
        return state['A'].numpy(), state['B'].numpy(), state['pi'].numpy()

    def measure(index):
        start = time.perf_counter()
        if spec.family == 'hmm':
            a, b, p = arrays()
            train_score = nll(hmm_filter(data.train, a, b, p), data.train)
            objective = float(-train_score*data.train.size+.001*(np.log(p).sum()+np.log(a).sum()+np.log(b).sum()))
            row_residual = validate_hmm(a, b, p)
        else:
            train_score = sampled_nll(log_predict(model, data.train), data.train)
            objective = None; row_residual = None
        state['training_elapsed_s'] += time.perf_counter()-start
        start = time.perf_counter()
        val_score = nll(hmm_filter(data.validation, *arrays()), data.validation) if spec.family == 'hmm' else sampled_nll(log_predict(model, data.validation), data.validation)
        state['validation_elapsed_s'] += time.perf_counter()-start
        if not np.isfinite(train_score) or not np.isfinite(val_score):
            raise ArithmeticError('nonfinite fit score')
        state['trace'].append({'iteration_or_epoch': index, 'training_nll': train_score,
                               'validation_nll': val_score, 'map_objective': objective,
                               'row_normalization_residual': row_residual})
        if val_score < state['best_score']:
            state['best_score'] = val_score; state['best_index'] = index
        if model is not None:
            state['model'] = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            state['optimizer'] = opt.state_dict()
        state['peak_process_rss_bytes'] = max(state['peak_process_rss_bytes'], peak_rss_bytes())

    if not state['trace']:
        measure(0)
        state['progress_score'] = state['best_score']
        persist(directory, state, 0)
    limit = (50 if spec.family == 'hmm' else 80) if cap is None else cap
    inputs = torch.from_numpy(shifted(data.train)) if model is not None else None
    targets = torch.tensor(data.train.copy(), dtype=torch.long) if model is not None else None
    for index in range(state['index']+1, limit+1):
        if (spec.family == 'hmm' and state['stale'] >= 3) or (spec.family != 'hmm' and state['index'] >= 20 and state['stale'] >= 10):
            break
        start = time.perf_counter()
        if spec.family == 'hmm':
            p, a, b = em_step(data.train, *arrays())
            state['pi'] = torch.from_numpy(p); state['A'] = torch.from_numpy(a); state['B'] = torch.from_numpy(b)
            state['token_exposures'] += data.train.size
        else:
            model.train()
            order = rng(spec.seed(1, index)).permutation(len(data.train))
            batch_size = min(32, len(data.train))
            for first in range(0, len(order), batch_size):
                ix = torch.from_numpy(order[first:first+batch_size])
                opt.zero_grad(set_to_none=True)
                logits = model(inputs[ix])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 8), targets[ix].reshape(-1))
                if not torch.isfinite(loss):
                    raise ArithmeticError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not torch.isfinite(norm):
                    raise ArithmeticError('nonfinite gradient norm')
                opt.step(); state['optimizer_steps'] += 1
                state['token_exposures'] += len(ix)*data.train.shape[1]
        state['training_elapsed_s'] += time.perf_counter()-start
        state['index'] = index; measure(index)
        if spec.family == 'hmm':
            previous, current = [entry['map_objective'] for entry in state['trace'][-2:]]
            if current < previous-1e-8*max(1., abs(previous)):
                raise ArithmeticError('MAP objective decreased')
            improvement = (current-previous)/data.train.size
            state['stale'] = state['stale']+1 if improvement < 1e-6 else 0
        else:
            score = state['trace'][-1]['validation_nll']
            if state['progress_score']-score >= 1e-4:
                state['progress_score'] = score; state['stale'] = 0
            else:
                state['stale'] += 1
        persist(directory, state, index)
        if interrupt_after is not None and index >= interrupt_after:
            raise InterruptedError('test interruption after committed checkpoint')
    resume, committed = verify_resume(directory)
    write_json(directory/'trace.json', committed['trace'])
    chosen = directory/f"checkpoint-{state['best_index']:03d}.pt"
    parameter_bytes = 52*8 if model is None else tensor_bytes(list(model.parameters()))
    record = {**spec.record(), 'checkpoint': state['best_index'], 'checkpoint_file': chosen.name,
              'validation_nll': state['best_score'], 'iterations_or_epochs': state['index'],
              'optimizer_steps': state['optimizer_steps'], 'em_iterations': state['index'] if model is None else 0,
              'token_exposures': state['token_exposures'], 'validation_evaluations': len(state['trace']),
              'training_tokens': data.train.size, 'validation_tokens': data.validation.size,
              'total_accessible_tokens': data.train.size+data.validation.size, 'data_hashes': data.hashes,
              'training_elapsed_s': state['training_elapsed_s'], 'validation_elapsed_s': state['validation_elapsed_s'],
              'checkpoint_io_s': state['checkpoint_io_s'], 'worker_pid': os.getpid(),
              'peak_process_rss_bytes': max(state['peak_process_rss_bytes'], peak_rss_bytes()),
              'library_overhead_rss_bytes': state['library_overhead_rss_bytes'],
              'parameter_count': 52 if model is None else count_parameters(model),
              'free_parameters': 43 if model is None else count_parameters(model),
              'parameter_tensor_bytes': parameter_bytes, 'optimizer_tensor_bytes': 0 if model is None else tensor_bytes(opt.state_dict()),
              'training_diagnostics_token_passes': len(state['trace'])*data.train.size,
              'artifact_hashes': dict(resume['inventory'])}
    record['artifact_hashes']['trace.json'] = hash_file(directory/'trace.json')
    write_json(directory/'record.json', record)
    write_json(directory/'completion.json', {'hashes': {name: hash_file(directory/name) for name in
               ('record.json', 'identity.json', 'resume.json', 'trace.json')}})
    return verify_candidate(directory, spec, data.hashes)


def sampled_nll(log_probabilities, tokens):
    return float(-np.take_along_axis(log_probabilities, tokens[..., None], -1).mean())


def worker(job_path):
    import json
    job_path = Path(job_path)
    if hash_file(job_path) != os.environ['BIGBRANE_JOB_HASH']:
        raise ValueError('worker job changed after dispatch')
    job = read_json(job_path)
    directory = Path(job['candidate_directory'])
    out = directory.parents[1]
    if ROOT != out/'frozen':
        raise ValueError('worker is not executing the frozen source tree')
    fd = int(os.environ['BIGBRANE_RUN_FD'])
    inherited = os.fstat(fd); expected = os.stat(out/'.run.lock')
    if (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError('worker does not hold the inherited run ownership')
    owner = {'pid': os.getpid(), 'token': os.environ['BIGBRANE_WORKER_TOKEN'],
             'driver_token': os.environ['BIGBRANE_DRIVER_TOKEN'], 'source_directory': str(ROOT)}
    write_json(directory/'worker-owner.json', owner)
    verify_sources(out, json.loads(os.environ['BIGBRANE_SOURCE_HASHES']))
    spec = FitSpec(**job['spec'])
    verify_inventory(job['learner_directory'], job['data_hashes'])
    data = LearnerData.load(job['learner_directory'], spec.size_index)
    if data.hashes != job['data_hashes']:
        raise ValueError('learner input hashes changed')
    fit_candidate(spec, data, directory)
