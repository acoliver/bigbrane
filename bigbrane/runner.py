"""Sequential, resumable orchestration with evaluator-only artifacts isolated from workers."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import numpy as np
from .artifacts import (ROOT, Ownership, environment, freeze_sources, hash_file, process_identity,
                        read_json, save_array, verify_inventory, verify_provenance, verify_sources, write_json)
from .core import L, generator, sample
from .decisions import matched_report, stress_report
from .evaluation import ANALYTICAL_ESTIMATES, evaluate_selected
from .protocol import FitSpec, grid, selection_key
from .fitting import verify_candidate


def preflight(out):
    command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
    log = out/'preflight.log'
    with log.open('w') as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return {'command': command, 'returncode': completed.returncode, 'log_hash': hash_file(log)}


def prepare_stage(out, root, stage, pilot):
    registry = out/f'data-stage-{stage}.json'
    if registry.exists():
        record = read_json(registry); verify_inventory(out, record['hashes'])
        return record
    hashes = {}; seeds = []
    for regime in (range(1) if pilot else range(2)):
        for g in (range(1) if pilot else range(4)):
            a, b, p, s, permutation = generator(root, regime, g)
            base = out/'data'/f's{stage}-r{regime}-g{g}'
            evaluator = base/'evaluator'; evaluator.mkdir(parents=True, exist_ok=True)
            np.savez(evaluator/'generator.npz', A=a, B=b, pi=p, S=s, symbol_permutation=permutation)
            seeds.append({'role': 'generator', 'regime': regime, 'generator': g, 'seed': (root, 0, regime, g)})
            for split, name, count in ((0, 'train', 256), (1, 'validation', 32), (2, 'test', 128)):
                seed = (root, 2 if stage else 1, regime, g, split)
                tokens, latent = sample(a, b, p, count, L, seed, stress=bool(stage))
                location = base/('evaluator' if split == 2 else 'learner')/f'{name}.npy'
                save_array(location, tokens)
                save_array(evaluator/f'{name}-latent.npy', latent)
                seeds.append({'role': name, 'regime': regime, 'generator': g, 'seed': seed,
                              'shape': tokens.shape, 'observed_tokens': tokens.size, 'path': str(location.relative_to(out))})
            for path in base.rglob('*'):
                if path.is_file():
                    hashes[str(path.relative_to(out))] = hash_file(path)
            if stage == 0 and not pilot:
                for length in (16, 32, 64, 128, 256):
                    seed = (root, 4, regime, g, length)
                    tokens, _ = sample(a, b, p, 32, length, seed)
                    path = out/'scaling'/f'r{regime}-g{g}-l{length}.npy'
                    save_array(path, tokens); hashes[str(path.relative_to(out))] = hash_file(path)
                    seeds.append({'role': 'scaling-evaluator-only', 'seed': seed, 'shape': tokens.shape,
                                  'path': str(path.relative_to(out))})
    record = {'stage': stage, 'hashes': hashes, 'seeds': seeds}
    write_json(registry, record)
    return read_json(registry)


def initialize(out, root, pilot, resume):
    out = Path(out).resolve()
    if (pilot and root != 20260931) or (not pilot and root != 20260930):
        raise ValueError('version-one roots are 20260931 for pilot and 20260930 for confirmatory study')
    existing = (out/'manifest.json').exists()
    if existing and not resume:
        raise FileExistsError('run already exists; use --resume')
    if resume and not existing:
        raise FileNotFoundError('no frozen manifest to resume')
    out.mkdir(parents=True, exist_ok=True)
    env = environment()
    if existing:
        manifest = read_json(out/'manifest.json')
        if (manifest['root'], manifest['pilot']) != (root, pilot):
            raise ValueError('resume root or stage mode mismatch')
        verify_sources(out, manifest['source_hashes'])
        if env != manifest['environment']:
            raise ValueError('environment changed since freeze')
        verify_inventory(out, {'frozen/environment.json': manifest['environment_hash']})
        for stage_record in manifest['data_stages'].values():
            verify_inventory(out, stage_record['hashes'])
        return out, manifest
    correctness = preflight(out)
    write_json(out/'frozen'/'environment.json', env)
    manifest = {'protocol_version': 1, 'root': root, 'pilot': pilot,
                'environment_hash': hash_file(out/'frozen'/'environment.json'),
                'excluded_from_scientific_outcomes': pilot, 'status': 'frozen',
                'source_hashes': freeze_sources(out), 'environment': env, 'preflight': correctness,
                'planned_candidate_counts': {'matched': 3 if pilot else 384, 'conditional_stress': 0 if pilot else 192},
                'candidate_plans': {'matched': [s.record() for s in grid(root, 0, pilot)],
                                    'conditional_stress': [] if pilot else [s.record() for s in grid(root, 1)]},
                'data_stages': {}, 'attempts': [], 'completed_candidates': [],
                'candidate_commits': {}, 'selections': {},
                'analytical_resource_estimates': ANALYTICAL_ESTIMATES,
                'resource_measurement_notes': [
                    'Candidate RSS is measured in independent sequential subprocesses and includes runtime/library overhead.',
                    'Library overhead RSS is a separately measured high-water mark before model construction, not tensor memory.',
                    'Tensor parameter and optimizer storage are counted separately from process RSS.',
                    'Training timing includes training-set objective diagnostics; validation and checkpoint I/O are separate.',
                    'Optimizer/EM token exposures exclude separately recorded diagnostic passes and validation calls.',
                    'Interrupted partial epochs may consume uncommitted work; attempts retain measured process wall time and explicitly flag unknown counters.',
                    'A driver killed before updating an attempt cannot recover its elapsed time; that attempt is retained as unknown, not omitted.'],
                'scientific_limitations': ['known four-state prior', 'two jointly varying generator regimes',
                                           'eight generators', 'fixed small neural widths and search budgets',
                                           'validation-assisted learning curves', 'descriptive bootstrap uncertainty',
                                           'optimization adequacy unestablished', 'no retrieval or composition claim']}
    write_json(out/'manifest.json', manifest)
    return out, manifest


def verify_boundary(out, manifest, *, candidates=False):
    verify_provenance(out, manifest)
    if not candidates:
        return
    plans = {plan['key']: plan for stage in manifest['candidate_plans'].values() for plan in stage}
    for key, digest in manifest['candidate_commits'].items():
        directory = out/'candidates'/key
        plan = plans[key]
        spec = FitSpec(**{name: plan[name] for name in FitSpec.__dataclass_fields__})
        verify_candidate(directory, spec, learner_hashes(out, manifest, spec)[1], digest)


def learner_hashes(out, manifest, spec):
    learner = out/'data'/f's{spec.stage}-r{spec.regime}-g{spec.generator}'/'learner'
    inventory = manifest['data_stages'][str(spec.stage)]['hashes']
    return learner, {name: inventory[str((learner/name).relative_to(out))]
                     for name in ('train.npy', 'validation.npy')}


def commit_candidate(out, manifest, spec, hashes):
    directory = out/'candidates'/spec.key
    commits = manifest.setdefault('candidate_commits', {})
    record = verify_candidate(directory, spec, hashes, commits.get(spec.key))
    digest = hash_file(directory/'completion.json')
    commits[spec.key] = digest
    if spec.key not in manifest['completed_candidates']:
        manifest['completed_candidates'].append(spec.key)
    write_json(out/'manifest.json', manifest)
    return record


def execute_candidate(out, manifest, spec, *, ownership=None):
    out = Path(out).resolve()
    if ownership is None:
        with Ownership(out/'.run.lock') as owner:
            # Standalone candidate dispatch freezes sources and uses the prepared registry.
            if 'protocol_version' not in manifest and 'source_hashes' not in manifest:
                manifest['source_hashes'] = freeze_sources(out)
                manifest['data_stages'] = {str(spec.stage): read_json(out/f'data-stage-{spec.stage}.json')}
            return _execute_candidate(out, manifest, spec, owner)
    return _execute_candidate(out, manifest, spec, ownership)


def _execute_candidate(out, manifest, spec, ownership):
    verify_boundary(out, manifest)
    directory = out/'candidates'/spec.key
    learner, hashes = learner_hashes(out, manifest, spec)
    if (directory/'completion.json').exists():
        record = commit_candidate(out, manifest, spec, hashes)
        for attempt in manifest['attempts']:
            if attempt['candidate'] == spec.key and attempt['status'] == 'running':
                attempt.update({'status': 'complete-recovered', 'elapsed_s': None,
                                'uncommitted_work_counters': 'none; driver elapsed time unavailable'})
        write_json(out/'manifest.json', manifest)
        return record
    directory.mkdir(parents=True, exist_ok=True)
    job = {'spec': asdict(spec), 'learner_directory': str(learner), 'data_hashes': hashes,
           'candidate_directory': str(directory)}
    write_json(directory/'job.json', job)
    for attempt in manifest['attempts']:
        if attempt['status'] == 'running':
            attempt['status'] = 'interrupted-driver'; attempt['elapsed_s'] = None
            attempt['uncommitted_work_counters'] = 'unknown'
    command = [sys.executable, '-m', 'bigbrane.cli', '--worker', str(directory/'job.json')]
    driver = process_identity()
    attempt = {'candidate': spec.key, 'command': command, 'status': 'running', 'driver_identity': driver,
               'log': str((directory/'worker.log').relative_to(out))}
    manifest['attempts'].append(attempt)
    write_json(out/'manifest.json', manifest)
    worker_token = process_identity()['token']
    worker_env = dict(os.environ, BIGBRANE_RUN_FD=str(ownership.fd), BIGBRANE_DRIVER_TOKEN=driver['token'],
                      BIGBRANE_WORKER_TOKEN=worker_token, BIGBRANE_JOB_HASH=hash_file(directory/'job.json'),
                      BIGBRANE_SOURCE_HASHES=json.dumps(manifest['source_hashes']))
    start = time.perf_counter()
    with (directory/'worker.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=out/'frozen', stdout=stream, stderr=subprocess.STDOUT,
                                   env=worker_env, pass_fds=(ownership.fd,))
        attempt['worker_identity'] = {'pid': process.pid, 'token': worker_token}
        write_json(out/'manifest.json', manifest)
        returncode = process.wait()
    attempt.update({'elapsed_s': time.perf_counter()-start, 'returncode': returncode,
                    'status': 'complete' if returncode == 0 else 'failed',
                    'uncommitted_work_counters': 'none' if returncode == 0 else 'unknown'})
    if returncode:
        manifest['status'] = 'candidate-failed'
        write_json(out/'manifest.json', manifest)
        raise subprocess.CalledProcessError(returncode, command)
    verify_boundary(out, manifest)
    return commit_candidate(out, manifest, spec, hashes)


def freeze_selections(out, stage, specs, records, *, manifest=None):
    if manifest is not None:
        verify_boundary(out, manifest)
    if len(specs) != len(records):
        raise ValueError('candidate selection inputs are incomplete')
    selections = []
    groups = {}
    for spec, record in zip(specs, records):
        directory = out/'candidates'/spec.key
        digest = None if manifest is None else manifest['candidate_commits'][spec.key]
        hashes = None if manifest is None else learner_hashes(out, manifest, spec)[1]
        trusted = verify_candidate(directory, spec, hashes, digest)
        if record != trusted:
            raise ValueError('selection input disagrees with committed record')
        coordinate = (spec.regime, spec.generator, spec.size_index, spec.family, spec.replica)
        groups.setdefault(coordinate, []).append((spec, record))
    for candidates in groups.values():
        chosen_spec, chosen = min(candidates, key=lambda pair: selection_key(pair[1]))
        checkpoint = out/'candidates'/chosen_spec.key/chosen['checkpoint_file']
        selections.append({'spec': asdict(chosen_spec), 'validation_nll': chosen['validation_nll'],
                           'checkpoint_index': chosen['checkpoint'],
                           'checkpoint_path': str(checkpoint.relative_to(out)),
                           'checkpoint_hash': chosen['artifact_hashes'][chosen['checkpoint_file']],
                           'candidate_keys': [spec.key for spec, _ in candidates],
                           'selection_cost': aggregate_cost([r for _, r in candidates]),
                           'retained_candidate_cost': aggregate_cost([chosen])})
    path = out/f'selections-stage-{stage}.json'
    if path.exists():
        if read_json(path) != selections:
            raise ValueError('frozen selection changed')
    else:
        write_json(path, selections)
    return selections


def aggregate_cost(records):
    summed = ('training_elapsed_s', 'validation_elapsed_s', 'checkpoint_io_s',
              'optimizer_steps', 'em_iterations', 'token_exposures', 'validation_evaluations',
              'training_diagnostics_token_passes')
    result = {key: sum(record[key] for record in records) for key in summed}
    result.update({'candidate_fits': len(records),
                   'peak_process_rss_bytes': max(record['peak_process_rss_bytes'] for record in records),
                   'parameter_tensor_bytes_per_candidate': [r['parameter_tensor_bytes'] for r in records],
                   'optimizer_tensor_bytes_per_candidate': [r['optimizer_tensor_bytes'] for r in records]})
    return result


def run_stage(out, manifest, stage, ownership):
    verify_boundary(out, manifest)
    pilot = manifest['pilot']; root = manifest['root']
    prepared = prepare_stage(out, root, stage, pilot)
    if str(stage) in manifest['data_stages'] and prepared != manifest['data_stages'][str(stage)]:
        raise ValueError('prepared stage registry changed')
    manifest['data_stages'][str(stage)] = prepared
    manifest['status'] = f'fitting-stage-{stage}'
    write_json(out/'manifest.json', manifest)
    specs = grid(root, stage, pilot)
    records = [execute_candidate(out, manifest, spec, ownership=ownership) for spec in specs]
    selections = freeze_selections(out, stage, specs, records, manifest=manifest)
    # All checkpoints in this stage are selected and frozen before any test/oracle access.
    manifest['selections'][str(stage)] = {'path': f'selections-stage-{stage}.json',
                                        'hash': hash_file(out/f'selections-stage-{stage}.json')}
    manifest['status'] = f'evaluating-stage-{stage}'
    write_json(out/'manifest.json', manifest)
    evaluations = [evaluate_selected(out, selection, manifest=manifest,
                                    scaling=(not pilot and stage == 0 and selection['spec']['size_index'] == 3 and selection['spec']['replica'] == 0))
                   for selection in selections]
    return evaluations, records


def run(root, pilot, out, resume=False, prepare_only=False):
    out = Path(out).resolve()
    with Ownership(out/'.run.lock') as ownership:
        return _run(root, pilot, out, resume, prepare_only, ownership)


def _run(root, pilot, out, resume, prepare_only, ownership):
    out, manifest = initialize(out, root, pilot, resume)
    verify_boundary(out, manifest)
    if manifest['status'] == 'complete':
        verify_inventory(out, manifest['result_hashes'])
        return read_json(out/'report.json')
    if prepare_only:
        manifest['data_stages']['0'] = prepare_stage(out, root, 0, pilot)
        verify_boundary(out, manifest)
        write_json(out/'manifest.json', manifest)
        return {'status': 'prepared; no candidates trained or test outcomes evaluated'}
    matched, records = run_stage(out, manifest, 0, ownership)
    if pilot:
        report = {'verdict': 'Pilot completed. These results are a pipeline sanity check only.',
                  'excluded_from_scientific_outcomes': True,
                  'pilot_metrics': {row['selection']['spec']['family']: row['metrics']['all'] for row in matched},
                  'stress_stage': 'not applicable to pilot'}
    else:
        report = matched_report(matched, root)
        if report['matched_advantage_passed']:
            stress, stress_records = run_stage(out, manifest, 1, ownership)
            records += stress_records
            report['stress_results'] = stress_report(matched, stress, root)
    report['resource_cost_all_candidates'] = aggregate_cost(records)
    report['resource_cost_by_family'] = {family: aggregate_cost([r for r in records if r['family'] == family])
                                        for family in ('hmm', 'rnn', 'attention')}
    report['process_attempts'] = manifest['attempts']
    report['analytical_resource_estimates'] = ANALYTICAL_ESTIMATES
    report['scientific_limitations'] = manifest['scientific_limitations']
    report['invalid_or_missing_candidates'] = []
    verify_boundary(out, manifest, candidates=True)
    write_json(out/'report.json', report)
    manifest['status'] = 'complete'
    manifest['result_hashes'] = {str(p.relative_to(out)): hash_file(p) for folder in ('candidates', 'evaluations')
                               for p in (out/folder).rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
    manifest['result_hashes'].update({p.name: hash_file(p) for p in out.glob('selections-stage-*.json')})
    manifest['result_hashes']['report.json'] = hash_file(out/'report.json')
    write_json(out/'manifest.json', manifest)
    print(report['verdict'], flush=True)
    return report
