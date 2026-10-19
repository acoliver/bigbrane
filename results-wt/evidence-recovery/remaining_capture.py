"""Observe complete untouched historical programs; never use their comparator outputs as evidence."""
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results-wt/evidence-recovery'
FIXTURES = ROOT / 'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-2c5dd003', 'card-6fc7efec', 'card-dc69816b')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if callable(x):
        return x.__name__
    raise TypeError(type(x).__name__)


def snapshot(x):
    return json.loads(json.dumps(x, default=plain))


def digest(x):
    return sha(json.dumps(x, sort_keys=True, separators=(',', ':')).encode())


def capture(card, source, filename):
    episodes, active, isotopes, soft = [], {}, [], []
    lines = source.splitlines()
    tensor = None

    def observer(frame, event, arg):
        nonlocal tensor
        if frame.f_code.co_filename != filename:
            return None
        name, v = frame.f_code.co_name, frame.f_locals
        if card == 'card-6fc7efec':
            if name == 'make_isotope':
                if event == 'return':
                    isotopes.append(snapshot(dict(zip(arg.__code__.co_freevars,
                                                     (c.cell_contents for c in arg.__closure__)))))
                return observer
            if name == 'step' and type(v['self']).__name__ == 'Card':
                obj = v['self']
                if event == 'call':
                    if tensor is None:
                        tensor = {'parameters': {'p': frame.f_globals['p']}, 'inputs': [],
                                  'card_outputs': [], 'states': [], 'initial_state': snapshot(vars(obj))}
                        episodes.append(tensor)
                    tensor['inputs'].append(snapshot(v['x']))
                elif event == 'return':
                    tensor['states'].append(snapshot(vars(obj)))
                    tensor['card_outputs'].append(snapshot(arg))
                return observer
            if name == 'contract' and frame.f_back.f_code.co_name == '<module>':
                if event == 'return':
                    soft.append({'u': snapshot(v['u']), 'v': snapshot(v['v']),
                                 'Z': snapshot(v['self'].Z), 'card_output': snapshot(arg)})
                return observer
            return None
        if name != 'card_run':
            return None
        if event == 'call':
            if card == 'card-2c5dd003':
                p = {k: v[k] for k in ('T', 'pi0', 'B0')}; inputs = v['xs']
            else:
                p = v['pr']; inputs = v['seq']
            e = {'parameters': snapshot(p), 'inputs': snapshot(inputs), 'states': [], 'card_outputs': []}
            episodes.append(e); active[id(frame)] = e
        elif event == 'line':
            line = lines[frame.f_lineno - 1].strip(); e = active[id(frame)]
            if card == 'card-2c5dd003' and line.startswith('ps.append(p.copy())'):
                e['states'].append(snapshot({k: v[k] for k in ('t', 'x', 'C', 'R', 'M', 'Phi', 'q', 'p')}))
            elif card == 'card-dc69816b' and (line == 'for t in range(n):' or line.startswith('return np.array(hs)')):
                if 't' in v and len(e['states']) == v['t']:
                    e['states'].append(snapshot({k: v[k] for k in ('t', 'H', 'u', 'z', 'kappa', 'h', 'ev')}))
        elif event == 'return':
            e = active.pop(id(frame))
            if card == 'card-2c5dd003':
                ps, hs, ss, pn = arg
                e['card_outputs'] = snapshot(ps)
                e['filtered'] = snapshot(hs); e['lag_one_credits'] = snapshot(ss); e['next_output'] = snapshot(pn)
            else:
                e['hidden'], e['card_outputs'], e['events'] = snapshot(arg)
        return observer

    stdout = io.StringIO(); previous = sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, filename, 'exec'), {'__name__': '__untouched_original__'})
    finally:
        sys.settrace(previous)
    assert episodes and not active
    for e in episodes:
        assert len(e['inputs']) == len(e['states']) == len(e['card_outputs'])
    return {'episodes': episodes, 'soft_contracts': soft, 'isotope_draws': isotopes,
            'reexecuted_stdout': stdout.getvalue()}


def main():
    run = ROOT.parent / 'bigbrane/runs/run-20261010T181300Z'
    entries = {e['card_id']: e for e in json.loads((ROOT / 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json').read_text())['cards']}
    events = [json.loads(line) for line in (run / 'ledger.jsonl').read_text().splitlines()]
    result = {}
    for card in CARDS:
        found = []
        for event in events:
            d = event['data']
            if event['type'] == 'artifact' and d.get('card_id') == card and d.get('kind') == 'executor':
                raw = (run / d['path']).read_bytes(); a = json.loads(raw)
                if a.get('purpose') == 'reduction_check':
                    assert sha(raw) == d['sha256']; found.append((event, a))
        assert len(found) == 1
        event, artifact = found[0]; raw = artifact['files']['main.py'].encode()
        original = run / 'jobs' / artifact['job_id'] / 'in/main.py'
        assert original.read_bytes() == raw
        dest = FIXTURES / 'programs' / f'remaining-{card}.py'; dest.write_bytes(raw)
        record = capture(card, raw.decode(), str(dest))
        specimen = {f'{k}_sha256': digest(entries[card][k]) for k in ('card', 'critic')}
        record['provenance'] = {'source': str(dest.relative_to(ROOT)), 'sha256': sha(raw),
            'original_source': str(original), 'artifact': str(run / event['data']['path']),
            'artifact_sha256': event['data']['sha256'], 'ledger_event': event,
            'specimen': specimen, 'observer_sha256': sha(Path(__file__).read_bytes()),
            'capture_insertions': [], 'method': 'observer-only sys.settrace of complete untouched original program',
            'draws_sha256': digest([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]),
            'trace_sha256': digest(record['episodes']), 'soft_contracts_sha256': digest(record['soft_contracts']),
            'isotope_draws_sha256': digest(record['isotope_draws']),
            'draw_limit': 'Original process hash seed and raw historical draws were not saved. Reexecution uses PYTHONHASHSEED=0. These three programs use seeded NumPy generators and no set-to-list random sampling; traces are source reexecution, not recovered historical bytes.'}
        record['original_stdout'] = artifact['stdout']; result[card] = record
        print(card, len(record['episodes']), sum(len(e['inputs']) for e in record['episodes']), flush=True)
    raw = (json.dumps(result, separators=(',', ':')) + '\n').encode()
    (REPORT / 'remaining-full-capture.json').write_bytes(raw)
    manifest = {'capture_sha256': sha(raw), 'invocation': {'python': sys.executable,
                'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'numpy': np.__version__, 'cwd': str(Path.cwd())},
                'cards': {c: {k: v for k, v in record.items() if k not in ('episodes', 'soft_contracts', 'isotope_draws')} for c, record in result.items()}}
    (FIXTURES / 'remaining_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
