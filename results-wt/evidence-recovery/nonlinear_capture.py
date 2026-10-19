"""Observe complete untouched reduction-check programs; no source insertion or draw replacement."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'results-wt/evidence-recovery'
FIXTURES = ROOT/'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-a36f1a3a', 'card-ddb071bd')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {'real': value.real.tolist(), 'imag': value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return snapshot(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    return value


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def capture(card, source, filename, namespace=None, invocation=None):
    episodes = []
    current = None
    lines = source.splitlines()
    pending = False
    step = None
    comparator = []
    art_states = []

    def observer(frame, event, arg):
        nonlocal current, pending, step, art_states
        if frame.f_code.co_filename != filename:
            return None
        name, v = frame.f_code.co_name, frame.f_locals
        if name == 'art_run' and event == 'call':
            art_states = []
        if name == 'present' and event == 'return':
            art_states.append(snapshot(vars(v['self'])))
        if name == 'art_run' and event == 'return':
            comparator.append({'outputs': snapshot(arg), 'pool_states': art_states})
        if name != 'card_run':
            return observer if name in ('art_run', 'present') else None
        if event == 'call':
            pnames = ('d', 'alpha', 'beta') if card == CARDS[0] else ('Theta', 'K', 'tau', 'beta', 'u')
            current = {'parameters': snapshot({k: v[k] for k in pnames}),
                       'inputs': snapshot(v['tokens' if card == CARDS[0] else 'seq']),
                       'states': [], 'card_outputs': []}
            episodes.append(current)
        if card == CARDS[0]:
            if pending:
                step['iterations'].append(snapshot({'G': v['G'], 'H': v['H']}))
                pending = False
            if event == 'line':
                line = lines[frame.f_lineno-1].strip()
                if line == 'if tid < 2 * d * d:':
                    step = {'iterations': []}
                if line == 'H = M * np.tanh(alpha * E + beta * G)':
                    pending = True
                if line == 'for tid in tokens:' and len(v['out']) > len(current['states']):
                    step['E'] = snapshot(v['E'])
                    current['states'].append(step)
            if event == 'return':
                current['card_outputs'] = snapshot(arg)
        else:
            if event == 'line':
                line = lines[frame.f_lineno-1].strip()
                if line == 'if t >= 1:':
                    step = {'action': 'initial'}
                    if v['t'] == 0:
                        current['embedding'] = snapshot(v['E'])
                if line == 'z = m[j] * P[j] + w':
                    step['action'] = 'merge'
                if line == 'del P[jev]':
                    step['evicted'] = snapshot({'index': v['jev'], 'P': v['P'][v['jev']], 'm': v['m'][v['jev']]})
                if line == 'P.append(w.copy())':
                    step['action'] = 'spawn'
                if line == 'outs.append(p)':
                    names = ('P', 'm', 'alpha', 's', 'lg', 'p', 'stats')
                    step.update(snapshot({k: v[k] for k in names}))
                    if v['t'] >= 1:
                        step.update(snapshot({k: v[k] for k in ('w', 'c', 'j')}))
                    if step['action'] == 'merge':
                        step.update(snapshot({k: v[k] for k in ('z', 'az')}))
                    current['states'].append(step)
            if event == 'return':
                current['card_outputs'], current['stats'] = snapshot(arg)
        return observer

    stdout = io.StringIO()
    ns = {'__name__': '__untouched_original__'} if namespace is None else namespace
    previous = sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, filename, 'exec'), ns)
            if invocation is not None:
                invocation(ns)
    finally:
        sys.settrace(previous)
    assert episodes and not pending
    for e in episodes:
        assert len(e['inputs']) == len(e['states']) == len(e['card_outputs'])
    return {'episodes': episodes, 'reexecuted_stdout': stdout.getvalue(), 'original_comparator_diagnostic': comparator}


def main():
    run = ROOT.parent/'bigbrane/runs/run-20261010T181300Z'
    entries = {e['card_id']: e for e in json.loads((FIXTURES.parent/'phase7_replay/run-20261010T181300Z.json').read_text())['cards']}
    ledger_raw = (run/'ledger.jsonl').read_bytes()
    events = [json.loads(line) for line in ledger_raw.splitlines()]
    records, specimens = {}, {}
    for card in CARDS:
        selected = []
        for event in events:
            d = event['data']
            if event['type'] == 'artifact' and d.get('card_id') == card and d.get('kind') == 'executor':
                raw = (run/d['path']).read_bytes(); artifact = json.loads(raw)
                if artifact.get('purpose') == 'reduction_check':
                    assert sha(raw) == d['sha256']
                    selected.append((event, artifact))
        assert len(selected) == 1
        event, artifact = selected[0]
        receipts = {}
        for kind in ('card_version', 'reduction_critic'):
            candidates = [e for e in events if e['data'].get('card_id') == card and
                (e['type'] == kind if kind == 'card_version' else e['type'] == 'artifact' and e['data'].get('kind') == kind)]
            receipt = candidates[-1]; raw = (run/receipt['data']['path']).read_bytes()
            assert sha(raw) == receipt['data']['sha256']
            obj = json.loads(raw)
            if kind == 'card_version':
                assert {k:v for k,v in obj.items() if k != 'status'} == entries[card]['card']
            else:
                assert all(obj[k] == value for k,value in entries[card]['critic'].items())
            receipts[kind] = {'path': str(run/receipt['data']['path']), 'sha256': sha(raw), 'ledger_event': receipt}
        raw = artifact['files']['main.py'].encode()
        original = run/'jobs'/artifact['job_id']/'in/main.py'
        assert original.read_bytes() == raw
        dest = FIXTURES/'programs'/f'nonlinear-{card}.py'
        dest.write_bytes(raw)
        record = capture(card, raw.decode(), str(dest))
        record['provenance'] = {
            'source': str(dest.relative_to(ROOT)), 'source_sha256': sha(raw), 'original_source': str(original),
            'artifact': str(run/event['data']['path']), 'artifact_sha256': event['data']['sha256'], 'ledger_event': event,
            'ledger_file': str(run/'ledger.jsonl'), 'ledger_sha256': sha(ledger_raw), 'purpose': artifact['purpose'],
            'observer_sha256': sha(Path(__file__).read_bytes()),
            'specimen': {f'{k}_sha256': digest(entries[card][k]) for k in ('card','critic')},
            'trace_sha256': digest(record['episodes']),
            'draws_sha256': digest([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]),
            'capture_insertions': [],
            'draw_limit': 'PYTHONHASHSEED=0 source reexecution, not recovered historical bytes: historical raw draws and hash seed were not saved. Program and seeded draws are untouched; no set-to-list random sampling occurs.'}
        record['original_stdout'] = artifact['stdout']
        records[card] = record
        specimens[card] = {'card': entries[card]['card'], 'critic': entries[card]['critic'], 'original_receipts': receipts}
        print(card, len(record['episodes']), sum(len(e['inputs']) for e in record['episodes']), flush=True)
    raw = (json.dumps(records, separators=(',', ':'))+'\n').encode()
    (REPORT/'nonlinear-full-capture.json').write_bytes(raw)
    manifest = {'capture_sha256': sha(raw), 'invocation': {'python': sys.executable, 'numpy': np.__version__, 'cwd': str(Path.cwd()), 'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')},
                'cards': {c: {k:v for k,v in rec.items() if k not in ('episodes','original_comparator_diagnostic')} for c,rec in records.items()}}
    (FIXTURES/'nonlinear_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (FIXTURES/'nonlinear_specimens.json').write_text(json.dumps(specimens,separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
