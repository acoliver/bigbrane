"""Observe CARD calls in byte-identical complete executor programs, without rewriting them."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results-wt/evidence-recovery'
FIXTURES = ROOT / 'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-14429dbc', 'card-68612a2a', 'card-8dc4db49')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(value):
    def plain(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.loads(json.dumps(value, default=plain))


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def capture(card, source, filename):
    episodes = []
    current = None
    lines = source.splitlines()
    pending = {}

    def table_state(v):
        return snapshot({'table': [[*key, *val] for key, val in sorted(v['table'].items())],
                         **{k: v[k] for k in ('t', 'w', 'a', 'x', 'y', 'evictions', 'gated_out')}})

    def observer(frame, event, arg):
        nonlocal current
        if frame.f_code.co_filename != filename:
            return None
        name, v = frame.f_code.co_name, frame.f_locals
        if card == 'card-14429dbc':
            if name == 'run' and event == 'call':
                current = {'parameters': snapshot({k: v[k] for k in ('d', 'B', 'extra_probes')}),
                           'inputs': snapshot(v['stream']), 'states': [], 'card_outputs': []}
                episodes.append(current)
            elif name == 'step' and type(v['s']).__name__ == 'Card' and event == 'return':
                current['states'].append(snapshot(vars(v['s'])))
                current['card_outputs'].append([])
            elif name == 'read' and type(v['s']).__name__ == 'Card' and event == 'return':
                current['card_outputs'][-1].append({'x': snapshot(v['x']), 'p': snapshot(arg)})
            return observer if name in ('run', 'step', 'read') else None
        if card == 'card-68612a2a':
            if name == 'step' and type(v['self']).__name__ == 'Card':
                if event == 'call':
                    if current is None:
                        parameters = {k: frame.f_globals[k] for k in ('m', 'd', 's', 'U', 'V', 'eps', 'tau', 'beta', 'C', 'E', 'vals')}
                        current = {'parameters': snapshot(parameters), 'inputs': [], 'states': [], 'card_outputs': [], 'pursuits': []}
                        episodes.append(current)
                    current['inputs'].append(snapshot([v['z'], v['X'], v['y']]))
                    current['pursuits'].append({'iterations': []})
                elif event == 'return':
                    current['states'].append(snapshot(vars(v['self'])))
                    current['card_outputs'].append(snapshot(arg))
                return observer
            if name == 'pursuit':
                if id(frame) in pending:
                    current['pursuits'][-1]['iterations'].append(snapshot({k: v[k] for k in ('S', 'alpha', 'r')}))
                    del pending[id(frame)]
                if event == 'line' and lines[frame.f_lineno-1].strip() == 'r = k - KS @ alpha':
                    pending[id(frame)] = True
                if event == 'return':
                    current['pursuits'][-1]['result'] = snapshot(arg)
                return observer
            return None
        if name != 'card_run':
            return None
        if id(frame) in pending:
            current['sweeps'][-1].append(snapshot({k: v[k] for k in ('x', 'y')}))
            del pending[id(frame)]
        if event == 'call':
            p = {k: frame.f_globals[k] for k in ('V', 'r', 'Pmax', 'J', 'kappa', 'E_raw', 'F_raw', 'E', 'F', 'beta', 'l', 'u', 'pairs')}
            current = {'parameters': snapshot(p), 'inputs': snapshot(v['seq']), 'states': [], 'card_outputs': [], 'sweeps': []}
            episodes.append(current)
        elif event == 'line':
            line = lines[frame.f_lineno-1].strip()
            if line == 'for _ in range(J):' and len(current['sweeps']) < v['t']:
                current['sweeps'].append([])
            if line == 'x = a / (1.0 + Ky)':
                pending[id(frame)] = True
            if line == 'outs.append(Kq * x[q] * y + x[q] * u)':
                current['states'].append(table_state(v))
        elif event == 'return':
            current['card_outputs'], current['evictions'], current['gated_out'] = snapshot(arg)
        return observer

    stdout = io.StringIO()
    previous = sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, filename, 'exec'), {'__name__': '__untouched_original__'})
    finally:
        sys.settrace(previous)
    assert episodes and not pending
    for e in episodes:
        assert len(e['inputs']) == len(e['states']) == len(e['card_outputs'])
    return {'episodes': episodes, 'reexecuted_stdout': stdout.getvalue()}


def main():
    run = ROOT.parent / 'bigbrane/runs/run-20261010T181300Z'
    entries = {e['card_id']: e for e in json.loads((ROOT/'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json').read_text())['cards']}
    events = [json.loads(line) for line in (run/'ledger.jsonl').read_text().splitlines()]
    records = {}
    specimens = {}
    for card in CARDS:
        selected = []
        receipts = {}
        for event in events:
            d = event['data']
            if event['type'] != 'artifact' or d.get('card_id') != card:
                continue
            if d.get('kind') in ('card', 'critic', 'reduction_critic'):
                raw = (run/d['path']).read_bytes()
                assert sha(raw) == d['sha256']
                receipts[d['kind']] = {'path': str(run/d['path']), 'sha256': sha(raw), 'ledger_event': event}
            if d.get('kind') == 'executor':
                raw = (run/d['path']).read_bytes(); a = json.loads(raw)
                if a.get('purpose') == 'reduction_check':
                    assert sha(raw) == d['sha256']
                    selected.append((event, a))
        for kind in ('card_version', 'reduction_critic'):
            candidates = [e for e in events if e['data'].get('card_id') == card and
                          (e['type'] == kind if kind == 'card_version' else
                           e['type'] == 'artifact' and e['data'].get('kind') == kind)]
            event = candidates[-1]
            raw = (run/event['data']['path']).read_bytes()
            assert sha(raw) == event['data']['sha256']
            obj = json.loads(raw)
            if kind == 'card_version':
                assert {k:v for k,v in obj.items() if k != 'status'} == entries[card]['card']
            else:
                assert all(obj[k] == value for k,value in entries[card]['critic'].items())
            receipts[kind] = {'path': str(run/event['data']['path']), 'sha256': sha(raw), 'ledger_event': event}
        assert len(selected) == 1
        event, artifact = selected[0]
        raw = artifact['files']['main.py'].encode()
        original = run/'jobs'/artifact['job_id']/'in/main.py'
        assert original.read_bytes() == raw
        dest = FIXTURES/'programs'/f'composites-{card}.py'
        dest.write_bytes(raw)
        record = capture(card, raw.decode(), str(dest))
        record['provenance'] = {
            'source': str(dest.relative_to(ROOT)), 'source_sha256': sha(raw),
            'original_source': str(original), 'artifact': str(run/event['data']['path']),
            'artifact_sha256': event['data']['sha256'], 'ledger_event': event,
            'purpose': artifact['purpose'], 'observer_sha256': sha(Path(__file__).read_bytes()),
            'specimen': {f'{k}_sha256': digest(entries[card][k]) for k in ('card', 'critic')},
            'trace_sha256': digest(record['episodes']),
            'draws_sha256': digest([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in record['episodes']]),
            'capture_insertions': [],
            'draw_limit': 'Historical raw draws and process hash seed were not saved. PYTHONHASHSEED=0 source reexecution is not recovered historical bytes; seeds and source bytes are unchanged. These programs do not draw from set-to-list conversions.'}
        record['original_stdout'] = artifact['stdout']
        records[card] = record
        specimens[card] = {'card': entries[card]['card'], 'critic': entries[card]['critic'], 'original_receipts': receipts}
        print(card, len(record['episodes']), sum(len(e['inputs']) for e in record['episodes']), flush=True)
    raw = (json.dumps(records, separators=(',', ':'))+'\n').encode()
    (REPORT/'composites-full-capture.json').write_bytes(raw)
    manifest = {'capture_sha256': sha(raw), 'invocation': {'python': sys.executable, 'numpy': np.__version__, 'cwd': str(Path.cwd()), 'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED')},
                'cards': {c: {k:v for k,v in rec.items() if k != 'episodes'} for c,rec in records.items()}}
    (FIXTURES/'composites_sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (FIXTURES/'composites_specimens.json').write_text(json.dumps(specimens, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
