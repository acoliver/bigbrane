"""Observer-only execution of three original reduction-check programs."""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'results-wt/evidence-recovery'
FIXTURES = ROOT / 'hybrid/tests/research/fixtures/reference_replays'
CARDS = ('card-7ec5e5c2', 'card-bc3935ac', 'card-d1d383d5')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def snapshot(x):
    return json.loads(json.dumps(x, default=plain))


def capture(card, source, filename):
    episodes, frames = [], {}
    lines = source.splitlines()
    target = 'run_card' if card == 'card-bc3935ac' else 'card_run'

    def observer(frame, event, arg):
        if frame.f_code.co_filename != filename or frame.f_code.co_name != target:
            return None
        v = frame.f_locals
        if event == 'call':
            if card == 'card-7ec5e5c2':
                p = {'source': v['T'], 'target': v['U']}; inputs = v['tokens']
            elif card == 'card-bc3935ac':
                p = {'vocabulary': frame.f_globals['VOCAB'], 'epsilon': frame.f_globals['EPS']}; inputs = v['stream']
            else:
                p = {k: v[k] for k in ('E', 'W', 'beta', 'gamma', 'K')}; inputs = v['seq']
            e = {'parameters': snapshot(p), 'inputs': snapshot(inputs), 'card_outputs': [], 'states': []}
            episodes.append(e); frames[id(frame)] = e
        elif event == 'line' and 'outs.append(' in lines[frame.f_lineno - 1]:
            e = frames[id(frame)]
            if card == 'card-7ec5e5c2':
                e['states'].append(snapshot({k: v[k] for k in ('F', 'c', 'b', 'r')}))
            elif card == 'card-bc3935ac':
                e['states'].append(snapshot({'M': v['M'], 'K': v['K']}))
            else:
                e['states'].append(snapshot({'h': v['h'], 'sigma': v['s']}))
        elif event == 'return':
            e = frames.pop(id(frame))
            if card == 'card-d1d383d5':
                e['card_outputs'] = snapshot([np.concatenate((h, s, p)) for h, s, p in arg])
            elif card == 'card-bc3935ac':
                e['card_outputs'] = snapshot(arg[0]); e['final_labels'] = snapshot(arg[1])
            else:
                e['card_outputs'] = snapshot(arg)
        return observer

    stdout = io.StringIO(); previous = sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, filename, 'exec'), {'__name__': '__original_capture__'})
    finally:
        sys.settrace(previous)
    assert episodes and not frames
    return episodes, stdout.getvalue()


def main():
    run = ROOT.parent / 'bigbrane/runs/run-20261010T181300Z'
    fixture = json.loads((ROOT / 'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json').read_text())
    entries = {e['card_id']: e for e in fixture['cards']}
    events = [json.loads(line) for line in (run / 'ledger.jsonl').read_text().splitlines()]
    records = {}
    for card in CARDS:
        found = []
        for event in events:
            d = event['data']
            if event['type'] == 'artifact' and d.get('card_id') == card and d.get('kind') == 'executor':
                raw = (run / d['path']).read_bytes(); a = json.loads(raw)
                if a.get('purpose') == 'reduction_check':
                    assert sha(raw) == d['sha256']; found.append((event, a))
        assert len(found) == 1
        event, a = found[0]; raw = a['files']['main.py'].encode()
        original = run / 'jobs' / a['job_id'] / 'in/main.py'
        assert original.read_bytes() == raw
        dest = FIXTURES / 'programs' / f'next-{card}.py'; dest.write_bytes(raw)
        episodes, stdout = capture(card, raw.decode(), str(dest))
        specimen = {f'{k}_sha256': sha(json.dumps(entries[card][k], sort_keys=True, separators=(',', ':')).encode())
                    for k in ('card', 'critic')}
        p = {'source': str(dest.relative_to(ROOT)), 'sha256': sha(raw), 'executed_source_sha256': sha(raw),
             'original_source': str(original), 'artifact': str(run / event['data']['path']),
             'artifact_sha256': event['data']['sha256'], 'ledger_event': event, 'purpose': a['purpose'],
             'source_run': run.name, 'observer_sha256': sha(Path(__file__).read_bytes()),
             'capture_insertions': [], 'method': 'observer-only sys.settrace; untouched complete original program',
             'draws_sha256': sha(json.dumps([{'parameters': e['parameters'], 'inputs': e['inputs']} for e in episodes],
                                         sort_keys=True, separators=(',', ':')).encode()),
             'specimen': specimen}
        if card == 'card-bc3935ac':
            p['draw_limit'] = 'rng.choice(list(placed)) depends on Python set iteration. Original runtime hash seed was not saved; this is an untouched-source reexecution, not recovery of byte-exact historical draws.'
        records[card] = {'provenance': p, 'specimen': specimen, 'episodes': episodes,
                         'original_stdout': a['stdout'], 'reexecuted_stdout': stdout}
        print(card, len(episodes), sum(len(e['inputs']) for e in episodes), flush=True)
    raw = (json.dumps(records, separators=(',', ':')) + '\n').encode()
    (REPORT / 'next-full-capture.json').write_bytes(raw)
    manifest = {'capture_sha256': sha(raw), 'cards': {c: {k: v for k, v in record.items() if k != 'episodes'}
                                                     for c, record in records.items()}}
    (FIXTURES / 'next_sources.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
