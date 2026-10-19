"""Execute byte-identical original programs with observer-only Python tracing, without source insertions."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT.parent / 'bigbrane/runs/run-20261010T181300Z'
OUT = ROOT / 'hybrid/tests/research/fixtures/reference_replays'
REPORT = ROOT / 'results-wt/evidence-recovery'
CARDS = ('card-582f8abf', 'card-d86dca4b', 'card-ffdbe331', 'card-36725a7c')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(type(value).__name__)


def snapshot(value):
    return json.loads(json.dumps(value, default=plain))


def capture(card, source, filename):
    episodes, instances = [], {}
    observed_frames = {}
    lines = source.splitlines()

    def episode(obj, parameters):
        key = id(obj)
        if key not in instances:
            e = {'parameters': snapshot(parameters), 'inputs': [], 'card_outputs': [], 'states': []}
            episodes.append(e)
            instances[key] = (obj, e)  # Keep the object alive so object ids cannot be reused.
        return instances[key][1]

    def observer(frame, event, arg):
        if frame.f_code.co_filename != filename:
            return None
        name = frame.f_code.co_qualname
        local = frame.f_locals
        parity = card == 'card-582f8abf' and name == 'Card.step'
        projective = card == 'card-d86dca4b' and name in ('run.<locals>.Card.dist', 'run.<locals>.Card.consume')
        concept = card == 'card-ffdbe331' and name == 'Card.step'
        sugs = card == 'card-36725a7c' and name == 'card_run'
        if not (parity or projective or concept or sugs):
            return None
        if sugs and event == 'call':
            e = {'parameters': snapshot({'phi': local['Phi'], 'gate': local['w'],
                 'ridge': frame.f_globals['lam'], 'variance': frame.f_globals['s2'],
                 'concentration': frame.f_globals['eta'], 'capacity': frame.f_globals['R']}),
                 'inputs': snapshot(local['seq']), 'card_outputs': [], 'states': [], 'accepted': []}
            episodes.append(e)
            observed_frames[id(frame)] = e
        if sugs and event == 'line' and lines[frame.f_lineno - 1].strip() == 'logits.append(Phi @ y)':
            e = observed_frames[id(frame)]
            e['states'].append(snapshot({k: local[k] for k in ('P', 'K', 'N', 'r', 'z', 'beta')}))
            if local['beta'] >= 0.5:
                e['accepted'].append(snapshot({'key': local['u'], 'value': local['v'], 'weight': local['beta'],
                                               'card_assignment': local['assign'][-1]}))
        if event == 'return':
            if parity:
                obj = local['self']; e = episode(obj, {'vertices': obj.d})
                e['inputs'].append(local['tok']); e['card_outputs'].append(snapshot(arg))
                e['states'].append(snapshot({k: getattr(obj, k) for k in ('P', 'H', 'M', 'C')}))
            elif projective:
                obj = local['self']; p = frame.f_back.f_locals['p']
                e = episode(obj, {'prime': p, 'mode': obj.mode})
                if name.endswith('.dist'):
                    e['card_outputs'].append(snapshot(arg))
                else:
                    e['inputs'].append(local['w'])
                    e['states'].append(snapshot({k: getattr(obj, k) for k in ('A', 'r', 'phi', 's')}))
            elif concept:
                obj = local['s']; e = episode(obj, {'embeddings': obj.E, 'words': obj.W,
                                                      'delta': obj.d, 'prior_rate': obj.l})
                e['inputs'].append([local['w'], local['op'], local['x']]); e['card_outputs'].append(snapshot(arg))
                e['states'].append(snapshot(obj.T))
            elif sugs:
                e = observed_frames.pop(id(frame)); e['card_outputs'] = snapshot(arg[0]); e['assignments'] = snapshot(arg[1])
        return observer

    stdout = io.StringIO()
    previous = sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, filename, 'exec'), {'__name__': '__saved_historical_capture__'})
    finally:
        sys.settrace(previous)
    assert episodes
    return episodes, stdout.getvalue()


def main():
    events = [json.loads(line) for line in (RUN / 'ledger.jsonl').read_text().splitlines()]
    recovered = {}
    for card in CARDS:
        matches = []
        for event in events:
            d = event['data']
            if event['type'] != 'artifact' or d.get('card_id') != card or d.get('kind') != 'executor':
                continue
            artifact_path = RUN / d['path']; raw = artifact_path.read_bytes(); artifact = json.loads(raw)
            if artifact.get('purpose') == 'reduction_check':
                assert digest(raw) == d['sha256']
                matches.append((event, artifact_path, artifact))
        assert len(matches) == 1, card
        event, artifact_path, artifact = matches[0]
        code = artifact['files']['main.py']
        original = RUN / 'jobs' / artifact['job_id'] / 'in/main.py'
        assert original.read_bytes() == code.encode()
        copied = OUT / 'programs' / f'historical-{card}.py'
        copied.write_bytes(original.read_bytes())
        episodes, stdout = capture(card, code, str(copied))
        provenance = {'source': str(copied.relative_to(ROOT)), 'sha256': digest(copied.read_bytes()),
                      'executed_source_sha256': digest(code.encode()), 'capture_insertions': [],
                      'method': 'observer-only sys.settrace on original CARD-side functions; no source changes',
                      'original_source': str(original), 'artifact': str(artifact_path),
                      'artifact_sha256': event['data']['sha256'], 'ledger_event': event,
                      'purpose': artifact['purpose'], 'source_run': RUN.name,
                      'observer_sha256': digest(Path(__file__).read_bytes())}
        recovered[card] = {'provenance': provenance, 'episodes': episodes,
                           'original_stdout': artifact['stdout'], 'reexecuted_stdout': stdout}
        print(card, len(episodes), 'episodes', sum(len(e['inputs']) for e in episodes), 'inputs', flush=True)
    path = OUT / 'historical.json'
    path.write_text(json.dumps(recovered, separators=(',', ':')) + '\n')
    print(path, path.stat().st_size, 'bytes', flush=True)


if __name__ == '__main__':
    main()
