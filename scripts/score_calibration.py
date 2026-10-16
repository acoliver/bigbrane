"""Score a calibration against expected.json with the system's acceptance items: the seven cells, the profile
verdict (which includes P4 learnability) and the target result. Earlier versions scored only the seven cells and hid
the profile failures; see STATUS.md."""
import glob
import json
import os
import sys

S = sys.argv[1]
exp = json.load(open('hybrid/fixtures/calibration/expected.json'))
cells = exp['cells']
tot = ok = 0
miss = []
for f in sorted(glob.glob(f'runs/calibration-math-{S}/*/ledger.jsonl')):
    run = os.path.basename(os.path.dirname(f))
    card = run.split('-')[0]
    want_card = exp['cards'][card]
    for line in open(f):
        e = json.loads(line)
        if e['type'] != 'run_end':
            continue
        d = e['data']
        got = d.get('cells') or {}
        items = [(c, want_card['cells'][c], got.get(c)) for c in cells]
        items.append(('profile', want_card['profile'], 'pass' if d.get('profile_passed') else 'fail'))
        items.append(('target', want_card['target'], d.get('target')))
        for name, want, g in items:
            tot += 1
            if g == want:
                ok += 1
            else:
                miss.append((run, name, want, g))
        validity = d.get('validity', '')
        print(run, ' '.join(f'{n}={g}' for n, _, g in items), validity)
print('items', tot, 'match', ok, '(7 cells + profile + target per card-run)')
for m in miss:
    print('  miss', *m)
