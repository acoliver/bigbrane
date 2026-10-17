"""Score a calibration against expected.json with the system's acceptance items: the seven cells, the profile
verdict (which includes P4 learnability) and the target result. Earlier versions scored only the seven cells and hid
the profile failures; see STATUS.md."""
import glob
import json
import os
import sys



def m_kill_codes(ledger_path):
    """Kill codes of the last stage M verdict, read from its 'math gate (M): ... kill codes a, b' reason (the
    calibration's own acceptance compares the required codes with M's kill codes)."""
    codes = []
    for line in open(ledger_path):
        e = json.loads(line)
        if e['type'] == 'verdict' and e['data'].get('stage') == 'M':
            for reason in e['data'].get('reasons', []):
                if reason.startswith('math gate (M):') and 'kill codes' in reason:
                    codes = [c.strip() for c in reason.split('kill codes', 1)[1].split(',')]
    return codes


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
        required = want_card.get('kill_codes_include') or []
        if required:
            codes = m_kill_codes(f)
            missing = [c for c in required if c not in codes]
            items.append(('kill_codes', 'include ' + ','.join(required),
                          'ok' if not missing else 'missing ' + ','.join(missing)))
        for name, want, g in items:
            tot += 1
            if g == want or (name == 'kill_codes' and g == 'ok'):
                ok += 1
            else:
                miss.append((run, name, want, g))
        validity = d.get('validity', '')
        print(run, ' '.join(f'{n}={g}' for n, _, g in items), validity)
print('items', tot, 'match', ok, '(7 cells, profile, target, and required kill codes where the key lists them)')
for m in miss:
    print('  miss', *m)
