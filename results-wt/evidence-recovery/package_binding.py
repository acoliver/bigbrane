"""Hash the tested binding tree without staging or changing fixtures."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results-wt/evidence-recovery'
FIXTURES = ROOT / 'hybrid/tests/research/fixtures/reference_replays'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


owned = ['hybrid/research/schemas/reduction_critic.schema.json',
         'hybrid/research/stages/published_references.py',
         'hybrid/tests/research/reference_replays.py', 'hybrid/tests/research/phase7_replay.py',
         'hybrid/tests/research/test_historical_binding.py', 'hybrid/tests/research/test_historical_references.py',
         'results-wt/evidence-recovery/capture_historical.py']
owned += [str((FIXTURES / name).relative_to(ROOT)) for name in
          ('.gitignore', 'SHA256SUMS', 'historical_declarations.json',
           'historical_negative_controls.json', 'historical_sources.json')]
owned += [str(p.relative_to(ROOT)) for p in sorted((FIXTURES / 'programs').glob('historical-*.py'))]
frozen = ['hybrid/research/stages/s2_prior_art.py', 'hybrid/research/stages/contracts.py',
          'hybrid/config/families.toml', 'hybrid/tests/research/test_phase7_replay.py',
          'hybrid/tests/research/test_reference_replays.py', 'hybrid/tests/research/test_review_gate_evidence.py',
          'hybrid/tests/research/fixtures/phase7_replay/run-20261010T181300Z.json']
base = json.loads((OUT / 'binding-baseline.json').read_text())
assert all(digest(ROOT / p) == base[p] for p in frozen)
originals = [FIXTURES / 'historical.json'] + list((FIXTURES / 'programs').glob('historical-*.py'))
assert all(digest(p) == base[str(p.relative_to(ROOT))] for p in originals)
files = [{'path': p, 'bytes': (ROOT / p).stat().st_size, 'sha256': digest(ROOT / p)}
         for p in sorted(owned + frozen)]
records = ''.join(f"{f['sha256']}  {f['path']}\n" for f in files)
fingerprint = hashlib.sha256(records.encode()).hexdigest()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(OUT / 'binding-tested-tree.json').write_text(json.dumps({
    'head': head, 'sha256': fingerprint, 'hash_format': 'SHA256 of sorted sha256-two-spaces-path-newline records',
    'files': files, 'frozen_paths_verified_against_binding_baseline': frozen}, indent=2) + '\n')
rows = json.loads((OUT / 'binding-phase7-rows.json').read_text())
prior = {r['card_id']: r for r in json.loads((OUT / 'phase7-current-rows.json').read_text())}
required = [r for r in rows if r['expected_codes']]
missed = [r for r in required if r['replay_codes'] != r['expected_codes']]
restored = [r for r in required if prior[r['card_id']]['current_codes'] != r['expected_codes']
            and r['replay_codes'] == r['expected_codes']]
assert {r['card_id'] for r in restored} == {'card-582f8abf', 'card-d86dca4b'}
assert len(missed) == 14 and len(required) == 19 and len(rows) == 21
assert all(r['replay_codes'] == prior[r['card_id']]['current_codes'] for r in rows
           if r['card_id'] not in {x['card_id'] for x in restored})
counts = defaultdict(Counter)
for case in ET.parse(OUT / 'binding-delivery-junit.xml').getroot().iter('testcase'):
    counts[case.attrib['classname'].rsplit('.', 1)[-1]][
        'failed' if case.find('failure') is not None else 'passed'] += 1
assert sum(c['passed'] for c in counts.values()) == 89
assert sum(c['failed'] for c in counts.values()) == 2
summary = {'exact_gaps_before': 16, 'exact_gaps_after': 14, 'exact_gaps_resolved': 2,
           'required_codes': 19, 'matching_required_codes': 5, 'source_rows': 21,
           'resolved': [r['card_id'] for r in restored], 'remaining': [r['card_id'] for r in missed],
           'tests': dict(counts), 'tested_tree_sha256': fingerprint}
(OUT / 'binding-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
local = [FIXTURES / 'historical.json'] + sorted(p for p in (OUT / 'sources').iterdir()
                                            if p.suffix in ('.pdf', '.html', '.txt'))
(OUT / 'binding-inventory.json').write_text(json.dumps({
    'source_and_test_files': files,
    'local_retained': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                        'sha256': digest(p)} for p in local],
    'owned_paths': owned,
    'outside_binding_ownership': ['hybrid/literature/sources/arxiv.py',
                                 'hybrid/tests/literature/test_parsing.py',
                                 'hybrid/tests/literature/test_arxiv_https.py']}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
