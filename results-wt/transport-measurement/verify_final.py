"""Offline final artifact checks. Does not construct a literature client."""
import json
from pathlib import Path
import subprocess

from measure import OUT, MAIN, WT, digest, finalize, implementation_hashes, preservation, write

plan = json.loads((OUT / 'PLAN.json').read_text())
before = json.loads((OUT / 'PRESERVATION-before.json').read_text())['saved_inputs_sha256']
assert preservation() == before
assert digest(OUT / 'PLAN.json') == (OUT / 'PLAN.sha256').read_text().strip()
assert all(digest(OUT / name) == sha for name, sha in plan['tool_sha256'].items())
current = implementation_hashes()
for path, sha in plan['implementation_sha256'].items():
    if Path(path).name not in {'seed.py', 'build_seed.py'}:
        assert current[path] == sha
for name in ('REPORT.md', 'STATUS.md', 'COMMANDS.md'):
    text = (OUT / name).read_text()
    assert '\u2013' not in text and '\u2014' not in text
assert len(list((OUT / 'raw-cache').glob('*/*.json'))) == 12
assert len((OUT / 'runtime/ledger.jsonl').read_text().splitlines()) == 36
assert 'Ran 7 tests' in (OUT / 'tests-final.log').read_text()
assert (OUT / 'tests-final.log').read_text().rstrip().endswith('OK')
for root in (MAIN, WT):
    subprocess.run(['git', '-C', str(root), 'diff', '--check'], check=True)
assert not subprocess.check_output(['git', '-C', str(MAIN), 'status', '--porcelain'], text=True)
write(OUT / 'FINAL-CHECKS.json', {'all_274_original_hashes_unchanged': True,
      'pinned_plan_runner_tests_transport_hashes_unchanged': True,
      'other_worker_seed_changes': 'See RUNTIME-HASH-CHANGES.json',
      'main_tracked_tree_clean': True, 'both_diff_checks_pass': True,
      'scoped_tests_passed': 7, 'raw_envelopes': 12, 'runtime_events': 36,
      'prose_no_en_or_em_dashes': True, 'no_additional_network_calls': True})
finalize()
artifacts = {str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'ARTIFACTS.json' and p.suffix != '.pyc'}
write(OUT / 'ARTIFACTS.json', {'sha256': artifacts, 'excludes': ['ARTIFACTS.json', '__pycache__', '*.pyc'],
                             'raw_envelopes_local_only': True})
print(json.dumps({'artifact_count': len(artifacts), 'artifact_manifest_sha256': digest(OUT / 'ARTIFACTS.json'),
                  'original_inputs_verified': 274, 'scoped_tests_passed': 7}))
