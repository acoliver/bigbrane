"""Pin delivery sources and verify byte preservation and repository state without network access."""
import ast
import json
import subprocess
from pathlib import Path
from harness import ROOT, file_sha, verify_freeze, check_pin

main = Path('/Volumes/XS1000/acoliver/projects/bigbrane')
sibling = ROOT.parent.parent
sources = [sibling / p for p in ['hybrid/literature/prior_art.py', 'hybrid/literature/client.py',
    'hybrid/literature/paper.py', 'hybrid/literature/transport.py', 'hybrid/literature/sources/arxiv.py',
    'hybrid/literature/sources/openalex.py', 'hybrid/literature/sources/semantic_scholar.py',
    'results-wt/acquisition-windows/REPORT.md', 'results-wt/acquisition-windows/SUMMARY.json',
    'results-wt/acquisition-windows/DELIVERY-RECEIPTS.json', 'results-wt/transport-measurement/REPORT.md']]
sources += list(ROOT.glob('*.py')) + [ROOT / 'PLAN.md', ROOT / 'policy.json']
hashes = {str(p): file_sha(p) for p in sorted(sources)}
(ROOT / 'SOURCE-HASHES.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
verify_freeze()
check_pin(ROOT / 'exclusions.json', ROOT / 'exclusions.sha256')
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text())
prose = [*ROOT.glob('*.md')]
for path in prose:
    text = path.read_text()
    assert not any(c in text for c in ('\u2013', '\u2014')), path
    assert not any(w in text.casefold() for w in ('canonical', 'load-bearing', 'to be honest', "here's the thing")), path
maintained = [*ROOT.glob('*.py'), *prose, ROOT / 'policy.json', ROOT / 'pytest.ini']
line_count = sum(len(p.read_text().splitlines()) for p in maintained)
assert line_count <= 1200, line_count
states = {}
for name, repo, prefix in [('main', main, '8347c7f'), ('sibling', sibling, '4dd4864')]:
    run = lambda *args: subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    states[name] = {'head': run('rev-parse', 'HEAD'), 'branch': run('branch', '--show-current'),
                    'status': run('status', '--porcelain', '--untracked-files=normal')}
    assert states[name]['head'].startswith(prefix)
assert not states['main']['status'], states['main']
assert all(file_sha(p) == sha for p, sha in hashes.items())
assert '44 passed' in (ROOT / 'tests-complete.log').read_text()
validation = {'repositories': states, 'source_hashes_rechecked': True, 'policy_and_exclusion_pins_valid': True,
    'maintained_lines': line_count, 'python_ast_valid': True, 'prose_dash_and_literal_checks': True,
    'tests': '44 passed', 'network': 'Audit hook active in CLI/tests; no live acquisition executed.',
    'study_status': json.loads((ROOT / 'PENDING.json').read_text())['status'],
    'preservation_scope': 'Main clean and HEAD unchanged; pinned sources unchanged across finalization. No full original-run rehash claimed.'}
(ROOT / 'VALIDATION.json').write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n')
artifacts = {p.name: {'sha256': file_sha(p), 'bytes': p.stat().st_size} for p in sorted(ROOT.iterdir())
             if p.is_file() and p.name != 'ARTIFACTS.json'}
(ROOT / 'ARTIFACTS.json').write_text(json.dumps(artifacts, indent=2, sort_keys=True) + '\n')
print(json.dumps(validation, indent=2, sort_keys=True))
