"""Pin read-only inputs before offline analysis. Does not import literature clients."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
WT = OUT.parents[1]
MAIN = WT.parent / 'bigbrane'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    historical = json.loads((WT / 'results-wt/transport-measurement/PRESERVATION-before.json').read_text())['saved_inputs_sha256']
    assert len(historical) == 274
    inputs = {}
    missing = []
    for name, expected in historical.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name)
            continue
        actual = sha(path)
        assert actual == expected, name
        inputs[name] = actual
    supplemental = []
    for directory, names in {
        'transport-measurement': ['PLAN.json', 'FIXTURES.json', 'RESULTS.json', 'SUMMARY.json', 'COVERAGE.json', 'REPORT.md', 'test_measure.py', 'runtime/ledger.jsonl'],
        'retrieval-diagnosis': ['REPORT.md', 'QUERIES.md', 'offline.json', 'raw-audit.json', 'replay-evidence.json', 'source-plan-results.json', 'primary-plan-results.json', 'postfix-results.json'],
    }.items():
        supplemental += [WT / 'results-wt' / directory / name for name in names]
    supplemental += list((WT / 'hybrid/literature/sources').glob('*.py'))
    supplemental += [WT / 'hybrid/literature' / name for name in ['paper.py', 'cache.py', 'prior_art.py', 'client.py']]
    supplemental += [WT / 'hybrid/research/stages/s2_prior_art.py', WT / 'results-wt/kd8/tools/extract_live.py']
    raw_root = MAIN / 'results-wt/transport-measurement/raw-cache'
    raw_paths = sorted(raw_root.glob('*/*.json'))
    supplemental += raw_paths
    manifest = {'original_expected_count': 274, 'original_found_count': len(inputs), 'original_sha256': inputs,
                'missing_original_paths': missing, 'fresh_raw_root': str(raw_root), 'fresh_raw_count': len(raw_paths),
                'supplemental_sha256': {str(p): sha(p) for p in supplemental if p.is_file()},
                'main_head': subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),
                'sibling_head': subprocess.check_output(['git','-C',str(WT),'rev-parse','HEAD'],text=True).strip()}
    target = OUT / 'INPUTS-before.json'
    with target.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('sha256')}, indent=2))

if __name__ == '__main__':
    main()
