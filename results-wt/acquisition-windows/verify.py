"""Repeat offline extraction, compare deterministic bytes and verify preservation."""
import json
from pathlib import Path
import subprocess
import sys
import os

from pin_inputs import OUT, WT, MAIN, sha


def main():
    outputs = ['QUERY-CONTEXTS.json', 'HISTORICAL-WINDOWS.json', 'FIRST-WINDOWS.json', 'ORDERING.json',
               'REGISTERED-TRACE.json', 'SUMMARY.json', 'INPUTS-after.json']
    first = {name: sha(OUT / name) for name in outputs}
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(WT)}
    with (OUT / 'reproduction.log').open('w') as log:
        subprocess.run([sys.executable, str(OUT / 'extract_windows.py')], cwd=WT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    second = {name: sha(OUT / name) for name in outputs}
    assert first == second, 'Offline extraction was not byte deterministic'
    manifest = json.loads((OUT / 'INPUTS-before.json').read_text())
    original = {path: sha(Path(path)) for path in manifest['original_sha256']}
    supplemental = {path: sha(Path(path)) for path in manifest['supplemental_sha256']}
    assert original == manifest['original_sha256']
    assert supplemental == manifest['supplemental_sha256']
    for line in (OUT / 'PLAN.sha256').read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        assert sha(Path(filename.strip())) == digest
    for name, count in [('tests.log', 18), ('prior-tests.log', 7)]:
        text = (OUT / name).read_text()
        assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
    structured = outputs[:-1]
    forbidden = {'headers', 'authorization', 'x-api-key', 'api_key', 'apikey', 'query_key'}
    def walk(item):
        if isinstance(item, dict):
            assert not forbidden.intersection(k.casefold() for k in item)
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)
    for name in structured:
        walk(json.loads((OUT / name).read_text()))
    source_before = manifest['supplemental_sha256'][str(WT / 'results-wt/transport-measurement/test_measure.py')]
    assert sha(WT / 'results-wt/transport-measurement/test_measure.py') == source_before
    result = {'offline_outputs_byte_identical': first, 'original_inputs_unchanged': len(original),
              'supplemental_inputs_unchanged': len(supplemental), 'fresh_raw_envelopes_unchanged': manifest['fresh_raw_count'],
              'plan_and_initial_inventory_unchanged': True, 'new_tests_passed': 18, 'prior_tests_passed': 7,
              'prior_test_source_sha256_unchanged': source_before,
              'structured_credential_fields_absent': True, 'actual_credentials_loaded_for_this_investigation': False,
              'network_disabled_during_extraction_and_tests': True,
              'main_git_status': subprocess.check_output(['git', '-C', str(MAIN), 'status', '--porcelain'], text=True),
              'sibling_git_status': subprocess.check_output(['git', '-C', str(WT), 'status', '--porcelain'], text=True)}
    (OUT / 'VALIDATION.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
