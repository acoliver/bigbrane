"""The newly captured traces are separate from old recorded decisions and are reference-checked as data."""
import hashlib
import json
from pathlib import Path

import pytest

from hybrid.research.stages.published_references import REFERENCE_OPERATIONS, reference_outputs
from hybrid.research.stages.s2_prior_art import independent_comparison
from hybrid.tests.research.reference_replays import FIXTURES, load


def test_fixture_checksums_and_source_provenance():
    for line in (FIXTURES / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == expected
    root = Path(__file__).resolve().parents[3]
    for source in json.loads((FIXTURES / 'sources.json').read_text()).values():
        assert hashlib.sha256((root / source['fixture']).read_bytes()).hexdigest() == source['sha256']
    for group in ('kd8', 'fix12', 'fix10', 'phase7'):
        for evidence in load(group).values():
            source = root / evidence['provenance']['source']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == evidence['provenance']['sha256']


@pytest.mark.parametrize('name', ['kd8', 'fix12', 'fix10', 'phase7'])
def test_saved_samples_have_finite_host_measurements(name):
    for key, e in load(name).items():
        operation = e['reference_spec']['operation']
        family, _, repairs = REFERENCE_OPERATIONS[operation]
        declared = {'base_family': family, 'reference_spec': e['reference_spec'],
                    'transformations': [{'type': t} for t in repairs]}
        measured = independent_comparison({'reference_samples': e['reference_samples'], 'max_abs_diff': 0}, declared)
        assert 'reference_problem' not in measured, (key, measured)
        matches = operation not in ('diagonal_attention_count', 'mesa_single', 'bimap_hard', 'bimap_weighted')
        assert (measured['max_abs_diff'] <= 1e-6) == matches, (key, measured)


def test_unsupported_operations_fail_fast():
    with pytest.raises(ValueError, match='unsupported host operation'):
        reference_outputs('unknown', {}, [1])
