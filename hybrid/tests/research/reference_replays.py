"""Separate executed trace evidence for immutable historical reports. No saved verdict is rewritten."""
import copy
import json
from functools import lru_cache
from pathlib import Path

FIXTURES = Path(__file__).parent / 'fixtures' / 'reference_replays'


@lru_cache(maxsize=None)
def load(name):
    return json.loads((FIXTURES / f'{name}.json').read_text())


def fix12_spec(comparator):
    evidence = load('fix12').get(comparator['id'])
    return dict(comparator, reference_spec=evidence['reference_spec']) if evidence else comparator


def kd8_record(record):
    replay = copy.deepcopy(record)
    card = record['card']
    evidence = load('kd8').get(card)
    if evidence is None:
        return replay
    ids = {
        'card-ea50345f': {'jeffrey_ipf_step': 'jeffrey_ipf'},
        'card-17100001': {'dpda_dyck2_textbook': 'dyck2', 'hard_stack_rnn': 'dyck2'},
        'card-17100002': {'queue_automaton': 'queue_copy'},
        'card-17100003': {'minsky_two_counter': 'two_counter_abc'},
        'card-17100004': {'inside_algorithm_pcfg': 'inside_pcfg'},
        'card-17100005': {'earley_1970': 'earley'},
        'card-17100006': {'viterbi_published': 'viterbi_max'},
        'card-17100007': {'sequitur_published': 'sequitur'},
        'card-17100009': {'selective_diag_ssm': 'selective_diagonal', 'mamba_zoh_n1': 'mamba_zoh'},
        'card-1710000a': {'deltanet_delta_rule': 'delta_rule', 'additive_linear_attention': 'gated_additive_attention'},
    }[card]
    for declaration in replay['comparators']:
        if declaration['id'] in ids:
            if card == 'card-17100003' and declaration['id'] == 'minsky_two_counter' and not declaration.get('components'):
                declaration['components'] = [{'role': 'transition', 'name': 'INC/DEC with zero-test branches',
                                               'origin': 'published', 'cite': 'Minsky 1967, chapter 14'}]
            declaration['reference_spec'] = dict(evidence['reference_spec'], operation=ids[declaration['id']])
    for comparison in replay['comparisons']:
        if comparison['declared_id'] in ids:
            comparison['reference_samples'] = evidence['reference_samples']
    return replay
