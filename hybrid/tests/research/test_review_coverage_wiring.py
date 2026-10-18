"""Offline B4 coverage propagation and bounded exposure regressions."""
import asyncio
import json

import pytest

from hybrid.literature.prior_art import listing_receipt
from hybrid.research.stages import difference, s2_prior_art
from hybrid.tests.research.test_difference import checker, critic, statement
from hybrid.tests.research.test_orchestrator_fakes import make_classifier, make_orchestrator, orch_config


def candidates(n):
    return [{'key': f'arxiv:p{i}', 'source_id': f'p{i}', 'source': 'arxiv', 'title': f'Unrelated hit {i}',
             'year': 2000, 'url': '', 'excerpt': f'Supplied source excerpt {i}', 'rank': i + 1,
             'query_fields': ['automata' if i == n - 2 else 'optimization']} for i in range(n)]


@pytest.mark.parametrize('adequate', [False, True])
def test_nonempty_coverage_flag_is_persisted_without_retry(tmp_path, adequate):
    orch, llm = make_orchestrator(tmp_path / 'run-20261018T010000Z')
    classify = make_classifier('novel', True)
    seen = []
    result = {}

    async def fake(card_text, complete, sink):
        seen.append(card_text)
        result.update(await classify(card_text, complete, sink))
        result.update(coverage={'errors': {'arxiv': '429'}, 'usable': adequate,
                                'alternate_source': 'openalex' if adequate else None},
                      listing=listing_receipt(result['candidates'], [p['key'] for p in result['candidates']], 60),
                      coverage_assessment={'adequate': adequate, 'why': 'alternate source covers update and readout'
                                           if adequate else 'the only hit is unrelated'},
                      novelty_unassessed=not adequate,
                      novelty_unassessed_reasons=[] if adequate else ['missing automata field coverage'])
        return result

    orch.ctx.classify_prior_art = fake
    asyncio.run(orch.run())
    card = orch.store.get_card(orch.store.list_cards()[0])
    assert len(seen) == 1
    assert card['status']['verdicts']['S2']['verdict'] == ('pass' if adequate else 'undetermined')
    stored = orch.ctx.latest_json_artifact(card['id'], 'prior_art')
    for field in ('coverage', 'listing', 'coverage_assessment', 'novelty_unassessed', 'novelty_unassessed_reasons'):
        assert stored[field] == result[field]
    prompt = next(c['user'] for c in llm.calls if c['prompt'] == 'reduction_critic')
    assert 'SEARCH COVERAGE AND LISTING RECEIPT' in prompt
    assert '429' in prompt and 'coverage_assessment' in prompt
    assert ('missing automata field coverage' in '\n'.join(card['status']['verdicts']['S2']['reasons'])) != adequate


def test_bounded_required_leads_and_actual_receipts_reach_both_difference_participants(tmp_path):
    cfg = orch_config()
    cfg['s2'].update(family_exclusion=True, difference_statement=True)
    orch, llm = make_orchestrator(tmp_path / 'run-20261018T020000Z', cfg=cfg, script={
        'reduction_critic': critic(known_work_sources=[{'kind': 'paper', 'ref': 'arxiv:p99', 'covers': 'unverified'}]),
        'difference': statement(literature=False, refs=[{
            'kind': 'literature', 'ref': 'arxiv:p99', 'title': 'lead',
            'what_was_tried': 'lead', 'why_it_failed': 'not established'}]),
        'difference_check': checker()})
    base = make_classifier('novel', True)

    async def fake(card_text, complete, sink):
        result = await base(card_text, complete, sink)
        result.update(candidates=candidates(100), evidence=[{
            'source_id': 'arxiv:p97', 'title': 'Required evidence', 'url': '', 'relation': 'untested', 'source': 'arxiv'}],
            coverage={'errors': {'arxiv': '429'}, 'usable': True},
            coverage_assessment={'adequate': True, 'why': 'alternate sources cover the mechanism'},
            novelty_unassessed=False, novelty_unassessed_reasons=[])
        return result

    orch.ctx.classify_prior_art = fake
    asyncio.run(orch.run())
    critic_prompt = next(c['user'] for c in llm.calls if c['prompt'] == 'reduction_critic')
    assert 'Supplied source excerpt 97' in critic_prompt and 'Supplied source excerpt 98' in critic_prompt
    assert json.loads(critic_prompt.split('SEARCH COVERAGE AND LISTING RECEIPT\n', 1)[1].split('\n\n', 1)[0])['listing']['shown_count'] == 80
    for prompt in ('difference', 'difference_check'):
        user = next(c['user'] for c in llm.calls if c['prompt'] == prompt)
        receipt = json.loads(user.split('SEARCH COVERAGE AND LISTING RECEIPT\n', 1)[1].split('\n\n', 1)[0])
        assert receipt['listing']['shown_count'] == 35
        keys = {p['key'] for p in receipt['listing']['shown']}
        assert {'arxiv:p97', 'arxiv:p98', 'arxiv:p99'} <= keys
        assert 'Supplied source excerpt 99' in user and '429' in user
        assert receipt['coverage_assessment']['adequate']


def test_required_sources_cannot_silently_overflow_the_listing_bound():
    result = {'candidates': candidates(36)}
    record = {'evidence': []}
    with pytest.raises(ValueError, match='exceed'):
        difference.bounded_candidates(record, result, 35, tuple(c['key'] for c in result['candidates']))
