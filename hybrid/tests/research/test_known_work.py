"""R2A, R2C, R2D: when known work gates at S2, and the difference gate's evidence rules."""
import pytest

from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages import difference, known_work
from hybrid.tests.research.critic_answers import KNOWN, NOT_KNOWN
from hybrid.tests.research.test_orchestrator_fakes import defaults

CANDIDATES = [{"key": "s2:abc123", "source_id": "abc123", "title": "A paper", "year": 2020, "excerpt": "x"},
              {"key": "arxiv:2306.04470", "source_id": "2306.04470", "title": "Another", "year": 2023, "excerpt": "y"}]
COUNTED = [{"declared_id": "published_form", "family": "other_known", "comparator": "the published form"}]


def critic(**changes):
    return dict(defaults()["reduction_critic"], **changes)


def source(kind="paper", ref="s2:abc123"):
    return {"kind": kind, "ref": ref, "covers": "the update and the readout"}


def test_not_claimed_does_not_gate():
    assert not known_work.judge(critic(**NOT_KNOWN), [], CANDIDATES).kills


def test_counted_comparator_gates_without_a_source():
    verdict = known_work.judge(critic(**KNOWN), COUNTED, CANDIDATES)
    assert verdict.kills and verdict.basis == "counted_comparator"


def test_bare_claim_with_a_paper_outside_the_search_results_does_not_gate():
    answer = critic(known_work=True, known_work_sources=[source(ref="s2:notinresults")])
    verdict = known_work.judge(answer, [], CANDIDATES)
    assert not verdict.kills and verdict.basis == "no_evidence"


@pytest.mark.parametrize("ref", ["s2:abc123", "abc123"])
def test_cited_paper_in_the_results_is_a_lead_not_checked_coverage(ref):
    verdict = known_work.judge(critic(known_work=True, known_work_sources=[source(ref=ref)]), [], CANDIDATES)
    assert not verdict.kills and verdict.basis == "coverage_unchecked"


def test_named_textbook_and_algorithm_are_unvalidated_leads():
    for kind in ("textbook", "standard_algorithm"):
        answer = critic(known_work=True, known_work_sources=[source(kind, "Earley")])
        assert not known_work.judge(answer, [], CANDIDATES).kills
        assert known_work.cited_names(answer, CANDIDATES) == [source(kind, "Earley")]


def test_resemblance_with_a_learned_part_fixed_cannot_carry_a_cited_source_kill():
    fixed = [{"part": "sigma", "capability": "learns the width of the match window"}]
    answer = critic(known_work=True, known_work_sources=[source()], fixed_learned_parts=fixed)
    verdict = known_work.judge(answer, [], CANDIDATES)
    assert not verdict.kills and verdict.basis == "learned_part_fixed"


def test_fixed_learned_part_prevents_general_mechanism_verification():
    fixed = [{"part": "sigma", "capability": "learns the width"}]
    assert not known_work.judge(critic(known_work=True, known_work_sources=[source()], fixed_learned_parts=fixed),
                            COUNTED, CANDIDATES).kills


def test_schema_requires_a_source_when_known_work_is_true():
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", critic(known_work=True, known_work_sources=[]))
    validate("reduction_critic", critic(**KNOWN))
    validate("reduction_critic", critic(**NOT_KNOWN))


def test_cited_papers_are_required_entries_and_names_go_to_the_checker():
    answer = critic(known_work=False, known_work_sources=[source(), source(ref="s2:zzz"), source("textbook", "CLRS 21")])
    assert known_work.cited_papers(answer, CANDIDATES) == ["s2:abc123"]
    assert [s["ref"] for s in known_work.cited_names(answer, CANDIDATES)] == ["CLRS 21"]


def test_search_listing_shows_every_citable_id():
    text = known_work.search_listing(CANDIDATES)
    assert "s2:abc123" in text and "arxiv:2306.04470" in text


def test_checker_schema_ties_accepted_to_problems():
    schema = difference.check_schema([], None)
    base = {"accepted": True, "problems": [], "excerpt_gaps": ["the excerpt does not state the update"],
            "causes": [], "family_difference_real": None, "family_why": "", "required_difference": ""}
    import jsonschema
    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(base))
    assert list(validator.iter_errors(dict(base, accepted=True, problems=["false of the equations"])))
    assert list(validator.iter_errors(dict(base, accepted=False, problems=[])))


def requirements(**changes):
    families, _ = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
    fields = dict(literature={"s2:abc123": CANDIDATES[0], "abc123": CANDIDATES[0]}, outcomes={}, r3=[], family=None,
                  need=0, families=families, relevant_papers=("s2:abc123",))
    return difference.Requirements(**(fields | changes)), families


def statement(*entries):
    return {"closest_prior": [{"kind": k, "ref": r, "title": "t", "what_was_tried": "w", "why_it_failed": "f"}
                              for k, r in entries], "what_was_tried": "w", "why_it_failed": "f", "precise_change": "p",
            "changed_equation": "learning", "cause_ids_addressed": []}


def test_family_entries_alone_satisfy_a_card_with_no_relevant_source():
    from hybrid.tests.research.builders import make_card
    req, families = requirements()
    card = make_card("card-r2test")
    assert "closest_prior:total" in difference.host_problems(statement(("family", families.known[0].id)), card, req)
    assert not difference.host_problems(statement(("family", families.known[0].id),
                                                  ("family", families.known[1].id)), card, req)


def test_relevant_sources_are_still_required_before_family_entries():
    from hybrid.tests.research.builders import make_card
    req, families = requirements(need=1)
    problems = difference.host_problems(statement(("family", families.known[0].id)), make_card("card-r2test"), req)
    assert "closest_prior:count" in problems


def test_a_paper_the_critic_cited_must_be_listed_in_closest_prior():
    from hybrid.tests.research.builders import make_card
    req, families = requirements(cited_papers=("s2:abc123",))
    card = make_card("card-r2test")
    family = families.known[0].id
    assert "cited_source:s2:abc123" in difference.host_problems(
        statement(("family", family), ("family", families.excluded[0].id)), card, req)
    assert "cited_source:s2:abc123" not in difference.host_problems(
        statement(("literature", "s2:abc123"), ("family", family)), card, req)
