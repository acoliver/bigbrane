"""Regressions from the Phase 6 draft run phase6-drafts-20261013T0611Z (results/phase6-20261013T0230/FIX-1.md).

Four generation slots failed there. Each test below replays the recorded model output (sanitized fixtures in
fixtures/phase6_failures/) or the recorded draw, and pins the fix:

- A2 schema failure: the retry reply held a fragment and then the full card; the parser took the fragment. The first
  top-level object that validates is now used, and the retry message names every failing field by its dotted path.
- A4 object_map failures: `sum_i w_i` and `C_ij*(T_j-T_i)` are expressions in the card's own symbols and now pass; a
  source-field word still fails.
- A1 image-equation failure: mechanism_ref named update[0] while the image was update[1]. The check still fails it
  (the rule stands), names the field and the matching equation, and the draft gets one structure retry.
- A1 lost slot: a pairing asked for 3 papers drew 1 + 1. The split now gives 2 + 1, and a lens that gives too few
  papers is topped up from neighbouring lenses and the seed list, recorded as `paper_fallback`.
"""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.llm import jsonmode
from hybrid.literature.seed import KNOWN_ALTERNATIVES_LENS, LENSES
from hybrid.research.stages import g_generate as gg
from hybrid.research.stages import s0_explore
from hybrid.research.stages.s0_explore import ExploreTask
from hybrid.tests.llm.fakes import FakeBackend
from hybrid.tests.llm.test_client import make_client
from hybrid.tests.research.test_generation_arms import Drafts, arms_cfg, events, run, script
from hybrid.tests.research.test_orchestrator_fakes import make_orchestrator

FIXTURES = Path(__file__).parent / "fixtures" / "phase6_failures"
PAIR = "pair:kernel+program_induction"
PAIR_SOURCES = ["kernel_gaussian_processes", "bayesian_program_induction"]
# The seed of the lost A1 slot: lens_seed("phase6-drafts-20261013T0611Z", 4); papers_min + seed % 4 asks for 3.
LOST_SLOT_SEED = 1088686876


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261013T000000Z"


# ----- A2: JSON schema failure after retry ---------------------------------------------------------


def test_recorded_a2_retry_reply_validates_through_its_second_object():
    fx = fixture("a2_schema_retry_two_objects.json")
    schema = fx["json_schema"]
    first, retry = (attempt["raw_output"] for attempt in fx["attempts"])
    # the recorded failure: the retry reply's first object is a fragment ending in a placeholder key
    assert "known_relatives" in fx["attempts"][1]["recorded_validation_error"]
    fragment = jsonmode.extract_first_json_object(retry)
    assert "constants_notes_placeholder_removed" in fragment["claims"] and "known_relatives" not in fragment
    assert len(jsonmode.json_objects(retry)) == 2
    parsed, error = jsonmode.parse_and_validate(retry, schema)
    assert error is None and parsed["title"].startswith("Self-Reducing Words")
    assert parsed["claims"]["complexity"]["constants_notes"] and parsed["known_relatives"]
    # the first reply still fails, and its error names the exact missing field
    parsed, error = jsonmode.parse_and_validate(first, schema)
    assert parsed is None
    assert error.startswith("schema validation failed at claims/complexity: 'constants_notes' is a required property")
    assert "claims.complexity.constants_notes is missing (required)" in error
    instruction = jsonmode.retry_instruction(first, error)
    assert "claims.complexity.constants_notes" in instruction and "complete JSON object" in instruction
    assert "placeholder" in instruction


def test_recorded_a2_replies_through_the_client_succeed_on_the_retry(tmp_path):
    fx = fixture("a2_schema_retry_two_objects.json")
    replies = [attempt["raw_output"] for attempt in fx["attempts"]]
    client, a, _, _ = make_client(tmp_path, a=FakeBackend(script=replies))
    result = asyncio.run(client.complete("r", "s", "draft", json_schema=fx["json_schema"]))
    assert result.parsed["known_relatives"] and len(a.requests) == 2
    assert "claims.complexity.constants_notes is missing (required)" in a.requests[1].user


def test_field_problems_name_missing_extra_and_nested_list_fields():
    schema = {"type": "object", "additionalProperties": False, "required": ["a", "items"],
              "properties": {"a": {"type": "string"},
                             "items": {"type": "array", "items": {
                                 "type": "object", "required": ["task"], "additionalProperties": False,
                                 "properties": {"task": {"type": "string"}}}}}}
    parsed, error = jsonmode.parse_and_validate('{"items": [{"task": "x"}, {"task": 3, "z": 1}], "b": 2}', schema)
    assert parsed is None
    for field in ("a is missing (required)", "b is not allowed (remove it)", "items[1].z is not allowed",
                  "items[1].task: 3 is not of type 'string'"):
        assert field in error, (field, error)
    # two objects, neither valid: the one with fewer errors is reported
    parsed, error = jsonmode.parse_and_validate('{"x": 1} then {"a": "ok", "items": [{"task": 4}]}', schema)
    assert parsed is None and "items[0].task" in error and "2 JSON objects" in error


# ----- A4: object_map expressions -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a4_sum_object.json", "a4_flux_expression.json"])
def test_recorded_a4_object_maps_in_the_cards_own_symbols_pass(name):
    fx = fixture(name)
    assert "object_map names mechanism objects" in fx["recorded_reason"]
    assert gg.check_structure("A4", fx["card"], gen=None, seeds=fx["seeds"]) == []


@pytest.mark.parametrize("obj", ["road density", "sum_i q_i", "vacancy of a destination cell", "Phi(w)"])
def test_object_map_still_rejects_objects_outside_the_mechanism(obj):
    fx = fixture("a4_sum_object.json")
    card = copy.deepcopy(fx["card"])
    card["arm_structure"]["object_map"][3]["mechanism_object"] = obj
    problems = gg.check_structure("A4", card, gen=None, seeds=fx["seeds"])
    assert problems == [p for p in problems if p.startswith("arm_structure.object_map[3].mechanism_object:")]
    assert len(problems) == 1 and repr(obj)[1:-1] in problems[0]


def test_a4_rule_on_the_given_concepts_and_the_image_equation_stands():
    fx = fixture("a4_flux_expression.json")
    card = copy.deepcopy(fx["card"])
    card["arm_structure"]["image_equation"]["mechanism_ref"] = "update[1]"
    card["arm_structure"]["concepts"] = ["traffic jams", "plate tectonics"]
    problems = gg.check_structure("A4", card, gen=None, seeds=fx["seeds"])
    assert [p.split(":", 1)[0] for p in problems] == ["arm_structure.image_equation.mechanism_ref",
                                                      "arm_structure.concepts"]
    assert "plain text is update[2] of the mechanism" in problems[0]


# ----- A1: image equation index ---------------------------------------------------------------------


def test_recorded_a1_wrong_index_still_fails_and_names_the_field_and_the_match():
    fx = fixture("a1_image_index.json")
    card = fx["card"]
    problems = gg.check_structure("A1", card, gen=None, seeds={})
    assert len(problems) == 1
    assert problems[0].startswith("arm_structure.image_equation.mechanism_ref: image equation")
    assert "is not update[0]" in problems[0] and "plain text is update[1] of the mechanism" in problems[0]
    fixed = copy.deepcopy(card)
    fixed["arm_structure"]["image_equation"]["mechanism_ref"] = "update[1]"
    assert gg.check_structure("A1", fixed, gen=None, seeds={}) == []
    # an image that is no mechanism equation fails with the copy instruction
    fixed["arm_structure"]["image_equation"]["plain"] = "z = z + eta * (xi - z)"
    fixed["arm_structure"]["image_equation"]["latex"] = r"z \leftarrow z + \eta(\xi - z)"
    problem = gg.check_structure("A1", fixed, gen=None, seeds={})[0]
    assert "no equation of the mechanism: copy the plain text" in problem
    fixed["arm_structure"]["image_equation"]["mechanism_ref"] = "update[7]"
    problem = gg.check_structure("A1", fixed, gen=None, seeds={})[0]
    assert "update[7] names no equation" in problem and "update[0]..update[3]" in problem


def test_structure_retry_instruction_names_fields_and_restates_the_rule():
    fx = fixture("a1_image_index.json")
    problems = gg.check_structure("A1", fx["card"], gen=None, seeds={})
    text = gg.structure_retry_instruction("A1", '{"title": "x"}', problems)
    assert "- arm_structure.image_equation.mechanism_ref:" in text and '{"title": "x"}' in text
    assert "prediction is derived through the map" in text and "zero-based" in text
    assert set(gg.STRUCTURE_RULES) == set(gg.ARM_IDS)


def test_structure_retry_repairs_a_wrong_mechanism_ref_and_counts_as_admitted(tmp_path, run_dir):
    seen = {"n": 0}

    def wrong_first(arm_id, card):
        seen["n"] += 1
        if seen["n"] == 1:
            card["arm_structure"]["image_equation"]["mechanism_ref"] = "update[1]"

    orch, llm = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}),
                                  script=script(Drafts(mutate=wrong_first)))
    run(orch)
    draft_calls = [c for c in llm.calls if c["prompt"] == "arm_cross_field"]
    assert len(draft_calls) == 2
    retry_user = draft_calls[1]["user"]
    assert "# ARM STRUCTURE CHECK FAILED" in retry_user
    assert "- arm_structure.image_equation.mechanism_ref: image equation" in retry_user
    assert "plain text is update[0] of the mechanism" in retry_user
    drafted = events(orch, "arm_draft")[0]
    assert drafted["outcome"] == "admitted" and len(drafted["calls"]) == 3
    assert drafted["structure_retry"]["fixed"] and drafted["structure_retry"]["call_id"] == drafted["calls"][1]
    card = orch.store.get_card(drafted["card_id"])
    assert card["created_by_call"] == drafted["calls"][1]
    assert card["arm_structure"]["image_equation"]["mechanism_ref"] == "update[0]"
    metrics = events(orch, "arm_metrics")[-1]["arms"]["A1"]
    assert metrics["schema_failures"] == 0 and metrics["structure_retries"] == 1 and metrics["structure_fixed"] == 1


def test_structure_retry_that_fails_again_kills_with_the_arm_code(tmp_path, run_dir):
    def unknown_object(arm_id, card):
        card["arm_structure"]["object_map"][0]["mechanism_object"] = "road density"

    orch, llm = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A4": 1}),
                                  script=script(Drafts(mutate=unknown_object)),
                                  models={"explorer_alt": "chatgpt/gpt-6.1-sol"})
    run(orch)
    assert [c["prompt"] for c in llm.calls] == ["arm_random", "arm_random"]
    drafted = events(orch, "arm_draft")[0]
    assert drafted["outcome"] == "arm_structure" and drafted["kill_code"] == "arm_structure:A4"
    assert drafted["structure_retry"]["fixed"] is False and len(drafted["calls"]) == 2
    reason = orch.store.get_card(drafted["card_id"])["status"]["verdicts"]["S0"]["reasons"][0]
    assert reason.startswith("gate: arm_structure:A4: arm_structure.object_map[0].mechanism_object:")
    metrics = events(orch, "arm_metrics")[-1]["arms"]["A4"]
    assert metrics["schema_failures"] == 1 and metrics["structure_retries"] == 1 and metrics["structure_fixed"] == 0


# ----- A1: paper sampling ----------------------------------------------------------------------------


def pair_task(seed: int = LOST_SLOT_SEED) -> ExploreTask:
    return ExploreTask(op="new", lens=PAIR, lens_sources=PAIR_SOURCES, role="explorer", explorer=1, seed=seed,
                       arm="A1")


def test_recorded_lost_slot_draw_now_gives_the_papers_it_asked_for():
    papers = s0_explore.sample_papers(pair_task(), 3)
    assert len(papers) == 3
    # every split of k over the two lenses gives k papers (the lenses hold far more than 6)
    for k in range(3, 7):
        assert len(s0_explore.sample_papers(pair_task(), k)) == k


def test_neighbour_lenses_follow_the_registry_order_and_skip_the_sources():
    order = [lens for lens in LENSES if lens != KNOWN_ALTERNATIVES_LENS]
    near = s0_explore.neighbour_lenses(PAIR_SOURCES)
    i = order.index("kernel_gaussian_processes")
    assert near[:1] == [order[i - 1]] and "bayesian_program_induction" not in near
    assert KNOWN_ALTERNATIVES_LENS not in s0_explore.neighbour_lenses([order[0], order[-1]])


def test_draw_papers_tops_up_from_neighbours_and_records_the_fallback(monkeypatch):
    emitted = []
    ctx = SimpleNamespace(cfg={"explore": {"papers_min": 3}}, emit=lambda kind, data: emitted.append((kind, data)))
    real = s0_explore.sample_papers
    monkeypatch.setattr(s0_explore, "sample_papers", lambda task, k: real(task, k)[:1])
    papers = s0_explore.draw_papers(ctx, pair_task(), 5)
    assert len(papers) == 5 and len({(p.source, p.source_id) for p in papers}) == 5
    (kind, data), = emitted
    assert kind == "paper_fallback" and data["from_lens"] == 1 and data["requested"] == 5
    assert data["lens"] == PAIR and data["arm"] == "A1" and len(data["added"]) == 4
    assert all(item["origin"].startswith("neighbour:") for item in data["added"])
    # no papers anywhere near: the seed list fills the draw
    monkeypatch.setattr(s0_explore, "sample_for_lens", lambda lens, k, seed: [])
    monkeypatch.setattr(s0_explore, "sample_papers", lambda task, k: [])
    emitted.clear()
    papers = s0_explore.draw_papers(ctx, pair_task(), 4)
    assert len(papers) == 4 and all(item["origin"].startswith("seed_list:") for item in emitted[0][1]["added"])
    # enough papers: no event
    monkeypatch.undo()
    emitted.clear()
    assert len(s0_explore.draw_papers(ctx, pair_task(), 3)) == 3 and emitted == []


def test_a1_slot_with_an_empty_lens_still_drafts(tmp_path, run_dir, monkeypatch):
    monkeypatch.setattr(s0_explore, "sample_papers", lambda task, k: [])
    orch, llm = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}), script=script())
    assert run(orch)["reason"] == "complete"
    fallback, = events(orch, "paper_fallback")
    assert fallback["from_lens"] == 0 and len(fallback["added"]) >= 3
    drafted, = events(orch, "arm_draft")
    assert drafted["outcome"] == "admitted" and not events(orch, "explore_failed")
    created = events(orch, "card_created")[0]
    assert created["papers"] == [item["paper"] for item in fallback["added"]]
