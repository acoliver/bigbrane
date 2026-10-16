"""Generation arms and the quality-diversity archive (DISCOVERY-PIPELINE-PLAN.md section 6, Phase 6), with fake LLMs.

Covers: arm selection by weight and by quota, each arm's output schema and structure check, archive cell assignment
with the consistency override, elite replacement, the empty-cell reward, duplicate rejection by mechanism signature
(within the run and against the outcome store), the excluded-family filter and closed cells, the base-arm fallback,
per-arm metrics, draft-only runs and resume.
"""
import asyncio
import copy
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from hybrid.llm.errors import SchemaValidationError
from hybrid.research import archive as arch
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore
from hybrid.research.stages import g_generate as gg
from hybrid.research.stages.common import load_prompt
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_orchestrator_fakes import draft, make_orchestrator, orch_config

CONFIG = ROOT / "hybrid" / "config"
LIMIT_TEXT = ("the output depends on the input only through the state after the separator (a fixed-state or "
              "generalized state space model)")
JELASSI = ("Jelassi, Brandfonbrener, Kakade, Malach, 'Repeat After Me: Transformers are Better than State Space "
           "Models at Copying', ICML 2024, arXiv 2402.01032")
OPEN_CELL = {"substrate": "geometric_topological", "learning_rule": "none_fixed", "unit": "word",
             "objective": "constraint_consistency", "structure_source": "builtin_prior"}
CLOSED_CELL = {"substrate": "dense_vector", "learning_rule": "gradient_backprop", "unit": "word",
               "objective": "next_unit", "structure_source": "learned_from_data"}
ARM_PROMPTS = {"A1": "arm_cross_field", "A2": "arm_limit_break", "A3": "arm_invert", "A4": "arm_random"}


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261013T000000Z"


# ----- fake drafts -------------------------------------------------------------------------------


def concepts_in(user: str) -> list[str]:
    block = user.split("CONCEPTS (", 1)[1].split("\n\n", 1)[0]
    return [line[2:] for line in block.splitlines()[1:] if line.startswith("- ")]


def commitment_in(user: str) -> str:
    return re.search(r"commitment_id (\w+)\)", user).group(1)


def closest() -> dict:
    return {"family": "none", "difference": "No listed family has a phase-indexed sum over a fixed codebook."}


def analogy(card: dict) -> dict:
    update = card["mechanism"]["update"][0]
    return {"source_field": "wave optics, superposition of phasors",
            "object_map": [{"source_object": "field amplitude", "mechanism_object": "S"},
                           {"source_object": "phase of a wave", "mechanism_object": "theta"}],
            "source_equation": {"latex": r"E = \sum_j a_j e^{i\phi_j}", "plain": "E = sum_j a_j * exp(i * phi_j)"},
            "image_equation": {"latex": update["latex"], "plain": update["plain"], "mechanism_ref": "update[0]"},
            "prediction": {"statement": "recall of 3 pairs is exact with 8 slots", "derived_via": "orthogonal phases",
                           "task": "associative recall", "metric": "accuracy", "value": 1.0}}


def structure(arm_id: str, card: dict, user: str) -> dict:
    if arm_id == "A1":
        out = analogy(card)
    elif arm_id == "A2":
        out = {"limit_id": "jelassi_copy", "assumption_id": "state_only", "assumption_quote": LIMIT_TEXT,
               "source": JELASSI, "how_broken": "the readout reads the slots, not one state",
               "mechanism_ref": "readout[0]",
               "cost": {"time_per_token": "O(1)", "memory": "O(N)", "statement": "fixed codebook, linear total"}}
    elif arm_id == "A3":
        out = {"commitment_id": commitment_in(user), "inverse": "retrieval by phase cancellation",
               "what_replaces_it": "the phase codebook theta", "mechanism_ref": "update[1]",
               "ablation": "with softmax retrieval the toy prints different recall values"}
    elif arm_id == "A4":
        out = {"concepts": concepts_in(user), "shared_relation": "both superpose periodic contributions",
               **analogy(card)}
    else:
        raise AssertionError(arm_id)
    return {**out, "closest_known": closest()}


class Drafts:
    """Arm drafts with distinct mechanisms (one operator variant per draft), unless `same` is set."""

    def __init__(self, same: bool = False, mutate=None):
        self.n = 0
        self.same = same
        self.mutate = mutate

    def __call__(self, arm_id):
        def make(_index, user):
            self.n += 1
            index = 1 if self.same else self.n
            card = draft(index, user=user if arm_id == "A1" else None)
            if self.same:
                card["title"] = f"Same mechanism, title {self.n}"
            if arm_id != "A1":
                card["references"] = []
            card["arm_structure"] = structure(arm_id, card, user)
            if self.mutate is not None:
                self.mutate(arm_id, card)
            return card
        return make


def classifier(cells=None, family="none", real=None):
    """Fake classifier output; `cells` is a list of commitments used in order (the last repeated)."""
    cells = cells or [OPEN_CELL]
    state = {"i": 0}

    def make(_index, _user):
        cell = cells[min(state["i"], len(cells) - 1)]
        state["i"] += 1
        return {**cell, "evidence": {axis: "update[0]" for axis in arch.AXES}, "family": family,
                "family_why": "fixture", "difference_is_real": real, "difference_why": "fixture"}
    return make


def arms_cfg(tmp_path, *, quota=None, draft_only=True, **overrides):
    cfg = orch_config(**overrides)
    cfg["generation"].update(mode="arms", draft_only=draft_only, quota=quota or {})
    cfg["outcomes"]["path"] = str(tmp_path / "outcomes")
    if quota:
        cfg["budgets"]["max_cards"] = sum(quota.values())
    return cfg


def script(drafts=None, cls=None):
    drafts = drafts or Drafts()
    out = {prompt: drafts(arm_id) for arm_id, prompt in ARM_PROMPTS.items()}
    out["arm_classifier"] = cls or classifier()
    return out


# ----- arm selection and the base arm ------------------------------------------------------------


def generation_config():
    return gg.load_generation_config(CONFIG / "generation.toml")


def test_arm_selection_follows_the_weights():
    weights = {"A1": 0.5, "A2": 0.3, "A3": 0.2}
    rng = random.Random(7)
    counts = Counter(gg.choose_arm(weights, rng) for _ in range(20000))
    assert set(counts) == {"A1", "A2", "A3"}
    for arm_id, weight in weights.items():
        assert abs(counts[arm_id] / 20000 - weight) < 0.02, (arm_id, counts)
    # the shipped weights with no base model: A5's 0.2 split over A1 and A2
    assert gg.effective_weights(generation_config(), False) == pytest.approx(
        {"A1": 0.3, "A2": 0.3, "A3": 0.2, "A4": 0.2})


def test_quota_schedule_interleaves_the_arms():
    assert gg.quota_schedule({"A4": 2, "A1": 3, "A2": 1}) == ["A1", "A2", "A4", "A1", "A4", "A1"]
    gen = SimpleNamespace(schedule=["A2", "A4"], weights={"A3": 1.0})
    assert gg.Generation.arm_for(gen, 0, random.Random(1)) == "A2"
    assert gg.Generation.arm_for(gen, 1, random.Random(1)) == "A4"
    assert gg.Generation.arm_for(gen, 2, random.Random(1)) == "A3"


def test_base_arm_falls_back_to_a1_and_a2_without_a_base_model():
    gen = generation_config()
    base = gen.arms["A5"]
    assert base.base_model == ""
    available, reason = gg.base_arm_status(base, routed_roles={"base_drafter", "base_formatter"}, listing=[])
    assert not available and "no base model" in reason
    configured = gg.Arm(**{**base.__dict__, "base_model": "some-base-7b"})
    roles = {"base_drafter", "base_formatter"}
    assert gg.base_arm_status(configured, routed_roles=roles, listing=[{"id": "some-base-7b"}])[0]
    assert "not listed" in gg.base_arm_status(configured, routed_roles=roles, listing=[{"id": "other"}])[1]
    assert "not routed" in gg.base_arm_status(configured, routed_roles={"base_drafter"},
                                              listing=[{"id": "some-base-7b"}])[1]
    assert "listing" in gg.base_arm_status(configured, routed_roles=roles, listing=None)[1]
    with_base = gg.effective_weights(gen, True)
    without = gg.effective_weights(gen, False)
    assert with_base["A5"] == 0.2 and "A5" not in without
    assert without["A1"] - with_base["A1"] == pytest.approx(0.1) == without["A2"] - with_base["A2"]
    assert without["A3"] == with_base["A3"] and without["A4"] == with_base["A4"]


def test_a_quota_naming_the_unavailable_base_arm_is_refused(tmp_path, run_dir):
    with pytest.raises(ValueError, match="cannot run"):
        make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A5": 1}))


# ----- arm schemas and structure checks ----------------------------------------------------------


@pytest.fixture
def gen(tmp_path, run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}), script=script())
    return orch.generation


def base_card(index=3) -> dict:
    card = draft(index)
    card.pop("source_transplant", None)
    return card


USERS = {"A3": "COMMITMENT TO INVERT (commitment_id flat_context): x",
         "A4": "CONCEPTS (drawn at random; copy them exactly):\n- tidal resonance\n- card shuffling\n\nnext"}
SEEDS = {"A3": {"commitment": {"id": "flat_context", "text": "x"}},
         "A4": {"concepts": ["tidal resonance", "card shuffling"]}}


@pytest.mark.parametrize("arm_id", ["A1", "A2", "A3", "A4"])
def test_each_arm_schema_accepts_its_structure_and_requires_its_fields(gen, arm_id):
    arm = gen.gen.arms[arm_id]
    schema = gg.arm_draft_schema(gen, arm)
    card = base_card() if arm_id != "A1" else draft(3, user="source_id: 2401.00001")
    card["arm_structure"] = structure(arm_id, card, USERS.get(arm_id, ""))
    jsonschema.validate(card, schema)
    assert gg.check_structure(arm_id, card, gen=gen, seeds=SEEDS.get(arm_id, {})) == []
    assert ("source_transplant" in schema["properties"]) == (arm_id == "A1")
    for field in list(card["arm_structure"]):
        broken = copy.deepcopy(card)
        del broken["arm_structure"][field]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)
    missing = {k: v for k, v in card.items() if k != "arm_structure"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, schema)


def test_analogy_arms_need_the_image_equation_in_the_mechanism_and_a_numeric_prediction(gen):
    card = base_card()
    card["arm_structure"] = structure("A1", card, "")
    card["arm_structure"]["image_equation"].update(plain="S_t = S_{t-1} * decay + v_t",
                                                   latex=r"S_t = \lambda S_{t-1} + v_t")
    problems = gg.check_structure("A1", card, gen=gen, seeds={})
    assert any("is not update[0]" in p for p in problems)
    card["arm_structure"] = structure("A1", card, "")
    card["arm_structure"]["image_equation"]["mechanism_ref"] = "readout[5]"
    assert any("names no equation" in p for p in gg.check_structure("A1", card, gen=gen, seeds={}))
    card["arm_structure"] = structure("A1", card, "")
    card["arm_structure"]["object_map"][0]["mechanism_object"] = "hidden_reservoir"
    assert any("object_map" in p for p in gg.check_structure("A1", card, gen=gen, seeds={}))
    schema = gg.arm_structure_schema("A1", gen.families, gen.limits, gen.gen.commitments)
    bad = structure("A1", card, "")
    bad["prediction"]["value"] = "high"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
    bad = structure("A4", card, USERS["A4"])
    bad["concepts"] = ["glacier calving", "card shuffling"]
    card["arm_structure"] = bad
    assert any("concepts" in p for p in gg.check_structure("A4", card, gen=gen, seeds=SEEDS["A4"]))


def test_limit_arm_needs_the_listed_assumption_quoted_with_its_source(gen):
    card = base_card()
    good = structure("A2", card, "")
    for change, expected in ((dict(assumption_quote="the state is small"), "assumption_quote"),
                             (dict(source="Some other paper, 2020"), "source"),
                             (dict(assumption_id="fixed_depth_heads"), "not an assumption of jelassi_copy"),
                             (dict(mechanism_ref="update[9]"), "names no equation")):
        card["arm_structure"] = {**good, **change}
        problems = gg.check_structure("A2", card, gen=gen, seeds={})
        assert any(expected in p for p in problems), (change, problems)
    card["arm_structure"] = {**good, "assumption_quote": LIMIT_TEXT[:60]}
    assert gg.check_structure("A2", card, gen=gen, seeds={}) == []
    schema = gg.arm_structure_schema("A2", gen.families, gen.limits, gen.gen.commitments)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**good, "limit_id": "made_up_limit"}, schema)


def test_invert_arm_must_invert_the_commitment_it_was_given(gen):
    card = base_card()
    card["arm_structure"] = structure("A3", card, "COMMITMENT TO INVERT (commitment_id softmax_retrieval): x")
    problems = gg.check_structure("A3", card, gen=gen, seeds=SEEDS["A3"])
    assert any("not the commitment given" in p for p in problems)


def test_base_arm_formatter_may_not_add_operators(gen):
    card = base_card(0)
    card["arm_structure"] = {"note_excerpt": "keep a running sum of phases", "added_by_formatter": [],
                             "closest_known": closest()}
    note = "Idea: keep a running sum of phases, add each value times exp of a phase, read by the conjugate phase."
    assert gg.check_structure("A5", card, gen=gen, seeds={}, base_text=note) == []
    card["mechanism"]["update"][0]["plain"] += " then softmax over slots"
    problems = gg.check_structure("A5", card, gen=gen, seeds={}, base_text=note)
    assert any("softmax_read" in p for p in problems)
    assert any("note_excerpt" in p for p in gg.check_structure("A5", card, gen=gen, seeds={}, base_text="other"))


def test_arm_prompts_ask_for_a_correspondence_a_prediction_and_the_broken_assumption_with_its_source():
    for name in ("arm_cross_field", "arm_random"):
        text = load_prompt(name)
        assert text.startswith(f"# prompt: {name}")
        assert "correspondence" in text.lower() or "map" in text
        assert "prediction" in text and "image_equation" in text and "mechanism_ref" in text
    limit = load_prompt("arm_limit_break")
    assert "assumption_quote" in limit and "source" in limit and "quoted exactly" in limit
    assert "commitment" in load_prompt("arm_invert") and "ablation" in load_prompt("arm_invert")
    assert "classify" in load_prompt("arm_classifier").lower()
    for name in ("arm_cross_field", "arm_limit_break", "arm_invert", "arm_random", "arm_base_formatter",
                 "arm_classifier"):
        assert "\u2014" not in load_prompt(name)


# ----- the excluded-family list ------------------------------------------------------------------


def test_families_config_excludes_the_d8_families_and_lists_the_known_ones():
    acfg = arch.load_archive_config(CONFIG / "archive.toml")
    families = arch.load_families(CONFIG / "families.toml", acfg)
    assert {f.id for f in families.excluded} == {"transformer", "ssm", "linear_attention", "delta_rule"}
    assert {f.id for f in families.known} == {"hopfield_associative_memory", "exact_match_cache"}
    assert arch.closed_by(CLOSED_CELL, families).id == "transformer"
    assert arch.closed_by(OPEN_CELL, families) is None
    assert arch.closed_by({**CLOSED_CELL, "unit": "byte_char"}, families) is not None   # unit is a wildcard
    assert arch.closed_by({**CLOSED_CELL, "objective": "description_length"}, families) is None


@pytest.mark.parametrize("family,real,cell,code", [
    ("linear_attention", None, OPEN_CELL, "excluded_family:linear_attention"),
    ("delta_rule", None, OPEN_CELL, "excluded_family:delta_rule"),
    ("none", None, CLOSED_CELL, "not_radical:R1"),
    ("hopfield_associative_memory", False, OPEN_CELL, "no_difference"),
    ("exact_match_cache", None, OPEN_CELL, "no_difference"),
    ("hopfield_associative_memory", True, OPEN_CELL, None),
])
def test_excluded_family_filter_and_closed_cells(tmp_path, run_dir, family, real, cell, code):
    cfg = arms_cfg(tmp_path, quota={"A1": 1})
    cfg["outcomes"]["enabled"] = True
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script(cls=classifier([cell], family, real)))
    run(orch)
    card_id = events(orch, "card_created")[0]["card_id"]
    verdict = orch.store.get_card(card_id)["status"]["verdicts"]["S0"]
    draft_event = events(orch, "arm_draft")[0]
    assigned = events(orch, "archive_assigned")[0]
    if code is None:
        assert verdict["verdict"] == "pass" and draft_event["outcome"] == "admitted" and assigned["admitted"]
        return
    assert verdict["verdict"] == "fail" and verdict["reasons"][0].startswith(f"gate: {code}: ")
    assert draft_event["kill_code"] == code and not assigned["admitted"] and assigned["gate"] == code
    # the outcome store records the G gate's kill code
    record = OutcomeStore(Path(cfg["outcomes"]["path"])).current()[-1].record
    assert record["kill_codes"] == [code] and record["terminal_stage"] == "S0"
    assert record["arm"] == "A1" and record["archive_cell"]["key"] == arch.cell_key(cell)
    # the user message showed the families from the config
    user = [c for c in llm.calls if c["prompt"] == "arm_cross_field"][0]["user"]
    assert "OUT OF SCOPE FAMILIES" in user and "delta_rule" in user and "hopfield_associative_memory" in user


# ----- a draft-only arms run ---------------------------------------------------------------------


def test_draft_only_run_drafts_each_arm_classifies_and_records_metrics(tmp_path, run_dir):
    quota = {"A1": 2, "A2": 1, "A3": 1, "A4": 1}
    cfg = arms_cfg(tmp_path, quota=quota)
    cfg["run"]["n_explorers"] = 2
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script(), models={"explorer_alt": "chatgpt/gpt-6.1-sol"})
    summary = run(orch)
    assert summary["reason"] == "complete"
    drafts = events(orch, "arm_draft")
    assert Counter(d["arm"] for d in drafts) == Counter(quota)
    assert all(d["outcome"] == "admitted" for d in drafts)
    # each arm used its own prompt; A4 went to the GPT-first role; nothing ran after S0
    prompts = Counter(llm.prompts())
    assert prompts == Counter({"arm_cross_field": 2, "arm_limit_break": 1, "arm_invert": 1, "arm_random": 1,
                               "arm_classifier": 5})
    assert {d["role"] for d in drafts if d["arm"] == "A4"} == {"explorer_alt"}
    assert {d["role"] for d in drafts if d["arm"] != "A4"} == {"explorer"}
    for card_id in orch.store.list_cards():
        card = orch.store.get_card(card_id)
        assert card["status"]["stage"] == "S0" and card["arm"] in quota
        assert card["commitments"] == OPEN_CELL and card["archive_cell"]["key"] == arch.cell_key(OPEN_CELL)
        assert ("source_transplant" in card) == (card["arm"] == "A1")
    # one cell: the first card filled it, the others visited it
    assigned = events(orch, "archive_assigned")
    assert [a["new_cell"] for a in assigned].count(True) == 1 and assigned[0]["visits_before"] == 0
    metrics = events(orch, "arm_metrics")[-1]["arms"]
    assert set(metrics) == set(quota)
    assert sum(m["new_cells"] for m in metrics.values()) == 1
    for arm_id, m in metrics.items():
        assert m["drafts"] == quota[arm_id] == m["admitted"] == m["survivors"]["S0"]
        assert m["schema_failures"] == 0 and m["schema_failure_rate"] == 0.0
        assert m["calls"] == 2 * quota[arm_id] and m["tokens_out"] == 100 * quota[arm_id]
    start = events(orch, "run_start")[0]["generation"]
    assert start["mode"] == "arms" and start["draft_only"] and start["base_arm"]["available"] is False
    assert "A5" not in start["weights"]
    assert [a for a in events(orch, "artifact") if a["kind"] == "archive_snapshot"]
    # the arm prompts: arm section first, then the shared explorer rules; A2 lists the limits, A4 the concepts
    a2 = [c for c in llm.calls if c["prompt"] == "arm_limit_break"][0]
    assert "KNOWN LIMITS" in a2["user"] and LIMIT_TEXT in a2["user"] and "General card rules" in a2["system"]
    a4 = [c for c in llm.calls if c["prompt"] == "arm_random"][0]
    seeds = [d["seeds"] for d in drafts if d["arm"] == "A4"][0]
    assert concepts_in(a4["user"]) == seeds["concepts"] and len(seeds["concepts"]) == 2
    classifier_user = [c for c in llm.calls if c["prompt"] == "arm_classifier"][0]["user"]
    assert "title and summary withheld" in classifier_user and "Fake card" not in classifier_user


def test_draft_only_run_holds_no_escape_reserve_and_fills_the_whole_quota(tmp_path, run_dir):
    quota = {"A1": 1, "A2": 1, "A3": 1, "A4": 1}
    cfg = arms_cfg(tmp_path, quota=quota)
    cfg["explore"]["escape_reserve"] = 0.3      # run.toml's value; ceil(0.3 * 4) = 2 slots outside draft-only runs
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script=script(), models={"explorer_alt": "chatgpt/gpt-6.1-sol"})
    assert run(orch)["reason"] == "complete"
    assert Counter(d["arm"] for d in events(orch, "arm_draft")) == Counter(quota)
    assert events(orch, "run_start")[0]["escape_reserve"]["slots"] == 0


def test_schema_failures_count_against_their_arm(tmp_path, run_dir):
    cfg = arms_cfg(tmp_path, quota={"A1": 1, "A3": 1})
    orch, _ = make_orchestrator(run_dir, cfg=cfg, script=script(),
                                fail={"arm_invert": SchemaValidationError("output invalid after one retry: x")})
    run(orch)
    metrics = events(orch, "arm_metrics")[-1]["arms"]
    assert metrics["A3"]["schema_failure"] == 1 and metrics["A3"]["schema_failure_rate"] == 1.0
    assert metrics["A1"]["schema_failures"] == 0 and metrics["A1"]["admitted"] == 1
    assert [e["op"] for e in events(orch, "explore_failed")] == ["new"]


def test_structure_failure_kills_the_card_with_the_arm_code(tmp_path, run_dir):
    def wrong_image(arm_id, card):
        card["arm_structure"]["image_equation"].update(plain="y = W x", latex="y = W x")

    cfg = arms_cfg(tmp_path, quota={"A1": 1})
    cfg["outcomes"]["enabled"] = True
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script(Drafts(mutate=wrong_image)))
    run(orch)
    card = orch.store.get_card(orch.store.list_cards()[0])
    assert card["status"]["verdicts"]["S0"]["reasons"][0].startswith(
        "gate: arm_structure:A1: arm_structure.image_equation.mechanism_ref: image equation")
    assert "arm_classifier" not in llm.prompts()           # no classifier call for a broken arm structure
    assert llm.prompts().count("arm_cross_field") == 2      # the draft and its one structure retry
    assert "commitments" not in card and not events(orch, "archive_assigned")
    metrics = events(orch, "arm_metrics")[-1]["arms"]["A1"]
    assert metrics["arm_structure"] == 1 and metrics["schema_failures"] == 1
    assert OutcomeStore(Path(cfg["outcomes"]["path"])).current()[-1].record["kill_codes"] == ["arm_structure:A1"]


def test_consistency_override_moves_backprop_out_of_a_gradient_free_cell(tmp_path, run_dir):
    def backprop(arm_id, card):
        card["toy_example"]["python"] = "# trains with loss.backward() in the full model\n" + card["toy_example"]["python"]

    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}),
                                script=script(Drafts(mutate=backprop)))
    run(orch)
    assigned = events(orch, "archive_assigned")[0]
    assert assigned["commitments"]["learning_rule"] == "gradient_backprop"
    assert assigned["overrides"] and "toy code" in assigned["overrides"][0]
    card = orch.store.get_card(assigned["card_id"])
    assert card["commitments"]["learning_rule"] == "gradient_backprop"
    assert any(r.startswith("consistency override") for r in card["status"]["verdicts"]["S0"]["reasons"])
    plain, overrides = gg.consistency(card | {"toy_example": {"python": "x = 1"},
                                              "mechanism": {**card["mechanism"], "learning": {
                                                  "description": "No backprop: counts only.",
                                                  "few_shot_argument": "-"}}}, OPEN_CELL)
    assert plain == OPEN_CELL and overrides == []


# ----- duplicates by signature -------------------------------------------------------------------


def test_same_mechanism_on_other_axes_is_a_duplicate_within_the_run(tmp_path, run_dir):
    other = {**OPEN_CELL, "substrate": "symbolic_graph_term"}
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1, "A2": 1}),
                                script=script(Drafts(same=True), classifier([OPEN_CELL, other])))
    run(orch)
    first, second = events(orch, "arm_draft")
    assert first["outcome"] == "admitted"
    assert second["outcome"] == "duplicate" and second["kill_code"] == f"duplicate:{first['card_id']}"
    assert events(orch, "archive_assigned")[1]["cell"] == arch.cell_key(other)


def test_signature_of_an_outcome_store_record_is_a_duplicate(tmp_path, run_dir):
    earlier = make_card("card-0000d001", title="Phase Ring Memory")
    earlier["mechanism"] = copy.deepcopy(draft(1)["mechanism"])
    view = RunView(build_run(tmp_path / "runs" / "run-20260901T000000Z", [(earlier, [
        ("S0", "pass", ["ok"]), ("S1", "fail", ["gate: fatal finding remains"])], {})]))
    store = OutcomeStore(tmp_path / "outcomes")
    store.append(build_record(view, "card-0000d001", "backfill"))
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1}), script=script(Drafts(same=True)))
    run(orch)
    drafted = events(orch, "arm_draft")[0]
    assert drafted["outcome"] == "duplicate" and drafted["kill_code"] == "duplicate:card-0000d001"
    reason = orch.store.get_card(drafted["card_id"])["status"]["verdicts"]["S0"]["reasons"][0]
    assert "run-20260901T000000Z/card-0000d001" in reason


# ----- archive units: elites, rewards, parents ---------------------------------------------------


class Cards:
    def __init__(self):
        self.cards = {}

    def add(self, card_id, passed=("S0",)):
        self.cards[card_id] = {"id": card_id, "status": {"stage": passed[-1], "verdicts": {
            stage: {"verdict": "pass"} for stage in passed}}}

    def get_card(self, card_id):
        return self.cards[card_id]

    def list_cards(self):
        return list(self.cards)


def small_config(elites=1, bonus=0.5):
    values = {axis: ("a", "b") for axis in arch.AXES}
    texts = {axis: {"a": "a", "b": "b"} for axis in arch.AXES}
    return arch.ArchiveConfig(values, texts, {axis: axis for axis in arch.AXES}, elites, bonus, "S1")


def entry(card_id, cell, new=False, arm="A1"):
    return arch.Entry(card_id, cell, arm, "new", True, new, 0)


ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")


def test_elite_replacement_by_furthest_stage_with_earlier_card_winning_ties():
    archive = arch.Archive(small_config(elites=1), arch.Families((), ()), ORDER)
    cards = Cards()
    cell = "a|a|a|a|a"
    for card_id in ("card-1", "card-2", "card-3"):
        cards.add(card_id)
        archive._apply(entry(card_id, cell))
    assert archive.elites(cell, cards) == ["card-1"]          # tie at S0: the earlier card stays
    cards.add("card-3", passed=("S0", "S1"))
    assert archive.elites(cell, cards) == ["card-3"]          # passing S1 replaces the elite
    cards.add("card-2", passed=("S0", "S1", "S2"))
    assert archive.elites(cell, cards) == ["card-2"]
    two = arch.Archive(small_config(elites=2), arch.Families((), ()), ORDER)
    two.members = archive.members
    assert two.elites(cell, cards) == ["card-2", "card-3"]


def test_empty_cell_reward_is_paid_once_at_the_reward_stage(tmp_path):
    ledger = Ledger(tmp_path / "run")
    archive = arch.Archive(small_config(), arch.Families((), ()), ORDER)
    archive._apply(entry("card-new", "a|a|a|a|a", new=True, arm="A2"))
    archive._apply(entry("card-old", "a|a|a|a|a", new=False))
    assert not archive.on_pass(ledger, "card-new", "S0")     # not the reward stage
    assert archive.on_pass(ledger, "card-new", "S1")
    assert not archive.on_pass(ledger, "card-new", "S1")     # once
    assert not archive.on_pass(ledger, "card-old", "S1")     # visited cell: no reward
    rewards = [e["data"] for e in ledger.iter_events("archive_reward")]
    assert rewards == [{"card_id": "card-new", "cell": "a|a|a|a|a", "arm": "A2", "op": "new", "stage": "S1"}]


def test_parent_selection_favours_cells_with_empty_neighbours():
    cfg = small_config(elites=1, bonus=0.5)
    archive = arch.Archive(cfg, arch.Families((), ()), ORDER)
    cards = Cards()
    crowded, open_cell = "a|a|a|a|a", "b|b|b|b|b"
    archive._apply(entry("card-crowded", crowded))
    archive._apply(entry("card-open", open_cell))
    cards.add("card-crowded")
    cards.add("card-open")
    for neighbour in arch.neighbours(crowded, cfg):         # every neighbour of the crowded cell is visited
        archive.visits[neighbour] += 1
    assert archive.empty_neighbours(crowded) == 0 and archive.empty_neighbours(open_cell) == 5
    rng = random.Random(3)
    picks = Counter(archive.choose_parents(["card-crowded", "card-open"], 1, rng, cards)[0] for _ in range(4000))
    expected = archive.cell_weight(open_cell) / (archive.cell_weight(open_cell) + archive.cell_weight(crowded))
    assert picks["card-open"] > picks["card-crowded"]
    assert abs(picks["card-open"] / 4000 - expected) < 0.03
    assert sorted(archive.choose_parents(["card-crowded", "card-open"], 2, rng, cards)) == ["card-crowded",
                                                                                            "card-open"]
    coverage = archive.coverage()
    assert coverage["cells_filled"] == 2 and coverage["per_axis"]["unit"] == {"filled": 2, "values": 2}


def test_full_line_in_arms_mode_rewards_the_arm_that_filled_an_empty_cell(tmp_path, run_dir):
    orch, llm = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A3": 1}, draft_only=False),
                                  script=script())
    summary = run(orch)
    assert summary["reason"] == "complete"
    card_id = orch.store.list_cards()[0]
    card = orch.store.get_card(card_id)
    assert card["status"]["verdicts"]["S1"]["verdict"] == "pass" and card["arm"] == "A3"
    assert events(orch, "archive_reward") == [{"card_id": card_id, "cell": arch.cell_key(OPEN_CELL), "arm": "A3",
                                               "op": "new", "stage": "S1"}]
    metrics = events(orch, "arm_metrics")[-1]["arms"]["A3"]
    assert metrics["rewards"] == 1 and metrics["survivors"]["S1"] == 1


# ----- resume ------------------------------------------------------------------------------------


def test_resume_rebuilds_the_archive_and_keeps_the_mode(tmp_path, run_dir):
    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1, "A2": 1}), script=script())
    run(orch)
    assert len(events(orch, "archive_assigned")) == 2
    cfg = arms_cfg(tmp_path, quota={"A1": 2, "A2": 2})
    again, llm = make_orchestrator(run_dir, cfg=cfg, script=script(), resume=True)
    assert again.generation.archive.visits[arch.cell_key(OPEN_CELL)] == 2
    assert len(again.generation.archive.signatures) >= 2
    run(again)
    assigned = events(again, "archive_assigned")
    assert len(assigned) == 4 and [a["new_cell"] for a in assigned] == [True, False, False, False]
    assert [d["arm"] for d in events(again, "arm_draft")] == ["A1", "A2", "A1", "A2"]
    assert Counter(llm.prompts())["arm_classifier"] == 2      # only the two new drafts were classified
    with pytest.raises(ValueError, match="generation.mode = 'arms'"):
        make_orchestrator(run_dir, cfg=orch_config(), script=script(), resume=True)


def test_a_lens_run_resumes_in_lens_mode_and_cannot_switch_to_arms(tmp_path, run_dir):
    orch, llm = make_orchestrator(run_dir)
    run(orch)
    assert events(orch, "run_start")[0]["generation"] == {"mode": "lenses", "draft_only": False}
    assert not events(orch, "arm_draft") and "arm_classifier" not in llm.prompts()
    with pytest.raises(ValueError, match="generation.mode = 'lenses'"):
        make_orchestrator(run_dir, cfg=arms_cfg(tmp_path), resume=True)


def test_shipped_config_keeps_lens_mode_and_validates_generation_keys(tmp_path):
    cfg = load_run_config(CONFIG / "run.toml")
    assert cfg["generation"]["mode"] == "lenses" and cfg["generation"]["quota"] == {}
    assert load_run_config(CONFIG / "run2.toml")["generation"]["mode"] == "lenses"
    with pytest.raises(ValueError, match="generation.mode"):
        load_run_config(CONFIG / "run.toml", {"generation.mode": "both"})
    with pytest.raises(ValueError, match="quota needs"):
        load_run_config(CONFIG / "run.toml", {"generation.quota": {"A1": 1}})


def test_preflight_checks_the_classifier_model_only_in_arms_mode():
    from hybrid import localmodels
    from hybrid.llm import load_config

    models = load_config(CONFIG / "models.toml", "llxprt")
    assert localmodels.requirements(models, localmodels.roles_for(("S0",))) == []
    reqs = localmodels.requirements(models, localmodels.roles_for(("S0",), arms=True))
    assert [(r.model, r.roles) for r in reqs] == [("qwen3.6-35b-a3b-mlx", ("arm_classifier",))]


def test_inspect_prints_the_per_arm_table(tmp_path, run_dir):
    from hybrid.research.inspect import render_text, summarize

    orch, _ = make_orchestrator(run_dir, cfg=arms_cfg(tmp_path, quota={"A1": 1, "A4": 1}), script=script())
    run(orch)
    text = render_text(summarize(orch.run_dir, orch.ledger, orch.store))
    assert "Generation arms" in text and "base-model arm A5: not available" in text
    rows = [line for line in text.splitlines() if line.strip().startswith(("A1 ", "A4 "))]
    assert len(rows) == 2 and all(" S0:1 " in row for row in rows)
    assert "archive: 1 cells filled" in text
