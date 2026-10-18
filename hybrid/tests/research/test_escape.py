"""Graveyard memory in S0, the escape operator, the escape sections of the S2 critic and check, and the
inspect report of reductions and escapes. Scripted LLM, LocalExecutor, no live model calls."""
import asyncio
import json
from pathlib import Path

import jsonschema
import pytest

from hybrid.research import graveyard
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config, resolve_graveyard_runs
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.stages.common import ESCAPE_FIELD, draft_schema, load_prompt
from hybrid.research.stages.s0_explore import ExploreTask
from hybrid.research.store import Store
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_orchestrator_fakes import declaring, draft, make_orchestrator, orch_config

ARGUMENT = {
    "known_family": "ssm: diagonal linear recurrence",
    "broken_invariant": "The family's state is a decayed sum of per-token writes; update u[0] solves a "
                        "shortest-path problem over the stored graph, so the state is not such a sum.",
    "why_unrepairable": "The strongest repair lifts the state to every stored edge, but the readout still needs "
                        "a min over paths, which no linear readout of a sum of per-token terms computes.",
}
REDUCED = declaring({"reduces_to": "ssm", "mapping": "S <- h, theta <- A. The card is a diagonal SSM.",
                     "confidence": 0.9, "difference_if_any": "", "radical_enough": False,
                     "why": "This is not a new mechanism. It is a diagonal recurrence with a phase readout.",
                     "textbook_transplant": True,
                     "textbook_why": "a diagonal linear recurrence as the textbook states it"})
NOT_REDUCED = REDUCED | {"reduces_to": "none", "mapping": None, "confidence": 0.2, "radical_enough": True,
                         "why": "No mapping onto a known form was found.", "comparators": []}


def explorer(offset: int = 0):
    """Explorer fake: distinct drafts; escape tasks and revisions of escape cards carry the argument."""
    def make(index, user):
        card = draft(index + offset, user=user)
        if "OPERATOR: escape" in user or f'"{ESCAPE_FIELD}"' in user:
            card[ESCAPE_FIELD] = ARGUMENT
        return card
    return make


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def calls(llm, prompt):
    return [c for c in llm.calls if c["prompt"] == prompt]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261010T000000Z"


def reduced_run(run_dir, **cfg_overrides):
    """One card that S2 reduces to `ssm`; with max_cards 1 its escape has no slot and is skipped."""
    cfg = orch_config(budgets__max_cards=1, **cfg_overrides)
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script={"explorer": explorer(), "reduction_critic": REDUCED})
    run(orch)
    return orch, llm


# ----- schema -------------------------------------------------------------------------------------

def test_non_reduction_argument_is_optional_on_cards_and_required_only_from_escape_drafts():
    old = make_card("card-aaaaaaaa")
    validate("theory_card", old)                                   # cards without the field still validate
    validate("theory_card", dict(old, **{ESCAPE_FIELD: ARGUMENT}))
    with pytest.raises(SchemaViolation):
        validate("theory_card", dict(old, **{ESCAPE_FIELD: {"known_family": "ssm"}}))

    plain, escape = draft_schema(), draft_schema(escape=True)
    assert ESCAPE_FIELD not in plain["properties"] and ESCAPE_FIELD not in plain["required"]
    assert ESCAPE_FIELD in escape["required"] and "core_idea" in escape["required"]
    base = draft(0, user="- source_id: 2401.00001")
    jsonschema.validate(base, plain)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(base, escape)
    jsonschema.validate(dict(base, **{ESCAPE_FIELD: ARGUMENT}), escape)
    with pytest.raises(jsonschema.ValidationError):                # a new draft may not carry it
        jsonschema.validate(dict(base, **{ESCAPE_FIELD: ARGUMENT}), plain)


def test_escape_task_needs_one_parent():
    ExploreTask(op="escape", lens="x", lens_sources=["x"], role="explorer", explorer=0, seed=1,
                parents=["card-aaaaaaaa"])
    with pytest.raises(ValueError, match="needs 1 parents"):
        ExploreTask(op="escape", lens="x", lens_sources=["x"], role="explorer", explorer=0, seed=1, parents=[])


# ----- graveyard ----------------------------------------------------------------------------------

def build_graveyard_run(run_dir: Path) -> Path:
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    ids = {name: f"card-{name * 8}" for name in "abcdef"}
    titles = {"a": "Reduced card", "b": "Fatal card", "c": "Finished card", "d": "Moving card", "e": "Revising card",
              "f": "Fresh card"}
    for name, card_id in ids.items():
        card = make_card(card_id, title=titles[name])
        if name != "f":
            card["core_idea"] = f"The {titles[name].lower()} binds tokens to phases and sums them."
        store.put_card(card)
    call = ["call-aaaaaaaaaaaa"]
    store.set_verdict(ids["a"], "S1", "pass", ["ok"], call)
    store.put_artifact("reduction_critic", "a.json", json.dumps(REDUCED).encode(), card_id=ids["a"])
    store.set_verdict(ids["a"], "S2", "reduces_to_known", ["gate: update rule reduces to ssm: S <- h"], call)
    store.set_verdict(ids["b"], "S1", "fail", ["gate: fatal salad-critic finding",
                                               "fatal dimension_error (mechanism.update[0]): W times x_t"], call)
    for stage in ("S1", "S2", "S3", "S4", "S5"):
        store.set_verdict(ids["c"], stage, "pass", ["ok"], call)
    # The graveyard labels an S6 card by its claims ledger: every claim supported, so the card finished.
    store.put_artifact("claims_ledger", "c.json", json.dumps({"card_id": ids["c"], "claims": [
        {"claim_id": "capability.1", "claim": "recall", "status": "supported", "decided_by": "analyst",
         "evidence": "acc 0.99"}]}).encode(), card_id=ids["c"])
    store.set_verdict(ids["c"], "S6", "pass", ["claims ledger: 1 supported, 0 refuted, 0 untested"], call)
    store.set_verdict(ids["d"], "S1", "pass", ["ok"], call)
    store.set_verdict(ids["e"], "S1", "revise", ["revision 1 of 2", "fixable missing_definition"], call)
    return run_dir


def test_graves_classify_every_outcome(tmp_path):
    run_dir = build_graveyard_run(tmp_path / "run-a")
    by_title = {grave.title: grave for grave in graveyard.graves(run_dir)}
    assert [g.title for g in graveyard.graves(run_dir)][0] == "Reduced card"           # oldest first
    reduced = by_title["Reduced card"]
    assert reduced.kind == "reduced"
    # the denial sentence is skipped; the summary says what the card is
    assert reduced.outcome == "reduced at S2 to ssm: It is a diagonal recurrence with a phase readout."
    assert by_title["Fatal card"].kind == "died"
    assert by_title["Fatal card"].outcome == ("died at S1 (fail): fatal salad-critic finding; fatal dimension_error "
                                              "(mechanism.update[0]): W times x_t")
    assert by_title["Finished card"].outcome == "finished S6: claims ledger: 1 supported, 0 refuted, 0 untested"
    assert by_title["Moving card"].outcome == "alive at S1"
    assert by_title["Revising card"].outcome == "alive at S1 (revising)"
    assert by_title["Fresh card"].outcome == "alive at S0"
    assert "(no core_idea)" in by_title["Fresh card"].line()


def test_graveyard_lines_are_bounded_and_reductions_come_first(tmp_path):
    run_dir = build_graveyard_run(tmp_path / "run-a")
    graves = graveyard.graves(run_dir)
    long = graveyard.Grave("card-99999999", "T" * 400, "I" * 400, "died", "died at S3 (fail): " + "x" * 400, 99)
    assert len(long.line()) == graveyard.ENTRY_CHARS and long.line().endswith("...")
    text = graveyard.render(graves, [], 3)
    lines = text.splitlines()
    assert lines[0] == "This run:" and len(lines) == 4
    assert "Reduced card" in lines[1] and "Fatal card" in lines[2] and "Finished card" in lines[3]
    assert graveyard.render([], [], 20) == "(none yet)"
    assert graveyard.render(graves, [], 0) == "(none yet)"


def test_graveyard_merges_earlier_runs_after_this_run_within_a_kind(tmp_path):
    mine = graveyard.graves(build_graveyard_run(tmp_path / "run-a"))
    theirs = [graveyard.Grave("card-12345678", "Old reduced card", "an old idea here", "reduced",
                              "reduced at S2 to linear_attention: a delta rule.", 0)]
    # limit 3 keeps both reductions (this run first) and then this run's death; finished and alive drop out
    lines = graveyard.render(mine, theirs, 3).splitlines()
    assert lines[:2] == ["This run:", mine[0].line()] and "Fatal card" in lines[2]
    assert lines[3:] == ["Earlier runs:", theirs[0].line()]


def test_read_events_skips_a_partly_written_last_line_only(tmp_path):
    run_dir = build_graveyard_run(tmp_path / "run-a")
    count = len(graveyard.read_events(run_dir))
    with open(run_dir / "ledger.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"seq": 99, "type": "card_ver')
    assert len(graveyard.read_events(run_dir)) == count
    (run_dir / "ledger.jsonl").write_text('{"bad\n{"seq": 1, "type": "x", "data": {}}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        graveyard.read_events(run_dir)


def test_reduced_card_without_critic_artifact_is_an_error():
    card = make_card("card-aaaaaaaa")
    card["status"] = {"stage": "S2", "verdicts": {"S2": {"verdict": "reduces_to_known", "reasons": ["r"],
                                                         "by_calls": ["call-aaaaaaaaaaaa"]}}}
    with pytest.raises(ValueError, match="no reduction_critic artifact"):
        graveyard.outcome(card, None)


def test_s0_prompt_carries_the_graveyard_bounded_by_graveyard_max(run_dir):
    reduced_run(run_dir)
    cfg = orch_config(budgets__max_cards=2)
    cfg["explore"]["p_escape"] = 0.0
    orch, llm = make_orchestrator(run_dir, cfg=cfg, resume=True, script={"explorer": explorer(10)})
    run(orch)
    user = calls(llm, "explorer")[0]["user"]
    assert "OPERATOR: new" in user and "GRAVEYARD" in user
    assert "reduced at S2 to ssm: It is a diagonal recurrence with a phase readout." in user
    assert calls(llm, "explorer")[0]["system"] == load_prompt("explorer")          # no escape section
    assert ESCAPE_FIELD not in calls(llm, "explorer")[0]["schema"]["properties"]
    assert events(orch, "escape_skipped")[-1]["reason"].startswith("draw ")

    cfg["explore"]["graveyard_max"] = 0
    orch2, llm2 = make_orchestrator(run_dir.parent / "run-20261010T000001Z", cfg=cfg, script={"explorer": explorer()})
    run(orch2)
    assert "GRAVEYARD" in calls(llm2, "explorer")[0]["user"]
    assert calls(llm2, "explorer")[0]["user"].split("GRAVEYARD")[1].splitlines()[1] == "(none yet)"


def test_graveyard_runs_add_earlier_runs_read_only(tmp_path):
    earlier = tmp_path / "run-20261009T000000Z"
    reduced_run(earlier)
    before = {path: path.read_bytes() for path in earlier.rglob("*") if path.is_file()}
    cfg = orch_config(budgets__max_cards=1)
    cfg["explore"]["graveyard_runs"] = [str(earlier)]
    orch, llm = make_orchestrator(tmp_path / "run-20261010T000000Z", cfg=cfg, script={"explorer": explorer(20)})
    assert orch.ctx.graveyard_runs == (earlier.resolve(),)
    run(orch)
    user = calls(llm, "explorer")[0]["user"]
    assert "Earlier runs:" in user and "reduced at S2 to ssm" in user and "This run:" not in user
    after = {path: path.read_bytes() for path in earlier.rglob("*") if path.is_file()}
    assert after == before


def test_graveyard_runs_are_checked_at_start(tmp_path, run_dir, monkeypatch):
    from hybrid.research import orchestrator
    with pytest.raises(FileNotFoundError, match="no ledger.jsonl"):
        resolve_graveyard_runs([str(tmp_path / "missing")], run_dir.resolve())
    reduced_run(run_dir)
    with pytest.raises(ValueError, match="this run itself"):
        resolve_graveyard_runs([str(run_dir)], run_dir.resolve())
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path.resolve())     # relative entries resolve from the repo root
    assert resolve_graveyard_runs([run_dir.name], Path("/elsewhere")) == (run_dir.resolve(),)


def test_run_config_defaults_and_validation(tmp_path):
    text = (ROOT / "hybrid" / "config" / "run.toml").read_text(encoding="utf-8")
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    assert (cfg["explore"]["graveyard_max"], cfg["explore"]["graveyard_runs"], cfg["explore"]["p_escape"]) == \
        (20, [], 1.0)
    old = tmp_path / "old.toml"
    old.write_text("\n".join(line for line in text.splitlines()
                             if not line.startswith(("graveyard_max", "graveyard_runs", "p_escape"))), encoding="utf-8")
    cfg = load_run_config(old)                                     # a config from before the keys existed
    assert (cfg["explore"]["graveyard_max"], cfg["explore"]["graveyard_runs"], cfg["explore"]["p_escape"]) == \
        (20, [], 1.0)
    for key, bad in (("p_escape", 1.5), ("graveyard_max", -1), ("graveyard_runs", "runs/x")):
        with pytest.raises(ValueError, match=f"explore.{key}"):
            load_run_config(old, {f"explore.{key}": bad})


# ----- escape operator ----------------------------------------------------------------------------

def test_reduction_without_a_card_slot_skips_the_escape(run_dir):
    orch, llm = reduced_run(run_dir)
    card_id = orch.store.list_cards()[0]
    assert orch.store.get_card(card_id)["status"]["verdicts"]["S2"]["verdict"] == "reduces_to_known"
    skipped = events(orch, "escape_skipped")
    assert [(s["card_id"], s["reason"]) for s in skipped] == [
        (card_id, "no card slot left (1 of 1 used, 0 escapes queued)")]
    assert skipped[0]["reserve"] == {"slots": 0, "claimed": 0, "queued": 0, "held": 0, "free": 0}
    assert events(orch, "escape_requested") == []


def test_p_escape_zero_never_queues(run_dir):
    orch, _ = reduced_run(run_dir, budgets__max_calls=200)
    orch.cfg["explore"]["p_escape"] = 0.0
    orch.mutations, orch.escapes = asyncio.Queue(), asyncio.Queue()
    orch.budget.max_cards = 5
    orch._request_escape(orch.store.list_cards()[0])
    assert orch.escapes.empty() and events(orch, "escape_skipped")[-1]["reason"].startswith("draw ")


def test_escape_requests_are_capped_by_free_slots(run_dir):
    orch, _ = reduced_run(run_dir)
    parent = orch.store.list_cards()[0]
    orch.mutations, orch.escapes = asyncio.Queue(), asyncio.Queue()
    orch.budget.max_cards = 3                                      # one card used, two slots free
    orch._request_escape(parent)
    orch._request_escape(parent)                                  # both free slots are now queued for escapes
    assert orch.escapes.qsize() == 2 and orch.escape_claims == 2
    orch._request_escape(parent)
    assert orch.escapes.qsize() == 2 and orch.escape_claims == 2
    skipped = events(orch, "escape_skipped")[-1]
    assert skipped["reason"] == "no card slot left (1 of 3 used, 2 escapes queued)"
    assert skipped["reserve"]["queued"] == 2 and skipped["reserve"]["free"] == 2


def test_queued_mutation_then_escape_follow_the_slots(run_dir):
    orch, _ = reduced_run(run_dir)
    parent = orch.store.list_cards()[0]
    orch.mutations, orch.escapes = asyncio.Queue(), asyncio.Queue()
    orch.budget.max_cards = 3                                      # one card used, two slots free
    orch.mutations.put_nowait(parent)
    orch._request_escape(parent)
    first, second = orch._next_task(0, 1), orch._next_task(0, 2)
    assert (first.op, second.op) == ("mutate", "escape")
    assert second.parents == [parent] and second.lens == orch.store.get_card(parent)["lens"]


def test_resume_queues_escapes_and_the_escape_card_goes_through_s2_with_the_argument(run_dir):
    orch, _ = reduced_run(run_dir)
    parent = orch.store.list_cards()[0]

    cfg = orch_config(budgets__max_cards=2)
    script = {"explorer": explorer(30), "reduction_critic": REDUCED}
    orch2, llm = make_orchestrator(run_dir, cfg=cfg, resume=True, script=script)
    summary = run(orch2)
    assert summary["reason"] == "complete"
    assert events(orch2, "escape_requested") == [{"card_id": parent}]
    created = [c for c in events(orch2, "card_created") if c["op"] == "escape"]
    assert len(created) == 1 and created[0]["parents"] == [parent]
    child_id = created[0]["card_id"]
    child = orch2.store.get_card(child_id)
    assert child["parent_ids"] == [parent] and child[ESCAPE_FIELD] == ARGUMENT
    assert child["lens"] == orch2.store.get_card(parent)["lens"]

    escape_call = calls(llm, "explorer")[0]
    assert escape_call["system"].startswith(load_prompt("explorer").rstrip())
    assert "# section: explorer_escape" in escape_call["system"]
    assert ESCAPE_FIELD in escape_call["schema"]["required"]
    user = escape_call["user"]
    assert "OPERATOR: escape" in user and f"PARENT CARD A ({parent})" in user
    assert f"REDUCTION OF THE PARENT ({parent}" in user and "S <- h, theta <- A" in user
    assert '"reduces_to": "ssm"' in user and '"reduction_outcome": "verified"' in user
    assert "numeric check confirms equivalence" in user                     # the S2 verdict line
    assert '"comparisons": [{' in user and "```python" in user              # printed report and check code
    assert "GRAVEYARD" in user and "reduced at S2 to ssm" in user

    # S2 on the escape card: the critic and the check get the escape sections, the argument, the parent's mapping
    critic_user = calls(llm, "reduction_critic")[0]["user"]
    assert critic_user.startswith("CARD:\n")
    assert "THIS CARD IS AN ESCAPE CARD" in critic_user and "strongest repair" in critic_user
    assert "THE CARD'S NON-REDUCTION ARGUMENT" in critic_user and ARGUMENT["broken_invariant"] in critic_user
    assert f"THE PARENT'S REDUCTION ({parent})" in critic_user and "S <- h, theta <- A" in critic_user
    check_user = calls(llm, "reduction_check")[0]["user"]
    assert "Write check_code now" in check_user and "strongest repaired mapping" in check_user
    assert ARGUMENT["why_unrepairable"] in check_user
    # the critic system prompt is the unchanged file
    assert calls(llm, "reduction_critic")[0]["system"] == load_prompt("reduction_critic")
    # the escape card reduced too; no slot is left for an escape of the escape
    assert child["status"]["verdicts"]["S2"]["verdict"] == "reduces_to_known"
    assert events(orch2, "escape_skipped")[-1]["card_id"] == child_id
    # resuming again does not queue a second escape for the parent. The escape card reduced to its parent's family,
    # so its own escape is a stall (Phase 8 budgets) and is never requested, on this resume or the next.
    assert events(orch2, "escape_skipped")[-1]["reason"].startswith("stall:escape")
    cfg3 = orch_config(budgets__max_cards=4)
    cfg3["explore"].update(p_mutate=0.0, p_cross=0.0)
    orch3, _ = make_orchestrator(run_dir, cfg=cfg3, resume=True,
                                 script={"explorer": explorer(40), "reduction_critic": NOT_REDUCED})
    run(orch3)
    requested = [e["card_id"] for e in events(orch3, "escape_requested")]
    assert requested.count(parent) == 1 and requested.count(child_id) == 0
    stops = [e for e in events(orch3, "loop_stop") if e["card_id"] == child_id]
    assert [stop["kill_code"] for stop in stops] == ["stall:escape"]


def test_plain_cards_get_no_escape_text_at_s2(run_dir):
    orch, llm = reduced_run(run_dir)
    for call in llm.calls:
        assert "ESCAPE CARD" not in call["user"] and "# section:" not in call["system"]


def test_s1_revision_of_an_escape_card_keeps_the_argument(run_dir):
    reduced_run(run_dir)
    bad = {"ok": False, "fatal": [], "notes": "theta undefined",
           "fixable": [{"kind": "missing_definition", "where": "mechanism.update[0]", "problem": "theta is undefined"}]}
    ok = {"ok": True, "fatal": [], "fixable": [], "notes": "ok"}
    script = {"explorer": explorer(50), "salad_critic": [bad, ok], "reduction_critic": NOT_REDUCED}
    orch, llm = make_orchestrator(run_dir, cfg=orch_config(budgets__max_cards=2), resume=True, script=script)
    run(orch)
    revise = [c for c in calls(llm, "explorer") if "OPERATOR: revise" in c["user"]]
    assert len(revise) == 1 and ESCAPE_FIELD in revise[0]["schema"]["required"]
    child = [c for c in events(orch, "card_created") if c["op"] == "escape"][0]["card_id"]
    card = orch.store.get_card(child)
    assert card[ESCAPE_FIELD] == ARGUMENT and card["status"]["verdicts"]["S2"]["verdict"] == "pass"


# ----- inspect ------------------------------------------------------------------------------------

def test_inspect_shows_reduction_families_and_escape_lineage(run_dir, capsys):
    reduced_run(run_dir)
    orch, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_cards=2), resume=True,
                                script={"explorer": explorer(60), "reduction_critic": REDUCED})
    run(orch)
    parent, child = [(c["parents"][0], c["card_id"]) for c in events(orch, "card_created") if c["op"] == "escape"][0]
    capsys.readouterr()
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert "Reductions at S2 (reduces_to_known cards by known family):\n  ssm 2" in out
    # skipped: the parent in the first invocation (no slot) and the escape child (no slot left)
    assert "Escapes: 1 requested, 2 skipped, 1 created" in out
    assert f"  {parent} (ssm) -> {child} [S2 reduces_to_known, reduced to ssm]" in out
    lineage = (run_dir / "report" / "lineage.html").read_text()
    assert "Escape lineage" in lineage and f"dossiers/{child}.html" in lineage and ", escape)" in lineage
    overview = (run_dir / "report" / "index.html").read_text()
    assert "Reductions and escapes" in overview and "<td>ssm</td>" in overview


def test_inspect_counts_calls_where_every_backend_failed(run_dir, capsys):
    reduced_run(run_dir)
    Ledger(run_dir).append("llm_call", "llm:explorer", {
        "call_id": "call-ffffffffffff", "role": "explorer", "backend": None, "model": None,
        "usage": {"input": None, "output": None, "cached": None}, "latency_s": 3.0,
        "transcript_path": str(run_dir / "llm" / "none.json"), "error": "TransportError: all failed"})
    capsys.readouterr()
    assert inspect_run(run_dir) == 0
    assert "(all failed)" in capsys.readouterr().out


# ----- prompts ------------------------------------------------------------------------------------

def test_explorer_prompt_lists_the_commonly_reinvented_families():
    text = load_prompt("explorer")
    assert "## Commonly re-invented families" in text
    for phrase in ("S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T", "Linear attention", "Recursive least squares",
                   "MesaNet", "Modern Hopfield", "Mamba", "Exact-match caches", "Union-find", "n-gram",
                   "textbook data-structure trap"):
        assert phrase in text, phrase
    assert ESCAPE_FIELD not in text                                # the field belongs to escape tasks only


def test_escape_prompt_sections_state_the_contract():
    escape = load_prompt("explorer_escape")
    for phrase in ("Operator `escape`", "known_family", "broken_invariant", "why_unrepairable", "associative",
                   "fixed size", "exact-match", "strongest repair"):
        assert phrase in escape, phrase
    assert "strongest repair" in load_prompt("reduction_critic_escape")
    assert "strongest repaired mapping" in load_prompt("reduction_check_escape")
    for name in ("explorer_escape", "reduction_critic_escape", "reduction_check_escape"):
        assert load_prompt(name).startswith("# section: "), name
