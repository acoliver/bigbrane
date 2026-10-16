"""S2 in Phase 7 (DISCOVERY-PIPELINE-PLAN.md sections 1.2, 1.3 and 8), with fake models: the family exclusion, the
difference statement, rule R3 against the outcome store, and the one send-back."""
import asyncio
import re

import pytest

from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore, check_kill_code
from hybrid.research.stages import difference
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_orchestrator_fakes import (analyst_supports_every_claim, declaring, defaults, draft,
                                                           make_orchestrator, orch_config)

EARLIER_RUN = "run-20260901T000000Z"
DEAD_CRITIC = {"reduces_to": "other_known", "mapping": "S = sum v exp(i theta)", "confidence": 0.9,
               "difference_if_any": "", "radical_enough": False,
               "why": "It is a phase-keyed holographic sum, which is diagonal linear attention.",
               "textbook_transplant": True, "textbook_why": "-"}


def run(orch):
    return asyncio.run(orch.run())


def events(orch, type):
    return [e["data"] for e in orch.ledger.iter_events(type)]


def calls(llm, prompt):
    return [c for c in llm.calls if c["prompt"] == prompt]


@pytest.fixture
def run_dir(tmp_path):
    return tmp_path / "run-20261013T080000Z"


@pytest.fixture
def store(tmp_path):
    """One dead idea of an earlier run whose mechanism is the fixture card's: the explorer's draft matches its
    signature exactly, so R3 applies."""
    earlier = make_card("card-0000d001", title="Phase Ring Memory")
    view = RunView(build_run(tmp_path / "runs" / EARLIER_RUN, [(earlier, [
        ("S0", "pass", ["ok"]), ("S1", "pass", ["ok"]),
        ("S2", "reduces_to_known", ["gate: update rule reduces to other_known: m"])],
        {"reduction_critic": DEAD_CRITIC})]))
    store = OutcomeStore(tmp_path / "outcomes")
    store.append(build_record(view, "card-0000d001", "backfill"))
    return store


@pytest.fixture
def empty(tmp_path):
    return OutcomeStore(tmp_path / "empty-outcomes")


def config(store, *, r3=True, exclusion=True, statement=True):
    cfg = orch_config()
    cfg["outcomes"].update(enabled=True, path=str(store.dir), enforce_r3=r3)
    cfg["s2"].update(family_exclusion=exclusion, difference_statement=statement)
    return cfg


def critic(**changes):
    """A reduction critic answer; unless `comparators` is given, it declares the published form of its reduces_to."""
    out = dict(defaults()["reduction_critic"], family="none", family_why="No family form fits the update.")
    out.update(changes)
    return out if "comparators" in changes else declaring(out)


# ----- the fake proposer and checker read the requirements from their prompts -----------------------------------------


def record_ids(user):
    return sorted(set(re.findall(r"^- (oc-[0-9a-f]{12}):", user, re.M)))


def r3_causes(user):
    lines = re.findall(r"^  - oc-[0-9a-f]{12} \(card-[0-9a-f]{8}\): causes (.+)$", user, re.M)
    return [cause.strip() for line in lines for cause in line.split(", ")]


def named_family(user):
    match = re.search(r"the critic places the core in the family (\w+)", user)
    return match.group(1) if match else None


def statement(*, records=True, causes=True, family=True, literature=True, refs=None, equation="update[0]"):
    """A writer: cites what the requirements ask for, unless told to leave a part out."""
    def write(i, user):
        prior = []
        if literature:
            prior.append({"kind": "literature", "ref": "2401.00001", "title": "A paper",
                          "what_was_tried": "phase binding", "why_it_failed": "interference between pairs"})
        if records:
            prior += [{"kind": "outcome", "ref": rid, "title": "Phase Ring Memory", "what_was_tried": "the same sum",
                       "why_it_failed": "reduced to a known form"} for rid in record_ids(user)]
        fam = named_family(user)
        if family and fam:
            prior.append({"kind": "family", "ref": fam, "title": fam, "what_was_tried": "the family form",
                          "why_it_failed": "it is that family"})
        prior += refs or []
        return {"closest_prior": prior, "what_was_tried": "a phase-keyed running sum",
                "why_it_failed": "it reduces to a known form", "precise_change": "The write is gated by a parity test.",
                "changed_equation": equation, "cause_ids_addressed": r3_causes(user) if causes else []}
    return write


def checker(*, accept=True, addressed=True, real=True, required="closest prior X used Y; state what replaces Y"):
    """A checker whose answers are fixed per call (a value, or a list used in order)."""
    def nth(value, i):
        return value[min(i, len(value) - 1)] if isinstance(value, list) else value

    def check(i, user):
        fam = named_family(user)
        return {"accepted": nth(accept, i), "problems": [] if nth(accept, i) else ["the change is a renaming"],
                "causes": [{"cause_id": c, "addressed": nth(addressed, i), "why": "the write equation changes"}
                           for c in r3_causes(user)],
                "family_difference_real": nth(real, i) if fam else None, "family_why": "checked" if fam else "",
                "required_difference": "" if nth(accept, i) else required}
    return check


def script(write, check, **more):
    return {"reduction_critic": more.pop("reduction_critic", critic()), "difference": write,
            "difference_check": check, **more}


def card_of(orch):
    return orch.store.get_card(orch.store.list_cards()[0])


def mine(store, run_dir):
    return {s.record["card_id"]: s.record for s in store.current() if s.record["run"] == run_dir.name}


# ----- the difference statement ---------------------------------------------------------------------------------------


def test_a_passing_card_carries_an_accepted_difference_statement(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker()))
    run(orch)
    card = card_of(orch)
    assert card["status"]["stage"] == "S6"
    s2 = card["status"]["verdicts"]["S2"]
    assert any(r.startswith("difference statement accepted") for r in s2["reasons"])
    assert card["difference_statement"]["closest_prior"][0]["ref"] == "2401.00001"
    assert len(calls(llm, "difference")) == 1 and len(calls(llm, "difference_check")) == 1
    # the checker is the challenger role, not the proposer's
    assert calls(llm, "difference")[0]["role"] == "explorer"
    assert calls(llm, "difference_check")[0]["role"] == "refine_challenger"
    record = mine(empty, run_dir)[card["id"]]
    assert record["difference_statement"]["changed_equation"] == "update[0]"
    assert "difference_statement" not in record["missing"]
    [judged] = events(orch, "difference_judged")
    assert judged["accepted"] and judged["failures"] == []


def test_a_difference_statement_is_required_and_invented_refs_drop_the_card_after_one_send_back(run_dir, empty):
    invented = statement(literature=False, refs=[{"kind": "literature", "ref": "9999.99999", "title": "x",
                                                   "what_was_tried": "x", "why_it_failed": "x"}])
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(invented, checker(accept=False)))
    run(orch)
    card = card_of(orch)
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["verdict"] == "fail" and s2["reasons"][0].startswith("gate: no_difference:")
    assert len(calls(llm, "difference")) == 2 and len(calls(llm, "difference_check")) == 2
    send_back = calls(llm, "difference")[1]["user"]
    assert "SEND-BACK" in send_back and "closest prior X used Y" in send_back and "9999.99999" in send_back
    assert [c["used"] for c in events(orch, "budget_charge") if c["loop"] == "send_back"] == [1]
    assert card["difference_statement"]["closest_prior"][0]["ref"] == "9999.99999"     # the last one is kept
    record = mine(empty, run_dir)[card["id"]]
    assert record["kill_codes"] == ["no_difference"] and record["outcome"] == "dead"
    assert {e["kind"] for e in record["evidence"]} >= {"reduction_critic", "difference"}


def test_host_checks_name_each_missing_part():
    families, _ = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
    req = difference.Requirements(literature={"2401.00001": "A paper", "arxiv:2401.00001": ""}, outcomes={},
                                  r3=[{"record_id": "oc-aaaaaaaaaaaa", "card_id": "card-0000d001", "exact": True,
                                       "jaccard": 1.0, "causes": [{"id": "oc-aaaaaaaaaaaa/reduces_to_known:x",
                                                                   "detail": "reduced"}]}],
                                  family=families.get("hopfield_associative_memory"), need=2, families=families)
    card = make_card("card-0000000a")
    bad = {"closest_prior": [{"kind": "literature", "ref": "2401.00001", "title": "", "what_was_tried": "a",
                              "why_it_failed": "b"}],
           "what_was_tried": "a", "why_it_failed": "b", "precise_change": "c", "changed_equation": "update[7]",
           "cause_ids_addressed": []}
    problems = difference.host_problems(bad, card, req)
    assert set(problems) == {"closest_prior:count", "changed_equation", "r3_prior:oc-aaaaaaaaaaaa",
                             "r3:oc-aaaaaaaaaaaa/reduces_to_known:x", "family:hopfield_associative_memory"}


# ----- R3 -------------------------------------------------------------------------------------------------------------


def test_r3_kills_a_variant_that_ignores_the_recorded_cause(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=config(store), script=script(statement(causes=False), checker()))
    run(orch)
    card = card_of(orch)
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["verdict"] == "fail" and s2["reasons"][0].startswith("gate: not_radical:R3:")
    dead = store.current()[0]
    cause = dead.causes()[0]["id"]
    assert cause in calls(llm, "difference")[0]["user"] and cause in calls(llm, "difference")[1]["user"]
    record = mine(store, run_dir)[card["id"]]
    assert record["kill_codes"] == ["not_radical:R3"]
    assert record["budget"]["loops"] == {"send_back": 1}


def test_r3_kills_a_variant_that_cites_the_cause_but_does_not_change_its_equation(run_dir, store):
    orch, _ = make_orchestrator(run_dir, cfg=config(store), script=script(statement(), checker(addressed=False)))
    run(orch)
    s2 = card_of(orch)["status"]["verdicts"]["S2"]
    assert s2["reasons"][0].startswith("gate: not_radical:R3:")
    assert "the challenger finds the cause not addressed" in " ".join(s2["reasons"])


def test_r3_passes_a_variant_that_addresses_the_recorded_cause(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=config(store), script=script(statement(), checker()))
    run(orch)
    card = card_of(orch)
    assert card["status"]["verdicts"]["S2"]["verdict"] == "pass"
    cause = store.current()[0].causes()[0]["id"]
    assert card["difference_statement"]["cause_ids_addressed"] == [cause]
    judged = events(orch, "difference_judged")[0]
    assert judged["r3_targets"] == [store.current()[0].record_id]


def test_r3_applies_to_a_near_signature_match(run_dir, store):
    """A variant with one extra operator is no exact match but is within Jaccard jaccard_min of the dead idea."""
    def variant(i, user):
        out = draft(0, distinct=False, user=user)
        out["mechanism"]["update"][1]["plain"] += " (variant: hash)"
        return out
    orch, _ = make_orchestrator(run_dir, cfg=config(store), script=script(statement(causes=False), checker(),
                                                                           explorer=variant))
    run(orch)
    history = events(orch, "own_history")[0]
    assert history["exact"] == [] and history["near"] == [store.current()[0].record_id]
    assert card_of(orch)["status"]["verdicts"]["S2"]["reasons"][0].startswith("gate: not_radical:R3:")


def test_without_enforce_r3_the_causes_are_shown_but_not_gated(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=config(store, r3=False), script=script(statement(causes=False),
                                                                                       checker()))
    run(orch)
    assert card_of(orch)["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert "RULE R3" not in calls(llm, "difference")[0]["user"]


def test_with_the_statement_off_enforced_r3_keeps_the_phase5_rule(run_dir, store):
    orch, llm = make_orchestrator(run_dir, cfg=config(store, statement=False), script=script(statement(), checker()))
    run(orch)
    assert card_of(orch)["status"]["verdicts"]["S2"]["reasons"][-1].startswith("gate: not_radical:R3")
    assert not calls(llm, "difference")


# ----- send-back once, then drop --------------------------------------------------------------------------------------


def test_a_send_back_answered_with_the_required_difference_passes(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(accept=[False, True])))
    run(orch)
    card = card_of(orch)
    assert card["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert any("accepted after the send-back" in r for r in card["status"]["verdicts"]["S2"]["reasons"])
    assert [j["accepted"] for j in events(orch, "difference_judged")] == [False, True]


def test_the_send_back_is_used_once_then_the_card_is_dropped(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(accept=False)))
    run(orch)
    assert len(calls(llm, "difference")) == 2 and len(calls(llm, "difference_check")) == 2
    assert [j["attempt"] for j in events(orch, "difference_judged")] == [1, 2]
    assert card_of(orch)["status"]["verdicts"]["S2"]["reasons"][0].startswith("gate: no_difference:")


def test_a_full_lineage_refuses_the_send_back_and_the_card_drops_at_once(run_dir, empty):
    cfg = config(empty)
    cfg["budgets"]["lineage"]["ideas"] = 1
    orch, llm = make_orchestrator(run_dir, cfg=cfg, script=script(statement(), checker(accept=False)))
    run(orch)
    assert len(calls(llm, "difference")) == 1
    record = mine(empty, run_dir)[card_of(orch)["id"]]
    assert record["kill_codes"] == ["no_difference", "budget:send_back"]


# ----- family exclusion -----------------------------------------------------------------------------------------------


RENAMED_LA = critic(reduces_to="linear_attention", confidence=0.9, family="none",
                    family_why="The ledger of rotations is a new object.",
                    mapping="S_t = S_{t-1} + phi(k_t) v_t^T with phi the phase map; the 'rotation ledger' is S.",
                    difference_if_any="LEARNING_RULE: the phases are learned by a local rule", radical_enough=False,
                    why="Renamed: the rotation ledger is an outer-product memory read with phi(q).")


def test_a_renamed_linear_attention_variant_is_excluded_despite_the_learning_rule_prefix(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(),
                                                                             reduction_critic=RENAMED_LA))
    run(orch)
    card = card_of(orch)
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["verdict"] == "reduces_to_known" and s2["reasons"][-1].startswith("gate: excluded_family:linear_attention:")
    assert not calls(llm, "difference")
    critic_call = calls(llm, "reduction_critic")[0]
    assert "OUT OF SCOPE FAMILIES" in critic_call["user"] and "family" in critic_call["schema"]["required"]
    record = mine(empty, run_dir)[card["id"]]
    assert record["kill_codes"] == ["excluded_family:linear_attention"]


def test_without_the_exclusion_the_learning_rule_prefix_still_passes_the_reduction(run_dir, empty):
    old = {k: v for k, v in RENAMED_LA.items() if k not in ("family", "family_why")}
    orch, llm = make_orchestrator(run_dir, cfg=config(empty, exclusion=False, statement=False),
                                  script=script(statement(), checker(), reduction_critic=dict(old, radical_enough=True)))
    run(orch)
    assert card_of(orch)["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert "OUT OF SCOPE FAMILIES" not in calls(llm, "reduction_critic")[0]["user"]


def test_a_delta_rule_reduction_is_named_as_such_by_its_keywords(run_dir, empty):
    delta = critic(reduces_to="linear_attention", family="none", radical_enough=False,
                   mapping="S_t = S_{t-1} + beta (v - S k) k^T: the gated DeltaNet delta rule",
                   why="This is the delta rule (fast weights), which RLS and MesaNet solve in closed form.")
    orch, _ = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(),
                                                                           reduction_critic=delta))
    run(orch)
    assert mine(empty, run_dir)[card_of(orch)["id"]]["kill_codes"] == ["excluded_family:delta_rule"]


def test_the_critics_family_wins_for_a_verified_reduction(run_dir, empty):
    ssm = critic(reduces_to="other_known", family="ssm", mapping="h_t = a h_{t-1} + b x_t", radical_enough=False,
                 why="A diagonal linear recurrence.")
    orch, _ = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(), reduction_critic=ssm))
    run(orch)
    assert mine(empty, run_dir)[card_of(orch)["id"]]["kill_codes"] == ["excluded_family:ssm"]


def test_an_excluded_family_needs_a_verified_reduction():
    families, keywords = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
    assert difference.excluded_reduction(RENAMED_LA, "refuted", families, keywords) is None
    assert difference.excluded_reduction(RENAMED_LA, "unverified", families, keywords) is None
    assert difference.excluded_reduction(RENAMED_LA, "verified", families, keywords).id == "linear_attention"
    hop = critic(reduces_to="hopfield_or_kernel_memory", family="none")
    assert difference.excluded_reduction(hop, "verified", families, keywords) is None
    # "rls" inside another word does not count as the delta rule
    worlds = dict(RENAMED_LA, why="Holographic worlds and girls", mapping="outer product")
    assert difference.excluded_reduction(worlds, "verified", families, keywords).id == "linear_attention"


# ----- known families: Hopfield ----------------------------------------------------------------------------------------


HOPFIELD = critic(family="hopfield_associative_memory",
                  family_why="The read is a softmax over stored patterns, as in modern Hopfield networks.")


def test_hopfield_is_allowed_with_a_real_recorded_difference(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(),
                                                                             reduction_critic=HOPFIELD))
    run(orch)
    card = card_of(orch)
    assert card["status"]["verdicts"]["S2"]["verdict"] == "pass"
    assert "hopfield_associative_memory" in calls(llm, "difference")[0]["user"]
    assert {"kind": "family", "ref": "hopfield_associative_memory"}.items() <= \
        [p for p in card["difference_statement"]["closest_prior"] if p["kind"] == "family"][0].items()
    assert events(orch, "difference_judged")[0]["family"] == "hopfield_associative_memory"


def test_hopfield_without_a_difference_is_killed(run_dir, empty):
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(family=False),
                                                                             checker(real=False),
                                                                             reduction_critic=HOPFIELD))
    run(orch)
    card = card_of(orch)
    s2 = card["status"]["verdicts"]["S2"]
    assert s2["reasons"][0].startswith("gate: no_difference:") and "hopfield_associative_memory" in s2["reasons"][0]
    assert mine(empty, run_dir)[card["id"]]["kill_codes"] == ["no_difference"]


def test_hopfield_listed_but_judged_not_different_is_killed(run_dir, empty):
    orch, _ = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(real=False),
                                                                           reduction_critic=HOPFIELD))
    run(orch)
    assert card_of(orch)["status"]["verdicts"]["S2"]["reasons"][0].startswith("gate: no_difference:")


def test_a_verified_hopfield_reduction_still_reduces(run_dir, empty):
    verified = critic(reduces_to="hopfield_or_kernel_memory", family="hopfield_associative_memory",
                      mapping="softmax read over stored keys", radical_enough=False)
    orch, llm = make_orchestrator(run_dir, cfg=config(empty), script=script(statement(), checker(),
                                                                             reduction_critic=verified))
    run(orch)
    assert mine(empty, run_dir)[card_of(orch)["id"]]["kill_codes"] == ["reduces_to_known:hopfield_or_kernel_memory"]
    assert not calls(llm, "difference")


# ----- outcome records and resume --------------------------------------------------------------------------------------


def test_every_new_kill_code_is_in_the_enumeration():
    for code in ("excluded_family:linear_attention", "excluded_family:delta_rule", "no_difference", "not_radical:R3",
                 "budget:send_back", "stall:send_back"):
        check_kill_code(code)


def test_a_full_phase7_line_from_s0_through_r_and_s2_to_s6_and_resume(run_dir, store):
    """Dialogue and difference statement together: R refines, S2 checks the statement against the dead idea. A run
    stopped inside S2 resumes there."""
    from hybrid.tests.research.test_refine import asks, proposer, question

    cfg = config(store)
    cfg["refine"]["enabled"] = True
    # every claim supported, so S6 passes and the record is finished (s6_verdict.ledger_verdict)
    s = script(statement(), checker(), refine_challenger=asks([question()], []), refine_proposer=proposer(),
               analyst=analyst_supports_every_claim)
    first = dict(cfg, budgets=dict(cfg["budgets"], max_calls=7))   # explorer, critic, 3 R calls, prior art, critic
    orch, llm = make_orchestrator(run_dir, cfg=first, script=s)
    run(orch)
    card_id = orch.store.list_cards()[0]
    assert orch.store.get_card(card_id)["status"]["stage"] == "R"
    resumed, llm2 = make_orchestrator(run_dir, cfg=cfg, script=s, resume=True)
    assert run(resumed)["reason"] == "complete"
    card = resumed.store.get_card(card_id)
    assert card["status"]["stage"] == "S6"
    assert set(card["status"]["verdicts"]) == {"S0", "S1", "R", "S2", "S3", "S4", "S5", "S6"}
    assert card["difference_statement"]["cause_ids_addressed"] == [store.current()[0].causes()[0]["id"]]
    assert not calls(llm2, "refine_challenger")
    record = mine(store, run_dir)[card_id]
    assert record["outcome"] == "finished" and record["premises"][0]["added_by"] == "dialogue"
    from hybrid.research.inspect import inspect_run
    assert inspect_run(run_dir) == 0
    from hybrid.research.inspect import phase7_metrics
    metrics = phase7_metrics(list(resumed.ledger.iter_events()), resumed.store)
    assert metrics["dialogue_ends"] == {"converged": 1} and metrics["difference"] == {"attempt 1 accepted": 1}
