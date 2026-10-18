"""Offline replay of the S2 reductions of runs/run-20261010T181300Z under the Phase 7 gate (the Phase 7 exit criterion of
DISCOVERY-PIPELINE-PLAN.md section 17), with recorded critic outputs and no live model.

`extract` reads the source run read-only (hybrid.research.outcome_records.RunView) and writes, for every card that
reached the reduction critic: the card, its recorded prior-art result, the recorded reduction critic answer, the
recorded outcome of the numeric check, the prior_art_delta answer, and the card's original S2 verdict and kill codes.

`replay` builds a new run with those cards at S1 pass and resumes it through the real orchestrator and the real S2
stage. A fake model client answers each call from the recording: the reduction critic with the recorded answer (plus
`family: "none"`, since no recorded critic was asked for a family, so the exclusion has to come from `reduces_to` and
the families.toml keywords, as it would for a critic misled by a rename), the check with a program reporting the
recorded difference as one comparison with the claimed family (FIX-10: the host decides; the replay trusts that the
recorded check compared the claimed form; FIX-12: the replayed critic declares that form as comparator `recorded`, and
the check reports it under that id), the prior-art classifier with the recorded label and sources. The
difference statement is off, so a card that passes the reduction gates passes S2 as before (it would need a live difference statement); outcomes
are off, so R3 is not part of this replay (test_difference.py covers it).

Expected codes. A verified reduction into the linear-attention or delta-rule family of families.toml becomes
excluded_family:<id> (the plan, section 1.2: "a verified reduction into one of these families fails the card
regardless of that prefix"); every other death keeps its original code."""
from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path

from hybrid.tests.research.critic_answers import novelty_fields

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = "run-20261010T181300Z"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "phase7_replay" / f"{SOURCE_RUN}.json"
CRITIC_KEYS = ("reduces_to", "mapping", "confidence", "difference_if_any", "why",
               "textbook_transplant", "textbook_why")
DIFF = re.compile(r"max_abs_diff ([0-9.eE+\-]+|None)")
CARD_ID = re.compile(r'"id": "(card-[0-9a-f]{8})"')


def extract(run_dir: Path) -> dict:
    from hybrid.research.graveyard import deciding_reason
    from hybrid.research.outcome_records import RunView, build_record

    view = RunView(run_dir)
    cards = []
    for card_id in view.card_ids():
        found = view.latest_artifact(card_id, "reduction_critic")
        if found is None:
            continue
        critic = found[0]
        card, _, _ = view.latest_card(card_id)
        prior = view.latest_artifact(card_id, "prior_art")[0]
        s2 = card["status"]["verdicts"]["S2"]
        diffs = [m.group(1) for r in s2["reasons"] for m in [DIFF.search(r)] if m]
        record = build_record(view, card_id, "replay")
        shown = {k: v for k, v in card.items() if k != "status"}
        cards.append({
            "card_id": card_id, "card": shown,
            "prior_art": {key: prior.get(key) for key in ("label", "evidence", "notes", "queries", "candidates")},
            "critic": {key: critic[key] for key in CRITIC_KEYS} | novelty_fields(critic),
            "reduction_outcome": critic["reduction_outcome"], "prior_art_delta": critic.get("prior_art_delta"),
            "max_abs_diff": None if not diffs or diffs[0] == "None" else float(diffs[0]),
            "original": {"verdict": s2["verdict"], "gate": deciding_reason(s2["reasons"])[:400],
                         "outcome": record["outcome"], "terminal_stage": record["terminal_stage"],
                         "kill_codes": record["kill_codes"]}})
    return {"source_run": view.run_id, "cards": cards}


def recorded_candidates(keys: list[str]) -> list[dict]:
    """Candidate records for the bare keys an earlier run stored (the title and excerpt were not kept then)."""
    return [{"key": key, "source": key.split(":", 1)[0], "source_id": key.split(":", 1)[1],
             "title": f"recorded paper {key}", "year": None, "url": "", "excerpt": "(not recorded)"}
            for key in keys]


def expected_codes(entry: dict) -> list[str]:
    """The Phase 7 code for a recorded card, from the plan's rule and the recorded critic text (not from the code under
    test): a verified linear-attention reduction is excluded, to the delta-rule family when the critic's own words
    name a delta-rule member (DeltaNet, RLS, MesaNet, fast weights, the delta rule), else to linear attention."""
    critic = entry["critic"]
    if entry["reduction_outcome"] == "verified" and critic["reduces_to"] == "linear_attention":
        text = " ".join([critic["mapping"] or "", critic["why"], critic["difference_if_any"]]).lower()
        delta = re.search(r"\b(delta rule|deltanet|mesanet|rls|fast[- ]weights?)\b", text) is not None
        return ["excluded_family:delta_rule" if delta else "excluded_family:linear_attention"]
    return list(entry["original"]["kill_codes"])


def _program(obj: dict) -> str:
    return "import json\nprint(json.dumps(%r))\n" % (obj,)


def replay(fixture: dict, work: Path) -> list[dict]:
    """Run every recorded card through the real S2 with recorded answers. Returns one row per card."""
    from hybrid.bench.local import LocalExecutor
    from hybrid.executor.lock import read_lock
    from hybrid.research.ledger import Ledger
    from hybrid.research.orchestrator import Orchestrator
    from hybrid.research.outcome_records import RunView, build_record
    from hybrid.research.store import Store
    from hybrid.tests.research.test_orchestrator_fakes import FakeBench, FakeLLM, orch_config

    entries = {e["card_id"]: e for e in fixture["cards"]}
    run_dir = (Path(work) / "run-20261013T000700Z").resolve()
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    for entry in fixture["cards"]:
        card = copy.deepcopy(entry["card"])
        # One version per card here; parents are dropped (an escape card's S2 reads its parent's reduction only when
        # the parent has one, and the fake critic answers from the recording whatever the message says).
        card.update(version=1, parent_ids=[], status={"stage": "S0", "verdicts": {}})
        store.put_card(card)
        ledger.append("card_created", "orchestrator", {"card_id": card["id"], "op": "new", "parents": [],
                                                       "role": "explorer", "model": "recorded"})
        for stage in ("S0", "S1"):
            store.set_verdict(card["id"], stage, "pass", [f"replayed from {fixture['source_run']}"],
                              ["call-000000000000"])

    def entry_for(user: str) -> dict:
        return entries[CARD_ID.search(user).group(1)]

    def critic(i, user):
        # FIX-12: the recorded critic declared no comparators; the replay declares the published form it named.
        recorded = entry_for(user)["critic"]
        comparators = [] if recorded["reduces_to"] == "none" else [{
            "id": "recorded", "base_family": recorded["reduces_to"], "base_form": "recorded known form",
            "update": recorded["mapping"], "read": recorded["mapping"], "transformations": [],
            "parameter_mapping": recorded["mapping"], "limit_only": False}]
        # the fixture was written before decision R2C: its radical_enough answer is read as the new known_work fields
        novelty = novelty_fields(recorded) if "radical_enough" in recorded else {}
        answer = {key: value for key, value in recorded.items() if key != "radical_enough"}
        return dict(answer, **novelty, family="none", comparators=comparators,
                    family_why="(recorded critic; no family question was asked in the source run)")

    def check(i, user):
        # FIX-10: the recorded outcome is replayed as one comparison with the claimed family, at the recorded
        # difference (the host decides from it). A recorded verification with no difference replays at 0.0, a
        # recorded refutation without one at 1.0, an unverified check as a report with no comparison.
        entry = entry_for(user)
        outcome, diff = entry["reduction_outcome"], entry["max_abs_diff"]
        comparisons = [{
            "declared_id": "recorded", "comparator": "recorded known form", "family": entry["critic"]["reduces_to"],
            "transformations": [], "limit_only": False, "reference": "recorded check", "inputs_tested": 1,
            "max_abs_diff": diff if diff is not None else (0.0 if outcome == "verified" else 1.0)}]
        if outcome not in ("verified", "refuted"):
            # an unverified recorded check replays as a report that leaves out the declared comparator
            comparisons = []
        return {"check_code": _program({"comparisons": comparisons, "notes": "replayed"}),
                "explanation": f"replays the recorded check outcome ({outcome})"}

    def delta(i, user):
        recorded = entry_for(user)["prior_art_delta"]
        return recorded or {"adds_untested_element": False, "element": "", "why": "none recorded"}

    async def classify(card_text, llm_complete, sink):
        entry = entries[CARD_ID.search(card_text).group(1)]
        prior = entry["prior_art"]
        await llm_complete("prior_art", "query system", card_text, {"type": "object"})
        evidence = [dict(item, source="arxiv") for item in prior["evidence"]]
        return {"label": prior["label"], "evidence": evidence, "notes": prior["notes"] or "recorded",
                "queries": prior["queries"] or [],
                "candidates": recorded_candidates(prior["candidates"] or ["arxiv:0000.00000"]),
                "rejected_evidence": []}

    cfg = orch_config()
    cfg["s2"]["family_exclusion"] = True
    cfg["budgets"].update(max_cards=len(entries), max_calls=2000)
    cfg["explore"]["p_escape"] = 0.0
    llm = FakeLLM(run_dir, ledger, {"reduction_critic": critic, "reduction_check": check, "prior_art_delta": delta})
    orch = Orchestrator(cfg, run_dir, llm=llm, executor=LocalExecutor(run_dir.name, run_dir / "jobs" / "receipts"),
                        bench=FakeBench(), image=read_lock()["reference"], ledger=ledger, classify_prior_art=classify,
                        resume=True)
    asyncio.run(orch.run())
    view = RunView(run_dir)
    rows = []
    for card_id, entry in sorted(entries.items()):
        record = build_record(view, card_id, "replay")
        s2 = view.latest_card(card_id)[0]["status"]["verdicts"]["S2"]
        rows.append({"card_id": card_id, "title": entry["card"]["title"],
                     "reduces_to": entry["critic"]["reduces_to"], "check": entry["reduction_outcome"],
                     "learning_rule_prefix": entry["critic"]["difference_if_any"].strip().startswith("LEARNING_RULE:"),
                     "original_codes": entry["original"]["kill_codes"], "original_outcome": entry["original"]["outcome"],
                     "replay_s2": s2["verdict"], "replay_gate": next((r for r in s2["reasons"] if r.startswith("gate:")),
                                                                     "")[:300],
                     "replay_codes": record["kill_codes"] if record["terminal_stage"] == "S2" else [],
                     "replay_terminal": record["terminal_stage"],
                     "expected_codes": expected_codes(entry) if entry["original"]["terminal_stage"] == "S2" else []})
    return rows


def write_fixture(run_dir: Path = ROOT / "runs" / SOURCE_RUN) -> Path:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(extract(run_dir), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    return FIXTURE


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))
