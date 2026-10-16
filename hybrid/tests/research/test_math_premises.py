"""Definitions, premises, circularity, conditional claims and cited sources in stages M and C (the fix after the
first live run on card-46dbb349, 2026-10-12: conventions listed as assumptions left every claim unproven, and a
premise that restated a claim stayed open). Scripted LLMs; the math jobs run through math_fakes.MathExecutor."""
import json
from pathlib import Path

import pytest

from hybrid.research import citations
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.store import Store
from hybrid.tests.literature.oai_support import build_fixture_index
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_math_stages import Harness

SAMPLING = "queries are drawn uniformly from the keys that were written, without replacement"
ACCOUNTING = "cost counts unit-cost scalar multiply-adds and stored scalar words; no autograd tape at inference"
COMPLEXITY = ["complexity.memory", "complexity.time_per_token", "complexity.total_sequence"]
RECORD = {"source": "arxiv", "id": "2401.00002", "title": "Hopfield networks and associative memory capacity",
          "abstract": "Modern Hopfield networks store exponentially many patterns for associative recall.",
          "year": 2024, "url": "https://arxiv.org/abs/2401.00002", "via": "arxiv_local"}
SUPPORTING = "Modern Hopfield networks store exponentially many patterns for associative recall."
CITATION = {"source": "arxiv", "id": "2401.00002", "quote": "store exponentially many patterns"}
LEGACY = Path(__file__).parent / "fixtures" / "math_legacy"


def harmless_magic_x(pid: str) -> dict:
    """Runnable magic-X code that carries no information about the answer (the baseline stays at chance)."""
    return {"premise_id": pid, "code": mf.references.POSITION_PARITY, "claim": dict(mf.references.MAGIC_X_CLAIM),
            "quote": mf.CAPABILITY_1_QUOTE, "why": "the premise as code"}


def lookup_of(records: dict):
    calls = []

    def lookup(source: str, ident: str) -> dict:
        calls.append((source, ident))
        if (source, ident) not in records:
            raise citations.LookupFailed(f"arXiv {ident} is not in the local index")
        return records[(source, ident)]

    lookup.calls = calls
    return lookup


# ----- definitions -------------------------------------------------------------------------------------------------

def test_conventions_are_definitions_frozen_with_the_statements_and_shown_to_the_match_checker(tmp_path):
    fixer = mf.fixer_output(definitions=[mf.definition("D1", SAMPLING), mf.definition("D2", ACCOUNTING, "accounting")],
                            capability_1_definitions=["D1"], complexity_definitions=["D2"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer))
    assert h.m() == "C"
    # No ledger entry, no status: every claim is decided by its checks alone.
    verdict = h.verdict("M")
    assert verdict["verdict"] == "pass" and verdict["definitions"] == ["D1", "D2"]
    assert {c["claim_id"]: c["status"] for c in verdict["claims"]}["capability.1"] == "proven"
    ledger = h.ledger_artifact("M")
    assert ledger["premises"] == [] and [d["id"] for d in ledger["definitions"]] == ["D1", "D2"]
    assert all(set(d) == {"id", "kind", "text", "sha256"} for d in ledger["definitions"])
    # Frozen and hashed with the statements.
    frozen = h.events("math_statements_frozen")[0]["data"]
    record = json.loads((h.run_dir / frozen["path"]).read_text())
    assert {d["id"]: d["sha256"] for d in record["definitions"]} == {k: frozen["hashes"][k] for k in ("D1", "D2")}
    assert next(o for o in record["obligations"] if o["id"] == "O5")["definitions"] == ["D1"]
    # The prover implements the definitions; the match checker reads them with the statements.
    prover_user = next(c["user"] for c in h.llm.calls if c["role"] == "prover")
    assert "frozen_definitions" in prover_user and SAMPLING in prover_user
    assert h.c() == "S4"
    back = next(c["user"] for c in h.llm.calls if c["prompt"] == "statement_backtranslate")
    match = next(c["user"] for c in h.llm.calls if c["prompt"] == "statement_match")
    assert SAMPLING in back and ACCOUNTING in back and "capability" not in back
    assert SAMPLING in match and "DEFINITIONS:" in match
    checks = h.verdict("C")["definition_checks"]
    assert [(c["definition_id"], c["verdict"]) for c in checks] == [("D1", "convention"), ("D2", "convention")]


def test_a_definition_cannot_carry_a_status():
    record = json.loads((LEGACY / "math_statements.json").read_text())
    new = {k: v for k, v in record.items() if k != "assumptions"}
    new["obligations"] = [dict({k: v for k, v in o.items() if k != "assumptions"}, premises=[], definitions=["D1"])
                          for o in record["obligations"]]
    definition = {"id": "D1", "kind": "sampling", "text": SAMPLING, "sha256": "0" * 64}
    validate("math_statement", dict(new, definitions=[definition], premises=[]))
    with pytest.raises(SchemaViolation):
        validate("math_statement", dict(new, definitions=[dict(definition, status="open")], premises=[]))


def test_a_definition_that_is_a_substantive_premise_is_a_mismatch(tmp_path):
    fixer = mf.fixer_output(definitions=[mf.definition("D1", "every key is stored without collision", "other")],
                            capability_1_definitions=["D1"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, match=mf.match_output(substantive=("D1",))))
    assert h.m() == "C"
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["statement_mismatch"])
    claim = h.claim("C", "capability.1")
    assert claim["status"] == "blocked" and "definition D1 is a substantive premise" in claim["phrase"]
    assert h.claim("C", "capability.2")["status"] == "observed"


# ----- circularity -------------------------------------------------------------------------------------------------

def test_a_circular_premise_fails_the_claims_it_restates_and_is_never_left_open(tmp_path):
    envelope = mf.premise("A6", "the recurrence costs (4M+12)d^2+2Vd operations and 6d^2+Vd words per token",
                          "other")
    unrelated = mf.premise("A2", "stored keys are nearly orthogonal", "store")
    fixer = mf.fixer_output(premises=[envelope, unrelated], complexity_premises=["A6"], capability_1_premises=["A2"])
    # A6 implies the complexity claims it supports; A2 implies a claim it does not support, which is not circular.
    match = mf.match_output(implies={"A6": COMPLEXITY, "A2": ["complexity.memory"]})
    challenger = mf.challenger_output(magic_x=[harmless_magic_x("A6"), harmless_magic_x("A2")])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=challenger, match=match))
    assert h.m() == "C"
    assert h.claim("M", "complexity.memory")["status"] == "conditional"
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["circular:A6"])
    for cid in COMPLEXITY:
        claim = h.claim("C", cid)
        assert claim["status"] == "circular" and claim["circular_premises"] == ["A6"]
        assert claim["tier"] == "sympy_exact" and "A6" in claim["phrase"]
    ledger = h.ledger_of("C")
    assert ledger["A6"]["status"] == "circular" and "complexity.time_per_token" in ledger["A6"]["evidence"]
    assert ledger["A2"]["status"] == "open"
    assert h.claim("C", "capability.1")["status"] == "conditional"
    flags = {c["premise_id"]: c["circular"] for c in verdict["circularity"]}
    assert flags == {"A6": True, "A2": False}
    # The last statement_match is C's; the pre-freeze match in M (FIX-19) carries no premises.
    match_user = [c["user"] for c in h.llm.calls if c["prompt"] == "statement_match"][-1]
    assert "CARD CLAIMS:" in match_user and envelope["text"] in match_user


# ----- conditional claims ------------------------------------------------------------------------------------------

def test_a_conditional_claim_with_a_testable_premise_advances_and_its_measurement_is_scheduled(tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "an index answers each probe from its own lookup", "store")],
                            capability_1_premises=["A1"])
    challenger = mf.challenger_output(magic_x=[harmless_magic_x("A1")])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=challenger))
    assert h.m() == "C"
    assert h.c() == "S4"
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("pass", [])
    claim = h.claim("C", "capability.1")
    assert claim["status"] == "conditional" and claim["open_premises"] == ["A1"] and claim["label"] == "lean-proven"
    assert verdict["conditional"] == ["capability.1"]
    assert verdict["scheduled_measurements"] == [{"premise_id": "A1", "text": "an index answers each probe from its "
                                                  "own lookup", "measurement_plan": mf.PLAN,
                                                  "claims": ["capability.1"], "status": "deferred_to_test_design"}]
    item = verdict["magic_x"][0]
    assert item["ran"] and item["status"] == "discharged" and not item["missing"]
    assert any("deferred to test design" in r for r in h.card_verdict("C")["reasons"])
    # Executed-on-examples is never called proven, conditional or not.
    observed = mf.fixer_output(premises=[mf.premise("A1", "an index answers each probe", "store")],
                               extra_obligations=[])
    observed["obligations"][5]["premises"] = ["A1"]
    h2 = Harness(tmp_path / "observed", mf.math_script(fixer=observed,
                                                         challenger=mf.challenger_output(
                                                             magic_x=[harmless_magic_x("A1")])))
    h2.m()
    claim = h2.claim("M", "capability.2")
    assert claim["status"] == "conditional" and "proven" not in claim["phrase"]
    assert claim["phrase"].startswith("conditional on A1: observed on the listed sizes")
    validate("math_verdict", h2.verdict("M"))


@pytest.mark.parametrize("premise, why", [
    (mf.premise("A1", "children hear every word at least twice", "prior", testable_by="citation"),
     "testable by citation but no source was given"),
    (mf.premise("A1", "an oracle segments the input", plan=None), "no measurement plan"),
])
def test_a_conditional_claim_with_an_untestable_open_premise_does_not_advance(tmp_path, premise, why):
    fixer = mf.fixer_output(premises=[premise], capability_1_premises=["A1"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer))
    assert h.m() is None
    verdict = h.verdict("M")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("undetermined", ["unproven:A1"])
    assert h.claim("M", "capability.1")["status"] == "conditional"
    assert why in h.ledger_of("M")["A1"]["basis"]
    first = json.loads((h.run_dir / h.events("math_statements_frozen")[0]["data"]["path"]).read_text())
    assert any("premise A1: testable_by" in p for p in first["first_problems"])


# ----- magic-X is required for code-testable premises ----------------------------------------------------------------

@pytest.mark.parametrize("entries, evidence", [
    ([], "no runnable code"),
    ([{"premise_id": "A1", "code": mf.references.POSITION_PARITY, "claim": None, "quote": "",
       "why": "the advantage is not a construction task"}], "no construction task fits"),
    ([{"premise_id": "A1", "code": "def x(prefix, layout, ops):\n    return undefined_name\n",
       "claim": dict(mf.references.MAGIC_X_CLAIM), "quote": mf.CAPABILITY_1_QUOTE, "why": "broken"}],
     "candidate_error"),
])
def test_missing_magic_x_code_for_a_testable_premise_fails_the_challenge(tmp_path, entries, evidence):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "an index answers each probe", "store")],
                            capability_1_premises=["A1"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=mf.challenger_output(magic_x=entries)))
    assert h.m() == "C"
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("fail", ["magic_x_missing:A1"])
    item = verdict["magic_x"][0]
    assert item["missing"] and item["required"] and evidence in item["evidence"]


# ----- cited premises ----------------------------------------------------------------------------------------------

def cited_harness(tmp_path, records, answers, *, citation=CITATION):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "a dense associative memory stores many patterns", "store",
                                                 testable_by="citation", citation=citation)],
                            capability_1_premises=["A1"])
    h = Harness(tmp_path, mf.math_script(fixer=fixer, challenger=mf.challenger_output(citations=answers)))
    h.ctx.lookup_source = lookup_of(records)
    return h


def test_a_cited_premise_with_a_verified_source_is_cited(tmp_path):
    answer = {"premise_id": "A1", "supports": True, "record_quote": SUPPORTING, "why": "the abstract states it"}
    h = cited_harness(tmp_path, {("arxiv", "2401.00002"): RECORD}, [answer])
    assert h.m() == "C"
    assert "verified at stage C" in h.ledger_of("M")["A1"]["basis"]
    assert h.c() == "S4"
    entry = h.ledger_of("C")["A1"]
    assert entry["status"] == "cited" and "arxiv:2401.00002" in entry["evidence"]
    assert h.claim("C", "capability.1")["status"] == "proven"
    challenger_user = next(c["user"] for c in h.llm.calls if c["role"] == "challenger")
    assert RECORD["title"] in challenger_user
    # A doing-work premise testable by citation does not need magic-X code.
    assert h.verdict("C")["magic_x"][0]["missing"] is False


@pytest.mark.parametrize("records, answers, reason", [
    ({}, [], "source not resolved: arXiv 2401.00002 is not in the local index"),
    ({("arxiv", "2401.00002"): RECORD},
     [{"premise_id": "A1", "supports": False, "record_quote": "", "why": "about Hopfield capacity, not this use"}],
     "does not support the premise"),
    ({("arxiv", "2401.00002"): RECORD},
     [{"premise_id": "A1", "supports": True, "record_quote": "dense memories never interfere with each other",
       "why": "invented"}], "not text of the resolved record"),
    ({("arxiv", "2401.00002"): RECORD}, [], "gave no verification"),
])
def test_a_cited_premise_without_a_verified_source_stays_open(tmp_path, records, answers, reason):
    h = cited_harness(tmp_path, records, answers)
    assert h.m() == "C"
    assert h.c() is None
    verdict = h.verdict("C")
    assert (verdict["verdict"], verdict["kill_codes"]) == ("undetermined", ["unproven:A1"])
    entry = h.ledger_of("C")["A1"]
    assert entry["status"] == "open" and entry["evidence"] is None and reason in entry["basis"]
    assert h.claim("C", "capability.1")["status"] == "conditional"
    assert h.ctx.lookup_source.calls == [("arxiv", "2401.00002")]


def test_the_default_lookup_reads_the_local_arxiv_index(tmp_path):
    index = build_fixture_index(tmp_path / "arxiv.sqlite")
    lookup = citations.default_lookup({"literature": {"local_index_path": str(index)}})
    found = lookup("arxiv", "arXiv:2401.00002v2")
    assert (found["id"], found["title"], found["via"]) == ("2401.00002", RECORD["title"], "arxiv_local")
    assert citations.quote_in_record(SUPPORTING, found)
    with pytest.raises(citations.LookupFailed, match="not in the local index"):
        lookup("arxiv", "2401.99999")
    with pytest.raises(citations.LookupFailed, match="not an arXiv id"):
        lookup("arxiv", "W12345")
    missing = citations.default_lookup({"literature": {"local_index_path": str(tmp_path / "none.sqlite")}})
    with pytest.raises(citations.LookupFailed, match="not at"):
        missing("arxiv", "2401.00002")


# ----- old records ---------------------------------------------------------------------------------------------------

def test_old_ledgers_still_load_for_inspect(tmp_path, capsys):
    """Artifacts of the first live run (card-46dbb349), written before definitions and premises were separated."""
    run_dir = (tmp_path / "plumb-math-card-46dbb349-20261012T193523Z").resolve()
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    card_id = "card-46dbb349"
    store.put_card(make_card(card_id))
    verdict = json.loads((LEGACY / "math_verdict.json").read_text())
    store.put_artifact("math_statements", Path(verdict["statements_artifact"]).name,
                       (LEGACY / "math_statements.json").read_bytes(), card_id)
    store.put_artifact("assumption_ledger", verdict["assumption_ledger"],
                       (LEGACY / "assumption_ledger.json").read_bytes(), card_id)
    store.put_artifact("math_verdict", "card-46dbb349-v1-2df29c.json", (LEGACY / "math_verdict.json").read_bytes(),
                       card_id)
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert f"{card_id} M undetermined [unproven:A1" in out
    assert "claim complexity.time_per_token: unproven (lean-proven)" in out
    assert "premise A6: open (kind other, testable by not recorded)" in out and "definitions: none" in out
    dossier = (run_dir / "report" / "dossiers" / f"{card_id}.html").read_text()
    assert "Math stages (M and C)" in dossier and "Under the scalar accounting convention" in dossier
    assert "no definitions recorded" in dossier
