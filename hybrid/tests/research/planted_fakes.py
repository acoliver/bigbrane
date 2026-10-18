"""Fakes for the planted-card driver tests (Phase 9): the fixture set, an earlier run holding the dead ideas the T2
and T6 fixtures vary, and a fake model whose answers depend on which fixture card a prompt is about.

The fixture cards (fixtures/planted/, written by results/phase9-20261013-053108/make_fixtures.py) are not planted
cards: each carries a unique name for its one state object, and the fake model looks that name up to decide how the
pipeline treats the card, as a planted card of its type should be treated:

- T1: the archive classifier flags an out-of-scope family at G (KD5: a flag, not a kill) and the reduction critic's
  counted comparison confirms it (killed at P, excluded_family);
- T2: its difference statement does not cite the recorded cause of the dead idea it varies (killed at P, R3);
- T3: M passes with a code-testable premise A1, and the challenger's magic-X code reaches the claim (killed at C);
- T4: the prover's constructions keep too little (killed at M, obligation_failed on the P2 slots);
- T5: the reduction critic finds a verified reduction (killed at P); M and C under the override pass;
- T6: nothing special: G finds the dead idea's mechanism signature (killed at G, duplicate);
- T7: everything passes (reaches T).

Stage M uses the calibration fakes: the eleven profile slots of a card that passes the profile (k2) or fails P2
(k3), plus one SymPy statement per card claim, so M's verdict depends on the profile alone. The fixture cards state no
resources (FIX-16), so stage R asks each card's author for them once; the scripted author answers with the resources
of the calibration card its stage M uses.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from hybrid.bench.local import LocalExecutor
from hybrid.executor.lock import read_lock
from hybrid.math.runner import MathRunner
from hybrid.research import planted as pl
from hybrid.research import planted_seal as ps
from hybrid.research.mathrecords import formal_claims
from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore
from hybrid.research.stages.falsify import claim_items
from hybrid.research.stages.math_contracts import mechanism_parts
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.outcome_fixtures import build_run
from hybrid.tests.research.test_difference import checker, critic, statement
from hybrid.tests.research.test_generation_arms import OPEN_CELL
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, make_classifier

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "planted"
EARLIER_RUN = "run-20260901T000000Z"
# marker (the card's state object name) -> fixture type
MARKERS = {"velarin": "T1", "quorrel": "T2", "tessaly": "T3", "mirrowen": "T4", "dunmore": "T5", "pellaro": "T6",
           "corvane": "T7", "halbrook": "T1", "ostrade": "T2", "frelling": "T3", "gantrel": "T4", "lumbard": "T5",
           "pellwick": "T6", "ashgrove": "T7"}
MAGIC_PREMISE = "an index that answers every probe from its own lookup at no counted cost"
DEAD_CRITIC = {"reduces_to": "other_known", "mapping": "tally[b(x)] += 1", "confidence": 0.9, "difference_if_any": "",
               "radical_enough": False, "why": "It is a counting table, a textbook structure.",
               "textbook_transplant": True, "textbook_why": "-"}


def fixture_cards() -> dict[str, dict]:
    """Every fixture card by marker (both sets)."""
    out = {}
    for path in FIXTURES.glob("*/cards/*.json"):
        card = json.loads(path.read_text())
        out[card["mechanism"]["state"][0]["name"]] = card
    return out


CARDS = fixture_cards()


def marker_of(text: str) -> str | None:
    found = [m for m in MARKERS if f'"name": "{m}"' in text]
    if len(found) > 1:
        raise AssertionError(f"a prompt about more than one fixture card: {found}")
    return found[0] if found else None


def per_card(behaviours: dict[str, object], default) -> callable:
    """A scripted output chosen by the fixture type of the card the prompt is about."""
    def answer(index: int, user: str):
        marker = marker_of(user)
        chosen = behaviours.get(MARKERS.get(marker), default)
        return chosen(marker, index, user) if callable(chosen) else chosen
    return answer


def claim_quote(card: dict, claim_id: str) -> str:
    kind, _, rest = claim_id.partition(".")
    if kind == "complexity":
        return card["claims"]["complexity"][rest]
    if claim_id == "sample_efficiency.prediction":
        return card["claims"]["sample_efficiency"]["prediction"]["statement"]
    return card["claims"]["capabilities"][int(rest) - 1]["prediction"]


def fixer_for(marker: str, *, key: str, premise_on: str | None = None) -> dict:
    """The calibration fixer's profile slots for ``key``, plus one SymPy statement per claim of the card, so every
    claim is formalized; with ``premise_on`` the claim's obligation rests on an open, code-testable premise A1."""
    card = CARDS[marker]
    out = cf.fixer_output(key, card)
    # FIX-17 (DECISIONS.md P3): the calibration limits name k2's or k3's own state entry, which a fixture card does
    # not have; the scripted limit (a bound on what the store holds) rests on the fixture card's one state object.
    anchor, (part, texts) = next((a, p) for a, p in mechanism_parts(card).items() if p[0] == "state")
    # New dicts: fixer_output hands out calibration_fakes.LIMITS itself, which other tests freeze for k2 and k3.
    out["limits"] = [dict(limit, basis={"part": part, "anchor": anchor, "quote": texts[0]}) for limit in out["limits"]]
    out["not_formalized"] = []
    claims = [item["id"] for item in formal_claims(claim_items(card))]
    for index, claim_id in enumerate(claims, len(out["obligations"]) + 1):
        out["obligations"].append(mf.obligation(f"O{index}", claim_id, claim_quote(card, claim_id),
                                                [mf.sympy_statement()], ["A1"] if claim_id == premise_on else []))
    if premise_on:
        out["premises"] = [mf.premise("A1", MAGIC_PREMISE, "store")]
    return out


def magic_x(marker: str) -> dict:
    card = CARDS[marker]
    quote = next(c["prediction"] for c in card["claims"]["capabilities"] if "Exact recall" in c["prediction"])
    return mf.challenger_output(magic_x=[{"premise_id": "A1", "code": mf.references.RETRIEVAL_ORACLE,
                                          "claim": dict(mf.references.MAGIC_X_CLAIM), "quote": quote,
                                          "why": "the index alone answers recall"}])


def resources_answer(key: str) -> dict:
    """FIX-16: the author's answer when stage R asks a fixture card (which states no resources) for them: the
    resources of the calibration card whose fakes script the card's stage M (k2, or k3 for T4). FIX-18: with that
    card's limits."""
    # FIX-18: and the card's P3 limits, those of the same calibration card (state limits; fixer_for rebases their
    # basis onto the fixture card's one state object).
    limits = [{"task": l["task"], "size": l["size"], "resources": l["resources"], "why": l["why"],
               "kind": l["basis"]["part"]} for l in cf.LIMITS[key]]
    return {"resources": cf.card_resources_of(None, key), "limits": limits,
            "why": f"the resources of calibration card {key}, whose scripted mechanism this fixture uses"}


def script(*, overrides: dict | None = None) -> dict:
    """Scripted outputs per prompt for every stage the planted line runs. ``overrides`` replaces the behaviour of a
    type for one prompt: {prompt name: {type: output or callable(marker, index, user)}}."""
    behaviours = {
        "arm_classifier": {"T1": lambda m, i, u: classification("linear_attention")},
        "salad_critic": {},
        "refine_challenger": {},
        "reduction_critic": {"T1": critic(reduces_to="linear_attention", mapping="S_t = S_{t-1} + v_t k_t^T",
                                          confidence=0.9, why="An additive outer-product memory read by a query."),
                             "T5": critic(reduces_to="other_known", mapping="push(x) on openers, pop on closers",
                                          confidence=0.9, why="A push and pop automaton from the textbook.")},
        "difference": {"T2": lambda m, i, u: statement(causes=False)(i, u)},
        "statement_fixer": {"T3": lambda m, i, u: fixer_for(m, key="k2", premise_on="capability.1"),
                            "T4": lambda m, i, u: fixer_for(m, key="k3")},
        "prover": {"T4": lambda m, i, u: cf.prover_output("k3")(i, u)},
        "challenger": {"T3": lambda m, i, u: magic_x(m)},
        "card_resources": {"T4": lambda m, i, u: resources_answer("k3")},
    }
    defaults = {
        "arm_classifier": lambda m, i, u: classification("none"),
        "salad_critic": {"ok": True, "fatal": [], "fixable": [], "notes": "Every symbol is defined."},
        "refine_challenger": {"questions": [], "notes": "nothing left open"},
        "reduction_critic": critic(),
        "difference": lambda m, i, u: statement()(i, u),
        "statement_fixer": lambda m, i, u: fixer_for(m, key="k2"),
        "prover": lambda m, i, u: cf.prover_output("k2")(i, u),
        "challenger": lambda m, i, u: mf.challenger_output(),
        "card_resources": lambda m, i, u: resources_answer("k2"),
    }
    for prompt, table in (overrides or {}).items():
        behaviours.setdefault(prompt, {}).update(table)
    out = {prompt: per_card(behaviours.get(prompt, {}), default) for prompt, default in defaults.items()}
    out.update({"difference_check": checker(), "statement_backtranslate": mf.backtranslate_output,
                "statement_match": mf.match_output()})
    return out


def classification(family: str) -> dict:
    return {**OPEN_CELL, "evidence": {axis: "update[0]" for axis in OPEN_CELL}, "family": family,
            "family_why": "fixture", "difference_is_real": None, "difference_why": "fixture"}


def deps_factory(*, overrides: dict | None = None, calls: list | None = None):
    """Fake dependencies per run directory: the scripted model, the host executor for toy examples and reduction
    checks, a math runner whose library jobs run in this process, and the fake prior-art classifier."""
    def build(ledger):
        run_dir = ledger.run_dir.resolve()
        llm = FakeLLM(run_dir, ledger, script(overrides=overrides))
        if calls is not None:
            calls.append(llm)
        executor = LocalExecutor(run_dir.name, run_dir / "jobs" / "receipts")
        math = MathRunner(cf.FastExecutor(run_dir / "exec"), mf.IMAGE, run_dir)
        return pl.Deps(llm, executor, None, read_lock()["reference"], math, make_classifier("novel", True))
    return build


# ----- the set, sealed, and the earlier run --------------------------------------------------------------------------

def copy_set(name: str, target: Path) -> Path:
    """The fixture set ``name`` copied to ``target`` (cards only; not sealed)."""
    shutil.copytree(FIXTURES / name / "cards", target / "cards")
    return target


def sealed_set(tmp_path: Path, name: str, *, store: OutcomeStore | None = None) -> tuple[ps.PlantedSet, ps.Sealed]:
    """Copy, seal (expectations moved to an owner-only file in tmp_path/sealed) and load a fixture set."""
    directory = copy_set(name, tmp_path / f"set-{name}")
    source = tmp_path / f"{name}.expectations.json"
    shutil.copyfile(FIXTURES / f"{name}.expectations.json", source)
    ps.seal(set_name=f"fixture_{name}", directory=directory, expectations_path=source,
            sealed_directory=tmp_path / "sealed", store=store)
    planted = ps.load_set(directory, name=f"fixture_{name}")
    return planted, ps.read_sealed(planted, tmp_path / "sealed")


def origin_card(card_id: str) -> dict:
    return json.loads((FIXTURES / "origins" / f"{card_id}.json").read_text())


def earlier_run(runs_root: Path) -> Path:
    """An earlier run whose two cards died: card-0000d002 reduced to a counting table at S2 and card-0000d006
    ill-formed at S1."""
    return build_run(runs_root / EARLIER_RUN, [
        (origin_card("card-0000d002"), [("S0", "pass", ["ok"]), ("S1", "pass", ["ok"]),
                                        ("S2", "reduces_to_known", ["gate: update rule reduces to other_known: m"])],
         {"reduction_critic": DEAD_CRITIC}),
        (origin_card("card-0000d006"), [("S0", "pass", ["ok"]), ("S1", "fail", ["gate: fatal finding: x"])], {}),
    ])


def store_with_origins(tmp_path: Path, runs_root: Path, cards=("card-0000d002", "card-0000d006")) -> OutcomeStore:
    """An outcome store holding the earlier run's dead cards named in ``cards``."""
    store = OutcomeStore(tmp_path / "source-outcomes")
    view = RunView(runs_root / EARLIER_RUN)
    for card_id in cards:
        store.append(build_record(view, card_id, "backfill"))
    return store
