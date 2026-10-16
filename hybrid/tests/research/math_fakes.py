"""Fakes for the stage M and C tests: a math executor and scripted outputs of the four math roles.

``MathExecutor`` answers the jobs a real ``hybrid.math.runner.MathRunner`` stages, so the runner's staging, manifest
check, two-run replay and Lean verdict code all run for real:
- Lean jobs (Check.lean): the scripted output a Lean run would print for the staged proof, with the job's nonce.
  A proof that uses sorry prints sorryAx; a proof whose theorem is not stated as ``claim_holds : Claim`` prints
  exact=false (a restated claim); anything else passes.
- library jobs: the kinds in ``host_kinds`` run the real jobmain.py with the host Python (as HostExecutor does);
  z3 and predicate_agreement (z3 is only in the tools image) and, by default, sympy are scripted.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

from hybrid.math import references
from hybrid.math.runner import CHECK_FILE
from hybrid.tests.math.test_library_manifest import HostExecutor

IMAGE = "localhost/bigbrane/tools@sha256:" + "c" * 64
CARD = "card-0000000a"

LEAN_STATEMENT = "import Mathlib\n\ndef Claim : Prop := ∀ n : ℕ, n + 0 = n\n"
GOOD_PROOF = "theorem claim_holds : Claim := fun _ => rfl\n"
SORRY_PROOF = "theorem claim_holds : Claim := by sorry\n"
# A weaker restated proposition: the theorem is not stated as the frozen constant Claim.
WEAKER_PROOF = "theorem claim_holds : ∀ n : ℕ, n + 0 = n ∨ True := fun _ => Or.inr trivial\n"
REDECLARE_PROOF = "def Claim : Prop := True\ntheorem claim_holds : Claim := trivial\n"
CONSTRUCTION_SOURCE = references.construction_source(references.RECALL_TABLE)

Z3_SPEC = {"variables": {"m": {"sort": "int", "lo": 1, "hi": 64}}, "hypotheses": [], "property": "m + 0 == m"}
Z3_PREDICATE = "def claim(m):\n    return m + 0 == m\n"


def lean_output(proof: str, nonce: str) -> str:
    if "sorry" in proof:
        return ("'claim_holds' depends on axioms: [sorryAx]\n"
                f"BIGBRANE_FACTS {nonce} kind=theorem exact=true claim_prop=true axioms=[sorryAx]\n")
    exact = "true" if re.search(r"^\s*theorem\s+claim_holds\s*:\s*Claim\s*:=", proof, re.M) else "false"
    return ("'claim_holds' does not depend on any axioms\n"
            f"BIGBRANE_FACTS {nonce} kind=theorem exact={exact} claim_prop=true axioms=[]\n")


def z3_default(request: dict) -> dict:
    spec = request["spec"]
    bound = ", ".join(f"{n} in [{v['lo']}, {v['hi']}]" for n, v in spec["variables"].items())
    if request["kind"] == "predicate_agreement":
        return {"status": "discharged", "counterexample": None, "checked": request.get("count", 1000),
                "evidence": "the predicate and the formal property agree on every assignment"}
    return {"status": "discharged", "tier": "smt_bounded", "bound": bound, "counterexample": None,
            "evidence": f"z3: the negation is unsat under the bounds and hypotheses; proven up to bound {bound}"}


def sympy_default(request: dict) -> dict:
    return {"status": "discharged", "tier": "sympy_exact", "counterexample": None,
            "evidence": "sympy: simplify(lhs - rhs) == 0; 20 rational spot checks agree; proven"}


class MathExecutor:
    def __init__(self, tmp_path: Path, *, z3=z3_default, host_kinds=("audited_construction", "conservation",
                                                                      "magic_x", "finding")):
        self.tmp_path = Path(tmp_path).resolve()
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.host = HostExecutor(self.tmp_path)
        self.z3 = z3
        self.host_kinds = set(host_kinds)
        self.jobs: list = []

    def _answer(self, job, stdout: str):
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text(stdout)
        err.write_text("")
        return SimpleNamespace(status="ok", detail="", exit_code=0, run_s=0.1, stdout_path=str(out),
                               stderr_path=str(err))

    def requests(self) -> list[dict]:
        return [json.loads((Path(job.input_dir) / "request.json").read_text()) for job in self.jobs
                if (Path(job.input_dir) / "request.json").is_file()]

    def run(self, job):
        self.jobs.append(job)
        staged = Path(job.input_dir)
        if (staged / CHECK_FILE).is_file():
            source = (staged / CHECK_FILE).read_text()
            nonce = re.search(r"bigbrane_check_([0-9a-f]+)", source).group(1)
            proof = source.split("-- proof\n", 1)[1].split("-- Bigbrane checker trailer", 1)[0]
            return self._answer(job, lean_output(proof, nonce))
        request = json.loads((staged / "request.json").read_text())
        if request["kind"] in self.host_kinds:
            return self.host.run(job)
        if request["kind"] in ("z3", "predicate_agreement"):
            return self._answer(job, json.dumps(self.z3(request), sort_keys=True) + "\n")
        if request["kind"] == "sympy":
            return self._answer(job, json.dumps(sympy_default(request), sort_keys=True) + "\n")
        raise AssertionError(f"no fake for library kind {request['kind']!r}")


# ----- scripted role outputs --------------------------------------------------------------------------------------

def sympy_statement(lhs: str = "2*n + 2*n", rhs: str = "4*n") -> dict:
    return {"form": "sympy", "sympy": {"symbols": {"n": "positive_integer"}, "lhs": lhs, "rhs": rhs,
                                       "relation": "=="}}


LEAN = {"form": "lean", "lean": LEAN_STATEMENT}
Z3 = {"form": "z3", "z3": Z3_SPEC}
CONSTRUCTION = {"form": "construction", "construction": {"task": "recall", "sizes": [4], "params": {"V": 16},
                                                          "count": 10}}
CAPABILITY_1_QUOTE = "Exact recall of the stored value at its own key"


def obligation(oid: str, claim_id: str | None, quote: str, statements: list[dict], premises=(),
               discharges=(), definitions=()) -> dict:
    return {"id": oid, "claim_id": claim_id, "claim_quote": quote, "statements": statements,
            "definitions": list(definitions), "premises": list(premises), "discharges": list(discharges)}


def fixer_output(*, premises=(), definitions=(), capability_1_premises=(), capability_1_definitions=(),
                 complexity_premises=(), complexity_definitions=(), extra_obligations=()) -> dict:
    """Statements for every claim of the example card: SymPy for the three complexity claims, a bounded Z3
    property for the sample-efficiency prediction, Lean for capability 1 and an executed construction for
    capability 2. A fresh copy on every call: a test that edits the statements it got (as the match-checker tests
    do) must not change the statements of the tests after it."""
    def complexity(oid, claim_id, quote, statement):
        return obligation(oid, claim_id, quote, [statement], complexity_premises,
                          definitions=complexity_definitions)
    return copy.deepcopy({"obligations": [
        complexity("O1", "complexity.time_per_token", "O(1) in n and d", sympy_statement()),
        complexity("O2", "complexity.memory", "O(1) complex scalar", sympy_statement("n*1", "n")),
        complexity("O3", "complexity.total_sequence", "O(n)", sympy_statement("3*n - 2*n", "n")),
        obligation("O4", "sample_efficiency.prediction",
                   "the readout at its key returns its value after one write", [Z3]),
        obligation("O5", "capability.1", CAPABILITY_1_QUOTE, [LEAN], capability_1_premises,
                   definitions=capability_1_definitions),
        obligation("O6", "capability.2", "Interference from other pairs is reported by the toy example",
                   [CONSTRUCTION]),
        *extra_obligations],
        "definitions": list(definitions), "premises": list(premises), "not_formalized": []})


PLAN = {"measure": "fraction of probes the index answers from its own lookup", "data": "the P2 recall task",
        "procedure": "run the reference index on 1000 recall instances", "accept_if": "fraction >= 0.99"}


def premise(aid: str, text: str, kind: str = "oracle", *, does_work: bool = True, testable_by: str = "code",
            plan=PLAN, citation=None, why_needed: str = "the readout step of the statement needs it",
            subject: str = "mechanism") -> dict:
    """``subject`` (FIX-6): where the premise is settled; ``frozen_definitions`` premises are settled at stage M."""
    return {"id": aid, "text": text, "kind": kind, "subject": subject, "does_work": does_work,
            "why_needed": why_needed, "testable_by": testable_by,
            "measurement_plan": plan if testable_by == "code" else None, "citation": citation}


def definition(did: str, text: str, kind: str = "sampling") -> dict:
    return {"id": did, "kind": kind, "text": text}


def _section(user: str, start: str, end: str | None = None) -> str:
    text = user.split(start, 1)[1]
    return text.split(end, 1)[0] if end else text


def prover_output(proofs) -> callable:
    """A prover that answers every statement it is shown: Lean with ``proofs[n]`` on the n-th call that shows a Lean
    statement (the last one repeated), executed statements with the reference recall construction. The index counts
    Lean calls, not prover calls: stage M sends a construction obligation in a call of its own (m_math.prover_batches),
    so the prover call index no longer equals the Lean attempt."""
    proofs = [proofs] if isinstance(proofs, str) else list(proofs)
    lean_calls = [0]

    def answer(index: int, user: str) -> dict:
        shown = json.JSONDecoder().raw_decode(_section(user, "FROZEN STATEMENTS:\n"))[0]
        proof = proofs[min(lean_calls[0], len(proofs) - 1)]
        if any(item["form"] == "lean" for item in shown):
            lean_calls[0] += 1
        attempts = []
        for item in shown:
            if item["form"] == "lean":
                attempts.append({"statement_id": item["statement_id"], "proof": proof,
                                 "source": None, "notes": "scripted"})
            else:
                attempts.append({"statement_id": item["statement_id"], "proof": None, "source": CONSTRUCTION_SOURCE,
                                 "notes": "scripted"})
        return {"attempts": attempts}

    return answer


def challenger_output(*, magic_x=(), citations=(), conservation=None, components=(), searches=(), new_edges=(),
                      findings=()) -> dict:
    """The challenger's answer in the FIX-4 schema (findings in place of new premises)."""
    conservation = dict(conservation or {"claims": None, "quotes": [], "why": "the card states no such numbers"})
    conservation.setdefault("derivations", [])
    return {"magic_x": list(magic_x), "citations": list(citations), "conservation": conservation,
            "components": list(components), "counterexample_searches": list(searches),
            "new_edges": list(new_edges), "findings": list(findings)}


def component(name: str, *, quote: str, statement_id=None, function=None, used_by=(), role: str = "mechanism",
              kind: str = "other", subject: str | None = None, premise_id=None, magic_x=None,
              why: str = "the conclusion needs it") -> dict:
    return {"name": name, "subject": subject or name, "quote": quote, "role": role, "kind": kind,
            "premise_id": premise_id, "statement_id": statement_id, "function": function, "used_by": list(used_by),
            "why": why, "magic_x": magic_x}


def finding(subject: str, *, kind: str = "missing_assumption", claim: str = "the result relies on it",
            evidence=None, used_by=("O5",), testable_by: str = "code", code=None, derivation=None, citation=None,
            record_quote=None, plan=None, magic_x=None, does_work: bool = True, premise_kind: str = "oracle",
            dependency: str = "the readout of the statement uses it", targets=()) -> dict:
    return {"subject": subject, "kind": kind, "claim": claim,
            "evidence": evidence or {"source": "card", "ref": None, "quote": CAPABILITY_1_QUOTE},
            "dependency": dependency, "used_by": list(used_by), "premise_kind": premise_kind, "does_work": does_work,
            "testable_by": testable_by, "code": code, "derivation": derivation, "citation": citation,
            "record_quote": record_quote, "measurement_plan": plan, "magic_x": magic_x, "targets": list(targets)}


def target(statement_id: str, deviation: str, edits, why: str = "removes the deviation and nothing else") -> dict:
    """A witness-violation target (FIX-13): the deviating text of one witness and the edit that removes it."""
    return {"statement_id": statement_id, "deviation": deviation,
            "edits": [{"anchor": a, "replacement": r} for a, r in edits], "why": why}


def backtranslate_output(index: int, user: str) -> dict:
    shown = json.loads(_section(user, "FORMAL STATEMENTS:\n", "\n\nDEFINITIONS:"))
    return {"translations": [{"statement_id": s["statement_id"], "plain": f"plain reading of {s['statement_id']}"}
                             for s in shown]}


def recorded_backtranslation(reply: dict) -> callable:
    """A recorded back-translation, answering for the statements the request shows. Since FIX-7 item 6 the request
    shows each distinct statement once, so a recording made before then also holds the repeated statements."""
    def answer(index: int, user: str) -> dict:
        shown = {s["statement_id"] for s in json.loads(_section(user, "FORMAL STATEMENTS:\n", "\n\nDEFINITIONS:"))}
        return {"translations": [t for t in reply["translations"] if t["statement_id"] in shown]}
    return answer


RUBRIC = ("quantifiers", "domains", "constants", "exponents", "hypotheses", "inequality_direction")


def claim_anchor(item: dict) -> str:
    """Exact text of an item's claim side (match_adjudication.claim_texts): its instantiated claim, else its claim."""
    return (item.get("instantiated_claim") or item["claim"])[:60]


def statement_anchor(spec) -> str:
    """Exact text of a formal statement: the line defining Claim of a Lean file, else the first scalar field."""
    if isinstance(spec, str):
        return next(line for line in spec.splitlines() if "Claim" in line)[:60]
    key, value = next((k, v) for k, v in spec.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
    return f"{json.dumps(key)}: {json.dumps(value)}"


def anchored(reply: dict, user: str) -> dict:
    """A comparison reply in the FIX-7 schema: every rubric item marked differs, and a direction other than match,
    gets an objection quoting exact claim and statement text (as the matcher must); a substantive definition gets an
    objection quoting its first words as a clause the card does not state (a capacity). Entries that already carry
    objections are left as they are."""
    items = {i["statement_id"]: i for i in json.loads(_section(user, "STATEMENTS TO COMPARE:\n", "\n\nDEFINITIONS:"))}
    definitions = {d["definition_id"]: d for d in json.loads(_section(user, "DEFINITIONS:\n", "\n\nCARD CLAIMS:"))}
    out = copy.deepcopy(reply)
    for entry in out["statements"]:
        if "objections" in entry:
            continue
        item = items[entry["statement_id"]]
        flagged = [name for name in RUBRIC if entry["rubric"][name]["verdict"] == "differs"]
        if entry["direction"] != "match":
            flagged.append("direction")
        entry["objections"] = [{"item": name, "claim_quote": claim_anchor(item),
                                "statement_quote": statement_anchor(item["formal_statement"]),
                                "difference": (entry["rubric"][name]["note"] if name in RUBRIC else
                                               f"the formal statement is {entry['direction']}: {entry['why']}")
                                or "differs"} for name in flagged]
    for entry in out["definitions"]:
        if "objection" in entry:
            continue
        entry["objection"] = None if entry["verdict"] != "substantive" else {
            "definition_quote": definitions[entry["definition_id"]]["text"][:40], "card_quote": None,
            "kind": "beyond_card", "asserts": "capacity", "difference": entry["why"]}
    return out


def match_output(differs: dict | None = None, *, substantive: tuple = (), implies: dict | None = None) -> callable:
    """Every statement matches its claim except those in ``differs`` ({statement_id: (direction, rubric item)});
    every definition is a convention except the ids in ``substantive``; a premise implies the claims ``implies``
    lists for it ({premise_id: [claim ids]}) and no others. Differences carry anchored objections (``anchored``)."""
    differs, implies = differs or {}, implies or {}

    def answer(index: int, user: str) -> dict:
        items = json.loads(_section(user, "STATEMENTS TO COMPARE:\n", "\n\nDEFINITIONS:"))
        definitions = json.loads(_section(user, "DEFINITIONS:\n", "\n\nCARD CLAIMS:"))
        premises = json.JSONDecoder().raw_decode(_section(user, "PREMISES:\n"))[0]
        statements = []
        for item in items:
            direction, bad = differs.get(item["statement_id"], ("match", None))
            rubric = {name: {"verdict": "differs" if name == bad else "same",
                             "note": "the exponent is n**2 in the formal statement, n in the claim" if name == bad
                             else ""} for name in RUBRIC}
            statements.append({"statement_id": item["statement_id"], "rubric": rubric, "direction": direction,
                               "predicate": Z3_PREDICATE if item["form"] == "z3" else None,
                               "why": "compared item by item"})
        reply = {"statements": statements,
                 "definitions": [{"definition_id": d["definition_id"],
                                  "verdict": "substantive" if d["definition_id"] in substantive else "convention",
                                  "why": "compared with the card"} for d in definitions],
                 "premises": [{"premise_id": p["premise_id"], "implies": list(implies.get(p["premise_id"], [])),
                               "why": "read against every claim"} for p in premises]}
        return anchored(reply, user)

    return answer


def crosscheck_output(verdicts: dict | None = None, *, default: str = "upheld") -> callable:
    """The cross-check judge: ``default`` for every objection, except the ids in ``verdicts`` ({objection id or
    statement/definition id: verdict})."""
    verdicts = verdicts or {}

    def answer(index: int, user: str) -> dict:
        shown = json.loads(_section(user, "OBJECTIONS TO ADJUDICATE:\n", "\n\nSTATEMENTS THEY CONCERN:"))
        out = []
        for objection in shown:
            oid = objection["objection_id"]
            verdict = verdicts.get(oid, verdicts.get(oid.split("#", 1)[0], default))
            out.append({"objection_id": oid, "verdict": verdict, "why": f"read the quoted texts: {verdict}"})
        return {"verdicts": out}

    return answer


def math_script(*, fixer=None, proofs=GOOD_PROOF, challenger=None, match=None) -> dict:
    return {"statement_fixer": fixer or fixer_output(), "prover": prover_output(proofs),
            "challenger": challenger or challenger_output(), "statement_backtranslate": backtranslate_output,
            "statement_match": match or match_output(), "statement_crosscheck": crosscheck_output()}
