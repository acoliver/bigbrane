"""Fakes for orchestrator tests: a scripted LLM client, a fake prior-art classifier, a fake bench.

Code checks run through hybrid.bench.local.LocalExecutor (host subprocess), so toy examples,
reduction and disproof checks and the S4 construct check really execute.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import tomllib
import uuid
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np

from hybrid.bench import falsifier as bench_falsifier
from hybrid.bench import falsifier_eval
from hybrid.bench.local import LocalExecutor
from hybrid.executor.lock import read_lock
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, Orchestrator
from hybrid.research.stages.common import ORCHESTRATOR_FIELDS
from hybrid.tests.bench.falsifier_programs import SHIFT
from hybrid.tests.research.builders import load_example

TEMPLATE = (ROOT / "hybrid" / "bench" / "candidate_template.py").read_text(encoding="utf-8")
WORDS = ("amber basalt cobalt dune ember fjord garnet harbor iris jasper kelp lumen marble nectar onyx prism quartz "
         "raven sable tundra umber velvet willow xenon yarrow zephyr").split()


def given_source_id(user: str) -> str:
    """The first paper id in an explorer prompt (S0 paper list, or a card's references when revising)."""
    match = re.search(r"source_id: (\S+)", user) or re.search(r'"source_id": "([^"]+)"', user)
    assert match, "the explorer prompt names no paper"
    return match.group(1)


# Operator words (hybrid.research.signature) added to the second update equation of distinct drafts, chosen by the
# bits of the draft index, so distinct drafts are distinct ideas with distinct mechanism signatures (Phase 8 budgets
# count per signature). Draft 0 keeps the fixture card's mechanism and signature.
VARIANT_OPS = ("hash", "sort", "count", "lookup", "pointer", "xor", "tanh", "stack")


def draft(index: int = 0, *, distinct: bool = True, user: str | None = None) -> dict:
    card = copy.deepcopy(load_example())
    for name in ORCHESTRATOR_FIELDS:
        card.pop(name)
    if distinct:
        words = " ".join(WORDS[(index * 5 + j) % len(WORDS)] for j in range(5))
        card["title"] = f"Fake card {index}: {words}"
        card["summary"] = f"Test fixture variant {index}. {words} {words} with a phase-bound running sum."
        ops = [op for bit, op in enumerate(VARIANT_OPS) if (index % 2 ** len(VARIANT_OPS)) >> bit & 1]
        if ops:
            card["mechanism"]["update"][1]["plain"] += " (variant: " + ", ".join(ops) + ")"
    card["core_idea"] = "Bind each token to a phase and read the running complex sum."
    if user is not None:
        source = given_source_id(user)
        card["references"] = [{"source_id": source, "why": "phase binding"}]
        card["source_transplant"] = {"object": "phasor superposition", "source_ids": [source],
                                     "change": "phases are learned per token"}
    return card


# The fixture card's capability 1 and falsifier 1, decided by one iterated-shift task (a stand-in protocol).
CAPABILITY_QUOTE = "Accuracy >= 0.9 at m = 2."
FALSIFIER_QUOTE = "The candidate beats the n-gram by less than 0.1 accuracy at m = 1."


def falsifier_rules() -> list[dict]:
    return [{"claims": ["capability.1"], "quote": CAPABILITY_QUOTE, "if_met": "supported",
             "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": "m=2", "op": ">=",
                             "value": 0.9}]},
            {"claims": ["falsifier.1"], "quote": FALSIFIER_QUOTE, "if_met": "refuted",
             "conditions": [{"entrant": "candidate", "minus": "ngram", "metric": "acc", "group": "m=1", "op": "<",
                             "value": 0.1}]}]


def falsifier_design(program_text: str = SHIFT, task_id: str = "iterated_shift", rules=None) -> dict:
    return {"task_id": task_id, "title": "iterated shift rule", "protocol": "demonstrations of a shift, then @m q",
            "deviations": "the fixture card states no protocol; a shift task stands in", "program": program_text,
            "rules": rules or falsifier_rules(), "protocol_params": []}


def program(obj: dict) -> str:
    return "import json\nprint(json.dumps(%r))\n" % (obj,)


CLAIM_MARK = "YOUR REDUCTION CLAIM:\n"


def claimed(user: str) -> dict:
    """The reduction claim quoted in a reduction_check user message."""
    claim, _ = json.JSONDecoder().raw_decode(user.split(CLAIM_MARK, 1)[1])
    return claim


def claimed_family(user: str) -> str:
    """The `reduces_to` of the reduction claim quoted in a reduction_check user message."""
    return claimed(user)["reduces_to"]


def declared(family: str, comparator_id: str = "known", transformations: tuple[str, ...] = (),
             limit_only: bool = False) -> dict:
    """One comparator a reduction critic declares (schemas/reduction_critic.schema.json, `comparators`; FIX-12)."""
    return {"id": comparator_id, "base_family": family, "base_form": "textbook form", "update": "S_t = S_{t-1} + v k^T",
            "read": "y_t = S_t q_t", "transformations": [{"type": t, "what": f"a {t}"} for t in transformations],
            "parameter_mapping": "card symbols map one to one", "limit_only": limit_only}


def declaring(critic: dict, *comparators: dict) -> dict:
    """A reduction critic answer with its comparators: the given ones, else one published form of its reduces_to
    (none when reduces_to is "none")."""
    if critic["reduces_to"] == "none":
        return dict(critic, comparators=[])
    return dict(critic, comparators=list(comparators) or [declared(critic["reduces_to"])])


def comparison(family: str, max_abs_diff: float, declared_id: str | None = "known", comparator: str = "known form",
               transformations: tuple[str, ...] = (), limit_only: bool = False, inputs_tested: int = 50) -> dict:
    """One entry of a reduction check's comparison report (contracts.REDUCTION_CHECK_REPORT)."""
    return {"declared_id": declared_id, "comparator": comparator, "family": family,
            "transformations": list(transformations), "limit_only": limit_only, "reference": "textbook form",
            "inputs_tested": inputs_tested, "max_abs_diff": max_abs_diff}


def reported(comparator: dict, max_abs_diff: float) -> dict:
    """The comparison a faithful check reports for a declared comparator."""
    return comparison(comparator["base_family"], max_abs_diff, comparator["id"], comparator["base_form"],
                      tuple(sorted({t["type"] for t in comparator["transformations"]})), comparator["limit_only"])


def comparison_check(max_abs_diff: float = 0.0):
    """A reduction_check answer whose program reports every comparator the critic declared, each at max_abs_diff."""
    def answer(i, user):
        report = {"comparisons": [reported(c, max_abs_diff) for c in claimed(user)["comparators"]],
                  "notes": "fake check"}
        return {"check_code": program(report), "explanation": "compares both forms"}
    return answer


def obligation_check(name: str, kind: str, program_text: str | None) -> dict:
    """One obligation_checker entry."""
    return {"name": name, "kind": kind, "program": program_text,
            "why": "a direct count" if program_text else "no program for this one"}


def crosscheck_upholds(index: int, user: str) -> dict:
    shown = json.loads(user.split("OBJECTIONS TO ADJUDICATE:\n", 1)[1].split("\n\nSTATEMENTS THEY CONCERN:", 1)[0])
    return {"verdicts": [{"objection_id": o["objection_id"], "verdict": "upheld", "why": "the quoted texts differ"}
                         for o in shown]}


def defaults() -> dict:
    """Canned, schema-valid outputs per prompt name. Values may be callables (call_index, user) -> output."""
    return {
        "explorer": lambda i, user: draft(i, user=user),
        "salad_critic": {"ok": True, "fatal": [], "fixable": [], "notes": "Every symbol is defined."},
        "prior_art_query": {"queries": ["phase running sum", "complex accumulator memory", "holographic recall"]},
        "reduction_critic": {"reduces_to": "none", "mapping": None, "confidence": 0.2, "difference_if_any": "",
                             "radical_enough": True, "why": "No mapping onto a known form was found.",
                             "textbook_transplant": False, "textbook_why": "The phases are learned, not fixed.",
                             "comparators": []},
        "reduction_check": comparison_check(0.0),
        "prior_art_delta": {"adds_untested_element": False, "element": "", "why": "Nothing new."},
        "formalizer": {"obligations": [
            {"name": f"o{i}", "kind": kind, "claim": "c", "statement": "s", "argument": "a", "known_bounds": "",
             "optional": False}
            for i, kind in enumerate(("complexity", "capacity", "learnability"))]},
        "disprover": {"attacks": [{"target_claim": "capacity", "counterexample": "many pairs", "status": "fails"}]},
        # o0 gets a program that discharges it; o1 is checkable but gets no program; o2 is left to the judges.
        "obligation_checker": {"checks": [
            obligation_check("o0", "complexity", program({"status": "discharged", "evidence": "3 ops per token",
                                                          "counterexample": None})),
            obligation_check("o1", "capacity", None),
            obligation_check("o2", "learnability", None)]},
        "feasibility_judge_a": {"verdict": "feasible", "obligations": [{"name": "o0", "status": "discharged",
                                                                        "argument": "ok"}],
                                "fatal_issues": [], "confidence": 0.7},
        "feasibility_judge_b": {"verdict": "feasible", "obligations": [{"name": "o0", "status": "discharged",
                                                                        "argument": "ok"}],
                                "fatal_issues": [], "confidence": 0.6},
        "builder": {"candidate_py": TEMPLATE, "notes": "template"},
        "falsifier_designer": {"tasks": [falsifier_design()],
                               "untestable": [{"claim_id": "complexity.memory", "why": "internal state size"}]},
        "analyst": {"claims": [{"claim_id": "capability.1", "claim": "single pair recall", "status": "supported",
                                "evidence": "hmm n32 kl 0.1", "falsifier_tasks": []}]},
        # FIX-7: the statement-match cross-check upholds every anchored objection unless a test scripts otherwise.
        "statement_crosscheck": crosscheck_upholds,
        # FIX-17: the statement tiebreak is unavailable unless a test scripts it, so a matcher/cross-check dispute
        # stays unresolved (the path a failed tiebreak call takes); tests of the tiebreak script it.
        "statement_tiebreak": tiebreak_unavailable,
    }


def analyst_supports_every_claim(index: int, user: str) -> dict:
    """An analyst that marks every claim id S6 shows it supported: the line where S6 passes and the card finishes
    (s6_verdict.ledger_verdict passes only a ledger whose every claim is supported). The default analyst comments on
    capability.1 alone, so its cards end S6 undetermined."""
    listed = user.split("CLAIM IDS (write one ledger entry per id):\n", 1)[1].split("\n\n", 1)[0]
    ids = [line[2:].split(": ", 1)[0] for line in listed.splitlines() if line.startswith("- ")]
    return {"claims": [{"claim_id": claim_id, "claim": claim_id, "status": "supported", "evidence": "hmm n32 kl 0.1",
                        "falsifier_tasks": []} for claim_id in ids]}


def tiebreak_unavailable(index: int, user: str) -> dict:
    from hybrid.llm.errors import TransportError

    raise TransportError("statement_tiebreak: no tiebreak scripted in this test (the call fails)")


class FakeLLM:
    """``models`` maps a role to the model id the fake reports (default ``fake-<role>``), so tests can put roles in
    the same or different model families (hybrid.research.families)."""

    def __init__(self, run_dir: Path, ledger: Ledger, script: dict | None = None, fail: dict | None = None,
                 models: dict | None = None):
        self.run_dir = Path(run_dir)
        self.ledger = ledger
        self.models = models or {}
        self.script = defaults()
        self.script.update(script or {})
        self.fail = fail or {}
        self.calls: list[dict] = []
        self.counts: dict[str, int] = {}

    def _prompt_name(self, role: str, system: str) -> str:
        first = system.splitlines()[0] if system else ""
        if first.startswith("# prompt: "):
            name = first[len("# prompt: "):].strip()
            if name == "feasibility_judge":
                return role
            return name
        if role == "prior_art":
            return "prior_art_query"
        raise AssertionError(f"unrecognised system prompt for role {role}")

    async def complete(self, role, system, user, *, json_schema=None, allow_web=False, max_output_tokens=None):
        await asyncio.sleep(0.01)   # yield like a real call so parallel workers interleave
        name = self._prompt_name(role, system)
        index = self.counts.get(name, 0)
        self.counts[name] = index + 1
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        self.calls.append({"role": role, "prompt": name, "call_id": call_id, "allow_web": allow_web, "user": user,
                           "system": system, "schema": json_schema})
        if name in self.fail:
            raise self.fail[name]
        output = self.script[name]
        if isinstance(output, list):
            output = output[min(index, len(output) - 1)]
        if callable(output):
            output = output(index, user)
        output = copy.deepcopy(output)
        if json_schema is not None:
            jsonschema.validate(output, json_schema)
        transcript = self.run_dir / "llm" / "calls" / f"{call_id}.json"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(json.dumps({"system": system, "user": user, "parsed": output}))
        usage = {"input": 100, "output": 50, "cached": 0}
        model = self.models.get(role, f"fake-{role}")
        self.ledger.append("llm_call", f"llm:{role}", {
            "call_id": call_id, "role": role, "backend": "fake", "model": model, "usage": usage,
            "latency_s": 0.01, "transcript_path": str(transcript), "error": None})
        return SimpleNamespace(call_id=call_id, role=role, backend="fake", model=model,
                               text=json.dumps(output), parsed=output, usage=usage, latency_s=0.01,
                               transcript_path=str(transcript))

    def prompts(self) -> list[str]:
        return [c["prompt"] for c in self.calls]


def make_classifier(labels, found=True):
    """Fake classify_prior_art: one model call through the injected wrapper, one search event per call.

    `labels` and `found` are per classifier call: a value, or a list used in order with its last item
    repeated. `found` says whether that call's search returned the paper 2401.00001."""
    state = {"i": 0}

    def nth(values, i):
        return values[min(i, len(values) - 1)] if isinstance(values, list) else values

    async def classify(card_text, llm_complete, sink):
        i = state["i"]
        state["i"] += 1
        await llm_complete("prior_art", "query system", card_text, {"type": "object"})
        hits = ["2401.00001"] if nth(found, i) else []
        sink.append("literature_search", "literature", {
            "query": "phase running sum", "sources": ["arxiv"], "limit": 6, "ts": "2026-10-10T00:00:00+00:00",
            "per_source_ids": {"arxiv": hits}, "errors": {}, "result_count": len(hits)})
        label = nth(labels, i)
        evidence = [] if label == "novel" or not hits else [{"source": "arxiv", "source_id": "2401.00001",
                                                             "title": "A paper",
                                                             "url": "https://arxiv.org/abs/2401.00001",
                                                             "relation": "close"}]
        return {"label": label, "evidence": evidence, "notes": "fake", "queries": ["phase running sum"],
                "candidates": [{"key": f"arxiv:{hit}", "source": "arxiv", "source_id": hit, "title": "A paper",
                                "year": 2024, "url": f"https://arxiv.org/abs/{hit}", "excerpt": "An abstract."}
                               for hit in hits],
                "rejected_evidence": []}

    return classify


def fake_bench_raw(status: str = "ok") -> dict:
    tasks = {"hmm": {"sizes": {"32": {"valid": True, "kl": 0.1, "acc": 0.5},
                               "128": {"valid": True, "kl": 0.05, "acc": 0.6}},
                     "tokens_to_target": {"reached": True, "interp_tokens": 4000.0}}}
    candidate = {"role": "candidate", "status": status, "job_status": "ok" if status == "ok" else "candidate_error",
                 "job_detail": "", "job_id": "job-aaaaaaaaaaaa", "claimed_exponent": 1.0}
    if status == "ok":
        candidate.update(tasks=tasks, scaling={"time_fit_tail": {"slope": 1.02}, "verdict": "match"})
    return {"schema": "bigbrane.bench_result/1",
            "entrants": {"candidate": candidate,
                         "ngram": {"role": "baseline", "status": "ok", "tasks": tasks, "job_id": "job-bbbbbbbbbbbb",
                                   "scaling": {"time_fit_tail": {"slope": 0.1}}}},
            "comparison": {}, "trusted": False,
            "host": {"receipts": {"candidate": {"job_id": "job-aaaaaaaaaaaa", "status": "ok"},
                                  "ngram": {"job_id": "job-bbbbbbbbbbbb", "status": "ok"}},
                     "evaluator_receipt": {"job_id": "job-cccccccccccc", "status": "ok"}}}


def fake_scores(acc: float, kl: float, labels: list[str]) -> dict:
    block = {"acc": acc, "kl": kl, "n": 64}
    return {**block, "groups": {label: dict(block) for label in labels}}


class FakeBench:
    """Scripted standard-task results. Falsifier tasks are real validated directories: their references come
    from the real oracle files and the rules are decided by the real falsifier_eval.decide, with scripted
    entrant scores ({entrant: (acc, kl)}, or None for an entrant without valid predictions)."""

    def __init__(self, status: str = "ok", falsifier_scores: dict | None = None, describe: dict | None = None):
        self.status = status
        self.describe = describe or {"hyper": {"d": 32}, "train_max_len": 2048}
        self.falsifier_scores = falsifier_scores or {"candidate": (0.95, 0.1), "ngram": (0.3, 2.0)}
        self.calls: list[tuple] = []
        self.falsifier_dirs: list[list[Path]] = []

    async def evaluate(self, card_id, candidate_dir, exponent, falsifiers=None):
        assert (Path(candidate_dir) / "candidate.py").is_file()
        self.calls.append((card_id, str(candidate_dir), exponent))
        self.falsifier_dirs.append([Path(d) for d in falsifiers or []])
        raw = fake_bench_raw(self.status)
        if falsifiers:
            raw["falsifiers"] = {}
            for directory in falsifiers:
                task = bench_falsifier.load_task(directory)
                with np.load(task.directory / "oracle.npz") as data:
                    oracle, group, unigram = data["oracle"], data["group"], data["unigram"]
                refs = falsifier_eval.references(task.oracle_kind, oracle, group, task.group_labels, unigram)
                scores = {name: (None if value is None else fake_scores(*value, task.group_labels))
                          for name, value in self.falsifier_scores.items()}
                entrants = {name: ({"status": "ok", "metrics": s} if s is not None else {"status": "job_failed"})
                            for name, s in scores.items()}
                block = {**bench_falsifier.eval_entry(task), "entrants": entrants, "references": refs}
                blocked = None if scores.get("candidate") is None else falsifier_eval.task_blocked(block, self.describe)
                block["decisions"] = [{"rule_index": i, "claims": r["claims"], "quote": r["quote"], "rule": r,
                                       **falsifier_eval.decide(r, {**scores, **refs}, task.design["rules"], blocked)}
                                      for i, r in enumerate(task.design["rules"])]
                block.pop("rules")
                raw["falsifiers"][task.task_id] = block
        return raw


def orch_config(**overrides) -> dict:
    with open(ROOT / "hybrid" / "config" / "run.toml", "rb") as handle:
        cfg = tomllib.load(handle)
    cfg["run"]["n_explorers"] = 1
    cfg["budgets"].update(max_cards=1, max_calls=200, max_wall_s=600)
    cfg["workers"] = {stage: 1 for stage in ("S1", "S2", "S3", "S4", "S5", "S6")}
    cfg["diversity"]["threshold"] = 0.99
    # Off unless a test sets it: at the shipped 0.3, a max_cards of 1 holds back its only slot.
    cfg["explore"]["escape_reserve"] = 0.0
    # Phase 7 switches, on in the shipped run.toml, off here so the tests written before Phase 7 keep their call
    # sequences; the Phase 7 tests (test_refine.py, test_difference.py) switch them on.
    cfg["refine"]["enabled"] = False
    cfg["s2"]["family_exclusion"] = False
    cfg["s2"]["difference_statement"] = False
    cfg["outcomes"]["enforce_r3"] = False
    # Phase 4 switches, on in the shipped run.toml, off here so the math tests written before Phase 4 keep their
    # fixer outputs; the Phase 4 tests (test_profile_stage.py, test_calibration.py) switch them on.
    cfg["math"]["profile"] = False
    cfg["target"]["enabled"] = False
    for dotted, value in overrides.items():
        section, key = dotted.split("__")
        cfg[section][key] = value
    return cfg


def make_orchestrator(run_dir: Path, *, cfg=None, script=None, fail=None, labels="novel", found=True, bench=None,
                      resume=False, llm=None, math=None, models=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir)
    llm = llm or FakeLLM(run_dir, ledger, script, fail, models)
    executor = LocalExecutor(run_dir.name, run_dir / "jobs" / "receipts")
    orch = Orchestrator(cfg or orch_config(), run_dir, llm=llm, executor=executor, bench=bench or FakeBench(),
                        image=read_lock()["reference"], ledger=ledger,
                        classify_prior_art=make_classifier(labels, found), resume=resume, math=math)
    return orch, llm
