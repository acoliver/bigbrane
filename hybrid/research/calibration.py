"""Calibration of the math stages on known models (DISCOVERY-PIPELINE-PLAN.md section 12, Phase 4).

Four cards in ``hybrid/fixtures/calibration/`` describe known models in neutral notation, written as theory cards in
the current schema without family names, citations or telltale terms: K1 (two rounds of weighted retrieval over stored
pairs with exponential scores), K2 (a running store regulated by multiplicative switches), K3 (a latent chain with k
states decoded to one state per token) and K4 (a continuation table over the last two symbols). ``expected.json``
holds the expected outcome of every scored cell, taken from the plan's table.

The driver runs each card through stages M and C, ``runs`` times, each run in a new run directory whose first ledger
event is ``operator_override`` (stages S0 to S2 are bypassed: the prior-art stage would recognize the models). C runs
after M whatever M's verdict, as in ``plumb``. M runs with the profile on and the target gate off, so the profile is
tested without the cost target interfering; the target gate (hybrid/math/target.py, decision D1 from run.toml) is then
applied to the verified P1 formula on its own.

Scored cells per card (plan section 12): P1, P2.copy, P2.recall, P2.nesting, P2.composition, P3 and P5, each
``pass`` (status proven, proven_up_to_bound or observed) or ``fail`` (failed or refuted); anything else (unproven,
not formalized, blocked by statement match, circular, conditional) is ``not_shown`` and never matches an expected
cell. A cell whose stage C claim record says ``match`` ``unresolved`` (FIX-7: the matcher objected with anchored quotes
and the separately recorded cross-check disagreed; FIX-17: and the third-family tiebreak did not settle it, because
its call failed) scores ``not_shown`` whatever its checks say (decisions XC and NS: an unsettled translation is not
checked evidence), and the run lists it under ``unresolved``. An unresolved cell is never a reproduced cell: the run
records it as a mismatch with the expected table ("statement match unresolved"), so acceptance needs every disputed
match settled. Each run also records the profile verdict (every slot passes, P4 included), the kill codes of M, and the target
result. Acceptance: every expected cell, profile verdict, required failure reason and target result is reproduced in
every run, and the runs agree on every cell. The report gives the share of scored items (seven cells, the profile
verdict and the target, per card) on which every run agrees; acceptance needs 100%. A disagreement fixes the stage,
not the table (plan section 12).

FIX-22 (decision ENV): a card-run that stopped on a stage error caused by the environment (the model backend or its
transport: HTTP 400 for too small a loaded context, "Model unloaded", connection refused, a gateway transport failure)
is ``invalid_environment``: it is no evidence about the stages and is never scored. Its cells are kept in the record.
Verdict-level outcomes (fail, not shown from a held check) and every other stop stay valid. The rule
(``stop_validity``): the stage error's cause is a hybrid/llm error recorded by class (``stop_error_kinds``, the cause's
LLMError classes); TransportError, BackendError or GatewayUnavailable make the run invalid, OutputExhausted (the model
spent its output cap) does not. Decision CT: so does JobHangGuard, a CPU-capped math library job stopped by its
wall-clock hang guard (hybrid/math/runner.py), which says the host starved the job, not how the check came out. Records made before FIX-22 have only the ``stopped`` text; there the class name the
stages write (``<role>: <ErrorClass>: ``) decides, and only a stop with no class name is matched against
ENVIRONMENT_PHRASES. Acceptance needs every card-run of the calibration valid. ``rerun_invalid`` (``calibrate
--rerun-invalid <out dir>``) reruns each invalid card-run in a fresh run directory ``<card>-run<N>-rerun<M>`` under the
same run root and rescores with the rerun in place of the invalid one; the invalid run directories stay as they are.

The anonymization check (``anonymization_problems``) rejects a card that names a model family, uses a term that gives
one away, or cites a source: run before any model call, and by a test over the fixtures.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from hybrid.executor.job import Caps
from hybrid.math import profile
from hybrid.research.ledger import Ledger
from hybrid.research.parallel import check_parallel, run_limited
from hybrid.research.schemas import validate
from hybrid.research.stages import c_challenge, card_resources, m_math
from hybrid.research.stages.common import Budget, CodeRunner, RunBudgetExhausted, StageContext, StageError
from hybrid.research.store import Store
from hybrid.slots import VMUnavailable, job_slots_from_config, slot_line

VALID = "valid"
INVALID_ENVIRONMENT = "invalid_environment"
# hybrid/llm error classes whose stage error is the environment's failure, and the one subclass that is the model's.
ENVIRONMENT_ERRORS = ("TransportError", "BackendError", "GatewayUnavailable", "JobHangGuard")
MODEL_ERRORS = ("OutputExhausted",)
LLM_ERROR_NAMES = ("LLMError", "ConfigError", "BudgetExceeded", "TransportError", "OutputExhausted", "BackendError",
                   "ToolPolicyViolation", "SchemaValidationError", "GatewayUnavailable")
# A pre-FIX-22 stop: "stage_error at <stage>: <role>: <ErrorClass>: <message>" (hybrid/research/stages/common.py).
STOP_CLASS = re.compile(r"^stage_error at \w+: [\w.-]+: (" + "|".join(LLM_ERROR_NAMES) + r"): ")
# Narrow text fallback, only for a stage-error stop that names no error class.
ENVIRONMENT_PHRASES = ("Model unloaded", "greater than the context length", "Connection refused", "ConnectError")

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "hybrid" / "fixtures" / "calibration"
CARD_KEYS = ("k1", "k2", "k3", "k4")
CELLS = ("P1", "P2.copy", "P2.recall", "P2.nesting", "P2.composition", "P3", "P5")
PASS_STATUSES = ("proven", "proven_up_to_bound", "observed")
FAIL_STATUSES = ("failed", "refuted")
OPERATOR = "operator"
ORCHESTRATOR = "orchestrator"
STATEMENT = ("calibration of stages M and C on an anonymized known model (plan section 12); stages S0 to S2 are "
             "bypassed and results are not evidence that the card passed any gate")

# Names and give-away terms of the families the cards describe and of the excluded families (hybrid/config/
# families.toml ids are added at run time), matched as whole words, case-insensitive, anywhere in the card.
BANNED_TERMS = (
    "transformer", "attention", "self-attention", "softmax", "multi-head", "multihead", "positional encoding",
    "lstm", "long short-term", "long short term", "gru", "gated recurrent", "forget gate", "input gate",
    "output gate", "cell state", "recurrent neural", "rnn", "hidden markov", "hmm", "markov", "viterbi",
    "baum-welch", "forward algorithm", "forward-backward", "emission", "emissions", "n-gram", "ngram", "bigram",
    "trigram", "kneser", "katz", "backoff", "back-off", "good-turing", "mamba", "state space", "state-space", "ssm",
    "deltanet", "delta rule", "linear attention", "hopfield", "language model",
)
# Authors and venues of the original papers, and citation markers.
CITATION_TERMS = ("vaswani", "hochreiter", "schmidhuber", "gers", "rabiner", "baum", "juang", "shannon", "jelinek",
                  "et al", "arxiv", "doi", "neurips", "nips", "icml", "iclr", "proceedings", "journal")
CITATION_PATTERNS = (re.compile(r"\b\d{4}\.\d{4,5}\b"), re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b"),
                     re.compile(r"https?://"))
# Terms that name an attention-style part when they appear in the mechanism (they are ordinary words in the task
# descriptions of the claims, where recall has keys, values and queries).
MECHANISM_TERMS = ("query", "queries", "key", "keys", "value", "values", "head", "heads", "gate", "gates", "cell")


def _words(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None


def family_terms(families_path: Path | None = None) -> tuple[str, ...]:
    """The ids of every family in families.toml, as words (``linear_attention`` -> ``linear attention``)."""
    data = tomllib.loads((families_path or ROOT / "hybrid" / "config" / "families.toml").read_text(encoding="utf-8"))
    return tuple(entry["id"].replace("_", " ") for kind in ("excluded", "known") for entry in data.get(kind, []))


def anonymization_problems(card: dict, *, extra_terms: tuple[str, ...] = ()) -> list[str]:
    """Why a calibration card is not anonymous: a family name or give-away term anywhere, a citation (an arXiv id,
    a year, an author or venue, a URL, a non-empty references list or a source_transplant), or an attention-style
    part name in the mechanism."""
    problems = []
    text = json.dumps(card, ensure_ascii=False).lower()
    for term in (*BANNED_TERMS, *extra_terms, *family_terms()):
        if _words(text, term.lower()):
            problems.append(f"names or gives away a model family: {term!r}")
    for term in CITATION_TERMS:
        if _words(text, term):
            problems.append(f"cites a source: {term!r}")
    for pattern in CITATION_PATTERNS:
        found = pattern.search(text)
        if found:
            problems.append(f"cites a source: {found.group(0)!r}")
    if card.get("references"):
        problems.append("references must be empty")
    if card.get("source_transplant"):
        problems.append("source_transplant must be absent (it names the source)")
    mechanism = json.dumps(card.get("mechanism", {}), ensure_ascii=False).lower()
    for term in MECHANISM_TERMS:
        if _words(mechanism, term):
            problems.append(f"the mechanism uses an attention-style part name: {term!r}")
    return sorted(set(problems))


def load_cards(keys=CARD_KEYS, directory: Path = FIXTURES) -> dict[str, dict]:
    """The calibration cards, each validated against the theory card schema and checked for anonymity. Decision P4P:
    every learning protocol a card supplies is read and its sha256 checked here, so a changed or missing learner
    source stops the load (card_resources.ProtocolError) before any model call."""
    cards = {}
    for key in keys:
        card = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
        validate("theory_card", card)
        problems = anonymization_problems(card)
        if problems:
            raise ValueError(f"calibration card {key} is not anonymous: {problems}")
        try:
            card_resources.learning_protocols(card)
        except card_resources.ProtocolError as exc:
            raise card_resources.ProtocolError(f"calibration card {key}: {exc}") from exc
        cards[key] = card
    check_complete(cards)
    return cards


def check_complete(cards: dict[str, dict]) -> None:
    """FIX-16: a calibration card whose resources leave items unstated is a fixture error. It is refused before any
    model call: in a calibration run there is no author to ask, and stage M would end the card with card_incomplete."""
    found = {key: card_resources.missing(card) for key, card in cards.items()}
    found = {key: items for key, items in found.items() if items}
    if found:
        raise ValueError("calibration fixture error (card_incomplete): " + "; ".join(
            f"{key} lacks {', '.join(items)}" for key, items in found.items()))


def load_expected(directory: Path = FIXTURES) -> dict:
    return json.loads((directory / "expected.json").read_text(encoding="utf-8"))


# ----- one card, one run -------------------------------------------------------------------------------------------

@dataclass
class Deps:
    llm: object
    executor: object
    image: str
    math: object


def calibration_config(cfg: dict) -> dict:
    """The run config for a calibration run: math mode, the profile on, the target gate off in stage M."""
    cfg = copy.deepcopy(cfg)
    cfg["s3"]["mode"] = "math"
    cfg["math"]["profile"] = True
    cfg["target"] = dict(cfg["target"], enabled=False)
    return cfg


def cell_outcome(status: str | None) -> str:
    if status in PASS_STATUSES:
        return "pass"
    if status in FAIL_STATUSES:
        return "fail"
    return "not_shown"


def _verdicts(ctx: StageContext, card_id: str) -> dict[str, dict]:
    found = {}
    for record in c_challenge._verdicts(ctx, card_id):
        found[record["stage"]] = record
    return found


async def run_card(*, key: str, card: dict, run_index: int, run_dir: Path, cfg: dict, reason: str,
                   deps_factory: Callable[[Ledger], Deps], target_rule: dict, rerun: dict | None = None) -> dict:
    """Run one calibration card through M and C in a new run directory; return its outcome record. ``rerun``
    (``{"rerun": M, "rerun_of": <run dir>}``) marks a rerun of an invalid card-run (FIX-22)."""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"{run_dir} is not empty")
    raw = (json.dumps(card, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    ledger.append("operator_override", OPERATOR, {
        "statement": STATEMENT, "reason": reason, "calibration": {"card": key, "run": run_index, **(rerun or {})},
        "source_run": str(FIXTURES), "source_run_id": "calibration-fixtures", "card_id": card["id"],
        "source_card": {"version": card["version"], "path": f"hybrid/fixtures/calibration/{key}.json",
                        "sha256": hashlib.sha256(raw).hexdigest()},
        "source_verdicts": {stage: None for stage in ("S0", "S1", "S2", "S3")}, "source_parent_ids": [],
        "stages_run": ["M", "C"], "math_note": "C runs after M whatever M's verdict (calibration)",
        "s5_mode": cfg["s5"]["mode"], "falsifiers": False, "routing": None})
    b = cfg["budgets"]
    budget = Budget(b["max_cards"], b["max_calls"], b["max_tokens"], b["max_wall_s"])
    ledger.append("run_start", ORCHESTRATOR, {"run_id": run_dir.name, "calibration": True, "plumb": True,
                                              "resume": False, "budget": budget.snapshot(), "stages": ["M", "C"],
                                              "s3_mode": "math", "profile": True, "target_in_m": False})
    store.put_card(card)
    deps = deps_factory(ledger)
    ex = cfg["executor"]
    caps = Caps(wall_s=ex["wall_s"], memory_mb=ex["memory_mb"], pids=ex["pids"], tmp_mb=ex["tmp_mb"])
    ctx = StageContext(run_id=run_dir.name, run_dir=run_dir, ledger=ledger, store=store, llm=deps.llm,
                       runner=CodeRunner(deps.executor, run_dir, store, deps.image, caps), bench=None, cfg=cfg,
                       budget=budget, lenses={}, math=deps.math)
    stopped = await run_math_stages(ctx, card["id"])
    outcome = outcome_record(ctx, key, card, run_index, target_rule)
    outcome["stopped"] = stopped
    errors = ctx.card_events(card["id"], "stage_error")
    outcome["stop_error_kinds"] = (errors[-1].get("error_kinds") or []) if stopped and errors else []
    outcome["validity"] = stop_validity(stopped, outcome["stop_error_kinds"])
    outcome.update(rerun or {})
    ctx.emit("run_end", {"calibration": True, "card": key, "run": run_index, "stopped": stopped,
                         "validity": outcome["validity"], **(rerun or {}),
                         "cells": outcome["cells"], "profile_passed": outcome["profile_passed"],
                         "target": outcome["target"]["status"],
                         "target_coupled": m_math.coupled_summary(outcome["target_coupled"])})
    return outcome


async def run_math_stages(ctx: StageContext, card_id: str) -> str | None:
    """Stage M, then stage C whatever M's verdict (as long as M froze its statements), under an operator override.
    Returns why the line stopped early (a stage error or the run budget), or None. Also used for the T5 cards of the
    planted-card test (hybrid/research/planted.py)."""
    for stage, runner in (("M", m_math.run), ("C", c_challenge.run)):
        if stage == "C" and m_math.load_frozen(ctx, card_id) is None:
            break
        ctx.emit("stage_start", {"card_id": card_id, "stage": stage})
        try:
            await runner(ctx, card_id)
            incomplete = ctx.card_events(card_id, "card_incomplete")
            if stage == "M" and incomplete and m_math.load_frozen(ctx, card_id) is None:
                # FIX-16: no author to ask under the override; the card is a fixture error and the line stops.
                return (f"{card_resources.KILL_CODE} at M (fixture error): "
                        + "; ".join(incomplete[-1]["missing"]))[:500]
        except StageError as exc:
            kinds = error_kinds(exc.__cause__)
            ctx.emit("stage_error", {"card_id": card_id, "stage": stage, "error": str(exc)[:4000],
                                     **({"error_kinds": kinds} if kinds else {})})
            return f"stage_error at {stage}: {exc}"[:500]
        except RunBudgetExhausted as exc:
            ctx.emit("run_stop", {"reason": str(exc), **ctx.budget.snapshot()})
            return f"budget: {exc}"
    return None


def error_kinds(cause: BaseException | None) -> list[str]:
    """The hybrid/llm error classes of a stage error's cause, most specific first, or ["JobHangGuard"] for a math job
    stopped by its hang guard (decision CT); empty for any other cause."""
    from hybrid.llm.errors import LLMError
    from hybrid.math.runner import JobHangGuard

    if isinstance(cause, JobHangGuard):
        return [JobHangGuard.__name__]
    if not isinstance(cause, LLMError):
        return []
    return [klass.__name__ for klass in type(cause).__mro__ if isinstance(klass, type) and issubclass(klass, LLMError)]


def stop_validity(stopped: str | None, kinds: list[str] | None = None) -> str:
    """``invalid_environment`` when the run stopped on a stage error the environment caused, else ``valid`` (see the
    module docstring for the rule)."""
    if not stopped or not stopped.startswith("stage_error at "):
        return VALID
    if not kinds:
        found = STOP_CLASS.match(stopped)
        kinds = [found.group(1)] if found else None
    if kinds:
        if any(kind in MODEL_ERRORS for kind in kinds):
            return VALID
        return INVALID_ENVIRONMENT if any(kind in ENVIRONMENT_ERRORS for kind in kinds) else VALID
    return INVALID_ENVIRONMENT if any(phrase in stopped for phrase in ENVIRONMENT_PHRASES) else VALID


def validity(outcome: dict) -> str:
    """A card-run's validity: recorded since FIX-22, derived from ``stopped`` for older records."""
    return outcome.get("validity") or stop_validity(outcome.get("stopped"), outcome.get("stop_error_kinds"))


def outcome_record(ctx: StageContext, key: str, card: dict, run_index: int, target_rule: dict) -> dict:
    """What a run showed: per-slot statuses at M and C, the scored cells (C when it ran, else M), M's verdict and
    kill codes, the profile verdict and the target gate on the verified P1 formula."""
    verdicts = _verdicts(ctx, card["id"])
    stages = {}
    for stage, record in verdicts.items():
        stages[stage] = {"verdict": record["verdict"], "kill_codes": record["kill_codes"],
                         "slots": {c["claim_id"]: {"status": c["status"], "tier": c["tier"],
                                                   **({"match": c["match"]} if "match" in c else {})}
                                   for c in record["claims"] if profile.is_slot(c["claim_id"])}}
    final = stages.get("C") or stages.get("M")
    slots = final["slots"] if final else {}
    unresolved = [cell for cell in CELLS if slots.get(cell, {}).get("match") == "unresolved"]
    # FIX-17 (decisions XC and NS): a claim whose statement match stayed unresolved after the tiebreak is not shown;
    # it never scores from its checks.
    cells = {cell: "not_shown" if cell in unresolved else cell_outcome(slots.get(cell, {}).get("status"))
             for cell in CELLS}
    profile_passed = bool(slots) and all(cell_outcome(slots.get(slot, {}).get("status")) == "pass"
                                         for slot in profile.SLOTS)
    target_result = {"status": "not_applicable", "reasons": ["stage M did not finish"]}
    coupled_result = {"status": "not_applicable", "applied": False, "reasons": ["stage M did not finish"]}
    record = m_math.load_frozen(ctx, card["id"])
    if record is not None and "M" in verdicts:
        checks = verdicts["M"]["checks"]
        target_result = m_math.target_block({"target": dict(target_rule, enabled=False)}, record, checks)
        target_result.pop("applied", None)
        # FIX-17: recorded, never scored; stage M computed it with the gate off (calibration_config).
        coupled_result = verdicts["M"].get("target_coupled") or m_math.coupled_block(
            {"target": dict(target_rule, enabled=False)}, record, checks)
    calls = {}
    for event in ctx.card_events(card["id"], "math_role_call"):
        calls[event["role"]] = calls.get(event["role"], 0) + 1
    return {"card": key, "card_id": card["id"], "run": run_index, "run_dir": str(ctx.run_dir), "stages": stages,
            "cells": cells, "unresolved": unresolved, "profile_passed": profile_passed,
            "m_kill_codes": stages.get("M", {}).get("kill_codes", []), "target": target_result,
            "target_coupled": coupled_result, "role_calls": calls, "model_calls": sum(calls.values())}


# ----- comparison with the expected table ---------------------------------------------------------------------------

def compare(expected: dict, outcomes: list[dict]) -> dict:
    """Per card and run, which expected items were reproduced; per card, whether the runs agree on every cell; and
    the acceptance verdict of plan section 12."""
    report = {"cards": {}, "accepted": True, "problems": []}
    scored, invalid = split_valid(outcomes)
    report["invalid"] = invalid
    pending = [item for item in invalid if item["replaced_by"] is None]
    if pending:
        report["accepted"] = False
        report["problems"].append(f"{len(pending)} card-run(s) invalid (environment failure), not scored: "
                                  + ", ".join(Path(item["run_dir"]).name for item in pending)
                                  + "; rerun them with --rerun-invalid")
    for key, want in expected["cards"].items():
        runs = sorted((o for o in scored if o["card"] == key), key=lambda o: o["run"])
        entry = {"runs": [], "agreement": {}, "agree": True, "reproduced_every_run": bool(runs)}
        if not runs:
            report["problems"].append(f"{key}: no runs")
        for outcome in runs:
            mismatches = [f"{cell}: expected {want['cells'][cell]}, got {outcome['cells'][cell]}"
                          for cell in CELLS if outcome["cells"][cell] != want["cells"][cell]]
            mismatches += [f"{cell}: statement match unresolved after the tiebreak (scored not shown)"
                           for cell in outcome.get("unresolved", [])]
            if outcome["profile_passed"] != (want["profile"] == "pass"):
                mismatches.append(f"profile: expected {want['profile']}, got "
                                  f"{'pass' if outcome['profile_passed'] else 'fail'}")
            missing = [code for code in want.get("kill_codes_include", []) if code not in outcome["m_kill_codes"]]
            if missing:
                mismatches.append(f"stage M kill codes lack {missing}")
            if outcome["target"]["status"] != want["target"]:
                mismatches.append(f"target: expected {want['target']}, got {outcome['target']['status']}")
            if outcome.get("stopped"):
                mismatches.append(f"run stopped: {outcome['stopped']}")
            entry["runs"].append({"run": outcome["run"], "rerun": outcome.get("rerun"), "cells": outcome["cells"],
                                  "unresolved": list(outcome.get("unresolved", [])),
                                  "profile": "pass" if outcome["profile_passed"] else "fail",
                                  "m_verdict": outcome["stages"].get("M", {}).get("verdict"),
                                  "m_kill_codes": outcome["m_kill_codes"], "target": outcome["target"]["status"],
                                  "target_coupled": m_math.coupled_summary(outcome.get("target_coupled")),
                                  "model_calls": outcome["model_calls"], "mismatches": mismatches,
                                  "reproduced": not mismatches})
            if mismatches:
                entry["reproduced_every_run"] = False
        for cell in (*CELLS, "profile", "target"):
            values = [r["cells"][cell] if cell in CELLS else r[cell] for r in entry["runs"]]
            entry["agreement"][cell] = {"values": values, "agree": len(set(values)) <= 1}
        entry["agree"] = all(item["agree"] for item in entry["agreement"].values())
        entry["agreement_rate"] = _rate(entry["agreement"].values())
        if not (entry["reproduced_every_run"] and entry["agree"]):
            report["accepted"] = False
        report["cards"][key] = entry
    # Share of scored items (cells, profile, target, per card) on which every run of the card agrees: acceptance
    # needs 100%; the plan's repeatability floor elsewhere (section 13) is 90%.
    report["agreement_rate"] = _rate(item for entry in report["cards"].values() if entry["runs"]
                                     for item in entry["agreement"].values())
    report["runs_per_card"] = {key: len(entry["runs"]) for key, entry in report["cards"].items()}
    if any(count < expected.get("runs_required", 3) for count in report["runs_per_card"].values()):
        report["accepted"] = False
        report["problems"].append(f"acceptance needs {expected.get('runs_required', 3)} runs per card")
    return report


def split_valid(outcomes: list[dict]) -> tuple[list[dict], list[dict]]:
    """The valid card-runs (at most one per card and run index: a rerun is made only while none is valid), and one
    entry per invalid card-run naming the run directory that replaced it (None while none did)."""
    scored: dict[tuple[str, int], dict] = {}
    for outcome in outcomes:
        if validity(outcome) == VALID:
            slot = (outcome["card"], outcome["run"])
            if slot in scored:
                raise ValueError(f"two valid card-runs for {slot[0]} run {slot[1]}: {scored[slot]['run_dir']} and "
                                 f"{outcome['run_dir']}")
            scored[slot] = outcome
    invalid = []
    for outcome in outcomes:
        if validity(outcome) != VALID:
            replacement = scored.get((outcome["card"], outcome["run"]))
            invalid.append({"card": outcome["card"], "run": outcome["run"], "rerun": outcome.get("rerun"),
                            "run_dir": outcome["run_dir"], "validity": validity(outcome),
                            "stopped": outcome.get("stopped"),
                            "replaced_by": replacement["run_dir"] if replacement else None})
    return list(scored.values()), invalid


def _rate(items) -> float | None:
    items = list(items)
    return sum(1 for item in items if item["agree"]) / len(items) if items else None


def _percent(rate: float | None) -> str:
    return "n/a" if rate is None else f"{100 * rate:.0f}%"


def render_report(report: dict, outcomes: list[dict], meta: dict) -> str:
    lines = ["# Math-stage calibration on anonymized known models", "",
             f"Started {meta['started']}, finished {meta['finished']}. Runs per card: {meta['runs']}. "
             f"Accepted (plan section 12): {'yes' if report['accepted'] else 'no'}.", "",
             "| Card | Run | " + " | ".join(CELLS) + " | Profile | Target | M verdict | Calls | Reproduced |",
             "|---" * (len(CELLS) + 7) + "|"]
    for key, entry in report["cards"].items():
        for run in entry["runs"]:
            label = f"{run['run']} (rerun {run['rerun']})" if run.get("rerun") else f"{run['run']}"
            lines.append(f"| {key} | {label} | " + " | ".join(
                run["cells"][c] + (" (unresolved)" if c in run.get("unresolved", []) else "") for c in CELLS)
                         + f" | {run['profile']} | {run['target']} | {run['m_verdict']} | {run['model_calls']} | "
                         + ("yes" if run["reproduced"] else "no") + " |")
    lines += ["", f"Run-to-run agreement: {_percent(report['agreement_rate'])} of scored items (acceptance needs "
                  "100%).", ""]
    for key, entry in report["cards"].items():
        split = [cell for cell, item in entry["agreement"].items() if not item["agree"]]
        lines.append(f"- {key} ({_percent(entry['agreement_rate'])}): " + (
            "all cells agree" if not split else "disagree on " + ", ".join(
            f"{cell} ({', '.join(entry['agreement'][cell]['values'])})" for cell in split)))
    lines += ["", "Mismatches with the expected table:", ""]
    found = False
    for key, entry in report["cards"].items():
        for run in entry["runs"]:
            for item in run["mismatches"]:
                found = True
                lines.append(f"- {key} run {run['run']}: {item}")
    if not found:
        lines.append("- none")
    for problem in report["problems"]:
        lines.append(f"- {problem}")
    lines += ["", "Invalid card-runs (environment failures, decision ENV; not scored):", ""]
    for item in report.get("invalid", []):
        lines.append(f"- {item['card']} run {item['run']} ({Path(item['run_dir']).name}): {item['stopped']}; "
                     + (f"replaced by {Path(item['replaced_by']).name}" if item["replaced_by"]
                        else "not replaced (rerun with --rerun-invalid)"))
    if not report.get("invalid"):
        lines.append("- none")
    lines += ["", "Skill scale at target (FIX-17, decision C17; recorded, not scored):", ""]
    for key, entry in report["cards"].items():
        for run in entry["runs"]:
            coupled = run.get("target_coupled") or {}
            slots = coupled.get("slots") or {}
            lines.append(f"- {key} run {run['run']}: {coupled.get('status', 'not recorded')}"
                         + (": " + ", ".join(f"{slot} {scale}" for slot, scale in slots.items()) if slots else ""))
    return "\n".join(lines) + "\n"


async def calibrate(*, cards: dict[str, dict], runs: int, run_root: Path, out_dir: Path, cfg: dict, reason: str,
                    deps_factory: Callable[[Ledger], Deps], expected: dict | None = None, parallel: int = 1,
                    job_slots: dict | None = None) -> dict:
    """Run every card ``runs`` times, then compare with the expected table. Writes ``calibration.json`` and
    ``report.md`` in ``out_dir``.

    ``parallel`` card-runs run at once (hybrid/research/parallel.py), started in the serial order (run 1 of every
    card, then run 2, ...). Each card-run has its own run directory and its own dependencies from ``deps_factory``, so
    the runs of one card stay independent: no prover memory, client or stage state is shared. They share the host's
    job slots and LLM lanes, so the GPT and local routes still take one request at a time. The acceptance rule is
    unchanged. ``job_slots`` (how the job slot limit was set) is recorded in the meta."""
    reason = reason.strip()
    if not reason:
        raise ValueError("a calibration run needs a non-empty reason")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    check_parallel(parallel)
    check_complete(cards)
    out_dir, run_root = Path(out_dir).resolve(), Path(run_root).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = expected or load_expected()
    target_rule = {k: cfg["target"][k] for k in m_math.TARGET_RULE_KEYS}
    run_cfg = calibration_config(cfg)
    started = datetime.now(timezone.utc).isoformat()
    order = [(run_index, key) for run_index in range(1, runs + 1) for key in cards]
    finished: dict[int, dict] = {}

    def thunk(run_index: int, key: str):
        return lambda: run_card(key=key, card=cards[key], run_index=run_index,
                                run_dir=run_root / f"{key}-run{run_index}", cfg=run_cfg, reason=reason,
                                deps_factory=deps_factory, target_rule=target_rule)

    def done(index: int, outcome: dict) -> None:
        # outcomes.jsonl holds the finished card-runs in the serial order, whichever finished first.
        finished[index] = outcome
        _write(out_dir / "outcomes.jsonl", [finished[i] for i in sorted(finished)])

    outcomes = await run_limited([thunk(run_index, key) for run_index, key in order], parallel, done)
    report = compare({"cards": {k: v for k, v in expected["cards"].items() if k in cards},
                      "runs_required": expected.get("runs_required", 3)}, outcomes)
    meta = {"started": started, "finished": datetime.now(timezone.utc).isoformat(), "runs": runs,
            "cards": list(cards), "run_root": str(run_root), "reason": reason, "target_rule": target_rule,
            "profile_version": profile.PROFILE_VERSION, "statement": STATEMENT, "parallel": parallel,
            "job_slots": job_slots}
    result = {"meta": meta, "expected": expected, "outcomes": outcomes, "report": report}
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                                              encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(report, outcomes, meta), encoding="utf-8")
    return result


def rerun_targets(outcomes: list[dict]) -> list[dict]:
    """One target per card and run index that has no valid card-run: the latest invalid card-run there, with the next
    rerun number. A valid card-run is never a target."""
    scored, _ = split_valid(outcomes)
    have = {(o["card"], o["run"]) for o in scored}
    latest: dict[tuple[str, int], dict] = {}
    for outcome in outcomes:
        slot = (outcome["card"], outcome["run"])
        if slot not in have and (slot not in latest or (outcome.get("rerun") or 0) > (latest[slot].get("rerun") or 0)):
            latest[slot] = outcome
    return [dict(outcome=o, rerun=(o.get("rerun") or 0) + 1) for _, o in sorted(latest.items())]


def check_rerunnable(outcome: dict, card: dict) -> None:
    """Refuse to rerun a valid card-run, or one whose card fixture changed since it ran (the rerun would not be the
    same card-run)."""
    if validity(outcome) == VALID:
        raise ValueError(f"{outcome['run_dir']} is a valid card-run; only {INVALID_ENVIRONMENT} card-runs are rerun "
                         "(verdicts are never retried)")
    ledger = Path(outcome["run_dir"]) / "ledger.jsonl"
    with ledger.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    if first["type"] != "operator_override":
        raise ValueError(f"{ledger} does not start with the calibration operator_override")
    raw = (json.dumps(card, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if first["data"]["source_card"]["sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"calibration card {outcome['card']} changed since {outcome['run_dir']} ran; a rerun would "
                         "not repeat that card-run")


async def rerun_invalid(*, out_dir: Path, cfg: dict, reason: str, deps_factory: Callable[[Ledger], Deps],
                        parallel: int = 1, cards: dict[str, dict] | None = None) -> dict:
    """FIX-22: rerun every invalid_environment card-run of the calibration in ``out_dir`` that no valid rerun has
    replaced, each in a fresh run directory ``<card>-run<N>-rerun<M>`` under the calibration's run root, with the
    calibration's target rule; then rescore with the reruns in place of the invalid card-runs. The previous
    ``calibration.json`` and ``report.md`` are kept as ``*-before-rerun<K>``; the run directories already there are
    never touched. Refuses when every card-run is valid."""
    reason = reason.strip()
    if not reason:
        raise ValueError("a rerun needs a non-empty reason")
    check_parallel(parallel)
    out_dir = Path(out_dir).resolve()
    data = json.loads((out_dir / "calibration.json").read_text(encoding="utf-8"))
    meta, outcomes = data["meta"], data["outcomes"]
    targets = rerun_targets(outcomes)
    if not targets:
        raise ValueError(f"{out_dir}: every card-run is valid; nothing to rerun")
    cards = cards or load_cards(tuple(meta["cards"]))
    check_complete(cards)
    for target in targets:
        check_rerunnable(target["outcome"], cards[target["outcome"]["card"]])
    run_root = Path(meta["run_root"])
    run_cfg = calibration_config(cfg)
    target_rule = meta["target_rule"]
    started = datetime.now(timezone.utc).isoformat()

    def thunk(target: dict):
        outcome = target["outcome"]
        run_dir = run_root / f"{outcome['card']}-run{outcome['run']}-rerun{target['rerun']}"
        return lambda: run_card(key=outcome["card"], card=cards[outcome["card"]], run_index=outcome["run"],
                                run_dir=run_dir, cfg=run_cfg, reason=reason, deps_factory=deps_factory,
                                target_rule=target_rule,
                                rerun={"rerun": target["rerun"], "rerun_of": outcome["run_dir"]})

    new = await run_limited([thunk(t) for t in targets], parallel)
    attempt = len(meta.get("reruns", [])) + 1
    for name in ("calibration.json", "report.md"):
        source = out_dir / name
        source.rename(out_dir / f"{source.stem}-before-rerun{attempt}{source.suffix}")
    meta.setdefault("reruns", []).append({
        "rerun": attempt, "started": started, "finished": datetime.now(timezone.utc).isoformat(), "reason": reason,
        "parallel": parallel, "replaced": [t["outcome"]["run_dir"] for t in targets],
        "run_dirs": [o["run_dir"] for o in new]})
    order = {key: index for index, key in enumerate(meta["cards"])}
    outcomes = sorted(outcomes + new, key=lambda o: (o["run"], order[o["card"]], o.get("rerun") or 0))
    expected = data["expected"]
    report = compare({"cards": {k: v for k, v in expected["cards"].items() if k in cards},
                      "runs_required": expected.get("runs_required", 3)}, outcomes)
    result = {"meta": meta, "expected": expected, "outcomes": outcomes, "report": report}
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                                              encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(report, outcomes, meta), encoding="utf-8")
    _write(out_dir / "outcomes.jsonl", outcomes)
    return result


def _write(path: Path, outcomes: list[dict]) -> None:
    path.write_text("".join(json.dumps(o, sort_keys=True, default=str) + "\n" for o in outcomes), encoding="utf-8")


# ----- the command --------------------------------------------------------------------------------------------------

def real_deps_factory(cfg: dict, routing: str | None, slots=None) -> Callable[[Ledger], Deps]:
    """New dependencies for every card-run. ``slots`` (hybrid.slots.JobSlots) are shared by every card-run's executor;
    every LLM client holds the host-wide lanes (hybrid/llm/lanes.py)."""
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.llm import LLMClient
    from hybrid.math.runner import MathRunner
    from hybrid.slots import SLOTS_DIR

    def build(ledger: Ledger) -> Deps:
        run_dir = ledger.run_dir.resolve()
        llm = LLMClient(str(ROOT / cfg["run"]["models_config"]), str(run_dir), ledger, routing=routing,
                        lanes_dir=SLOTS_DIR)
        executor = Executor(run_dir.name, run_dir / "jobs" / "receipts", slots=slots)
        return Deps(llm, executor, read_lock()["reference"],
                    MathRunner(executor, read_lock("tools")["reference"], run_dir))

    return build


def calibrate_command(args) -> int:
    import sys

    from hybrid.llm import ConfigError, GatewayUnavailable, check_gateways, load_config
    from hybrid.research.orchestrator import load_run_config

    if args.rerun_invalid is not None:
        return rerun_command(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("results") / f"calibration-math-{stamp}"
    run_root = args.run_root or Path("runs") / f"calibration-math-{stamp}"
    keys = tuple(k.strip() for k in args.cards.split(",") if k.strip())
    unknown = [k for k in keys if k not in CARD_KEYS]
    try:
        if unknown or not keys:
            raise ValueError(f"--cards takes keys from {', '.join(CARD_KEYS)}, got {args.cards!r}")
        cfg = load_run_config(args.config)
        parallel = check_parallel(args.parallel)
        models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
        check_gateways(models)
        cards = load_cards(keys)
        slots, slot_record = job_slots_from_config(cfg)
    except (GatewayUnavailable, ConfigError, ValueError, VMUnavailable) as exc:
        print(f"cannot start the calibration: {exc}", file=sys.stderr)
        return 2
    print("CALIBRATION (operator override): " + STATEMENT, flush=True)
    print(f"cards: {', '.join(keys)}; runs: {args.runs}; out: {Path(out_dir).resolve()}; runs under "
          f"{Path(run_root).resolve()}", flush=True)
    print(f"card-runs at once: {parallel}; " + slot_line(slot_record), flush=True)
    try:
        result = asyncio.run(calibrate(cards=cards, runs=args.runs, run_root=run_root, out_dir=out_dir, cfg=cfg,
                                       reason=args.reason, deps_factory=real_deps_factory(cfg, args.routing, slots),
                                       parallel=parallel, job_slots=slot_record))
    except ValueError as exc:
        print(f"cannot start the calibration: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["report"], indent=2))
    return 0 if result["report"]["accepted"] else 1



def rerun_command(args) -> int:
    """``calibrate --rerun-invalid <out dir>``: the calibration's own cards, runs, run root and target rule; --out,
    --run-root, --runs and --cards do not apply."""
    import sys

    from hybrid.llm import ConfigError, GatewayUnavailable, check_gateways, load_config
    from hybrid.research.orchestrator import load_run_config

    if args.out is not None or args.run_root is not None:
        print("cannot rerun: --rerun-invalid uses the calibration's own results directory and run root; drop --out "
              "and --run-root", file=sys.stderr)
        return 2
    try:
        cfg = load_run_config(args.config)
        parallel = check_parallel(args.parallel)
        models = load_config(ROOT / cfg["run"]["models_config"], args.routing)
        check_gateways(models)
        slots, slot_record = job_slots_from_config(cfg)
        print(f"CALIBRATION RERUN (operator override) of the invalid card-runs in {Path(args.rerun_invalid).resolve()}"
              f"; card-runs at once: {parallel}; " + slot_line(slot_record), flush=True)
        result = asyncio.run(rerun_invalid(out_dir=args.rerun_invalid, cfg=cfg, reason=args.reason,
                                           deps_factory=real_deps_factory(cfg, args.routing, slots),
                                           parallel=parallel))
    except (GatewayUnavailable, ConfigError, ValueError, VMUnavailable, FileNotFoundError) as exc:
        print(f"cannot rerun the calibration: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["report"], indent=2))
    return 0 if result["report"]["accepted"] else 1
