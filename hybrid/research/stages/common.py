"""Shared stage machinery: the stage context, prompt loading, budgets, code execution, helpers.

Stage functions receive a `StageContext` and a card id. They return the next stage name or None
when the card's line ends. External failures (model transport or schema errors, executor infra
errors, literature errors) raise `StageError`; the card stays at its last recorded verdict and the
orchestrator records a `stage_error` event. Run budgets raise `RunBudgetExhausted`.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hybrid.research.ledger import Ledger
from hybrid.research.schemas import load_schema
from hybrid.research.store import Store

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
# Text appended to a role's user message in code. prompts/*.md are read live by a running process, so a
# contract change goes into a new file here instead of an edit there.
SECTION_DIR = Path(__file__).resolve().parents[1] / "prompt_sections"
# Roles that share one prompt file. Every other prompt name maps to prompts/<name>.md.
PROMPT_FILES = {
    "explorer_alt": "explorer",
    "feasibility_judge_a": "feasibility_judge",
    "feasibility_judge_b": "feasibility_judge",
    # Phase 7: the refinement challenger for cards a GPT model proposed (Claude first) shares the challenger prompt.
    "refine_challenger_alt": "refine_challenger",
}
ACTOR = "orchestrator"
STAGE_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
# s3.mode = "math": stages M (math) and C (challenge) replace S3 (DISCOVERY-PIPELINE-PLAN.md sections 4 and 5).
MATH_ORDER = ("S0", "S1", "S2", "M", "C", "S4", "S5", "S6")
S3_MODES = ("judges", "math")
# refine.enabled (Phase 7, DISCOVERY-PIPELINE-PLAN.md section 7): the refinement dialogue R runs between S1 and S2.
REFINE = "R"


def pipeline_order(cfg: dict) -> tuple[str, ...]:
    """The stage order a run uses: S3 with its judges (the default, and every run started before Phase 3), or the
    math stages M and C in its place; with refine.enabled the dialogue stage R follows S1 (load_run_config gives a
    config written before Phase 7 refine.enabled = false, so its runs have no stage R)."""
    order = MATH_ORDER if cfg["s3"]["mode"] == "math" else STAGE_ORDER
    if refine_enabled(cfg):
        index = order.index("S1") + 1
        order = order[:index] + (REFINE,) + order[index:]
    return order


def refine_enabled(cfg: dict) -> bool:
    return cfg["refine"]["enabled"]


def after_s1(cfg: dict) -> str:
    """The stage a card goes to when it passes S1."""
    return REFINE if refine_enabled(cfg) else "S2"


class StageError(Exception):
    """An external boundary failed while a stage ran. The card is left where it was."""


class RunBudgetExhausted(Exception):
    """A run budget (cards, calls, tokens, wall time) is used up. The run stops cleanly."""


# The card a stage worker is working on, and the stage, while it runs (set by the orchestrator). Model calls made in
# that context are charged to the card's idea and lineage (hybrid.research.lineage). Unset outside a stage worker:
# explorer calls are charged once their card exists, and plumb runs charge nothing.
CURRENT_CARD: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("current_card", default=None)


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{PROMPT_FILES.get(name, name)}.md"
    return path.read_text(encoding="utf-8")


def load_section(name: str) -> str:
    return (SECTION_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def new_card_id() -> str:
    return f"card-{uuid.uuid4().hex[:8]}"


def new_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


# ----- budgets -------------------------------------------------------------------------------


@dataclass
class Budget:
    max_cards: int
    max_calls: int
    max_tokens: int
    max_wall_s: float
    cards: int = 0
    calls: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    def exhausted_reason(self) -> str | None:
        if self.calls >= self.max_calls:
            return f"max_calls {self.max_calls} reached"
        if self.tokens >= self.max_tokens:
            return f"max_tokens {self.max_tokens} reached ({self.tokens})"
        if self.elapsed() >= self.max_wall_s:
            return f"max_wall_s {self.max_wall_s} reached"
        return None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        reason = self.exhausted_reason()
        if reason is not None:
            raise RunBudgetExhausted(reason)

    def snapshot(self) -> dict:
        return {"cards": self.cards, "calls": self.calls, "tokens": self.tokens,
                "elapsed_s": round(self.elapsed(), 1), "max_cards": self.max_cards,
                "max_calls": self.max_calls, "max_tokens": self.max_tokens, "max_wall_s": self.max_wall_s}


# ----- dependencies --------------------------------------------------------------------------


class LLM(Protocol):
    async def complete(self, role: str, system: str, user: str, *, json_schema: dict | None = None,
                       allow_web: bool = False, max_output_tokens: int | None = None) -> Any: ...


class BenchRunner(Protocol):
    async def evaluate(self, card_id: str, candidate_dir: Path, claimed_exponent: float | None) -> dict: ...


@dataclass
class ExecResult:
    job_id: str
    status: str
    detail: str
    stdout: str
    stderr: str
    run_s: float
    out_dir: Path | None = None   # the job's output directory (files the job wrote under /out)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def printed_json(self) -> dict | None:
        """The last stdout line that parses as a JSON object, else the whole stdout, else None."""
        for line in reversed(self.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    return value
        try:
            value = json.loads(self.stdout)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def report(self, limit: int = 3000) -> dict:
        return {"job_id": self.job_id, "status": self.status, "detail": self.detail,
                "stdout_tail": self.stdout[-limit:], "stderr_tail": self.stderr[-limit:]}


class CodeRunner:
    """Runs small Python programs through an executor (Podman `Executor` or `LocalExecutor`).

    Each job gets `<run>/jobs/<job-id>/{in,out}`; the files are written to `in/` and the entry
    file is run with `python3 /in/<entry>`. The receipt, code, stdout and stderr are stored as an
    `executor` artifact tied to the card, so the dossier links every executed check.
    """

    def __init__(self, executor, run_dir: Path, store: Store, image: str, caps):
        self.executor = executor
        self.run_dir = Path(run_dir)
        self.store = store
        self.image = image
        self.caps = caps

    async def run(self, card_id: str, purpose: str, files: dict[str, str], *, entry: str = "main.py",
                  caps=None) -> ExecResult:
        from hybrid.executor.job import Job

        job_id = new_job_id()
        job_dir = self.run_dir / "jobs" / job_id
        in_dir, out_dir = job_dir / "in", job_dir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        for name, text in files.items():
            (in_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (in_dir / name).write_text(text, encoding="utf-8")
        job = Job(job_id, self.image, in_dir, out_dir, ("python3", f"/in/{entry}"), caps or self.caps)
        receipt = await asyncio.to_thread(self.executor.run, job)
        stdout = Path(receipt.stdout_path).read_text(encoding="utf-8", errors="replace")
        stderr = Path(receipt.stderr_path).read_text(encoding="utf-8", errors="replace")
        record = {"job_id": job_id, "purpose": purpose, "status": receipt.status, "detail": receipt.detail,
                  "exit_code": receipt.exit_code, "run_s": receipt.run_s, "files": files,
                  "stdout": stdout[-20000:], "stderr": stderr[-8000:], "receipt": asdict(receipt)}
        self.store.put_artifact("executor", f"{job_id}.json",
                                json.dumps(record, indent=2, sort_keys=True, default=str).encode("utf-8"),
                                card_id=card_id)
        if receipt.status == "infra_error":
            raise StageError(f"executor infra_error for {purpose} ({job_id}): {receipt.detail}")
        return ExecResult(job_id, receipt.status, receipt.detail, stdout, stderr, receipt.run_s, out_dir)


# ----- context -------------------------------------------------------------------------------


@dataclass
class StageContext:
    run_id: str
    run_dir: Path
    ledger: Ledger
    store: Store
    llm: LLM
    runner: CodeRunner
    bench: BenchRunner
    cfg: dict
    budget: Budget
    lenses: dict
    # classify_prior_art(card_text, llm_complete, sink) from hybrid.literature; tests inject a fake.
    classify_prior_art: Any = None
    # Called with a card id when S6 finds supported claims; the orchestrator queues a mutate task.
    on_mutation: Any = None
    # Called with a card id when S2 gives reduces_to_known; the orchestrator queues an escape task.
    on_reduction: Any = None
    # Other run directories whose cards S0 adds to the graveyard, read-only (explore.graveyard_runs).
    graveyard_runs: tuple[Path, ...] = ()
    # hybrid.math.runner.MathRunner for stages M and C (tools image through the executor); None when s3.mode is
    # "judges".
    math: Any = None
    # lookup(source, id) for stage C's cited premises (hybrid.research.citations.default_lookup when None); tests
    # inject a fake.
    lookup_source: Any = None
    # hybrid.research.outcomes.OutcomeStore when outcomes.enabled: S0 lists nearby dead ideas, S2 searches own history.
    outcomes: Any = None
    # hybrid.research.lineage.BudgetBook: per-idea and per-lineage caps. The orchestrator always sets it; plumb runs
    # single stages without one.
    book: Any = None
    # hybrid.research.stages.g_generate.Generation when generation.mode = "arms" (Phase 6): arms, archive, families,
    # limits. None in lens mode, so S0 runs as before.
    generation: Any = None

    def emit(self, type: str, data: dict, actor: str = ACTOR) -> dict:
        return self.ledger.append(type, actor, data)

    async def ask(self, role: str, prompt: str, user: str, schema: dict | None = None,
                  allow_web: bool = False):
        """One model call with the system prompt from prompts/<prompt>.md. Counts against the run budget."""
        return await self.call(role, load_prompt(prompt), user, schema, allow_web)

    async def call(self, role: str, system: str, user: str, schema: dict | None = None,
                   allow_web: bool = False):
        from hybrid.llm.errors import BudgetExceeded, LLMError

        self.budget.check()
        current = CURRENT_CARD.get()
        if current is not None:
            self.book.reserve_call(current[0])
        self.budget.calls += 1
        try:
            result = await self.llm.complete(role, system, user, json_schema=schema, allow_web=allow_web)
        except BaseException as exc:
            if current is not None:
                self.book.release_call(current[0])
            if isinstance(exc, BudgetExceeded):
                raise RunBudgetExhausted(f"LLMClient budget: {exc}") from exc
            if isinstance(exc, LLMError):
                raise StageError(f"{role}: {type(exc).__name__}: {exc}") from exc
            raise
        if current is not None:
            self.book.charge_call(current[0], result.call_id, current[1], reserved=True)
        usage = result.usage or {}
        self.budget.tokens += int(usage.get("input") or 0) + int(usage.get("output") or 0)
        if schema is not None and result.parsed is None:
            raise StageError(f"{role}: no parsed JSON in {result.call_id}")
        return result

    def card(self, card_id: str) -> dict:
        return self.store.get_card(card_id)

    def verdict(self, card_id: str, stage: str, verdict: str, reasons: list[str], by_calls: list[str]) -> dict:
        record = self.store.set_verdict(card_id, stage, verdict, reasons, by_calls)
        self.emit("stage_verdict", {"card_id": card_id, "stage": stage, "verdict": verdict,
                                    "card_version": record["card_version"]})
        return record

    def put_json_artifact(self, kind: str, name: str, value: Any, card_id: str) -> dict:
        data = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return self.store.put_artifact(kind, name, data, card_id=card_id)

    def latest_json_artifact(self, card_id: str, kind: str) -> Any | None:
        """The JSON content of the card's most recent artifact of `kind`, or None when it has none."""
        recorded = [data for data in self.card_events(card_id, "artifact") if data["kind"] == kind]
        if not recorded:
            return None
        return json.loads((self.run_dir / recorded[-1]["path"]).read_text(encoding="utf-8"))

    def artifact_name(self, card_id: str, suffix: str = "json") -> str:
        return f"{card_id}-v{self.store.latest_version(card_id)}-{uuid.uuid4().hex[:6]}.{suffix}"

    def card_events(self, card_id: str, type: str) -> list[dict]:
        return [event["data"] for event in self.ledger.iter_events(type) if event["data"].get("card_id") == card_id]


# ----- card helpers --------------------------------------------------------------------------

ORCHESTRATOR_FIELDS = ("id", "version", "parent_ids", "created_by_call", "lens", "status")
# Optional in theory_card (older cards lack them) but required from every new or revised explorer draft.
DRAFT_REQUIRED = ("core_idea", "source_transplant")
# Optional in theory_card, required from escape drafts and from revisions of cards that carry it.
ESCAPE_FIELD = "non_reduction_argument"
# Fields of generation mode "arms" (Phase 6): filled at G by the orchestrator (arm), the arm draft (arm_structure, an
# arm-specific schema added by g_generate) and the classifier (commitments, archive_cell). Never part of the generic
# draft schema, and carried unchanged through revisions.
GENERATION_FIELDS = ("arm", "arm_structure", "commitments", "archive_cell")
# Fields a later stage writes (Phase 7: S2's difference statement). Never part of a draft, carried through revisions.
STAGE_FIELDS = ("difference_statement",)
TRANSPLANT_FIELD = "source_transplant"


def draft_schema(*, escape: bool = False, transplant: bool = True) -> dict:
    """theory_card schema without the fields the orchestrator fills in, with DRAFT_REQUIRED required.

    With escape=True, `non_reduction_argument` is required as well (escape drafts and revisions of cards
    that carry it). Otherwise the field is left out, so new, mutate and cross drafts never see it.
    With transplant=False, `source_transplant` is left out: generation arms other than the cross-field arm get no
    papers, so they have no transplant to name (and revisions of their cards keep it out).
    """
    schema = copy.deepcopy(load_schema("theory_card"))
    schema.pop("$id", None)
    schema["title"] = "Theory card draft (explorer output)"
    for name in ORCHESTRATOR_FIELDS + GENERATION_FIELDS + STAGE_FIELDS:
        schema["properties"].pop(name)
    if not escape:
        schema["properties"].pop(ESCAPE_FIELD)
    if not transplant:
        schema["properties"].pop(TRANSPLANT_FIELD)
    schema["required"] = [name for name in schema["required"] if name not in ORCHESTRATOR_FIELDS]
    required = (tuple(name for name in DRAFT_REQUIRED if transplant or name != TRANSPLANT_FIELD)
                + ((ESCAPE_FIELD,) if escape else ()))
    schema["required"] += [name for name in required if name not in schema["required"]]
    return schema


def ground_draft(draft: dict, allowed: set[str]) -> tuple[dict, list[dict], list[str]]:
    """Drop references and transplant sources that name papers outside `allowed`.

    Returns (draft, dropped references, dropped transplant source ids). A transplant left with no
    source names no paper the explorer was given, which is a broken draft (StageError).
    """
    draft = dict(draft)
    kept = [ref for ref in draft.get("references", []) if ref["source_id"] in allowed]
    dropped = [ref for ref in draft.get("references", []) if ref["source_id"] not in allowed]
    draft["references"] = kept
    dropped_sources: list[str] = []
    transplant = draft.get("source_transplant")
    if transplant is not None:
        sources = [sid for sid in transplant["source_ids"] if sid in allowed]
        dropped_sources = [sid for sid in transplant["source_ids"] if sid not in allowed]
        if not sources:
            raise StageError(f"source_transplant cites no paper the explorer was given: {transplant['source_ids']}")
        draft["source_transplant"] = dict(transplant, source_ids=sources)
    return draft, dropped, dropped_sources


def parsimony(card: dict) -> dict:
    mech = card["mechanism"]
    return {"state_objects": len(mech["state"]), "update_equations": len(mech["update"]),
            "readout_equations": len(mech["readout"]), "extensions": len(card.get("extensions", []))}


def card_for_prompt(card: dict) -> str:
    shown = {key: value for key, value in card.items() if key not in ("status", "created_by_call", "version")}
    return json.dumps(shown, indent=2, ensure_ascii=False)


def card_text_for_similarity(card: dict) -> str:
    mech = card["mechanism"]
    parts = [card["title"], card["summary"], card.get("core_idea", "")]
    parts += [f"{item['name']} {item['meaning']}" for item in mech["state"]]
    parts += [eq["plain"] for eq in mech["update"]]
    parts += [eq["plain"] for eq in mech["readout"]]
    parts += [mech["learning"]["description"], mech["learning"]["few_shot_argument"]]
    return "\n".join(parts)


def compare_expected(expected: Any, got: Any, rtol: float, atol: float, path: str = "") -> list[str]:
    """Differences between the card's `expected` values and what the toy example printed."""
    where = path or "<root>"
    if isinstance(expected, bool) or isinstance(got, bool):
        same = isinstance(expected, bool) and isinstance(got, bool) and expected == got
        return [] if same else [f"{where}: expected {expected!r}, got {got!r}"]
    if isinstance(expected, (int, float)):
        if not isinstance(got, (int, float)):
            return [f"{where}: expected number {expected!r}, got {got!r}"]
        if not (math.isfinite(float(got)) and abs(float(got) - float(expected)) <= atol + rtol * abs(float(expected))):
            return [f"{where}: expected {expected!r}, got {got!r}"]
        return []
    if isinstance(expected, dict):
        if not isinstance(got, dict):
            return [f"{where}: expected an object, got {got!r}"]
        problems = []
        for key, value in expected.items():
            if key not in got:
                problems.append(f"{path}/{key}: missing from printed output")
            else:
                problems.extend(compare_expected(value, got[key], rtol, atol, f"{path}/{key}"))
        return problems
    if isinstance(expected, list):
        if not isinstance(got, list) or len(got) != len(expected):
            return [f"{where}: expected a list of {len(expected)}, got {got!r}"]
        problems = []
        for index, (a, b) in enumerate(zip(expected, got)):
            problems.extend(compare_expected(a, b, rtol, atol, f"{path}[{index}]"))
        return problems
    return [] if expected == got else [f"{where}: expected {expected!r}, got {got!r}"]


def claimed_exponent(complexity_expr: str) -> float | None:
    """Exponent of n in a total-sequence complexity such as O(n), O(n log n), O(n^2 d)."""
    import re

    text = complexity_expr
    match = re.search(r"(?<![A-Za-z_])n\s*(?:\^|\*\*)\s*\{?\s*(\d+(?:\.\d+)?)\s*\}?", text)
    if match:
        return float(match.group(1))
    if "n²" in text:
        return 2.0
    if re.search(r"(?<![A-Za-z_])n(?![A-Za-z_])", text):
        return 1.0
    return 0.0
