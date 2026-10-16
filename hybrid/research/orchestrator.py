"""Research loop scheduler (RESEARCH-SYSTEM-SPEC.md section 6).

N explorer workers create cards (S0). Each later stage has its own asyncio queue and workers.
A stage function returns the next stage, and the card id is put on that queue. Per-role model
concurrency is enforced inside LLMClient. A card that S2 reduces to a known family gets an `escape`
explore task (explore.p_escape, while card slots remain); explorers take queued mutations first, then
queued escapes, then draw new, mutate or cross. New, mutate and cross tasks leave explore.escape_reserve of
the card slots for escapes, so an escape finds a slot even when the explorers have drawn the rest. Budgets
(cards, calls, tokens, wall time) stop the run cleanly: workers are cancelled, the in-flight call is lost,
and every card keeps its last recorded verdict, from which `--resume` rebuilds the queues.

Generation arms (Phase 6, `[generation] mode = "arms"`, hybrid.research.stages.g_generate): new tasks draw a generation
arm by weight (or from `generation.quota`), every card is classified into the quality-diversity archive
(hybrid.research.archive), and mutate and cross parents are drawn by archive cell weight. A pass at
`archive.reward_stage` rewards a card that filled an empty cell. `generation.draft_only` stops every card at S0 (the
Phase 6 live check of the arms); no stage workers run. The mode is recorded in run_start and a resumed run keeps it.

Loop budgets (Phase 8, hybrid.research.lineage): every card has a lineage and an idea key. Model calls made while a
stage works on a card are charged to both, and a card whose idea or lineage has used its calls ends at that stage
with `budget:model_calls` or `budget:lineage_calls`. Mutation and escape requests, and parents drawn for mutate and
cross tasks, need room in the parent's lineage (`budgets.lineage.ideas`); an escape is not requested when the card
was itself an escape and S2 reduced it to the same family as its parent (`stall:escape`).
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import random
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hybrid.executor.job import Caps
from hybrid.research import graveyard, lineage
from hybrid.research.ledger import Ledger
from hybrid.research.stages import STAGE_RUNNERS
from hybrid.research.stages.common import (CURRENT_CARD, S3_MODES, Budget, CodeRunner,
                                           RunBudgetExhausted, StageContext, StageError, pipeline_order)
from hybrid.research.stages.escape import escape_chain, reduction_record
from hybrid.research.stages.s0_explore import ExploreTask, explore, lens_seed
from hybrid.research.store import Store

ROOT = Path(__file__).resolve().parents[2]
ACTOR = "orchestrator"
# Keys that run.toml may leave out. run.seed unset means "derive the lens seed from the run id".
OPTIONAL_KEYS = {"run.seed"}
# The seed every run used before the lens seed was derived from the run id (run.toml shipped seed = 0).
LEGACY_LENS_SEED = 0
# [explore] keys added after runs had started; a config file without them gets these values.
EXPLORE_DEFAULTS = {"graveyard_max": 20, "graveyard_runs": [], "p_escape": 1.0, "escape_reserve": 0.3}


# [literature] keys added after runs had started. The path is relative to the repository root.
LITERATURE_DEFAULTS = {"local_index_path": "data/arxiv-index/arxiv.sqlite"}
# [s5] keys added with the card falsifier tasks; a config file without them gets these values.
S5_DEFAULTS = {"falsifiers": True, "falsifier_wall_s": 300}
# [s3] keys added with the executed obligation checks; a config file without them gets these values. s3.mode
# "judges" is the S3 stage; "math" runs stages M and C in its place (Phase 3 of DISCOVERY-PIPELINE-PLAN.md). Every
# run started before Phase 3 has no mode and resumes with the judges.
S3_DEFAULTS = {"symbolic_checks": True, "max_obligation_checks": 6, "mode": "judges"}
# [math] keys for stages M and C; a config file without the section gets these values.
MATH_DEFAULTS = {"max_prover_attempts": 4, "max_obligations": 12, "max_statements": 16, "match_instances": 1000}
# [math] profile (Phase 4): hold every card to the transformer profile P1 to P5 (hybrid/math/profile.py). A config
# without the key gets false, so runs started before Phase 4 resume without it.
MATH_PROFILE_DEFAULT = False
# [math] max_parallel_jobs: executor jobs that may run at once on this host, across every process (hybrid/slots.py).
# "auto" is half the Podman VM's CPUs; any value is clamped to the VM's CPUs and memory when a run starts. A config
# without the key gets "auto".
MAX_PARALLEL_JOBS_DEFAULT = "auto"
# [target] keys (Phase 4, the compute target gate, hybrid/math/target.py). The numbers are decision D1's defaults
# (DISCOVERY-PIPELINE-PLAN.md sections 1.1 and 18, applied 2026-10-13 until Andrew decides): total cost exponent in n
# at most 1.5, and inference state fixed or O(log n). A config without the section gets the gate off. The coupled
# bounds (FIX-17, decision C17 as amended 2026-10-15) judge each P2 skill at the card's resources: task size s = n^b
# with total cost at most n^1.5 and state at most n^0.5 (hybrid/math/coupled.py).
TARGET_DEFAULTS = {"enabled": False, "max_cost_exponent": 1.5, "state_growth": "log",
                   "coupled_max_cost_exponent": 1.5, "coupled_max_state_exponent": 0.5}
# [outcomes] keys (outcome store, DISCOVERY-PIPELINE-PLAN.md section 9); a config file without the section gets these
# values, so every run started before Phase 5 resumes with the store off.
OUTCOMES_DEFAULTS = {"enabled": False, "path": "results/outcomes", "prompt_max": 12, "history_max": 8,
                     "jaccard_min": 0.6, "tfidf_min": 0.3, "enforce_r3": False}
# [generation] keys (Phase 6); a config file without the section gets these values, so every run started before
# Phase 6 resumes in lens mode. Paths are relative to the repository root.
# [refine] keys (Phase 7, refinement dialogue R between S1 and S2, DISCOVERY-PIPELINE-PLAN.md section 7); a config
# without the section gets these values, so every run started before Phase 7 resumes without R. The number of rounds
# is budgets.idea.refinement_rounds.
REFINE_DEFAULTS = {"enabled": False, "growth_limit": 0.25, "max_questions": 5}
# [s2] keys added in Phase 7 (family exclusion and the difference statement, section 8); off when absent.
S2_DEFAULTS = {"family_exclusion": False, "difference_statement": False}
GENERATION_MODES = ("lenses", "arms")
GENERATION_DEFAULTS = {"mode": "lenses", "config": "hybrid/config/generation.toml",
                       "archive_config": "hybrid/config/archive.toml", "families_config": "hybrid/config/families.toml",
                       "limits_config": "hybrid/config/limits.toml", "quota": {}, "draft_only": False}


def stage_workers(cfg: dict, stage: str) -> int:
    """Workers for a stage queue; M and C default to the S3 count and R to the S1 count when the config does not name
    them."""
    workers = cfg["workers"]
    if stage in ("M", "C"):
        return workers.get(stage, workers.get("S3", 1))
    if stage == "R":
        return workers.get(stage, workers.get("S1", 1))
    return workers.get(stage, 0)


def recorded_s3_mode(ledger: Ledger) -> str | None:
    """The s3.mode a run started with, from its first run_start event; "judges" for a run started before the mode
    was recorded (every run before Phase 3), None for a ledger with no run_start."""
    for event in ledger.iter_events("run_start"):
        return event["data"].get("s3_mode", "judges")
    return None


def check_resume_mode(ledger: Ledger, mode: str) -> None:
    """A resumed run keeps the stage order it started with: S3 with its judges, or M and C in its place."""
    started = recorded_s3_mode(ledger)
    if started is not None and started != mode:
        raise ValueError(f"this run started with s3.mode = {started!r}; resume it with that mode, not {mode!r} "
                         f"(a config without the key gets 'judges')")


def recorded_refine(ledger: Ledger) -> bool | None:
    """Whether a run started with the refinement dialogue; False for a run started before Phase 7, None without
    run_start."""
    for event in ledger.iter_events("run_start"):
        return bool(event["data"].get("refine", {}).get("enabled", False))
    return None


def check_resume_refine(ledger: Ledger, enabled: bool) -> None:
    """A resumed run keeps its stage order: with stage R or without it."""
    started = recorded_refine(ledger)
    if started is not None and started != enabled:
        raise ValueError(f"this run started with refine.enabled = {str(started).lower()}; resume it with that setting, "
                         f"not {str(enabled).lower()} (a config without [refine] gets false)")


def recorded_generation_mode(ledger: Ledger) -> str | None:
    """The generation.mode a run started with; "lenses" for a run started before Phase 6, None without run_start."""
    for event in ledger.iter_events("run_start"):
        return (event["data"].get("generation") or {}).get("mode", "lenses")
    return None


def recorded_profile_target(ledger: Ledger) -> dict | None:
    """The math.profile and [target] settings a run started with; the pre-Phase 4 values (profile off, gate off) for
    a run started before they were recorded, None without run_start."""
    for event in ledger.iter_events("run_start"):
        data = event["data"]
        return {"profile": bool(data.get("math_profile", False)),
                "target": data.get("target") or {"enabled": False}}
    return None


def check_resume_profile_target(ledger: Ledger, cfg: dict) -> None:
    """A math run resumes with the profile and target settings it started with, so every card of the run is held to
    the same obligations and the same gate (a judges run never reads them)."""
    if cfg["s3"]["mode"] != "math":
        return
    started = recorded_profile_target(ledger)
    if started is None:
        return
    now = {"profile": cfg["math"]["profile"], "target": {key: cfg["target"][key] for key in TARGET_DEFAULTS}}
    if started["profile"] != now["profile"]:
        raise ValueError(f"this run started with math.profile = {str(started['profile']).lower()}; resume it with "
                         f"that setting, not {str(now['profile']).lower()} (a run started before Phase 4 has it off)")
    if started["target"].get("enabled", False) != now["target"]["enabled"] or (
            started["target"].get("enabled") and started["target"] != now["target"]):
        raise ValueError(f"this run started with [target] {started['target']}; resume it with that setting, not "
                         f"{now['target']} (a run started before Phase 4 has the gate off)")


def check_resume_generation(ledger: Ledger, mode: str) -> None:
    started = recorded_generation_mode(ledger)
    if started is not None and started != mode:
        raise ValueError(f"this run started with generation.mode = {started!r}; resume it with that mode, not {mode!r} "
                         f"(a config without the key gets 'lenses')")


def validate_generation(g: dict) -> None:
    unknown = set(g) - set(GENERATION_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown [generation] keys {sorted(unknown)}")
    if g["mode"] not in GENERATION_MODES:
        raise ValueError(f"generation.mode must be one of {GENERATION_MODES}, got {g['mode']!r}")
    if not isinstance(g["draft_only"], bool):
        raise ValueError(f"generation.draft_only must be true or false, got {g['draft_only']!r}")
    for key in ("config", "archive_config", "families_config", "limits_config"):
        if not isinstance(g[key], str) or not g[key]:
            raise ValueError(f"generation.{key} must be a non-empty path string, got {g[key]!r}")
    quota = g["quota"]
    if not isinstance(quota, dict) or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                                          for n in quota.values()):
        raise ValueError(f"generation.quota must map arm ids to integers >= 0, got {quota!r}")
    if quota and g["mode"] != "arms":
        raise ValueError("generation.quota needs generation.mode = 'arms'")


def derive_lens_seed(run_id: str) -> int:
    """A stable seed for the lens order of one run: the same run id always gives the same order."""
    return int(hashlib.sha256(f"lens-order:{run_id}".encode()).hexdigest()[:8], 16)


def validate_profile_target(cfg: dict) -> None:
    """[math] profile and the [target] section (Phase 4)."""
    from hybrid.math import target

    if not isinstance(cfg["math"]["profile"], bool):
        raise ValueError(f"math.profile must be true or false, got {cfg['math']['profile']!r}")
    rule = cfg["target"]
    if not isinstance(rule["enabled"], bool):
        raise ValueError(f"target.enabled must be true or false, got {rule['enabled']!r}")
    exponent = rule["max_cost_exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)) or exponent <= 0:
        raise ValueError(f"target.max_cost_exponent must be a positive number, got {exponent!r}")
    if rule["state_growth"] not in target.STATE_RULES:
        raise ValueError(f"target.state_growth must be one of {target.STATE_RULES}, got {rule['state_growth']!r}")
    for key in ("coupled_max_cost_exponent", "coupled_max_state_exponent"):
        value = rule[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"target.{key} must be a positive number, got {value!r}")
    if rule["enabled"] and not cfg["math"]["profile"]:
        raise ValueError("target.enabled needs math.profile = true: the gate reads the verified P1 formula")


def load_run_config(path: Path, overrides: dict | None = None) -> dict:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    for key, value in EXPLORE_DEFAULTS.items():
        cfg["explore"].setdefault(key, value)
    for key, value in LITERATURE_DEFAULTS.items():
        cfg.setdefault("literature", {}).setdefault(key, value)
    for key, value in S5_DEFAULTS.items():
        cfg["s5"].setdefault(key, value)
    for key, value in S3_DEFAULTS.items():
        cfg["s3"].setdefault(key, value)
    for key, value in MATH_DEFAULTS.items():
        cfg.setdefault("math", {}).setdefault(key, value)
    cfg["math"].setdefault("profile", MATH_PROFILE_DEFAULT)
    cfg["math"].setdefault("max_parallel_jobs", MAX_PARALLEL_JOBS_DEFAULT)
    for key, value in TARGET_DEFAULTS.items():
        cfg.setdefault("target", {}).setdefault(key, value)
    for key, value in OUTCOMES_DEFAULTS.items():
        cfg.setdefault("outcomes", {}).setdefault(key, value)
    for key, value in GENERATION_DEFAULTS.items():
        cfg.setdefault("generation", {}).setdefault(key, value)
    for key, value in REFINE_DEFAULTS.items():
        cfg.setdefault("refine", {}).setdefault(key, value)
    for key, value in S2_DEFAULTS.items():
        cfg["s2"].setdefault(key, value)
    lineage.fill_defaults(cfg)
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".")
        if key not in cfg[section] and dotted not in OPTIONAL_KEYS:
            raise KeyError(f"unknown run config key {dotted}")
        cfg[section][key] = value
    seed = cfg["run"].get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"run.seed must be an integer when set, got {seed!r}")
    residual = cfg["s1"].setdefault("pass_with_residual_fixable", True)
    if not isinstance(residual, bool):
        raise ValueError(f"s1.pass_with_residual_fixable must be true or false, got {residual!r}")
    index_path = cfg["literature"]["local_index_path"]
    if not isinstance(index_path, str) or not index_path:
        raise ValueError(f"literature.local_index_path must be a non-empty path string, got {index_path!r}")
    if not isinstance(cfg["s5"]["falsifiers"], bool):
        raise ValueError(f"s5.falsifiers must be true or false, got {cfg['s5']['falsifiers']!r}")
    wall = cfg["s5"]["falsifier_wall_s"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:
        raise ValueError(f"s5.falsifier_wall_s must be a positive number of seconds, got {wall!r}")
    if not isinstance(cfg["s3"]["symbolic_checks"], bool):
        raise ValueError(f"s3.symbolic_checks must be true or false, got {cfg['s3']['symbolic_checks']!r}")
    max_obligation_checks = cfg["s3"]["max_obligation_checks"]
    if isinstance(max_obligation_checks, bool) or not isinstance(max_obligation_checks, int) or max_obligation_checks < 0:
        raise ValueError(f"s3.max_obligation_checks must be an integer >= 0, got {max_obligation_checks!r}")
    empty_search = cfg["s2"]["empty_search_counts_as_novel"]
    if not isinstance(empty_search, bool):
        raise ValueError(f"s2.empty_search_counts_as_novel must be true or false, got {empty_search!r}")
    if cfg["s3"]["mode"] not in S3_MODES:
        raise ValueError(f"s3.mode must be one of {S3_MODES}, got {cfg['s3']['mode']!r}")
    for key in MATH_DEFAULTS:
        value = cfg["math"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"math.{key} must be an integer >= 1, got {value!r}")
    jobs = cfg["math"]["max_parallel_jobs"]
    if jobs != "auto" and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1):
        raise ValueError(f"math.max_parallel_jobs must be an integer >= 1 or \"auto\", got {jobs!r}")
    validate_profile_target(cfg)
    for stage in pipeline_order(cfg)[1:]:
        if stage_workers(cfg, stage) < 1:
            raise ValueError(f"workers.{stage} must be at least 1")
    if "revisions" in cfg["s1"]:
        raise ValueError("s1.revisions is now s1.max_revisions")
    revisions = cfg["s1"].get("max_revisions")
    if isinstance(revisions, bool) or not isinstance(revisions, int) or revisions < 0:
        raise ValueError(f"s1.max_revisions must be an integer >= 0, got {revisions!r}")
    validate_explore(cfg["explore"])
    validate_outcomes(cfg["outcomes"])
    validate_generation(cfg["generation"])
    validate_refine(cfg)
    lineage.validate_budgets(cfg)
    return cfg


def validate_refine(cfg: dict) -> None:
    r = cfg["refine"]
    unknown = set(r) - set(REFINE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown [refine] keys {sorted(unknown)}")
    if not isinstance(r["enabled"], bool):
        raise ValueError(f"refine.enabled must be true or false, got {r['enabled']!r}")
    limit = r["growth_limit"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit < 0:
        raise ValueError(f"refine.growth_limit must be a number >= 0, got {limit!r}")
    questions = r["max_questions"]
    if isinstance(questions, bool) or not isinstance(questions, int) or not 1 <= questions <= 5:
        raise ValueError(f"refine.max_questions must be an integer from 1 to 5 (plan section 7), got {questions!r}")
    for key in S2_DEFAULTS:
        if not isinstance(cfg["s2"][key], bool):
            raise ValueError(f"s2.{key} must be true or false, got {cfg['s2'][key]!r}")


def validate_outcomes(oc: dict) -> None:
    unknown = set(oc) - set(OUTCOMES_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown [outcomes] keys {sorted(unknown)}")
    for key in ("enabled", "enforce_r3"):
        if not isinstance(oc[key], bool):
            raise ValueError(f"outcomes.{key} must be true or false, got {oc[key]!r}")
    if not isinstance(oc["path"], str) or not oc["path"]:
        raise ValueError(f"outcomes.path must be a non-empty path string, got {oc['path']!r}")
    for key in ("prompt_max", "history_max"):
        if isinstance(oc[key], bool) or not isinstance(oc[key], int) or oc[key] < 0:
            raise ValueError(f"outcomes.{key} must be an integer >= 0, got {oc[key]!r}")
    for key in ("jaccard_min", "tfidf_min"):
        value = oc[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
            raise ValueError(f"outcomes.{key} must be a number in (0, 1], got {value!r}")


def open_store_readonly(cfg: dict):
    """The outcome store at outcomes.path for reading (archive seeds and signature duplicates) when it exists, whatever
    outcomes.enabled says; None when there is no store there yet. Nothing is written through it."""
    from hybrid.research.outcomes import JSONL, OutcomeStore

    path = Path(cfg["outcomes"]["path"])
    path = path if path.is_absolute() else ROOT / path
    return OutcomeStore(path) if (path / JSONL).is_file() else None


def open_outcome_store(cfg: dict):
    """The outcome store when outcomes.enabled, else None. outcomes.path is relative to the repository root."""
    if not cfg["outcomes"]["enabled"]:
        return None
    from hybrid.research.outcomes import OutcomeStore

    path = Path(cfg["outcomes"]["path"])
    return OutcomeStore(path if path.is_absolute() else ROOT / path)


def validate_explore(ex: dict) -> None:
    limit = ex["graveyard_max"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError(f"explore.graveyard_max must be an integer >= 0, got {limit!r}")
    runs = ex["graveyard_runs"]
    if not isinstance(runs, list) or not all(isinstance(run, str) and run for run in runs):
        raise ValueError(f"explore.graveyard_runs must be a list of run directory paths, got {runs!r}")
    p = ex["p_escape"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        raise ValueError(f"explore.p_escape must be a number in [0, 1], got {p!r}")
    reserve = ex["escape_reserve"]
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not 0.0 <= reserve <= 1.0:
        raise ValueError(f"explore.escape_reserve must be a number in [0, 1], got {reserve!r}")


def escape_reserve_slots(max_cards: int, fraction: float) -> int:
    """Card slots held back for escapes: the fraction of max_cards, rounded up so that any nonzero fraction
    holds at least one slot. The round before ceil keeps 0.3 * 10 from becoming 4 through float error."""
    return math.ceil(round(fraction * max_cards, 9))


def resolve_graveyard_runs(entries: list[str], own_run_dir: Path) -> tuple[Path, ...]:
    """Run directories (absolute, or relative to the repository root) read for the S0 graveyard. Each must
    hold a ledger and be readable now, so a bad entry stops the run at start rather than at the first S0."""
    out = []
    for entry in entries:
        path = Path(entry) if Path(entry).is_absolute() else ROOT / entry
        path = path.resolve()
        if path == own_run_dir:
            raise ValueError(f"explore.graveyard_runs names this run itself: {entry}")
        if not (path / "ledger.jsonl").is_file():
            raise FileNotFoundError(f"explore.graveyard_runs entry has no ledger.jsonl: {entry}")
        graveyard.graves(path)
        out.append(path)
    return tuple(out)


def load_lenses(path: Path) -> tuple[dict, list[str]]:
    """Returns ({name: {"description", "sources"}}, explore order of single lenses then pairings)."""
    from hybrid.literature.seed import LENSES

    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    lenses: dict[str, dict] = {}
    order: list[str] = []
    for entry in data["lenses"]:
        if entry["name"] not in LENSES:
            raise ValueError(f"lens {entry['name']!r} is not a seed lens")
        lenses[entry["name"]] = {"description": entry["description"], "sources": [entry["name"]]}
        if entry["explore"]:
            order.append(entry["name"])
    if set(lenses) != set(LENSES):
        raise ValueError(f"lenses.toml must list every seed lens; missing {sorted(set(LENSES) - set(lenses))}")
    for pair in data.get("pairings", []):
        for lens in pair["lenses"]:
            if lens not in LENSES:
                raise ValueError(f"pairing {pair['name']} names unknown lens {lens}")
        lenses[pair["name"]] = {"description": pair["description"], "sources": list(pair["lenses"])}
        order.append(pair["name"])
    return lenses, order


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Orchestrator:
    def __init__(self, cfg: dict, run_dir: Path, *, llm: Any, executor: Any, bench: Any, image: str,
                 ledger: Ledger | None = None, classify_prior_art: Any = None, resume: bool = False,
                 config_bytes: bytes | None = None, routing: dict | None = None, math: Any = None,
                 models_info: dict | None = None, job_slots: dict | None = None):
        self.cfg = cfg
        # How the host-wide executor job slots were sized (hybrid/slots.py), recorded in run_start.
        self.job_slots = job_slots
        self.order = pipeline_order(cfg)
        self.routing = routing
        self.run_dir = Path(run_dir).resolve()
        self.run_id = self.run_dir.name
        self.resume = resume
        self.ledger = ledger or Ledger(self.run_dir)
        if resume:
            check_resume_mode(self.ledger, cfg["s3"]["mode"])
            check_resume_generation(self.ledger, cfg["generation"]["mode"])
            check_resume_refine(self.ledger, cfg["refine"]["enabled"])
            check_resume_profile_target(self.ledger, cfg)
        self.math_mode = cfg["s3"]["mode"] == "math"
        self.draft_only = cfg["generation"]["draft_only"]
        if self.math_mode and math is None:
            raise ValueError("s3.mode = 'math' needs a math runner (hybrid.math.runner.MathRunner)")
        self.store = Store(self.run_dir, self.ledger)
        self.config_bytes = config_bytes
        b = cfg["budgets"]
        self.budget = Budget(b["max_cards"], b["max_calls"], b["max_tokens"], b["max_wall_s"])
        self.lenses, self.lens_order = load_lenses(ROOT / cfg["run"]["lenses_config"])
        ex = cfg["executor"]
        caps = Caps(wall_s=ex["wall_s"], memory_mb=ex["memory_mb"], pids=ex["pids"], tmp_mb=ex["tmp_mb"])
        runner = CodeRunner(executor, self.run_dir, self.store, image, caps)
        validate_explore(cfg["explore"])
        outcomes = open_outcome_store(cfg)
        self.book = lineage.BudgetBook(cfg, self.ledger, self.store, run_id=self.run_id, outcomes=outcomes)
        self.generation = None
        if cfg["generation"]["mode"] == "arms":
            from hybrid.research.stages.g_generate import build_generation

            info = models_info or {}
            self.generation = build_generation(cfg, ROOT, ledger=self.ledger, store=self.store, run_id=self.run_id,
                                               order=self.order, routed_roles=info.get("routed_roles"),
                                               listing=info.get("lmstudio_listing"),
                                               outcome_store=outcomes or open_store_readonly(cfg))
        self.ctx = StageContext(run_id=self.run_id, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=llm, runner=runner, bench=bench, cfg=cfg, budget=self.budget,
                                lenses=self.lenses, classify_prior_art=classify_prior_art,
                                on_mutation=self._request_mutation, on_reduction=self._on_reduction,
                                graveyard_runs=resolve_graveyard_runs(cfg["explore"]["graveyard_runs"],
                                                                      self.run_dir),
                                math=math, outcomes=outcomes, book=self.book, generation=self.generation)
        self.queues: dict[str, asyncio.Queue] = {}
        self.mutations: asyncio.Queue | None = None
        self.escapes: asyncio.Queue | None = None
        # Escape slots claimed so far: accepted escape requests, plus on resume the escape tasks the ledger
        # already started. Queued escapes are included, so the reserve is claimed when it is requested.
        self.escape_claims = 0
        self.pending = 0
        self.explorers_live = 0
        # Set when something changes that an explorer waiting on the escape reserve may act on.
        self.slot_wakeup: asyncio.Event | None = None
        self.stop_event: asyncio.Event | None = None
        self.stop_reason: str | None = None
        self.failure: BaseException | None = None
        self.lens_seed = self._resolve_lens_seed()
        self.lens_perm = list(range(len(self.lens_order)))
        random.Random(self.lens_seed).shuffle(self.lens_perm)

    def _resolve_lens_seed(self) -> int:
        """run.seed when set; else, on resume, the seed the run recorded at its start (a run from before
        the seed was recorded used the shipped seed 0); else a stable hash of the run id."""
        explicit = self.cfg["run"].get("seed")
        if explicit is not None:
            return explicit
        if not self.resume:
            return derive_lens_seed(self.run_id)
        for event in self.ledger.iter_events("run_start"):
            if "lens_seed" in event["data"]:
                return event["data"]["lens_seed"]
        return LEGACY_LENS_SEED

    # ----- events and control ----------------------------------------------------------------

    def emit(self, type: str, data: dict) -> dict:
        return self.ledger.append(type, ACTOR, data)

    def _stop(self, reason: str) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
            self.emit("run_stop", {"reason": reason, **self.budget.snapshot()})
        self.stop_event.set()
        self._wake()

    def _wake(self) -> None:
        self.slot_wakeup.set()

    def _fail(self, exc: BaseException) -> None:
        if self.failure is None:
            self.failure = exc
            self.emit("orchestrator_error", {"error": f"{type(exc).__name__}: {exc}"})
        self._stop(f"internal error: {type(exc).__name__}")

    def _check_idle(self) -> None:
        if self.explorers_live == 0 and self.pending == 0:
            self._stop("complete")

    def _enqueue(self, stage: str, card_id: str) -> None:
        index = self.order.index(stage)
        if index >= 2:
            card = self.store.get_card(card_id)
            previous = self.order[index - 1]
            got = card["status"]["verdicts"].get(previous, {}).get("verdict")
            if got != "pass":
                raise AssertionError(f"{card_id} queued for {stage} with {previous} verdict {got!r}")
        gates = ("M", "C") if self.math_mode else ("S3",)
        verdicts = self.store.get_card(card_id)["status"]["verdicts"]
        if stage == "S4" and any(verdicts.get(gate, {}).get("verdict") != "pass" for gate in gates):
            raise AssertionError(f"{card_id} queued for S4 without a pass at {' and '.join(gates)}")
        self.pending += 1
        self.queues[stage].put_nowait(card_id)
        self.emit("queued", {"card_id": card_id, "stage": stage})

    def _record_outcome(self, card_id: str) -> None:
        """Write the outcome record of a card whose line just ended (outcomes.enabled). The store skips a record it
        already holds."""
        if self.ctx.outcomes is None:
            return
        from hybrid.research.outcome_records import record_for_live_card

        line = record_for_live_card(self.run_dir, card_id, self.ctx.outcomes)
        if line is not None:
            record = line["record"]
            self.emit("outcome_recorded", {"card_id": card_id, "seq": line["seq"], "hash": line["hash"],
                                           "outcome": record["outcome"], "kill_codes": record["kill_codes"],
                                           "signature": record["mechanism_signature"]["signature"]})

    def _request_mutation(self, card_id: str) -> None:
        stop = self.book.room(card_id)
        if stop is not None:
            stop = lineage.LoopStop(lineage.MUTATION, "budget", stop.detail)
            lineage.record_stop(self.ledger, card_id=card_id, stage="S6", stop=stop, terminal=False)
            self.emit("mutation_skipped", {"card_id": card_id, "reason": f"{stop.kill_code}: {stop.detail}"})
            return
        self.emit("mutation_requested", {"card_id": card_id})
        self.mutations.put_nowait(card_id)

    def _on_reduction(self, card_id: str) -> None:
        """S2 reduced the card to a known family: end the escape loop here when it stalled or its lineage is full
        (recorded once per card as a loop_stop, and as an escape_skipped event), else request an escape."""
        stop = self._escape_loop_stop(card_id)
        if stop is None:
            self._request_escape(card_id)
            return
        if not any(e["data"]["card_id"] == card_id and e["data"]["loop"] == lineage.ESCAPE
                   for e in self.ledger.iter_events("loop_stop")):
            lineage.record_stop(self.ledger, card_id=card_id, stage="S2", stop=stop, terminal=False)
        self.emit("escape_skipped", {"card_id": card_id, "reason": f"{stop.kill_code}: {stop.detail}",
                                     "reserve": self._reserve_state()})

    def _escape_loop_stop(self, card_id: str) -> lineage.LoopStop | None:
        """Why the escape loop ends at this reduced card, or None. Each card of an escape chain is one round of the
        loop and its failure is the family S2 reduced it to; a card made by escape that reduces to its parent's family
        repeats the previous round (a stall). A lineage at its idea cap is the budget."""
        chain = escape_chain(self.ledger, card_id)
        failing = {f"reduces_to_known:{reduction_record(self.ctx, card_id)['reduces_to']}": None}
        if len(chain) > 1:
            previous = {f"reduces_to_known:{reduction_record(self.ctx, chain[1])['reduces_to']}": None}
            if lineage.is_stall(lineage.fingerprint(previous), failing):
                return lineage.LoopStop(lineage.ESCAPE, "stall", f"{card_id} reduces to the same family as its escape "
                                                                 f"parent {chain[1]}: {', '.join(failing)}")
        stop = self.book.room(card_id)
        if stop is not None:
            return lineage.LoopStop(lineage.ESCAPE, "budget", stop.detail)
        return None

    def _reserve_slots(self) -> int:
        # A draft-only run stops every card at S0, so S2 never reduces one and no escape can be requested; holding
        # slots back would only cut the drafts short (the Phase 6 smoke run drafted 1 of its 2 quota slots).
        if self.draft_only:
            return 0
        return escape_reserve_slots(self.budget.max_cards, self.cfg["explore"]["escape_reserve"])

    def _escape_held(self) -> int:
        """Card slots that a new, mutate or cross task must leave free: one for each queued escape, plus the
        reserve that no escape has claimed yet. The claims already include the queued escapes."""
        unclaimed = max(0, self._reserve_slots() - self.escape_claims)
        return self.escapes.qsize() + unclaimed

    def _normal_slot_free(self) -> bool:
        """True when a new, mutate or cross task may take a card slot without leaving fewer free slots than
        the escape reserve still needs."""
        free_after = self.budget.max_cards - self.budget.cards - 1
        return free_after >= self._escape_held()

    def _reserve_state(self) -> dict:
        return {"slots": self._reserve_slots(), "claimed": self.escape_claims, "queued": self.escapes.qsize(),
                "held": self._escape_held(), "free": self.budget.max_cards - self.budget.cards}

    def _request_escape(self, card_id: str) -> None:
        """Queue an escape task for a card S2 reduced to a known family. The p_escape draw is fixed per run
        and card, so a resumed run makes the same choice. An escape may take any free card slot, so a request
        is skipped only when every free slot is already queued for another escape; skips are recorded with
        the reserve state."""
        p = self.cfg["explore"]["p_escape"]
        draw = random.Random(f"escape:{self.run_id}:{card_id}").random()
        if draw >= p:
            self.emit("escape_skipped", {"card_id": card_id, "reason": f"draw {draw:.3f} >= p_escape {p}",
                                         "reserve": self._reserve_state()})
            return
        free = self.budget.max_cards - self.budget.cards
        queued = self.escapes.qsize()
        if free - queued < 1:
            self.emit("escape_skipped", {"card_id": card_id,
                                         "reason": f"no card slot left ({self.budget.cards} of "
                                                   f"{self.budget.max_cards} used, {queued} escapes queued)",
                                         "reserve": self._reserve_state()})
            return
        self.escape_claims += 1
        self.emit("escape_requested", {"card_id": card_id})
        self.escapes.put_nowait(card_id)
        self._wake()

    # ----- explorers -------------------------------------------------------------------------

    def _eligible_parents(self) -> list[str]:
        out = []
        for card_id in self.store.list_cards():
            verdicts = self.store.get_card(card_id)["status"]["verdicts"]
            if verdicts.get("S1", {}).get("verdict") == "pass":
                out.append(card_id)
        return out

    def _next_task(self, explorer: int, k: int) -> ExploreTask:
        ex = self.cfg["explore"]
        seed = lens_seed(self.run_id, k)
        role = "explorer" if k % 2 == 0 else "explorer_alt"
        rng = random.Random(seed)
        lens = self.lens_order[self.lens_perm[k % len(self.lens_perm)]]
        op, parents = "new", []
        if not self.mutations.empty() and self._normal_slot_free():
            op, parents = "mutate", [self.mutations.get_nowait()]
        elif not self.escapes.empty():
            op, parents = "escape", [self.escapes.get_nowait()]
        else:
            # Only reached when a normal slot is free (see _await_task).
            eligible = [card_id for card_id in self._eligible_parents() if self.book.room(card_id) is None]
            draw = rng.random()
            if self.generation is not None:
                # Arms mode: parents are archive elites drawn by cell weight (1 / (1 + visits) plus a bonus per empty
                # neighbouring cell, plan 6.2).
                archive = self.generation.archive
                if draw < ex["p_cross"]:
                    pair = archive.choose_parents(eligible, 2, rng, self.store)
                    if len(pair) == 2:
                        op, parents = "cross", pair
                elif draw < ex["p_cross"] + ex["p_mutate"]:
                    one = archive.choose_parents(eligible, 1, rng, self.store)
                    if one:
                        op, parents = "mutate", one
            elif len(eligible) >= 2 and draw < ex["p_cross"]:
                op, parents = "cross", rng.sample(eligible, 2)
            elif eligible and draw < ex["p_cross"] + ex["p_mutate"]:
                op, parents = "mutate", [rng.choice(eligible)]
        if op in ("mutate", "escape"):
            parent_lens = self.store.get_card(parents[0])["lens"]
            if parent_lens in self.lenses:
                lens = parent_lens
        arm = None
        if self.generation is not None and op == "new":
            arm = self.generation.arm_for(k, random.Random(f"arm:{seed}"))
            role = self.generation.gen.arms[arm].role
        return ExploreTask(op=op, lens=lens, lens_sources=self.lenses[lens]["sources"], role=role,
                           explorer=explorer, seed=seed, parents=parents, arm=arm)

    async def _await_task(self, explorer: int) -> ExploreTask | None:
        """The explorer's next task, or None when it should stop. A new, mutate or cross task waits while
        taking a slot would cut into the escape reserve. The wait ends when an escape is queued or a card
        leaves the pipeline. With nothing in flight no escape can arrive, so the explorer stops."""
        while True:
            if self.stop_event.is_set() or self.budget.cards >= self.budget.max_cards:
                return None
            reason = self.budget.exhausted_reason()
            if reason is not None:
                self._stop(reason)
                return None
            if not self.escapes.empty() or self._normal_slot_free():
                return self._next_task(explorer, self.budget.cards)
            if self.pending == 0:
                return None
            self.slot_wakeup.clear()
            await self.slot_wakeup.wait()

    async def _explorer(self, explorer: int) -> None:
        try:
            while True:
                task = await self._await_task(explorer)
                if task is None:
                    break
                self.budget.cards += 1
                self.pending += 1
                try:
                    card_id = await explore(self.ctx, task)
                    if card_id is not None and not self.draft_only:
                        self._enqueue("S1", card_id)
                    elif task.card_id is not None:
                        self._record_outcome(task.card_id)
                except StageError as exc:
                    self.emit("explore_failed", {"explorer": explorer, "op": task.op, "lens": task.lens,
                                                 "role": task.role, "parents": task.parents, "error": str(exc)})
                except RunBudgetExhausted as exc:
                    self._stop(str(exc))
                finally:
                    self.pending -= 1
                    self._wake()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # internal bug: stop the run and re-raise from run()
            self._fail(exc)
        finally:
            self.explorers_live -= 1
            if not self.stop_event.is_set():
                self._check_idle()

    # ----- stage workers ---------------------------------------------------------------------

    def _route(self, stage: str) -> str:
        """S2 names S3 as its next stage; a math-mode run sends the card to M instead."""
        if stage == "S3" and self.math_mode:
            return "M"
        if stage not in self.order:
            raise AssertionError(f"stage {stage} is not in this run's order {self.order}")
        return stage

    async def _stage_worker(self, stage: str) -> None:
        queue = self.queues[stage]
        runner = STAGE_RUNNERS[stage]
        while True:
            card_id = await queue.get()
            try:
                self.emit("stage_start", {"card_id": card_id, "stage": stage})
                current = CURRENT_CARD.set((card_id, stage))
                try:
                    next_stage = await runner(self.ctx, card_id)
                except lineage.LoopBudgetExhausted as exc:
                    self._end_on_budget(card_id, stage, exc.stop)
                    next_stage = None
                finally:
                    CURRENT_CARD.reset(current)
                if self.generation is not None:
                    verdict = self.store.get_card(card_id)["status"]["verdicts"].get(stage, {}).get("verdict")
                    if verdict == "pass":
                        self.generation.archive.on_pass(self.ledger, card_id, stage)
                if next_stage is not None:
                    self._enqueue(self._route(next_stage), card_id)
                else:
                    self._record_outcome(card_id)
            except StageError as exc:
                self.emit("stage_error", {"card_id": card_id, "stage": stage, "error": str(exc)[:4000]})
            except RunBudgetExhausted as exc:
                self._stop(str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # internal bug
                self._fail(exc)
            finally:
                self.pending -= 1
                self._wake()
                queue.task_done()
                if not self.stop_event.is_set():
                    self._check_idle()

    def _end_on_budget(self, card_id: str, stage: str, stop: lineage.LoopStop) -> None:
        """A model-call cap stopped the card mid-stage: the stop ends the card at that stage. The verdict names the
        model calls charged to the card, which are what used the budget."""
        lineage.record_stop(self.ledger, card_id=card_id, stage=stage, stop=stop, terminal=True)
        spent = [e["data"]["call_id"] for e in self.ledger.iter_events("budget_charge")
                 if e["data"]["card_id"] == card_id and e["data"]["loop"] == lineage.MODEL_CALLS]
        self.ctx.verdict(card_id, stage, "fail", [f"gate: {stop.kill_code}: {stop.detail}"], spent[-200:])

    async def _watchdog(self) -> None:
        await asyncio.sleep(max(0.0, self.budget.max_wall_s - self.budget.elapsed()))
        self._stop(f"max_wall_s {self.budget.max_wall_s} reached")

    # ----- resume ----------------------------------------------------------------------------

    def _restore_budget(self) -> None:
        calls = tokens = 0
        for event in self.ledger.iter_events("llm_call"):
            calls += 1
            usage = event["data"].get("usage") or {}
            tokens += int(usage.get("input") or 0) + int(usage.get("output") or 0)
        failed = sum(1 for _ in self.ledger.iter_events("explore_failed"))
        self.budget.calls, self.budget.tokens = calls, tokens
        self.budget.cards = len(self.store.list_cards()) + failed
        # Every escape task the ledger started (a card made or an explore that failed) already holds a slot.
        escape_starts = [event["data"]["op"] for event in self.ledger.iter_events("card_created")]
        escape_starts += [event["data"]["op"] for event in self.ledger.iter_events("explore_failed")]
        self.escape_claims = escape_starts.count("escape")

    def _requeue(self) -> dict[str, list[str]]:
        requeued: dict[str, list[str]] = {}
        for card_id in self.store.list_cards():
            status = self.store.get_card(card_id)["status"]
            stage = status["stage"]
            verdict = status["verdicts"].get(stage, {}).get("verdict")
            target = None
            if stage == "S0":
                target = "S1" if verdict in (None, "pass") else None
            elif verdict == "pass" and stage != "S6":
                if stage not in self.order:
                    raise ValueError(f"{card_id} passed {stage}, which s3.mode {self.cfg['s3']['mode']!r} does not "
                                     f"run; resume this run with the s3.mode it was started with")
                target = self.order[self.order.index(stage) + 1]
            elif stage == "S1" and verdict == "revise":
                target = "S1"
            if target is not None:
                self._enqueue(target, card_id)
                requeued.setdefault(target, []).append(card_id)
        return requeued

    def _requeue_escapes(self) -> None:
        """On resume, request an escape for every card reduced at S2 that has no escape child yet. This also
        covers runs that reached S2 before the escape operator existed."""
        escaped = {event["data"]["parents"][0] for event in self.ledger.iter_events("card_created")
                   if event["data"]["op"] == "escape"}
        for card_id in self.store.list_cards():
            verdict = self.store.get_card(card_id)["status"]["verdicts"].get("S2", {}).get("verdict")
            if verdict == "reduces_to_known" and card_id not in escaped:
                self._on_reduction(card_id)

    # ----- main ------------------------------------------------------------------------------

    async def run(self) -> dict:
        self.stop_event = asyncio.Event()
        self.slot_wakeup = asyncio.Event()
        self.queues = {stage: asyncio.Queue() for stage in self.order[1:]}
        self.mutations = asyncio.Queue()
        self.escapes = asyncio.Queue()
        if self.config_bytes is not None:
            self.store.put_artifact("config", f"run-{_utc()}.toml", self.config_bytes)
        if self.generation is not None:
            g = self.cfg["generation"]
            for key in ("config", "archive_config", "families_config", "limits_config"):
                path = Path(g[key]) if Path(g[key]).is_absolute() else ROOT / g[key]
                self.store.put_artifact("generation_config", f"{key}-{_utc()}-{uuid.uuid4().hex[:6]}.toml",
                                        path.read_bytes())
        requeued: dict[str, list[str]] = {}
        if self.resume:
            self._restore_budget()
        self.emit("run_start", {"run_id": self.run_id, "resume": self.resume, "budget": self.budget.snapshot(),
                                "n_explorers": self.cfg["run"]["n_explorers"], "workers": self.cfg["workers"],
                                "s5_mode": self.cfg["s5"]["mode"], "routing": self.routing,
                                "s3_mode": self.cfg["s3"]["mode"], "stages": list(self.order),
                                "math_profile": self.cfg["math"]["profile"],
                                "target": {key: self.cfg["target"][key] for key in TARGET_DEFAULTS},
                                "refine": {key: self.cfg["refine"][key] for key in REFINE_DEFAULTS},
                                "s2_phase7": {key: self.cfg["s2"][key] for key in S2_DEFAULTS},
                                "outcomes": {key: self.cfg["outcomes"][key] for key in ("enabled", "path", "enforce_r3")},
                                "generation": {"mode": self.cfg["generation"]["mode"],
                                               "draft_only": self.draft_only,
                                               **(self.generation.record() if self.generation else {})},
                                "lens_seed": self.lens_seed, "loop_budgets": self.book.caps_record(),
                                **({"job_slots": self.job_slots} if self.job_slots is not None else {}),
                                "escape_reserve": {"fraction": self.cfg["explore"]["escape_reserve"],
                                                   "slots": self._reserve_slots(),
                                                   "max_cards": self.budget.max_cards,
                                                   "claimed": self.escape_claims}})
        if self.resume:
            missing = self.book.assign_missing()
            if missing:
                self.emit("lineages_assigned_at_resume", {"cards": missing})
            if not self.draft_only:
                requeued = self._requeue()
                self.emit("resume_requeued", {"requeued": requeued})
                self._requeue_escapes()

        n = self.cfg["run"]["n_explorers"]
        self.explorers_live = n
        tasks = [asyncio.create_task(self._explorer(i), name=f"explorer-{i}") for i in range(n)]
        for stage in ([] if self.draft_only else self.order[1:]):
            tasks += [asyncio.create_task(self._stage_worker(stage), name=f"{stage}-{j}")
                      for j in range(stage_workers(self.cfg, stage))]
        tasks.append(asyncio.create_task(self._watchdog(), name="watchdog"))
        await self.stop_event.wait()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        left = {stage: list(queue._queue) for stage, queue in self.queues.items() if queue.qsize()}
        if self.escapes.qsize():
            left["escape"] = list(self.escapes._queue)
        summary = {"run_id": self.run_id, "reason": self.stop_reason, "budget": self.budget.snapshot(),
                   "queued_unfinished": left, "requeued_at_start": requeued, "cards": self.store.list_cards()}
        # FIX-17: each card's skill scale at target, from its latest stage M decision.
        coupled_by_card = {e["data"]["card_id"]: e["data"]["target_coupled"]
                           for e in self.ledger.iter_events("math_decision")
                           if e["data"].get("stage") == "M" and e["data"].get("target_coupled")}
        if coupled_by_card:
            summary["target_coupled"] = coupled_by_card
        if self.generation is not None:
            from hybrid.research.stages.g_generate import arm_metrics

            metrics = arm_metrics(self.ledger, self.store, self.order)
            self.emit("arm_metrics", {"arms": metrics, "coverage": self.generation.archive.coverage()})
            snapshot = self.generation.archive.snapshot(self.store)
            self.ctx.put_json_artifact("archive_snapshot", f"archive-{_utc()}-{uuid.uuid4().hex[:6]}.json", snapshot,
                                     card_id=None)
            summary["arm_metrics"] = metrics
        self.emit("run_end", summary)
        if self.failure is not None:
            raise self.failure
        return summary


def build_orchestrator(config_path: Path, run_dir: Path, *, resume: bool, overrides: dict | None = None,
                       routing: str | None = None) -> Orchestrator:
    """Wire the real dependencies: LLMClient, Podman Executor, the trusted bench.

    `routing` (the --routing flag) selects the models.toml routing table, over BIGBRANE_ROUTING and the
    file's own `routing`. When the selected table uses a gateway backend, the gateway is checked before
    anything is written: reachable, token accepted, every routed model id served. GatewayUnavailable
    carries the serve and login commands.
    """
    from hybrid.executor.executor import Executor
    from hybrid.executor.lock import read_lock
    from hybrid.literature.keys import load_api_keys
    from hybrid.llm import LLMClient, check_gateways, load_config
    from hybrid.research.stages.s5_evaluate import BenchEvaluator
    from hybrid.slots import SLOTS_DIR, job_slots_from_config

    load_api_keys()  # a literature key file that others can read stops the run before anything is created
    cfg = load_run_config(config_path, overrides)
    models = load_config(ROOT / cfg["run"]["models_config"], routing)
    gateways = check_gateways(models)
    run_dir = Path(run_dir).resolve()
    if resume:
        if not (run_dir / "ledger.jsonl").is_file():
            raise FileNotFoundError(f"--resume needs an existing run with ledger.jsonl: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty; use --resume to continue it")
    slots, slot_record = job_slots_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(run_dir)
    llm = LLMClient(str(ROOT / cfg["run"]["models_config"]), str(run_dir), ledger, routing=routing,
                    lanes_dir=SLOTS_DIR)
    executor = Executor(run_dir.name, run_dir / "jobs" / "receipts", slots=slots)
    bench = BenchEvaluator(cfg["s5"]["mode"], ROOT / cfg["s5"]["bench_config"], run_dir.name, run_dir)
    routing_record = {"table": models.routing, "source": models.routing_source, "gateways": gateways}
    math = None
    if cfg["s3"]["mode"] == "math":
        from hybrid.math.runner import MathRunner

        math = MathRunner(executor, read_lock("tools")["reference"], run_dir)
    models_info = None
    if cfg["generation"]["mode"] == "arms":
        models_info = {"routed_roles": set(models.roles), "lmstudio_listing": lmstudio_listing(models)}
    return Orchestrator(cfg, run_dir, llm=llm, executor=executor, bench=bench, image=read_lock()["reference"],
                        ledger=ledger, resume=resume, config_bytes=Path(config_path).read_bytes(),
                        routing=routing_record, math=math, models_info=models_info, job_slots=slot_record)


def lmstudio_listing(models) -> list[dict] | None:
    """LM Studio's model list (GET /api/v0/models, read only) for the base-model arm's availability (decision D4);
    None when no LM Studio backend is configured or it does not answer."""
    from hybrid import localmodels

    for backend in models.backends.values():
        if backend.kind == localmodels.LOCAL_KIND:
            try:
                return localmodels.fetch_models(localmodels.server_root(backend.base_url))
            except localmodels.LocalModelUnavailable:
                return None
    return None
