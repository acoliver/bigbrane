"""Local-model preflight: refuse to start a research run or plumbing check whose routing sends a role to a local LM
Studio model that is not loaded, or is loaded with less context than the config needs.

Why. The Phase 3 live card failed in stage C because LM Studio had reloaded the local Qwen at 8192 tokens, and
LM Studio had quit entirely during the second run (results/phase3-20261012T183415Z/summary.md). Both show up only
when a stage first calls the model, hours into a run.

What is checked. Every role the command will call (the roles of the stages it runs, STAGE_ROLES), through every entry
of that role in the active routing table (fallbacks included, since a fallback is used when the first entry fails),
on a backend of kind `openai_compat` (the local LM Studio server). For each such model, `GET <server>/api/v0/models`
must list it with `state: "loaded"` and `loaded_context_length` at least the backend's `min_loaded_context`
(models.toml; DEFAULT_MIN_CONTEXT when the key is absent). Entries with `model = "auto"` are skipped: no research
stage routes to them. Nothing is loaded or unloaded here; the check only reads.

A refusal names each model with the measured state and context and the required context, and the command exits 2
before anything is created, like the disk precheck (hybrid/diskguard.py) it sits next to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

DEFAULT_MIN_CONTEXT = 131072
LOCAL_KIND = "openai_compat"
AUTO = "auto"
TIMEOUT_S = 10.0

# Roles each research stage calls (hybrid/research/stages). S1 revisions go back to the card's explorer role.
STAGE_ROLES = {
    "S0": ("explorer", "explorer_alt"),
    "S1": ("salad_critic", "explorer", "explorer_alt"),
    # Phase 7: the dialogue's challenger roles and the proposer (the card's explorer role); with s2.difference_statement
    # the proposer writes the difference statement and a challenger role checks it.
    "R": ("refine_challenger", "refine_challenger_alt", "explorer", "explorer_alt"),
    "S2": ("prior_art", "reduction_critic", "explorer", "explorer_alt", "refine_challenger", "refine_challenger_alt"),
    "S3": ("formalizer", "disprover", "obligation_checker", "feasibility_judge_a", "feasibility_judge_b"),
    "M": ("statement_fixer", "prover"),
    "C": ("challenger", "statement_match", "statement_crosscheck", "statement_tiebreak"),
    "S4": ("builder",),
    "S5": ("falsifier_designer",),
    "S6": ("analyst",),
}


# Roles S0 calls only in generation mode "arms" (Phase 6, hybrid/research/stages/g_generate.py).
ARMS_ROLES = ("arm_classifier",)


class LocalModelUnavailable(RuntimeError):
    """LM Studio does not answer, or a routed model is not loaded with the context the config needs."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Requirement:
    backend: str
    server: str            # LM Studio root URL (base_url without the /v1 suffix)
    model: str
    min_context: int
    roles: tuple[str, ...] = field(default=())


def server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def roles_for(stages: Iterable[str], *, arms: bool = False) -> list[str]:
    """Roles the given stages call; with arms=True (generation mode "arms") S0 also calls ARMS_ROLES."""
    roles: list[str] = []
    for stage in stages:
        for role in STAGE_ROLES[stage] + (ARMS_ROLES if arms and stage == "S0" else ()):
            if role not in roles:
                roles.append(role)
    return roles


def requirements(models, roles: Iterable[str]) -> list[Requirement]:
    """One requirement per (local backend, model) the given roles route to in the active table."""
    found: dict[tuple[str, str], list[str]] = {}
    for role in roles:
        for entry in models.roles.get(role, ()):
            backend = models.backends[entry.backend]
            if backend.kind == LOCAL_KIND and entry.model != AUTO:
                found.setdefault((entry.backend, entry.model), []).append(role)
    out = []
    for (name, model), used_by in found.items():
        backend = models.backends[name]
        out.append(Requirement(name, server_root(backend.base_url), model,
                               backend.min_loaded_context or DEFAULT_MIN_CONTEXT, tuple(used_by)))
    return out


def fetch_models(server: str, *, get: Callable | None = None, timeout: float = TIMEOUT_S) -> list[dict]:
    """LM Studio's model list (`GET <server>/api/v0/models`). Raises LocalModelUnavailable when it does not answer
    with a JSON model list."""
    import httpx

    get = get or httpx.get
    url = f"{server}/api/v0/models"
    try:
        response = get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()["data"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise LocalModelUnavailable([f"LM Studio does not answer at {url}: {type(exc).__name__}: {exc}"]) from exc
    if not isinstance(data, list):
        raise LocalModelUnavailable([f"LM Studio at {url} returned no model list"])
    return data


def check(reqs: list[Requirement], *, fetch: Callable[[str], list[dict]] | None = None) -> list[str]:
    """Measure every requirement. Returns one line per model that passed; raises LocalModelUnavailable listing
    every model that did not (state and loaded context measured, context required). `fetch` defaults to
    fetch_models, looked up at call time so tests can replace it."""
    fetch = fetch or fetch_models
    ok, problems = [], []
    listings: dict[str, list[dict]] = {}
    for req in reqs:
        if req.server not in listings:
            listings[req.server] = fetch(req.server)
        entry = next((item for item in listings[req.server] if item.get("id") == req.model), None)
        roles = ", ".join(req.roles)
        if entry is None:
            problems.append(f"{req.model} (roles {roles}) is not known to LM Studio at {req.server}; required: loaded "
                            f"with at least {req.min_context} tokens of context")
            continue
        state = entry.get("state")
        loaded = entry.get("loaded_context_length")
        if state != "loaded" or not isinstance(loaded, int):
            problems.append(f"{req.model} (roles {roles}) is {state or 'in an unknown state'} at {req.server} "
                            f"(measured loaded context: none); required: loaded with at least {req.min_context} "
                            f"tokens of context")
        elif loaded < req.min_context:
            problems.append(f"{req.model} (roles {roles}) is loaded with {loaded} tokens of context at {req.server}; "
                            f"required: at least {req.min_context}")
        else:
            ok.append(f"{req.model}: loaded, context {loaded} (required {req.min_context})")
    if problems:
        raise LocalModelUnavailable(problems)
    return ok


def guard_cli(models, stages: Iterable[str], what: str, *, fetch: Callable[[str], list[dict]] | None = None,
              report: Callable[[str], None] | None = None, arms: bool = False) -> bool:
    """Run the preflight for a command-line entry point. On a refusal prints every problem and the fix to stderr and
    returns False (the caller exits with status 2)."""
    import sys

    reqs = requirements(models, roles_for(stages, arms=arms))
    try:
        lines = check(reqs, fetch=fetch)
    except LocalModelUnavailable as exc:
        print(f"refusing to start {what}: the local model check failed (routing table {models.routing!r}):",
              file=sys.stderr)
        for problem in exc.problems:
            print(f"  {problem}", file=sys.stderr)
        print("Load the model in LM Studio with the required context length (this check never loads or unloads "
              "models), or route the roles elsewhere with --routing.", file=sys.stderr)
        return False
    for line in lines if report else ():
        report(f"local model: {line}")
    return True


# ----- per-call guard (FIX-22, decision ENV) -----------------------------------------------------------------------
# The preflight above runs once, at run start. During calibration 20261016T141847Z LM Studio was restarted by a
# process outside this project hours into the run; the next request auto-loaded the model at the 8192-token default
# and a stage failed with HTTP 400. hybrid/llm/backends/lmstudio_guard.py therefore checks before every request, using
# the helpers below. The guard reads instances from GET /api/v1/models, whose `loaded_instances` lists every loaded
# instance of a model key (ids such as `key` and `key:2`, each with its own context length); GET /api/v0/models has
# one row per model key, so it cannot tell two instances of one key apart.


@dataclass(frozen=True)
class Instance:
    id: str                       # instance identifier; the `model` value that reaches this instance
    context: int | None           # loaded context length, None when LM Studio does not report one


def loaded_instances(listing: dict, model: str) -> list[Instance]:
    """The loaded instances of model key ``model`` in an answer of ``GET <server>/api/v1/models``. Raises
    LocalModelUnavailable when the answer has no model list."""
    models = listing.get("models") if isinstance(listing, dict) else None
    if not isinstance(models, list):
        raise LocalModelUnavailable([f"LM Studio's /api/v1/models answer has no model list: {str(listing)[:200]}"])
    found = []
    for item in models:
        if not isinstance(item, dict) or item.get("key") != model:
            continue
        for instance in item.get("loaded_instances") or ():
            context = (instance.get("config") or {}).get("context_length")
            found.append(Instance(str(instance["id"]), context if isinstance(context, int) else None))
    return found


def adequate_instance(instances: list[Instance], model: str, min_context: int) -> Instance | None:
    """An instance loaded with at least ``min_context`` tokens; the one named by the bare key first."""
    fits = [i for i in instances if i.context is not None and i.context >= min_context]
    fits.sort(key=lambda i: i.id != model)
    return fits[0] if fits else None


def undersized_instances(instances: list[Instance], min_context: int) -> list[Instance]:
    """The instances loaded with less than ``min_context`` tokens (or no reported context)."""
    return [i for i in instances if i.context is None or i.context < min_context]
