"""Generation stage G: measured arms over a quality-diversity archive (DISCOVERY-PIPELINE-PLAN.md section 6, Phase 6).

Switched on by `[generation] mode = "arms"` in run.toml (default "lenses": S0 as before). In arms mode every explore
task runs here instead of `s0_explore.explore`; the card still records its S0 verdict, so the rest of the pipeline,
resume and inspect see an S0 stage as before.

A `new` task carries an arm (hybrid/config/generation.toml): A1 cross-field mapping, A2 break an assumption of a known
limit (hybrid/config/limits.toml), A3 invert a transformer commitment, A4 distant random concepts, A5 base model
plus formatter. The arm's prompt file goes before the shared explorer rules; its required output is the draft field
`arm_structure`, whose schema depends on the arm and which `check_structure` checks before anything else: the image
equation of an analogy arm must be one of the card's mechanism equations, the A2 quote must be the listed assumption
of the named limit with its source, A3 must invert the commitment it was given, A4 must use the concepts it was
given, A5's formatter may not add operators the base draft does not contain. An object_map entry may name a state
object, a symbol of the update or readout equations, or an expression built only from them (`sum_i w_i`). Every
problem names its field (`arm_structure.image_equation.mechanism_ref: ...`). A draft that fails the check gets one
structure retry: the same role sees its draft and the failing fields and returns a full corrected draft, which is held
to the same check; a second failure kills the card with the arm code (recorded as `structure_retry` on arm_draft).
Mutate, cross and escape tasks keep the explorer's operator prompts and carry no arm.

Papers (A1, A5, operator tasks) come from `s0_explore.draw_papers`: when the task's lenses give fewer than
explore.papers_min, the draw is topped up from the neighbouring lenses and the seed list and a `paper_fallback` event
records it, instead of losing the slot.

Every card that passes its arm check is classified by the `arm_classifier` role (a family other than the proposer's;
local Qwen in every routing table) onto the five archive axes, with the closest known family. A deterministic
consistency check overrides an obvious contradiction (a card trained with backpropagation is not in a `none_fixed` or
`local_hebbian` learning-rule cell). Gates, in order, each with its kill code: the arm check (`arm_structure:<arm>`),
a closed archive cell (`not_radical:R1`), a known
family without a real difference (`no_difference`), a mechanism signature already in the archive or the outcome store
(`duplicate:<card>`), then the TF-IDF duplicate check and the lineage caps of S0 (Phase 8 BudgetBook). A classifier
placement in an out-of-scope family (families.toml) is not a gate (KD5): it rests on argument alone, so it is recorded
as an `S0 family flag` reason and S2's counted comparison decides `excluded_family:<id>`. Every model
call of the task (draft, formatter, classifier) is charged to the card's idea and lineage.

Per-arm metrics. Each task records one `arm_draft` event: arm (or `op:<operator>`), outcome (admitted,
schema_failure, card_invalid, transport_failure, classifier_failure, arm_structure, excluded_family, closed_cell,
no_difference, duplicate, budget), the cell and whether it filled an empty one, and the call ids. `arm_metrics`
aggregates them with the survivors per stage and the tokens per arm; the orchestrator writes the aggregate as an
`arm_metrics` event at the end of every invocation, and `inspect` prints it.
"""
from __future__ import annotations

import json
import random
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from hybrid.research.archive import (Archive, ArchiveConfig, Families, cell_key, closed_by, load_archive_config,
                                     load_families)
from hybrid.research.schemas import SchemaViolation, validate
from hybrid.research.signature import AXES, detect, mask
from hybrid.research.stages.common import (StageContext, StageError, draft_schema, load_prompt, card_for_prompt)
from hybrid.research.stages.s0_explore import (GRAVEYARD_HEADER, ExploreTask, _paper_block, admit, build_user_prompt,
                                               card_from_draft, draw_papers, existing_summaries, graveyard_block,
                                               nearby_dead, system_prompt)

ARM_IDS = ("A1", "A2", "A3", "A4", "A5")
BASE_ARM = "A5"
ANALOGY_ARMS = ("A1", "A4")
CLASSIFIER_ROLE = "arm_classifier"
EQUATION_REF = re.compile(r"^(update|readout)\[(\d+)\]$")
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
# Function and LaTeX words an object_map expression may use besides the card's symbols (`sum_i w_i`).
OPERATOR_WORDS = frozenset({
    "sum", "prod", "max", "min", "exp", "log", "ln", "abs", "sign", "clip", "mean", "sqrt", "tanh", "sigmoid",
    "softmax", "argmax", "argmin", "conj", "norm", "det", "tr", "diag", "frac", "left", "right", "cdot", "times",
    "operatorname", "mathbf", "mathrm", "mathcal", "text", "bar", "hat", "tilde", "vec", "over", "for", "in"})
# Outcomes of one generation task (event arm_draft). The first group fails before or at the arm check and counts as a
# schema failure of the arm (plan Phase 6 exit criterion: at most 10% per arm).
SCHEMA_OUTCOMES = ("schema_failure", "card_invalid", "arm_structure")
OUTCOMES = ("admitted", *SCHEMA_OUTCOMES, "transport_failure", "classifier_failure", "excluded_family", "closed_cell",
            "no_difference", "duplicate", "budget")
# Backpropagation in the toy code or the learning rule text (the consistency check's one rule, plan 6.2).
GRAD_CODE = re.compile(r"\.backward\(|autograd|torch\.optim|optim\.(?:SGD|Adam)|jax\.grad|GradientTape")
GRAD_TEXT = re.compile(r"\b(backprop(?:agation|agated)?|trained end[- ]to[- ]end|autograd)\b", re.IGNORECASE)
GRAD_NEGATED = re.compile(r"\b(no|without|not|never|instead of|unlike|rather than)\s+(?:\w+\s+){0,2}"
                          r"(backprop|end[- ]to[- ]end|autograd)", re.IGNORECASE)
GRADIENT_FREE = ("none_fixed", "local_hebbian")
ARCHIVE_CELLS_SHOWN = 15


# ----- config ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    id: str
    name: str
    title: str
    prompt: str
    role: str
    weight: float
    papers: bool
    formatter_role: str | None = None
    base_model: str = ""
    fallback: tuple[str, ...] = ()


@dataclass(frozen=True)
class Limit:
    id: str
    name: str
    source: str
    statement: str
    assumptions: dict[str, str]


@dataclass(frozen=True)
class GenerationConfig:
    arms: dict[str, Arm]
    commitments: dict[str, str]
    concepts: tuple[str, ...]
    per_task: int


def load_generation_config(path: Path) -> GenerationConfig:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    arms = {}
    for arm_id, entry in data["arms"].items():
        if arm_id not in ARM_IDS:
            raise ValueError(f"{path}: unknown arm {arm_id!r}; arms are {ARM_IDS}")
        weight = entry["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"{path}: arms.{arm_id}.weight must be a number >= 0, got {weight!r}")
        arms[arm_id] = Arm(arm_id, entry["name"], entry["title"], entry["prompt"], entry["role"], float(weight),
                           bool(entry["papers"]), entry.get("formatter_role"), entry.get("base_model", ""),
                           tuple(entry.get("fallback", ())))
    if BASE_ARM in arms:
        base = arms[BASE_ARM]
        if not base.formatter_role or not base.fallback or any(f not in arms or f == BASE_ARM for f in base.fallback):
            raise ValueError(f"{path}: arm A5 needs formatter_role and a fallback naming other arms")
    commitments = {entry["id"]: entry["text"] for entry in data["commitments"]}
    concepts = tuple(data["random_concepts"]["concepts"])
    per_task = data["random_concepts"]["per_task"]
    if len(set(concepts)) != len(concepts) or not 1 <= per_task <= len(concepts):
        raise ValueError(f"{path}: random_concepts needs distinct concepts and 1 <= per_task <= their number")
    return GenerationConfig(arms, commitments, concepts, per_task)


def load_limits(path: Path) -> dict[str, Limit]:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    out = {}
    for entry in data["limits"]:
        assumptions = {a["id"]: a["text"] for a in entry["assumptions"]}
        if entry["id"] in out or not assumptions:
            raise ValueError(f"{path}: limit {entry['id']} is repeated or lists no assumptions")
        out[entry["id"]] = Limit(entry["id"], entry["name"], entry["source"], entry["statement"], assumptions)
    return out


def base_arm_status(arm: Arm | None, *, routed_roles: set[str] | None, listing: list[dict] | None) -> tuple[bool, str]:
    """Whether arm A5 can run (decision D4): a base model is configured, both its roles are routed, and LM Studio lists
    the model. The listing is read only; nothing is loaded."""
    if arm is None:
        return False, "arm A5 is not configured"
    if not arm.base_model:
        return False, "no base model configured (generation.toml arms.A5.base_model is empty)"
    missing = [role for role in (arm.role, arm.formatter_role) if routed_roles is None or role not in routed_roles]
    if missing:
        return False, f"roles not routed in models.toml: {', '.join(missing)}"
    if listing is None:
        return False, "LM Studio model listing not available"
    if not any(model.get("id") == arm.base_model for model in listing):
        return False, f"base model {arm.base_model} is not listed by LM Studio"
    return True, f"base model {arm.base_model} listed by LM Studio"


def effective_weights(gen: GenerationConfig, base_available: bool) -> dict[str, float]:
    """Arm weights in force: A5's weight moves in equal parts to its fallback arms when it cannot run."""
    weights = {arm_id: arm.weight for arm_id, arm in gen.arms.items()}
    if BASE_ARM in weights and not base_available:
        share = weights.pop(BASE_ARM) / len(gen.arms[BASE_ARM].fallback)
        for target in gen.arms[BASE_ARM].fallback:
            weights[target] += share
    weights = {arm_id: w for arm_id, w in weights.items() if w > 0}
    if not weights:
        raise ValueError("generation arms: every arm weight is 0")
    return weights


def choose_arm(weights: dict[str, float], rng: random.Random) -> str:
    ids = sorted(weights)
    return rng.choices(ids, weights=[weights[i] for i in ids], k=1)[0]


def quota_schedule(quota: dict[str, int]) -> list[str]:
    """Arms for a fixed number of drafts per arm, interleaved: one draft of each arm with quota left, in arm order,
    repeated. Interleaving keeps the paced GPT arm running beside the others instead of last."""
    left = {arm_id: n for arm_id, n in sorted(quota.items())}
    out = []
    while any(left.values()):
        for arm_id in sorted(left):
            if left[arm_id] > 0:
                out.append(arm_id)
                left[arm_id] -= 1
    return out


@dataclass
class Generation:
    """Everything stage G needs, built once per invocation (Orchestrator)."""
    gen: GenerationConfig
    limits: dict[str, Limit]
    archive_cfg: ArchiveConfig
    families: Families
    archive: Archive
    weights: dict[str, float]
    base_status: tuple[bool, str]
    schedule: list[str] = field(default_factory=list)

    def arm_for(self, k: int, rng: random.Random) -> str:
        """The arm of the k-th explore task: the quota schedule while it lasts, else a weighted draw."""
        if k < len(self.schedule):
            return self.schedule[k]
        return choose_arm(self.weights, rng)

    def record(self) -> dict:
        return {"weights": self.weights, "base_arm": {"available": self.base_status[0], "reason": self.base_status[1]},
                "schedule": list(self.schedule), "arms": {a.id: {"name": a.name, "role": a.role, "prompt": a.prompt}
                                                          for a in self.gen.arms.values()}}


def build_generation(cfg: dict, root: Path, *, ledger, store, run_id: str, order: tuple[str, ...],
                     routed_roles: set[str] | None = None, listing: list[dict] | None = None,
                     outcome_store=None) -> Generation:
    g = cfg["generation"]
    paths = {key: (Path(g[key]) if Path(g[key]).is_absolute() else root / g[key])
             for key in ("config", "archive_config", "families_config", "limits_config")}
    gen = load_generation_config(paths["config"])
    archive_cfg = load_archive_config(paths["archive_config"])
    families = load_families(paths["families_config"], archive_cfg)
    limits = load_limits(paths["limits_config"])
    base_status = base_arm_status(gen.arms.get(BASE_ARM), routed_roles=routed_roles, listing=listing)
    weights = effective_weights(gen, base_status[0])
    quota = dict(g["quota"])
    unknown = sorted(set(quota) - set(weights))
    if unknown:
        raise ValueError(f"generation.quota names arms that cannot run in this run: {unknown} (weights {weights})")
    archive = Archive(archive_cfg, families, order)
    if outcome_store is not None:
        archive.seed_from_store(outcome_store.current(), run_id=run_id)
    archive.replay(ledger, store)
    return Generation(gen, limits, archive_cfg, families, archive, weights, base_status, quota_schedule(quota))


# ----- arm output schemas ------------------------------------------------------------------------

_TEXT = {"type": "string", "minLength": 1}
_EQUATION = {"type": "object", "additionalProperties": False, "required": ["latex", "plain"],
             "properties": {"latex": _TEXT, "plain": _TEXT}}
_REF = {"type": "string", "pattern": r"^(update|readout)\[[0-9]+\]$"}
_REF_OR_LEARNING = {"type": "string", "pattern": r"^((update|readout)\[[0-9]+\]|learning)$"}
_PREDICTION = {"type": "object", "additionalProperties": False,
               "required": ["statement", "derived_via", "task", "metric", "value"],
               "properties": {"statement": _TEXT, "derived_via": _TEXT, "task": _TEXT, "metric": _TEXT,
                              "value": {"type": "number"}}}
_MAP = {"type": "array", "minItems": 2,
        "items": {"type": "object", "additionalProperties": False, "required": ["source_object", "mechanism_object"],
                  "properties": {"source_object": _TEXT, "mechanism_object": _TEXT}}}


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def arm_structure_schema(arm_id: str, families: Families, limits: dict[str, Limit],
                         commitments: dict[str, str]) -> dict:
    closest = _object({"family": {"enum": ["none", *families.ids]}, "difference": _TEXT})
    analogy = {"source_field": _TEXT, "object_map": _MAP, "source_equation": _EQUATION,
               "image_equation": _object({"latex": _TEXT, "plain": _TEXT, "mechanism_ref": _REF}),
               "prediction": _PREDICTION}
    if arm_id == "A1":
        props = analogy
    elif arm_id == "A2":
        props = {"limit_id": {"enum": sorted(limits)}, "assumption_id": _TEXT, "assumption_quote": _TEXT,
                 "source": _TEXT, "how_broken": _TEXT, "mechanism_ref": _REF_OR_LEARNING,
                 "cost": _object({"time_per_token": _TEXT, "memory": _TEXT, "statement": _TEXT})}
    elif arm_id == "A3":
        props = {"commitment_id": {"enum": sorted(commitments)}, "inverse": _TEXT, "what_replaces_it": _TEXT,
                 "mechanism_ref": _REF_OR_LEARNING, "ablation": _TEXT}
    elif arm_id == "A4":
        props = {"concepts": {"type": "array", "minItems": 1, "items": _TEXT}, "shared_relation": _TEXT, **analogy}
    elif arm_id == BASE_ARM:
        props = {"note_excerpt": _TEXT, "added_by_formatter": {"type": "array", "items": _TEXT}}
    else:
        raise ValueError(f"unknown arm {arm_id}")
    return _object({**props, "closest_known": closest})


def arm_draft_schema(gen: Generation, arm: Arm) -> dict:
    schema = draft_schema(transplant=arm.papers)
    schema["properties"]["arm_structure"] = arm_structure_schema(arm.id, gen.families, gen.limits,
                                                                 gen.gen.commitments)
    schema["required"].append("arm_structure")
    return schema


# ----- arm checks --------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(".;").lower()


def resolve_ref(card: dict, ref: str) -> dict | None:
    """The equation an `update[i]` or `readout[i]` reference points to, or None."""
    match = EQUATION_REF.match(ref)
    if match is None:
        return None
    equations = card["mechanism"][match.group(1)]
    index = int(match.group(2))
    return equations[index] if index < len(equations) else None


def _mechanism_names(card: dict) -> tuple[frozenset[str], str]:
    names = frozenset(item["name"] for item in card["mechanism"]["state"])
    text = "\n".join(f"{eq['plain']}\n{eq['latex']}"
                     for key in ("update", "readout") for eq in card["mechanism"][key])
    return names, text


def _symbols(text: str) -> frozenset[str]:
    """Identifiers of equation text, each also without its subscript: `C_{ij}`, `C[i,j]` and `C_ij` all give C, and
    `x_t` gives x_t and x."""
    out: set[str] = set()
    for token in IDENTIFIER.findall(text):
        token = token.rstrip("_")
        out.update((token, token.split("_", 1)[0]))
    return frozenset(out)


def _mechanism_object_known(obj: str, names: frozenset[str], equations: str, symbols: frozenset[str]) -> bool:
    """Whether an object_map entry lands on the mechanism: a state object, text of an update or readout equation, or
    an expression whose every identifier (subscripts aside, operator words such as sum or max skipped) is a state
    object or a symbol of those equations, with at least one such identifier. `sum_i w_i` and `C_ij*(T_j-T_i)` are
    expressions in the card's symbols (Phase 6 A4 drafts card-19c22c05, card-0427c534); `road density` is not."""
    obj = obj.strip()
    if obj in names or (obj and obj in equations):
        return True
    known = False
    for token in IDENTIFIER.findall(obj):
        token = token.rstrip("_")
        base = token.split("_", 1)[0]
        if token in names or token in symbols or base in names or base in symbols:
            known = True
        elif base.lower() not in OPERATOR_WORDS:
            return False
    return known


def _same_equation(image: dict, target: dict, names: frozenset[str]) -> bool:
    if _norm(image["plain"]) == _norm(target["plain"]) or _norm(image["latex"]) == _norm(target["latex"]):
        return True
    return _norm(mask(image["plain"], names)) == _norm(mask(target["plain"], names))


def _equation_refs(card: dict) -> list[tuple[str, dict]]:
    return [(f"{key}[{i}]", eq) for key in ("update", "readout") for i, eq in enumerate(card["mechanism"][key])]


def _ref_range(card: dict) -> str:
    counts = {key: len(card["mechanism"][key]) for key in ("update", "readout")}
    return ", ".join(f"{key}[0]..{key}[{n - 1}]" if n else f"no {key} equation" for key, n in counts.items())


def check_analogy(card: dict, structure: dict) -> list[str]:
    """Problems with an analogy arm's correspondence, each naming its field: the image equation must be the mechanism
    equation its mechanism_ref names, and every object_map entry must land on the mechanism."""
    problems = []
    names, equations = _mechanism_names(card)
    symbols = _symbols(equations)
    image = structure["image_equation"]
    ref = image["mechanism_ref"]
    target = resolve_ref(card, ref)
    matches = [other for other, eq in _equation_refs(card) if _same_equation(image, eq, names)]
    hint = (f"; the image equation's plain text is {' and '.join(matches)} of the mechanism" if matches else
            "; the image equation's plain text is no equation of the mechanism: copy the plain text of the mechanism "
            "equation the source equation maps to")
    if target is None:
        problems.append(f"arm_structure.image_equation.mechanism_ref: {ref} names no equation of the mechanism "
                        f"(zero-based; this card has {_ref_range(card)}){hint}")
    elif not _same_equation(image, target, names):
        problems.append(f"arm_structure.image_equation.mechanism_ref: image equation '{image['plain'][:200]}' is not "
                        f"{ref} of the mechanism ('{target['plain'][:200]}'){hint}")
    for i, pair in enumerate(structure["object_map"]):
        if not _mechanism_object_known(pair["mechanism_object"], names, equations, symbols):
            problems.append(f"arm_structure.object_map[{i}].mechanism_object: object_map names "
                            f"'{pair['mechanism_object']}', which is neither a state object nor a symbol of the "
                            f"update or readout equations nor an expression in them (state objects: "
                            f"{', '.join(sorted(names))})")
    return problems


def _check_ref_or_learning(card: dict, ref: str) -> list[str]:
    if ref == "learning" or resolve_ref(card, ref) is not None:
        return []
    return [f"arm_structure.mechanism_ref: mechanism_ref {ref} names no equation of the mechanism (zero-based; this "
            f"card has {_ref_range(card)}, or `learning`)"]


def formatter_added_operators(note: str, card: dict) -> list[str]:
    """Operators of the card's update and readout equations that the base draft never uses (arm A5)."""
    names, _ = _mechanism_names(card)
    in_note = detect(mask(note, names), "write")
    in_card: set[str] = set()
    for key in ("update", "readout"):
        for eq in card["mechanism"][key]:
            in_card |= detect(mask(f"{eq['plain']}\n{eq['latex']}", names), "write")
    return sorted(in_card - in_note)


def check_structure(arm_id: str, card: dict, *, gen: Generation, seeds: dict, base_text: str | None = None) -> list[str]:
    """Problems with the card's arm_structure; empty when the arm's required output holds (plan 6.1)."""
    structure = card["arm_structure"]
    if arm_id in ANALOGY_ARMS:
        problems = check_analogy(card, structure)
        if arm_id == "A4":
            given = sorted(_norm(c) for c in seeds["concepts"])
            if sorted(_norm(c) for c in structure["concepts"]) != given:
                problems.append(f"arm_structure.concepts: concepts {structure['concepts']} are not the concepts "
                                f"given: {seeds['concepts']}")
        return problems
    if arm_id == "A2":
        limit = gen.limits[structure["limit_id"]]
        problems = []
        text = limit.assumptions.get(structure["assumption_id"])
        if text is None:
            problems.append(f"arm_structure.assumption_id: assumption {structure['assumption_id']} is not an "
                            f"assumption of {limit.id} ({', '.join(limit.assumptions)})")
        else:
            quote = _norm(structure["assumption_quote"])
            if not (quote == _norm(text) or (len(quote) >= 20 and quote in _norm(text))):
                problems.append(f"arm_structure.assumption_quote: assumption_quote is not the listed text of "
                                f"{limit.id}/{structure['assumption_id']}: '{text}'")
        source = _norm(structure["source"])
        if not (source == _norm(limit.source) or _norm(limit.source) in source):
            problems.append(f"arm_structure.source: source is not the listed source of {limit.id}: '{limit.source}'")
        return problems + _check_ref_or_learning(card, structure["mechanism_ref"])
    if arm_id == "A3":
        problems = _check_ref_or_learning(card, structure["mechanism_ref"])
        if structure["commitment_id"] != seeds["commitment"]["id"]:
            problems.append(f"arm_structure.commitment_id: commitment_id {structure['commitment_id']} is not the "
                            f"commitment given: {seeds['commitment']['id']}")
        return problems
    if arm_id == BASE_ARM:
        problems = []
        if base_text is None or _norm(structure["note_excerpt"]) not in _norm(base_text):
            problems.append("arm_structure.note_excerpt: note_excerpt is not text of the base draft")
        added = formatter_added_operators(base_text or "", card)
        if added:
            problems.append(f"mechanism: the formatter added operators the base draft does not contain: {added}")
        return problems
    raise ValueError(f"unknown arm {arm_id}")


# ----- classifier --------------------------------------------------------------------------------


def classifier_schema(archive_cfg: ArchiveConfig, families: Families) -> dict:
    props = {axis: {"enum": list(archive_cfg.values[axis])} for axis in AXES}
    props["evidence"] = _object({axis: _TEXT for axis in AXES})
    props["family"] = {"enum": ["none", *families.ids]}
    props["family_why"] = _TEXT
    props["difference_is_real"] = {"type": ["boolean", "null"]}
    props["difference_why"] = {"type": "string"}
    return _object(props)


def families_block(families: Families) -> str:
    lines = ["OUT OF SCOPE FAMILIES (a card whose core matches one of these, under any name, is flagged for the exact "
             "reduction check at S2. Name it only when the card's own equations map to the form without a state that "
             "grows with the input or a one-hot lift over reachable configurations):"]
    lines += [f"- {f.id}: {f.name}. Form: {f.form}" for f in families.excluded]
    lines.append("KNOWN FAMILIES (allowed only with a real difference that changes an equation of the form, not its "
                 "vocabulary, parameters or application):")
    lines += [f"- {f.id}: {f.name}. Form: {f.form}" for f in families.known]
    return "\n".join(lines)


def axes_block(archive_cfg: ArchiveConfig) -> str:
    lines = []
    for axis in AXES:
        lines.append(f"{axis} ({archive_cfg.titles[axis]}):")
        lines += [f"  - {value}: {archive_cfg.texts[axis][value]}" for value in archive_cfg.values[axis]]
    return "\n".join(lines)


def classifier_user(card: dict, gen: Generation) -> str:
    shown = {"mechanism": {key: value for key, value in card["mechanism"].items()},
             "complexity": card["claims"]["complexity"], "known_relatives": card["known_relatives"],
             "toy_example_python": card["toy_example"]["python"],
             "closest_known": (card.get("arm_structure") or {}).get("closest_known")}
    return "\n\n".join([
        "AXES AND VALUES:\n" + axes_block(gen.archive_cfg),
        families_block(gen.families),
        "THE CARD (equations, learning rule, cost, known relatives and toy code; title and summary withheld):\n"
        + json.dumps(shown, indent=2, ensure_ascii=False),
        "Return the classification as JSON."])


def consistency(card: dict, assigned: dict) -> tuple[dict, list[str]]:
    """The classifier's values after the deterministic consistency check (plan 6.2): code or a learning rule that
    trains with backpropagation cannot sit in a none_fixed or local_hebbian learning-rule cell."""
    commitments = {axis: assigned[axis] for axis in AXES}
    learning = card["mechanism"]["learning"]
    text = "\n".join([learning["description"]] + [eq["plain"] for eq in learning.get("equations", [])])
    by_text = bool(GRAD_TEXT.search(text)) and not GRAD_NEGATED.search(text)
    by_code = bool(GRAD_CODE.search(card["toy_example"]["python"]))
    overrides = []
    if (by_text or by_code) and commitments["learning_rule"] in GRADIENT_FREE:
        where = "toy code" if by_code else "learning rule text"
        overrides.append(f"learning_rule {commitments['learning_rule']} -> gradient_backprop: the {where} trains with "
                         f"backpropagation")
        commitments["learning_rule"] = "gradient_backprop"
    return commitments, overrides


# ----- prompts -----------------------------------------------------------------------------------


def arm_system_prompt(arm: Arm, formatter: bool = False) -> str:
    """The arm's prompt file (A5: its formatter's), then the shared explorer rules without their header line."""
    head = load_prompt("arm_base_formatter" if formatter else arm.prompt).rstrip()
    shared = load_prompt("explorer").split("\n", 1)[1].strip()
    if not arm.papers:
        head += ("\n\nThis arm gives you no papers. Your draft has no `source_transplant` field and its `references` "
                 "list stays empty; the rules below about citing given papers and naming a transplant do not apply. "
                 "Everything else in them does.")
    return head + "\n\n## General card rules (shared with every explorer)\n\n" + shared + "\n"


def arm_seeds(gen: Generation, task: ExploreTask) -> dict:
    """Seeded draws for the task (recorded in arm_draft): A3's commitment, A4's concepts."""
    rng = random.Random(f"arm-seeds:{task.seed}")
    if task.arm == "A3":
        commitment = rng.choice(sorted(gen.gen.commitments))
        return {"commitment": {"id": commitment, "text": gen.gen.commitments[commitment]}}
    if task.arm == "A4":
        return {"concepts": rng.sample(list(gen.gen.concepts), gen.gen.per_task)}
    return {}


def limits_block(limits: dict[str, Limit]) -> str:
    lines = ["KNOWN LIMITS (pick one limit and one assumption; quote the assumption and the source exactly):"]
    for limit in limits.values():
        lines.append(f"- limit_id: {limit.id}\n  name: {limit.name}\n  source: {limit.source}\n"
                     f"  statement: {limit.statement}\n  assumptions:")
        lines += [f"    - {aid}: {text}" for aid, text in limit.assumptions.items()]
    return "\n".join(lines)


def archive_block(ctx: StageContext, gen: Generation) -> str:
    """The axes, then the visited cells (most visited first) with their cards and the causes recorded there (G gates of
    this run, kill codes of earlier runs' records), so a draft sees which cells are crowded and why cards died."""
    archive = gen.archive
    cells = sorted((cell for cell, n in archive.visits.items() if n),
                   key=lambda cell: (-archive.visits[cell], cell))[:ARCHIVE_CELLS_SHOWN]
    lines = ["ARCHIVE. A separate classifier assigns every card one value per axis from its equations (not from its "
             "wording); a card in a cell no earlier card has reached fills an empty cell and earns this arm a reward "
             "once it passes the next stage. Cells of the out-of-scope families are closed. Axes and values:",
             axes_block(gen.archive_cfg)]
    if not cells:
        lines.append("No cell has been visited yet.")
        return "\n".join(lines)
    lines.append("Cells visited so far (visits; admitted cards with stage and verdict; recorded causes of death):")
    for cell in cells:
        shown = []
        for cid in archive.members.get(cell, [])[-3:]:
            card = ctx.store.get_card(cid)
            stage = card["status"]["stage"]
            shown.append(f"{card['title'][:60]} [{stage} {card['status']['verdicts'].get(stage, {}).get('verdict', '-')}]")
        causes = ", ".join(f"{code} x{n}" for code, n in archive.causes.get(cell, Counter()).most_common(4))
        lines.append(f"- {cell} ({archive.visits[cell]}): {'; '.join(shown) or 'none admitted'}"
                     + (f"; causes: {causes}" if causes else ""))
    return "\n".join(lines)


def build_arm_user(ctx: StageContext, task: ExploreTask, arm: Arm, papers: list, seeds: dict,
                   nearby: list | None) -> str:
    gen: Generation = ctx.generation
    ex = ctx.cfg["explore"]
    parts = ["OPERATOR: new", f"ARM: {arm.id} {arm.name} ({arm.title})"]
    if arm.papers:
        parts.append(f"LENS: {task.lens}\n{ctx.lenses[task.lens]['description']}")
        parts.append("PAPERS (cite by source_id in references when you use one):\n"
                     + _paper_block(papers, ex["abstract_chars"]))
    else:
        parts.append("This arm gives no papers: leave references empty and give no source_transplant.")
    if arm.id == "A2":
        parts.append(limits_block(gen.limits))
    if arm.id == "A3":
        commitment = seeds["commitment"]
        parts.append(f"COMMITMENT TO INVERT (commitment_id {commitment['id']}): {commitment['text']}")
    if arm.id == "A4":
        parts.append("CONCEPTS (drawn at random; copy them exactly into arm_structure.concepts):\n"
                     + "\n".join(f"- {c}" for c in seeds["concepts"]))
    parts.append(families_block(gen.families))
    parts.append(archive_block(ctx, gen))
    parts.append("CARDS ALREADY IN THIS RUN (do not repeat them):\n"
                 + existing_summaries(ctx, ex["existing_summaries"], ex["summary_chars"]))
    parts.append(GRAVEYARD_HEADER + "\n" + graveyard_block(ctx))
    if nearby is not None:
        from hybrid.research import outcomes as outcome_store
        from hybrid.research.stages.common import load_section

        parts.append(load_section("outcome_store") + "\n" + outcome_store.render_nearby(nearby))
    parts.append("Return one theory card draft as JSON with arm_structure filled in for this arm. In "
                 "arm_structure.closest_known give the id of the closest family from the lists above (or none) and "
                 "what your core changes in that family's equations. Do not include id, version, parent_ids, "
                 "created_by_call, lens or status; the orchestrator fills them in.")
    return "\n\n".join(parts)


def base_user(ctx: StageContext, task: ExploreTask, papers: list) -> str:
    """The base model's continuation prompt for A5: the arm prompt is a document prefix; this adds a seed."""
    titles = "; ".join(paper.title for paper in papers[:3])
    return f"Note 1. Inspired by {ctx.lenses[task.lens]['description']} ({titles}).\nIdea:"


# ----- the stage ---------------------------------------------------------------------------------


def _failure_kind(exc: StageError) -> str:
    from hybrid.llm.errors import SchemaValidationError

    if isinstance(exc.__cause__, SchemaValidationError) or "no parsed JSON" in str(exc):
        return "schema_failure"
    if isinstance(exc.__cause__, SchemaViolation) or "source_transplant cites" in str(exc):
        return "card_invalid"
    return "transport_failure"


def _outcome_of(kill_code: str | None) -> str:
    if kill_code is None:
        return "admitted"
    family = kill_code.split(":", 1)[0]
    return {"arm_structure": "arm_structure", "excluded_family": "excluded_family", "not_radical": "closed_cell",
            "no_difference": "no_difference", "duplicate": "duplicate"}.get(family, "budget")


def _record_draft(ctx: StageContext, task: ExploreTask, *, outcome: str, calls: list[str], seeds: dict,
                  card: dict | None = None, kill_code: str | None = None, error: str | None = None,
                  filled: bool = False, model: str | None = None, structure_retry: dict | None = None) -> None:
    ctx.emit("arm_draft", {
        "arm": task.arm, "op": task.op, "key": task.arm or f"op:{task.op}", "role": task.role, "model": model,
        "explorer": task.explorer, "card_id": card["id"] if card else None, "outcome": outcome,
        "kill_code": kill_code, "cell": (card or {}).get("archive_cell", {}).get("key"), "filled_empty_cell": filled,
        "calls": list(calls), "seeds": seeds, "error": error[:2000] if error else None,
        "structure_retry": structure_retry})


# What each arm's structure check holds a draft to; repeated in the structure retry so a repair cannot drop it.
STRUCTURE_RULES = {
    "A1": "the image equation is one of the card's mechanism equations, named by its zero-based mechanism_ref, and "
          "every object_map entry lands on a state object or on symbols of the update or readout equations; the "
          "prediction is derived through the map",
    "A2": "limit_id and assumption_id are listed ids, assumption_quote is the listed assumption text and source is "
          "the listed source of that limit, and mechanism_ref names the equation (or `learning`) that breaks it",
    "A3": "commitment_id is the commitment given and mechanism_ref names the equation (or `learning`) where the "
          "replacement acts",
    "A5": "note_excerpt is text of the base draft and the mechanism adds no operator the base draft lacks",
}
STRUCTURE_RULES["A4"] = "concepts are the concepts given, and " + STRUCTURE_RULES["A1"]


def structure_retry_instruction(arm_id: str, previous_output: str, problems: list[str]) -> str:
    """The structure retry: the previous draft, then every failing field by its path, then the arm's rule."""
    return ("\n\n# ARM STRUCTURE CHECK FAILED\n"
            f"Your previous draft was:\n<<<\n{previous_output}\n>>>\n"
            "It failed this arm's structure check. Each line names the failing field (dotted path in the draft) and "
            "what is wrong:\n" + "\n".join(f"- {problem}" for problem in problems) + "\n"
            f"The rule stands: {STRUCTURE_RULES[arm_id]}. Fix the named fields. Equation indices are zero-based: "
            "update[0] is the first entry of mechanism.update. If you change the mechanism to fix a field, keep "
            "arm_structure consistent with the changed mechanism. Return the complete corrected draft as one JSON "
            "object with every required field, not a fragment or a patch.\n")


async def generate(ctx: StageContext, task: ExploreTask) -> str | None:
    gen: Generation = ctx.generation
    arm = gen.gen.arms[task.arm] if task.arm else None
    seeds = arm_seeds(gen, task) if arm else {}
    ex = ctx.cfg["explore"]
    papers = []
    if arm is None or arm.papers:
        k = ex["papers_min"] + (task.seed % (ex["papers_max"] - ex["papers_min"] + 1))
        papers = draw_papers(ctx, task, k)
    nearby = nearby_dead(ctx, task, papers) if ctx.outcomes is not None else None
    calls: list[str] = []
    base_text = None
    if arm is None:
        user = build_user_prompt(ctx, task, papers, nearby)
    else:
        user = build_arm_user(ctx, task, arm, papers, seeds, nearby)
    extra = {"arm": arm.id} if arm else None
    try:
        if arm is None:
            result = await ctx.call(task.role, system_prompt(task), user, draft_schema(escape=task.op == "escape"))
        else:
            if arm.id == BASE_ARM:
                base = await ctx.call(arm.role, load_prompt(arm.prompt), base_user(ctx, task, papers))
                calls.append(base.call_id)
                base_text = base.text
                user = user + "\n\nTHE NOTE TO FORMAT (drafted by a base model):\n" + base_text
                draft_role, draft_system = arm.formatter_role, arm_system_prompt(arm, formatter=True)
            else:
                draft_role, draft_system = task.role, arm_system_prompt(arm)
            result = await ctx.call(draft_role, draft_system, user, arm_draft_schema(gen, arm))
        calls.append(result.call_id)
        card, dropped, dropped_sources = card_from_draft(ctx, task, result, papers, extra=extra)
    except StageError as exc:
        _record_draft(ctx, task, outcome=_failure_kind(exc), calls=calls, seeds=seeds, error=str(exc))
        raise

    problems = check_structure(arm.id, card, gen=gen, seeds=seeds, base_text=base_text) if arm else []
    structure_retry = None
    if problems:
        # One repair call naming every failing field. The draft is held to the same check afterwards; a draft that
        # still fails, or whose repair fails the JSON schema, dies with the arm code as before.
        structure_retry = {"problems": problems, "call_id": None, "fixed": False, "error": None}
        try:
            again = await ctx.call(draft_role, draft_system,
                                   user + structure_retry_instruction(arm.id, result.text, problems),
                                   arm_draft_schema(gen, arm))
            calls.append(again.call_id)
            structure_retry["call_id"] = again.call_id
            repaired = card_from_draft(ctx, task, again, papers, extra=extra)
        except StageError as exc:
            structure_retry["error"] = str(exc)[:2000]
        else:
            result = again
            card, dropped, dropped_sources = repaired
            problems = check_structure(arm.id, card, gen=gen, seeds=seeds, base_text=base_text)
            structure_retry["fixed"] = not problems
    common = dict(call_ids=calls, model=result.model, papers=papers, nearby=nearby, user=user, dropped=dropped,
                  dropped_sources=dropped_sources)
    created = {"arm": task.arm, "arm_seeds": seeds}
    if structure_retry is not None:
        created["structure_retry"] = structure_retry
    if problems:
        admission = admit(ctx, task, card, gate=(f"arm_structure:{arm.id}", "; ".join(problems)),
                          created_extra=created, **common)
        _record_draft(ctx, task, outcome="arm_structure", calls=calls, seeds=seeds, card=card,
                      kill_code=admission.kill_code, model=result.model, structure_retry=structure_retry)
        return None

    try:
        classified = await classify_and_gate(ctx, card)
    except StageError as exc:
        _record_draft(ctx, task, outcome="classifier_failure", calls=calls, seeds=seeds, error=str(exc),
                      model=result.model, structure_retry=structure_retry)
        raise
    calls.append(classified.call_id)
    created.update(archive_cell=card["archive_cell"], family=classified.family)
    admission = admit(ctx, task, card, gate=classified.gate, created_extra=created, extra_reasons=classified.reasons,
                      **common)
    entry = gen.archive.record(ctx.ledger, card=card, op=task.op, admitted=admission.passed,
                               classifier_call=classified.call_id, overrides=classified.overrides,
                               gate=admission.kill_code)
    _record_draft(ctx, task, outcome=_outcome_of(admission.kill_code), calls=calls, seeds=seeds, card=card,
                  kill_code=admission.kill_code, filled=entry.new_cell, model=result.model,
                  structure_retry=structure_retry)
    return card["id"] if admission.passed else None


@dataclass(frozen=True)
class Classified:
    """The classifier's reading of a card and the G gate it decides (None when the card may be admitted)."""
    call_id: str
    family: str
    overrides: list[str]
    gate: tuple[str, str] | None
    reasons: list[str]


async def classify_and_gate(ctx: StageContext, card: dict) -> Classified:
    """Classify a drafted card onto the archive axes (filling `commitments` and `archive_cell` on the card) and decide
    the G gates in order: closed cell, known family without a real difference, signature duplicate. An out-of-scope
    family is a flag in the reasons, not a gate (KD5). Shared by generation and by the planted-card driver
    (hybrid/research/planted.py), whose cards enter here in place of a draft."""
    gen: Generation = ctx.generation
    verdict = await ctx.call(CLASSIFIER_ROLE, load_prompt(CLASSIFIER_ROLE), classifier_user(card, gen),
                             classifier_schema(gen.archive_cfg, gen.families))
    parsed = verdict.parsed
    commitments, overrides = consistency(card, parsed)
    key = cell_key(commitments)
    card["commitments"] = commitments
    card["archive_cell"] = {"key": key, "new_cell": gen.archive.is_empty(key)}
    validate("theory_card", card)

    family = parsed["family"]
    closed = closed_by(commitments, gen.families)
    duplicate = gen.archive.duplicate_of(card)
    gate = None
    flag = None
    if gen.families.is_excluded(family):
        # KD5: the classifier's placement is an argument, not an executed check, and a lifted or growing state can
        # make any mechanism look like an out-of-scope form (KD4). It is recorded and left to S2, whose counted
        # comparison decides an exact reduction (`excluded_family:<id>` at S2).
        flag = f"{family}: {parsed['family_why']}"
    if closed is not None:
        gate = ("not_radical:R1", f"archive cell {key} is closed: it lies in the reference cells of the excluded "
                                  f"family {closed.id}, so no foundational commitment differs from it")
    elif gen.families.is_known(family) and parsed["difference_is_real"] is not True:
        gate = ("no_difference", f"the classifier places the core in the known family {family} and finds no real "
                                 f"difference: {parsed['difference_why'] or parsed['family_why']}")
    elif duplicate is not None:
        gate = (f"duplicate:{duplicate.split('/')[-1]}",
                f"mechanism signature equals that of {duplicate} (archive or outcome store)")
    reasons = [f"archive cell {key} ({'empty before this card' if card['archive_cell']['new_cell'] else 'visited'}); "
               f"closest family {family}"] + [f"consistency override: {line}" for line in overrides]
    if flag is not None:
        reasons.append(f"S0 family flag, unconfirmed by any executed check; S2's counted comparison decides: {flag}")
    return Classified(verdict.call_id, family, overrides, gate, reasons)


# ----- metrics -----------------------------------------------------------------------------------


def _zero() -> dict:
    return {"drafts": 0, **{outcome: 0 for outcome in OUTCOMES}, "new_cells": 0, "rewards": 0, "calls": 0,
            "tokens_in": 0, "tokens_out": 0, "structure_retries": 0, "structure_fixed": 0, "survivors": {}}


def arm_metrics(ledger, store, order: tuple[str, ...]) -> dict:
    """Per arm (or `op:<operator>`): drafts, each outcome, schema failures and their rate, new cells filled, rewards,
    survivors per stage (cards with a pass there), model calls and tokens of the generation calls."""
    usage: dict[str, dict] = {}
    drafts = []
    rewards = Counter()
    for event in ledger.iter_events():
        data = event["data"]
        if event["type"] == "llm_call":
            usage[data["call_id"]] = data.get("usage") or {}
        elif event["type"] == "arm_draft":
            drafts.append(data)
        elif event["type"] == "archive_reward":
            rewards[data["arm"] or f"op:{data['op']}"] += 1
    per: dict[str, dict] = {}
    known_cards = set(store.list_cards())
    for data in drafts:
        m = per.setdefault(data["key"], _zero())
        m["drafts"] += 1
        m[data["outcome"]] += 1
        m["new_cells"] += 1 if data["filled_empty_cell"] else 0
        retry = data.get("structure_retry")
        if retry is not None:
            m["structure_retries"] += 1
            m["structure_fixed"] += 1 if retry["fixed"] else 0
        for call_id in data["calls"]:
            m["calls"] += 1
            m["tokens_in"] += int(usage.get(call_id, {}).get("input") or 0)
            m["tokens_out"] += int(usage.get(call_id, {}).get("output") or 0)
        if data["card_id"] in known_cards:
            verdicts = store.get_card(data["card_id"])["status"]["verdicts"]
            for stage in order:
                if verdicts.get(stage, {}).get("verdict") == "pass":
                    m["survivors"][stage] = m["survivors"].get(stage, 0) + 1
    for key, m in per.items():
        m["rewards"] = rewards[key]
        m["schema_failures"] = sum(m[o] for o in SCHEMA_OUTCOMES)
        m["schema_failure_rate"] = round(m["schema_failures"] / m["drafts"], 4) if m["drafts"] else None
    return dict(sorted(per.items()))
