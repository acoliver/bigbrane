"""Card falsifier tasks for S5: the falsifier_designer role writes them, the executor validates them.

The designer sees the card (mechanism, claims, capabilities, falsifiers, complexity) and the bench's training
budget and baselines, never the candidate code. It returns one to three task programs with pre-registered
pass rules quoted from the card (contract in hybrid/bench/falsifier.py). Each program is checked on the host
(AST, rule names, the quote is card text) and then run by the falsifier harness in the executor (Podman in
real runs), which also generates the task data. A task that fails is recorded; failed tasks are sent back to
the designer once, quoting the failures, and a task that fails again is dropped.

Validated tasks are written to <run>/falsifiers/<card>/<tag>-<task id>/ and listed in a `falsifier_tasks`
artifact. A resumed S5 reuses that artifact instead of designing again.

Host gates (host_problems) on top of the rule names and the card quote: falsifier polarity and claim-scoped
quotes, thresholds and groups taken from the quote, a positive control for rules that can support the card from
a low score, protocol parameters that match the card, and a reference mechanism that uses the card's d. The validation job adds the data gates (see
hybrid/bench/falsifier_harness.py). Every validation attempt and the artifact record the sha256 of the staged
harness and evaluator code, because an edit to that code between two attempts can change the outcome.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from hybrid.bench import falsifier as bench_falsifier
from hybrid.bench.config import load_config as load_bench_config
from hybrid.bench.falsifier_eval import REFERENCES, better_than, needs_control
from hybrid.executor.job import Caps
from hybrid.research.stages.common import StageContext, card_for_prompt, load_section
from hybrid.research.stages.contracts import falsifier_design_schema

ROLE = "falsifier_designer"
ARTIFACT = "falsifier_tasks"
ROOT = Path(__file__).resolve().parents[3]
MAX_TASKS = 3
GATES_VERSION = 2   # a falsifier_tasks artifact without this value was designed under older gates and is not reused
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def claim_items(card: dict) -> list[dict]:
    """The card's claims with stable ids, in card order. S6 and the designer use the same ids."""
    claims = card["claims"]
    items = [{"id": f"complexity.{key}", "text": f"complexity {key}: {claims['complexity'][key]}"}
             for key in ("time_per_token", "memory", "total_sequence")]
    sample = claims["sample_efficiency"]
    prediction = sample["prediction"]
    items.append({"id": "sample_efficiency.mechanism", "text": f"sample efficiency mechanism: {sample['mechanism']}"})
    items.append({"id": "sample_efficiency.prediction",
                  "text": f"sample efficiency prediction: {prediction['statement']} (metric: {prediction['metric']}; "
                          f"value: {prediction['value']})"})
    for index, capability in enumerate(claims["capabilities"], 1):
        items.append({"id": f"capability.{index}",
                      "text": f"capability {index}: task: {capability['task']} prediction: {capability['prediction']}"})
    for index, text in enumerate(card["falsifiers"], 1):
        items.append({"id": f"falsifier.{index}", "text": f"falsifier {index}: {text}"})
    return items


SYMBOLS = str.maketrans({"\u2265": ">=", "\u2264": "<=", "\u2212": "-", "\u2013": "-", "\u00d7": "x",
                         "\u2192": "->", "\u2208": "in", "\u2018": "'", "\u2019": "'", "\u201c": '"',
                         "\u201d": '"'})


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(SYMBOLS)).strip().lower()


def quote_in_card(quote: str, card: dict) -> bool:
    """The quote is card text: a substring, after whitespace and case normalization, of the card's claims or
    falsifiers."""
    corpus = _normal(json.dumps({"claims": card["claims"], "falsifiers": card["falsifiers"]}, ensure_ascii=False)
                     .replace("\\n", " "))
    plain = _normal(" ".join(item["text"] for item in claim_items(card)))
    needle = _normal(quote).strip(" .;:'\"")
    return bool(needle) and (needle in plain or needle in corpus)


def bench_settings(ctx: StageContext) -> dict:
    bench = load_bench_config(ROOT / ctx.cfg["s5"]["bench_config"])
    return {"baselines": list(bench["baselines"]), "seed": int(bench["seed"]), "budget": bench["budget"]}


def candidate_train_max_len(ctx: StageContext, card_id: str) -> int | None:
    """The guaranteed training length the card's candidate reported in describe() (S4 enforces the field for new
    candidates), or None for a candidate built before the field existed."""
    built = ctx.card_events(card_id, "candidate_built")
    limit = ((built[-1].get("describe") or {}).get("train_max_len") if built else None)
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else None


def designer_prompt(card: dict, claims: list[dict], settings: dict, train_max_len: int | None = None) -> str:
    entrants = ["candidate", *settings["baselines"], *REFERENCES]
    length = ("The candidate fits training sequences in full up to train_max_len = %d tokens; longer training "
              "sequences are rejected." % train_max_len if train_max_len is not None
              else "The candidate's training length is not known; keep training episodes short.")
    return "\n\n".join([
        "CARD:\n" + card_for_prompt(card),
        "CLAIM IDS (use these in rules[].claims and untestable[].claim_id):\n"
        + "\n".join(f"- {item['id']}: {item['text']}" for item in claims),
        "ENTRANTS a condition may name: " + ", ".join(entrants)
        + "\nThe baselines are standard sequence models trained on your data with the same budget: ngram (count "
          "table over the last tokens), gru, transformer (small, softmax attention), ssm (diagonal state space), "
          "linear_attention (fixed-size outer-product state).",
        "TRAINING BUDGET per model and task: " + json.dumps(settings["budget"], sort_keys=True)
        + ". The seed passed to generate() is " + str(settings["seed"]) + ".",
        "CAPS: " + json.dumps(bench_falsifier.CAPS, sort_keys=True),
        length,
        load_section("falsifier_gates"),
    ])


def retry_prompt(first_user: str, failures: list[dict]) -> str:
    lines = ["THESE TASKS FAILED VALIDATION. Return corrected versions of these tasks only, with the same task_id, "
             "fixing every problem listed. Keep the rules' thresholds and quotes from the card."]
    for failure in failures:
        lines.append(f"\nTASK {failure['task_id']}:\nPROBLEMS:\n" + "\n".join(f"- {p}" for p in failure["problems"])
                     + "\nYOUR PREVIOUS DESIGN:\n" + json.dumps(failure["design"], indent=2))
    return first_user + "\n\n" + "\n".join(lines)


def claim_problems(rule: dict, index: int, claims_by_id: dict[str, str]) -> list[str]:
    """Falsifier polarity and claim-scoped quotes. A met falsifier refutes the card while a met capability
    supports it, so the two never share a rule, and a falsifier rule states the falsifying observation."""
    ids = rule["claims"]
    fals = [c for c in ids if c.startswith("falsifier.")]
    problems = []
    if fals and len(fals) != len(ids):
        problems.append(f"rule {index} mixes falsifier claims with other claims; a met falsifier refutes the card "
                        "while a met capability supports it, so write separate rules")
    if fals and rule["if_met"] != "refuted":
        problems.append(f"rule {index} decides {', '.join(fals)} with if_met 'supported'; a falsifier rule states "
                        "the falsifying observation from the falsifier's own text and if_met must be 'refuted'")
    needle = _normal(rule["quote"]).strip(" .;:'\"")
    for cid in ids:
        if needle not in _normal(claims_by_id[cid]):
            problems.append(f"rule {index} quote is not text from {cid}")
    return problems


def threshold_problems(rule: dict, index: int) -> list[str]:
    """Condition values and groups come from the rule's quote, and absolute thresholds stay absolute."""
    text = _normal(rule["quote"])
    numbers = {float(x) for x in NUMBER.findall(text)}
    problems = []
    for j, cond in enumerate(rule["conditions"]):
        if float(cond["value"]) not in numbers:
            problems.append(f"rule {index} condition {j} value {cond['value']} is not a number in the quote")
        if cond.get("minus") in REFERENCES:
            problems.append(f"rule {index} condition {j} subtracts {cond['minus']}; the card states absolute "
                            "thresholds, so subtracting a trivial reference changes them")
        if cond["group"] and "=" in cond["group"]:
            name, val = (part.strip() for part in cond["group"].split("=", 1))
            try:
                value = float(val)
            except ValueError:
                problems.append(f"rule {index} condition {j} group {cond['group']} has no numeric value")
                continue
            bound = re.search(rf"(?<![\w.]){re.escape(name.lower())}\s*(<=|>=|=)\s*(\d+(?:\.\d+)?)", text)
            ok = value in numbers or (bound is not None and (
                (bound[1] == "<=" and value <= float(bound[2])) or (bound[1] == ">=" and value >= float(bound[2]))))
            if not ok:
                problems.append(f"rule {index} condition {j} group {cond['group']} is not covered by the quote")
    return problems


def control_problems(rules: list[dict]) -> list[str]:
    """A rule that can support the card from a low score names a control_rule in the same task, and the control
    is a rule where the candidate alone must show performance above a bar."""
    problems = []
    for index, rule in enumerate(rules):
        control = rule.get("control_rule")
        if control is None:
            if needs_control(rule):
                problems.append(f"rule {index} can support the card by a low score; name a control_rule in this task "
                                "where the candidate must first show the capability")
            continue
        if not 0 <= control < len(rules) or control == index:
            problems.append(f"rule {index} control_rule {control} does not name another rule of this task")
            continue
        target = rules[control]
        if target["if_met"] != "supported" or any(c["entrant"] != "candidate" or c.get("minus") or not better_than(c)
                                                  for c in target["conditions"]):
            problems.append(f"rule {index} control_rule {control} must be a 'supported' rule whose conditions all ask "
                            "the candidate alone for performance above a bar")
    return problems


def _number_pattern(value: float) -> str:
    return re.escape("%g" % value) + r"(?!\.?\d)"


def protocol_problems(design: dict, claims_by_id: dict[str, str]) -> list[str]:
    """Each protocol parameter's card value is stated as `name = value` in a claim the rules name, and a task
    value that differs is listed in deviations."""
    corpus = " ".join(_normal(claims_by_id[c]) for rule in design["rules"] for c in rule["claims"])
    deviations = _normal(design["deviations"])
    problems = []
    for param in design["protocol_params"]:
        name = _normal(param["name"])
        if not re.search(rf"(?<![\w.]){re.escape(name)}\s*=\s*{_number_pattern(param['card'])}", corpus):
            problems.append(f"protocol parameter {param['name']} = {param['card']:g} is not stated as "
                            f"'{param['name']} = {param['card']:g}' in a claim the rules name")
        if param["task"] != param["card"] and not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", deviations):
            problems.append(f"protocol parameter {param['name']} is {param['task']:g} in the task and "
                            f"{param['card']:g} in the card, and deviations does not mention {param['name']}")
    return problems


def reference_declaration_problems(design: dict) -> list[str]:
    """A declared reference mechanism uses the card's d: the task lists d among its protocol_params with the
    same card value, so the capacity control tests the card's operator and not another size."""
    reference = design.get("reference")
    if reference is None:
        return []
    card_d = [param["card"] for param in design["protocol_params"] if _normal(param["name"]) == "d"]
    if reference["d"] not in card_d:
        return [f"reference d = {reference['d']} is not the card's d; list protocol_params d with card = "
                f"{reference['d']}"]
    return []


def retry_problems(first: dict, second: dict, failed: list[str]) -> list[str]:
    """A retry keeps the rules' claims, quotes, operators and thresholds. Rules named in the first failure's
    problems may change; the others may not."""
    named = {int(m) for p in failed for m in re.findall(r"\brule (\d+)\b", p)}

    def signature(rule: dict) -> tuple:
        return (tuple(rule["claims"]), rule["quote"], rule["if_met"],
                tuple((c["op"], float(c["value"])) for c in rule["conditions"]))

    problems = []
    for index, rule in enumerate(second["rules"]):
        if index < len(first["rules"]) and index not in named and signature(rule) != signature(first["rules"][index]):
            problems.append(f"the retry changed rule {index}'s threshold, operator, quote or claims; the failure "
                            "did not name this rule")
    return problems


def host_problems(design: dict, card: dict, baselines: list[str]) -> list[str]:
    problems = bench_falsifier.static_problems(design["program"])
    problems += bench_falsifier.rule_problems(design["rules"], baselines)
    claims_by_id = {item["id"]: item["text"] for item in claim_items(card)}
    for index, rule in enumerate(design["rules"]):
        if not quote_in_card(rule["quote"], card):
            problems.append(f"rule {index} quote is not text from the card's claims or falsifiers: {rule['quote'][:200]!r}")
        problems += claim_problems(rule, index, claims_by_id)
        problems += threshold_problems(rule, index)
    problems += control_problems(design["rules"])
    problems += protocol_problems(design, claims_by_id)
    problems += reference_declaration_problems(design)
    return problems


def validation_caps(ctx: StageContext) -> Caps:
    ex = ctx.cfg["executor"]
    return Caps(wall_s=ctx.cfg["s5"]["falsifier_wall_s"], memory_mb=ex["construct_memory_mb"], pids=ex["pids"],
                tmp_mb=ex["tmp_mb"])


async def validate(ctx: StageContext, card_id: str, card: dict, design: dict, settings: dict, tag: str,
                   train_max_len: int | None = None, extra_problems: list[str] | None = None) -> dict:
    """Host checks, then the harness job. Returns {"task_id", "ok", "problems", "report", "job_id", "dir",
    "code_sha256"}; code_sha256 is the hashes of the harness and evaluator code the job ran (None when the task
    failed on the host and no job ran)."""
    entry = {"task_id": design["task_id"], "job_id": None, "dir": None, "report": None, "code_sha256": None}
    problems = host_problems(design, card, settings["baselines"]) + list(extra_problems or [])
    if problems:
        return dict(entry, ok=False, problems=problems, stage="host")
    caps = validation_caps(ctx)
    files = bench_falsifier.harness_files(design["program"], settings["seed"], design["rules"], caps.wall_s * 0.4,
                                          train_max_len=train_max_len, solvable_by=design.get("solvable_by"),
                                          reference=design.get("reference"))
    hashes = bench_falsifier.code_hashes(files)
    result = await ctx.runner.run(card_id, f"falsifier_validation:{design['task_id']}", files, caps=caps)
    report = dict(bench_falsifier.parse_report(result.stdout, result.status, result.detail), code_sha256=hashes)
    entry.update(job_id=result.job_id, report=report, code_sha256=hashes)
    if not report["ok"]:
        return dict(entry, ok=False, problems=report["problems"], stage="executor")
    destination = ctx.run_dir / "falsifiers" / card_id / f"{tag}-{design['task_id']}"
    bench_falsifier.write_task(destination, result.out_dir, design, report)
    return dict(entry, ok=True, problems=[], stage="executor", dir=destination.relative_to(ctx.run_dir).as_posix())


async def prepare(ctx: StageContext, card_id: str) -> tuple[list[Path], list[str]]:
    """Validated falsifier task directories for the card (designing them unless an earlier S5 attempt did) and
    the ids of the calls and jobs used."""
    if not ctx.cfg["s5"]["falsifiers"]:
        return [], []
    recorded = ctx.latest_json_artifact(card_id, ARTIFACT)
    if recorded is not None and recorded.get("gates") == GATES_VERSION:
        return [ctx.run_dir / task["dir"] for task in recorded["validated"]], []
    card = ctx.card(card_id)
    claims = claim_items(card)
    settings = bench_settings(ctx)
    schema = falsifier_design_schema([item["id"] for item in claims],
                                     ["candidate", *settings["baselines"], *REFERENCES])
    tag = uuid.uuid4().hex[:6]
    train_max_len = candidate_train_max_len(ctx, card_id)
    user = designer_prompt(card, claims, settings, train_max_len)
    first = await ctx.ask(ROLE, ROLE, user, schema)
    calls = [first.call_id]
    models = [first.model]
    attempts, validated, failures = [], [], []
    seen: set[str] = set()
    for design in first.parsed["tasks"]:
        if design["task_id"] in seen:
            attempts.append({"round": 0, "task_id": design["task_id"], "ok": False, "stage": "host",
                             "problems": ["duplicate task_id"], "job_id": None})
            continue
        seen.add(design["task_id"])
        outcome = await validate(ctx, card_id, card, design, settings, tag, train_max_len)
        attempts.append({"round": 0, **{k: v for k, v in outcome.items() if k != "dir"}})
        if outcome["job_id"]:
            calls.append(outcome["job_id"])
        if outcome["ok"]:
            validated.append({"task_id": design["task_id"], "dir": outcome["dir"], "design": design,
                              "code_sha256": outcome["code_sha256"]})
        else:
            failures.append({"task_id": design["task_id"], "problems": outcome["problems"], "design": design})
    untestable = list(first.parsed["untestable"])
    dropped = []
    if failures:
        second = await ctx.ask(ROLE, ROLE, retry_prompt(user, failures), schema)
        calls.append(second.call_id)
        models.append(second.model)
        retried = {failure["task_id"]: failure for failure in failures}
        done = {task["task_id"] for task in validated}
        for design in second.parsed["tasks"]:
            if design["task_id"] not in retried or design["task_id"] in done or len(validated) >= MAX_TASKS:
                attempts.append({"round": 1, "task_id": design["task_id"], "ok": False, "stage": "host",
                                 "problems": ["not one of the failed tasks; ignored"], "job_id": None})
                continue
            done.add(design["task_id"])
            changed = retry_problems(retried[design["task_id"]]["design"], design,
                                     retried[design["task_id"]]["problems"])
            outcome = await validate(ctx, card_id, card, design, settings, tag + "r", train_max_len, changed)
            attempts.append({"round": 1, **{k: v for k, v in outcome.items() if k != "dir"}})
            if outcome["job_id"]:
                calls.append(outcome["job_id"])
            if outcome["ok"]:
                validated.append({"task_id": design["task_id"], "dir": outcome["dir"], "design": design,
                                  "code_sha256": outcome["code_sha256"]})
                del retried[design["task_id"]]
            else:
                retried[design["task_id"]] = {"task_id": design["task_id"], "problems": outcome["problems"],
                                              "design": design, "first_problems": retried[design["task_id"]]["problems"]}
        dropped = [{"task_id": tid, "problems": f["problems"], "first_problems": f.get("first_problems", f["problems"])}
                   for tid, f in retried.items()]
    record = {"card_id": card_id, "gates": GATES_VERSION,
              "code_sha256": bench_falsifier.current_code_hashes(), "candidate_train_max_len": train_max_len,
              "designer_calls": [c for c in calls if c.startswith("call-")],
              "designer_models": models, "untestable": untestable, "attempts": attempts,
              "validated": [{"task_id": t["task_id"], "dir": t["dir"], "title": t["design"]["title"],
                             "rules": t["design"]["rules"], "code_sha256": t["code_sha256"]} for t in validated],
              "dropped": dropped}
    ctx.put_json_artifact(ARTIFACT, ctx.artifact_name(card_id), record, card_id)
    ctx.emit(ARTIFACT, {"card_id": card_id, "validated": [t["task_id"] for t in validated],
                        "dropped": [d["task_id"] for d in dropped], "designer_models": models,
                        "rounds": 2 if failures else 1})
    return [ctx.run_dir / t["dir"] for t in validated], calls
