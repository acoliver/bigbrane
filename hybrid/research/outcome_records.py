"""Outcome records from run directories, and the backfill of every card under runs/ (plan section 9).

Reading is read-only: `RunView` parses ledger.jsonl without the ledger lock (as the graveyard does) and reads card
and artifact files that the ledger recorded, checking each file against the sha256 the ledger holds. Nothing is
written inside a run directory, so the backfill can point at old runs and at runs another process is writing.

The same code writes a record for a card of a live run at its terminal verdict (the orchestrator, when
`outcomes.enabled`), so the mapping from verdicts to kill codes has one implementation.

Kill-code mapping, by the stage where the card's line ended:
- S0 fail: `duplicate:<nearest card>`; in generation mode "arms" (Phase 6) also the G gate's code, written in the
  verdict as `gate: <kill code>: <detail>` (`arm_structure:<arm>`, `excluded_family:<family>`, `not_radical:R1` for a
  closed archive cell, `no_difference` for a known family without a real difference, `duplicate:<card>` for a
  mechanism signature already in the archive or the outcome store).
- S1 fail: `ill_formed:fatal` (a fatal salad-critic finding), `ill_formed:toy` (toy example mismatch or crash),
  `ill_formed:fixable` (fixable findings left after the last revision).
- R (refinement dialogue, Phase 7) fail: the gate's code, written as `gate: <kill code>: <detail>` (`ill_formed:toy`
  when an accepted revision broke the toy example).
- S2: `reduces_to_known:<family>` (the reduction critic's `reduces_to`), `excluded_family:<family>` (Phase 7: a
  verified reduction into an out-of-scope family of families.toml, gate reason `gate: excluded_family:<id>: ...`),
  `known_disproven`, `no_difference` (prior art tested and no untested element; or, Phase 7, a difference statement
  rejected after its send-back), `not_radical:R2` (the reduction critic's radical_enough false), `not_radical:R3`
  (the own-history gate, or a difference statement that does not address a recorded cause), `novelty_unassessed`
  (empty literature search).
- S3: `infeasible:disproof`, `obligation_failed:<obligation>` (a replayed check violation), `infeasible:judges`,
  `undetermined:S3`.
- M and C: the kill codes in the stage's `math_verdict` artifact.
- S4 fail `construct_failed`; S5 fail `bench_failed`; S6 `falsified:<task>` when a pre-registered rule found a
  card falsifier fired, otherwise the card finished.
A card whose latest verdict is a pass (or S1 revise) has not ended: its record is `open`, with no kill code.

Causes (FIX-10). Each kill code's cause carries its own deciding explanation, bounded at CAUSE_CHARS with an explicit
truncation marker, while `kill_detail` stays a short summary: an S2 reduction or family exclusion gives the critic's
mapping, the numeric check lines and the critic's whole argument; `not_radical:R2` the critic's whole argument (the
verdict line clips it); a rejected difference statement the gate line and every failure of the last attempt; each
S3 `obligation_failed:<name>` the reason that names that obligation; `infeasible:disproof` each confirmed attack's
whole target and counterexample, recovered from the run's `disproof` artifacts (FIX-12, F2: S3 verdict reasons cut them
at 200 and 400 characters before FIX-12); an M or C code the claim, mismatch, circularity
finding or premise it names; a loop code its own stop. R3 shows these explanations to the difference writer and
checker. The store is append-only and keeps the records it has; a backfill into a new store directory writes the
corrected records.

Loop budgets (Phase 8, hybrid.research.lineage). A `loop_stop` event marked terminal at the card's terminal stage
(a lineage at its idea cap at S0, a model-call cap reached mid-stage, S6's analyst call included) is the death: its
code (`budget:<loop>`) is the kill code, read before any stage's own mapping (FIX-14: S6 was mapped from its claims
ledger first, so a card stopped at S6 was recorded as finished). A non-terminal stop at the terminal stage (a revision loop that ran out or stalled before the gate decided)
adds its `budget:<loop>` or `stall:<loop>` code to the stage's codes when the card died there. The record's `budget`
block holds the card's lineage, idea key, revisions charged, model calls and stops; its `lineage_id` is the one the
run assigned (which may come from an earlier run through the signature), else the root of its parents in the run.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from hybrid.research import graveyard, lineage
from hybrid.research.graveyard import clip, deciding_reason, reduction_summary
from hybrid.research.outcomes import OutcomeStore, check_kill_code, kill_family
from hybrid.research.signature import card_signature
from hybrid.research.stages.s3_feasibility import disproof_reason

DETAIL_CHARS = 600
# FIX-10: the bound on one cause's deciding explanation (Death.explanations), past which `bounded` marks the cut.
CAUSE_CHARS = 8000
REASON_CHARS = 400
DUPLICATE = re.compile(r"duplicate: TF-IDF cosine [0-9.]+ with (card-[0-9a-f]{8})")
# A gate verdict of generation stage G names its kill code (stages/g_generate.py).
GATE_CODE = re.compile(r"^gate: ([a-z_]+(?::[\w.\-]+)?): ")
VIOLATED = re.compile(r"executed check violated stated obligation '([^']+)'")
# Artifacts that hold the evidence for a death at each stage (latest of each kind is cited).
STAGE_EVIDENCE = {
    "S0": (), "S1": ("salad_critic",), "R": ("refinement", "dialogue_assumptions"),
    "S2": ("reduction_critic", "prior_art", "difference"),
    "S3": ("s3_gate", "judges", "disproof"), "M": ("math_verdict", "assumption_ledger"),
    "C": ("math_verdict", "assumption_ledger", "challenge"), "S4": ("construct_check", "candidate"),
    "S5": ("bench_result",), "S6": ("claims_ledger",),
}
OPEN_VERDICTS = ("pass", "revise")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RunView:
    """A read-only view of one run directory through its ledger."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        self.run_id = self.dir.name
        self.versions: dict[str, list[dict]] = {}
        self.artifacts: dict[str, list[dict]] = {}
        self.created: dict[str, dict] = {}
        self.verdicts: dict[str, list[dict]] = {}
        self.tokens: dict[str, int] = {}
        self.override: dict | None = None
        self.events = list(graveyard.read_events(self.dir))
        self.stops: dict[str, list[dict]] = {}
        self.lineages: dict[str, str] = {}
        for event in self.events:
            data, kind = event["data"], event["type"]
            if kind == "loop_stop":
                self.stops.setdefault(data["card_id"], []).append(data)
            elif kind == "lineage_assigned":
                self.lineages[data["card_id"]] = data["lineage_id"]
            if kind == "card_version":
                self.versions.setdefault(data["card_id"], []).append(data)
            elif kind == "artifact" and data.get("card_id"):
                self.artifacts.setdefault(data["card_id"], []).append(data)
            elif kind == "card_created":
                self.created[data["card_id"]] = data
            elif kind == "verdict":
                self.verdicts.setdefault(data["card_id"], []).append(data)
            elif kind == "llm_call" and data.get("call_id"):
                usage = data.get("usage") or {}
                self.tokens[data["call_id"]] = int(usage.get("input") or 0) + int(usage.get("output") or 0)
            elif kind == "operator_override" and self.override is None:
                self.override = data

    def card_ids(self) -> list[str]:
        return sorted(self.versions)

    def _read(self, entry: dict) -> bytes:
        data = (self.dir / entry["path"]).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise ValueError(f"{self.run_id}/{entry['path']}: sha256 differs from the ledger record")
        return data

    def latest_card(self, card_id: str) -> tuple[dict, dict, bytes]:
        """(card, its card_version event, file bytes) of the newest version the ledger recorded."""
        entry = max(self.versions[card_id], key=lambda e: e["version"])
        raw = self._read(entry)
        return json.loads(raw), entry, raw

    def all_artifacts(self, card_id: str, kind: str) -> list[tuple[dict, dict]]:
        """Every recorded artifact of a kind for the card, newest first."""
        return [(json.loads(self._read(entry)), entry) for entry in reversed(self.artifacts.get(card_id, []))
                if entry["kind"] == kind]

    def latest_artifact(self, card_id: str, kind: str) -> tuple[dict, dict] | None:
        recorded = [entry for entry in self.artifacts.get(card_id, []) if entry["kind"] == kind]
        if not recorded:
            return None
        return json.loads(self._read(recorded[-1])), recorded[-1]

    def math_verdict(self, card_id: str, stage: str) -> tuple[dict, dict] | None:
        found = None
        for entry in self.artifacts.get(card_id, []):
            if entry["kind"] == "math_verdict":
                record = json.loads(self._read(entry))
                if record.get("stage") == stage:
                    found = (record, entry)
        return found


@dataclass
class Death:
    """The codes a card died with, the short `kill_detail` summary, the evidence kinds, and one deciding explanation
    per code (`explanations`, FIX-10): each cause of the record carries its own argument, so two codes never share a
    detail that explains only one of them."""
    codes: list[str]
    detail: str
    kinds: tuple[str, ...]
    explanations: dict[str, str]

    def __post_init__(self):
        missing = [code for code in self.codes if code not in self.explanations]
        if missing:
            raise ValueError(f"no deciding explanation for {missing}")


def bounded(text: str) -> str:
    """A cause's deciding explanation: whitespace flattened, in full up to CAUSE_CHARS, then cut with an explicit
    marker (the evidence pointers lead to the rest)."""
    flat = " ".join(text.split())
    if len(flat) <= CAUSE_CHARS:
        return flat
    return flat[:CAUSE_CHARS] + (f" [truncated: {CAUSE_CHARS} of {len(flat)} characters shown; full text in the "
                                 f"evidence]")


def _single(code: str, detail: str, kinds: tuple[str, ...], explanation: str) -> Death:
    return Death([code], detail, kinds, {code: bounded(explanation)})


def _slug(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", text.strip())[:120] or "unnamed"


def _gate(reasons: list[str]) -> str:
    return next((r for r in reasons if r.startswith("gate")), "")


def _stop_text(stop: dict) -> str:
    return f"{stop['kill_code']}: {stop['detail']}"


def classify(view: RunView, card: dict) -> tuple[str, str | None, Death | None]:
    """(outcome, terminal stage, death) for the card's latest status. death is None unless the outcome is dead."""
    outcome, stage, death = _classify_stage(view, card)
    if outcome != "dead":
        return outcome, stage, death
    # A mutation that found no room ends the lineage's growth, not the card: it never explains a death.
    stops = [stop for stop in view.stops.get(card["id"], [])
             if stop["stage"] == stage and stop["loop"] != lineage.MUTATION]
    if stops:
        # Each loop code is explained by its own stop; a code the stage already has keeps the stage's explanation.
        explanations = {stop["kill_code"]: bounded(_stop_text(stop)) for stop in stops} | death.explanations
        death = Death(list(dict.fromkeys(death.codes + [stop["kill_code"] for stop in stops])), death.detail,
                      death.kinds, explanations)
    return outcome, stage, death


def _terminal_stop(view: RunView, card_id: str, stage: str) -> dict | None:
    stops = [s for s in view.stops.get(card_id, []) if s["terminal"] and s["stage"] == stage]
    return stops[-1] if stops else None


def _classify_stage(view: RunView, card: dict) -> tuple[str, str | None, Death | None]:
    stage = card["status"]["stage"]
    entry = card["status"]["verdicts"].get(stage)
    if entry is None:
        return "open", stage, None
    verdict, reasons = entry["verdict"], entry["reasons"]
    full = deciding_reason(reasons)
    detail = clip(full, DETAIL_CHARS)
    kinds = STAGE_EVIDENCE.get(stage, ())
    # FIX-14: a terminal stop ends the card at any stage, S6 included (the analyst's call can reach the idea's model
    # calls when cards of one idea share them), so it is read before the S6 claims ledger.
    stop = _terminal_stop(view, card["id"], stage) if verdict not in OPEN_VERDICTS else None
    if stop is not None:
        return "dead", stage, _single(stop["kill_code"], clip(_stop_text(stop), DETAIL_CHARS), (), _stop_text(stop))
    if stage == "S6":
        if verdict != "pass":
            raise ValueError(f"{view.run_id}/{card['id']}: S6 {verdict} without a terminal loop stop")
        claims = view.latest_artifact(card["id"], "claims_ledger")
        fired = (claims[0].get("falsified_by") or []) if claims else []
        if fired:
            falsified = [r for r in reasons if "falsified" in r]
            explanations = {}
            for f in fired:
                code = f"falsified:{_slug(f['task_id'])}"
                own = [r for r in falsified if f["task_id"] in r] or falsified or [full]
                explanations[code] = bounded("; ".join(own))
            return "dead", stage, Death(sorted(explanations), clip("; ".join(falsified) or detail, DETAIL_CHARS),
                                        STAGE_EVIDENCE["S6"], explanations)
        return "finished", stage, None
    if verdict in OPEN_VERDICTS:
        return "open", stage, None
    if stage == "S0":
        gate = next((GATE_CODE.match(r).group(1) for r in reasons if GATE_CODE.match(r)), None)
        if gate is not None:
            return "dead", stage, _single(check_kill_code(gate), detail, kinds, full)
        match = next((DUPLICATE.match(r) for r in reasons if DUPLICATE.match(r)), None)
        if match is None:
            raise ValueError(f"{view.run_id}/{card['id']}: S0 {verdict} without a duplicate reason")
        return "dead", stage, _single(f"duplicate:{match.group(1)}", detail, kinds, full)
    if stage == "S1":
        gate = _gate(reasons)
        if "fatal" in gate:
            code = "ill_formed:fatal"
        elif any("toy" in r.lower() for r in reasons):
            code = "ill_formed:toy"
        else:
            code = "ill_formed:fixable"
        return "dead", stage, _single(code, detail, kinds, full)
    if stage == "R":
        gate = next((GATE_CODE.match(r).group(1) for r in reasons if GATE_CODE.match(r)), None)
        if gate is None:
            raise ValueError(f"{view.run_id}/{card['id']}: R {verdict} without a gate code")
        return "dead", stage, _single(check_kill_code(gate), detail, kinds, full)
    if stage == "S2":
        return "dead", stage, _s2_death(view, card, verdict, reasons, detail)
    if stage == "S3":
        return "dead", stage, _s3_death(view, card, verdict, reasons, detail)
    if stage in ("M", "C"):
        found = view.math_verdict(card["id"], stage)
        if found is None or not found[0]["kill_codes"]:
            raise ValueError(f"{view.run_id}/{card['id']}: {stage} {verdict} without math_verdict kill codes")
        codes = list(dict.fromkeys(found[0]["kill_codes"]))
        premises, _ = premise_ledger(view, card["id"])
        explanations = {code: bounded(_math_cause(found[0], premises, code) or full) for code in codes}
        return "dead", stage, Death(codes, detail, kinds, explanations)
    if stage == "S4":
        return "dead", stage, _single("construct_failed", detail, kinds, full)
    if stage == "S5":
        return "dead", stage, _single("bench_failed", detail, kinds, full)
    raise ValueError(f"{view.run_id}/{card['id']}: no kill-code mapping for {stage} {verdict}")


def _math_cause(verdict: dict, premises: list[dict], code: str) -> str | None:
    """The part of a stage M or C math_verdict that decides one kill code: the claim the code names (its status, tier
    and detail), the statement-match mismatches, the circularity finding or the premise the code names. None when the
    artifact holds nothing specific to the code. Older artifacts lack some keys, so each is read when present."""
    family, _, subject = code.partition(":")
    if code == "obligation_failed:target_coupled" and verdict.get("target_coupled"):
        block = verdict["target_coupled"]
        slots = ", ".join(f"{slot} {entry['scale']}" for slot, entry in (block.get("slots") or {}).items())
        return f"skill scale at target (FIX-17): {slots}; " + "; ".join(block.get("reasons") or [])
    if code == "obligation_failed:target" and verdict.get("target"):
        return "target gate: " + "; ".join(verdict["target"].get("reasons") or [])
    claims = {c["claim_id"]: c for c in verdict.get("claims", [])}
    if subject in claims:
        claim = claims[subject]
        return f"claim {subject}: {claim['status']} [{claim['label']}]: {claim.get('detail') or claim.get('phrase')}"
    if family == "statement_mismatch":
        rows = [f"statement {m.get('statement_id')} ({m['claim_id']}): {'; '.join(m.get('reasons') or [])}"
                for m in verdict.get("statement_match", []) if m.get("mismatch")]
        return ("statement mismatch: " + " | ".join(rows)) if rows else None
    if family == "circular":
        found = [c for c in verdict.get("circularity", []) if c.get("premise_id") == subject]
        resting = [f"claim {c['claim_id']}: {c.get('detail')}" for c in claims.values()
                   if subject in (c.get("circular_premises") or [])]
        parts = [f"premise {subject} is circular: {c.get('why')}" for c in found] + resting
        return "; ".join(parts) or None
    for premise in premises:
        if premise.get("id") == subject:
            evidence = premise.get("evidence")
            return (f"premise {subject} ({premise.get('status')}): {premise.get('text')}"
                    + (f"; evidence: {evidence}" if evidence else ""))
    return None


def _reduction_explanation(critic: dict, reasons: list[str], head: str) -> str:
    """The deciding argument of an S2 reduction: the known form the critic mapped the card onto, the numeric check's
    verdict lines and the critic's argument, in full. (The critic's first sentence, used for kill_detail, can be about
    something else: for an escape card it opens with the escape test.)"""
    checks = [r for r in reasons if r.startswith("reduction claim")]
    parts = [head, f"Mapping: {critic.get('mapping') or '(none)'}"]
    if checks:
        parts.append("Numeric check: " + " ".join(checks))
    parts.append(f"Critic's argument: {critic['why']}")
    return " ".join(parts)


def _reduction_summary(card: dict, critic: dict) -> str:
    """kill_detail of a reduction: the first sentence that says what the card is. An escape card's critic argument
    opens with the escape test, so its summary comes from the mapping."""
    if card.get("non_reduction_argument") and critic.get("mapping"):
        return reduction_summary(critic["mapping"])
    return reduction_summary(critic["why"])


def _difference_explanation(view: RunView, card: dict, full: str) -> str:
    """A death by a rejected difference statement (Phase 7): the gate line, then every failure of the last attempt
    (host problems and the challenger's findings), each in full."""
    found = view.latest_artifact(card["id"], "difference")
    if found is None:
        return full
    attempt = found[0]
    failures = "; ".join(f"{key}: {value}" for key, value in attempt["failures"].items())
    return f"{full} Failures of attempt {attempt['attempt']}: {failures or 'none recorded'}"


def _s2_death(view: RunView, card: dict, verdict: str, reasons: list[str], detail: str) -> Death:
    kinds = STAGE_EVIDENCE["S2"]
    found = view.latest_artifact(card["id"], "reduction_critic")
    critic = found[0] if found else None
    gate = _gate(reasons)
    full = deciding_reason(reasons)
    if gate.startswith("gate: excluded_family:"):
        code = check_kill_code(GATE_CODE.match(gate).group(1))
        if critic is None:
            return _single(code, detail, kinds, full)
        head = (f"{code}: a verified reduction to {critic['reduces_to']} into a family out of scope "
                f"(families.toml); a LEARNING_RULE: difference does not apply.")
        return _single(code, detail, kinds, _reduction_explanation(critic, reasons, head))
    if verdict == "reduces_to_known":
        if critic is None:
            raise ValueError(f"{view.run_id}/{card['id']}: reduces_to_known without a reduction_critic artifact")
        family = critic["reduces_to"]
        summary = clip(f"reduced to {family}: {_reduction_summary(card, critic)}", DETAIL_CHARS)
        return _single(f"reduces_to_known:{family}", summary, kinds,
                       _reduction_explanation(critic, reasons, f"reduced to {family}."))
    if verdict == "undetermined":
        if any("novelty unassessed" in r for r in reasons):
            return _single("novelty_unassessed", detail, kinds, full)
        return _single("undetermined:S2", detail, kinds, full)
    if gate.startswith("gate: known_disproven"):
        return _single("known_disproven", detail, kinds, full)
    if gate.startswith("gate: prior_art_tested"):
        return _single("no_difference", detail, kinds, full)
    if gate.startswith("gate: no_difference"):
        return _single("no_difference", detail, kinds, _difference_explanation(view, card, full))
    if gate.startswith("gate: not radical enough"):
        # The verdict line clips the critic's argument; the critic artifact holds all of it.
        why = critic["why"] if critic is not None else gate.removeprefix("gate: not radical enough:").strip()
        return _single("not_radical:R2", detail, kinds, f"not radical enough (reduction critic): {why}")
    if gate.startswith("gate: not_radical:R3"):
        return _single("not_radical:R3", detail, kinds, _difference_explanation(view, card, full))
    raise ValueError(f"{view.run_id}/{card['id']}: S2 {verdict} with an unknown gate reason: {gate[:120]!r}")


def _s3_death(view: RunView, card: dict, verdict: str, reasons: list[str], detail: str) -> Death:
    kinds = STAGE_EVIDENCE["S3"]
    full = deciding_reason(reasons)
    if verdict == "undetermined":
        return _single("undetermined:S3", detail, kinds, full)
    gate = view.latest_artifact(card["id"], "s3_gate")
    rule = gate[0]["rule"] if gate else None
    if rule == "check_violation":
        # One code per violated obligation, each explained by the reason that names that obligation.
        explanations: dict[str, str] = {}
        for reason in reasons:
            match = VIOLATED.search(reason)
            if match:
                code = f"obligation_failed:{_slug(match.group(1))}"
                explanations[code] = bounded(f"{explanations[code]} {reason}" if code in explanations else reason)
        if not explanations:
            raise ValueError(f"{view.run_id}/{card['id']}: S3 check_violation without a violated obligation reason")
        return Death(list(explanations), detail, kinds, explanations)
    if rule == "disproof" or (rule is None and any(r.startswith("confirmed disproof") for r in reasons)):
        disproofs = [_full_disproof(view, card["id"], r) for r in reasons if r.startswith("confirmed disproof")]
        return _single("infeasible:disproof", detail, kinds, "; ".join(disproofs) or full)
    return _single("infeasible:judges", detail, kinds, full)


def _legacy_disproof_reason(attack: dict) -> str:
    """The verdict reason S3 wrote for a confirmed attack before FIX-12, with the target cut at 200 characters and the
    counterexample at 400."""
    return f"confirmed disproof of '{attack['target_claim'][:200]}': {attack['counterexample'][:400]}"


def _full_disproof(view: RunView, card_id: str, reason: str) -> str:
    """FIX-12, F2: a confirmed-disproof verdict reason with the attack's whole target and counterexample, recovered from
    the card's `disproof` artifacts in this run (the reason S3 wrote before FIX-12 cut both). A reason that matches no
    recorded confirmed attack of this run (an override run whose S3 verdict was copied from its source run) is kept as
    recorded, with a marker saying so."""
    for record, _ in view.all_artifacts(card_id, "disproof"):
        for attack in record["attacks"]:
            if attack["confirmed"] and reason in (disproof_reason(attack), _legacy_disproof_reason(attack)):
                return disproof_reason(attack)
    return (f"{reason} [verdict reason as recorded; no confirmed attack in this run's disproof artifacts matches it, "
            f"so its target and counterexample may be cut at 200 and 400 characters]")


def lineage_root(view: RunView, card_id: str) -> str:
    seen = {card_id}
    current = card_id
    while True:
        card, _, _ = view.latest_card(current)
        parents = [p for p in card["parent_ids"] if p in view.versions and p not in seen]
        if not parents:
            return current
        current = parents[0]
        seen.add(current)


def evidence_refs(view: RunView, card_id: str, kinds: tuple[str, ...], by_calls: list[str]) -> list[dict]:
    refs = []
    for kind in kinds:
        found = view.latest_artifact(card_id, kind)
        if found is not None:
            refs.append({"kind": kind, "ref": f"{view.run_id}/{found[1]['path']}", "sha256": found[1]["sha256"]})
    for ident in by_calls:
        refs.append({"kind": "call" if ident.startswith("call-") else "job", "ref": f"{view.run_id}/{ident}"})
    return refs


def premise_ledger(view: RunView, card_id: str) -> tuple[list[dict], list[dict]]:
    """(premises, definitions) from the card's latest assumption_ledger artifact (stages M and C). A record written
    before definitions existed holds one `assumptions` list, which is returned as the premises. A card that never
    reached M has the list its refinement dialogue wrote (Phase 7, `dialogue_assumptions`), when it had one."""
    found = view.latest_artifact(card_id, "assumption_ledger")
    if found is None:
        dialogue = view.latest_artifact(card_id, "dialogue_assumptions")
        if dialogue is not None:
            return list(dialogue[0]["premises"]), list(dialogue[0]["definitions"])
        return [], []
    ledger = found[0]
    return list(ledger.get("premises", ledger.get("assumptions", []))), list(ledger.get("definitions", []))


def cost(view: RunView, card_id: str) -> dict:
    """Model calls, their tokens and executor jobs named in the card's verdicts (every version)."""
    ids = {i for verdict in view.verdicts.get(card_id, []) for i in verdict["by_calls"]}
    calls = [i for i in ids if i.startswith("call-")]
    return {"model_calls": len(calls), "tokens": sum(view.tokens.get(i, 0) for i in calls),
            "executor_jobs": len(ids) - len(calls)}


def build_record(view: RunView, card_id: str, written_by: str) -> dict:
    card, entry, raw = view.latest_card(card_id)
    outcome, stage, death = classify(view, card)
    verdict_entry = card["status"]["verdicts"].get(stage) or {"by_calls": []}
    evidence = evidence_refs(view, card_id, death.kinds if death else STAGE_EVIDENCE.get(stage, ()),
                             verdict_entry["by_calls"])
    codes = [check_kill_code(code) for code in (death.codes if death else [])]
    causes = [{"code": code, "detail": death.explanations[code], "evidence": evidence} for code in codes]
    premises, definitions = premise_ledger(view, card_id)
    difference = card.get("difference_statement")
    missing = [name for name, value in (("difference_statement", difference), ("premises", premises),
                                        ("definitions", definitions), ("commitments", card.get("commitments")),
                                        ("arm", card.get("arm")), ("archive_cell", card.get("archive_cell")))
               if not value]
    created = view.created.get(card_id, {})
    record = {
        "run": view.run_id, "card_id": card_id, "card_version": entry["version"], "card_sha256": _sha256(raw),
        "card_path": f"{view.run_id}/{entry['path']}", "written_by": written_by,
        "legacy": "difference_statement" in missing or "premises" in missing, "missing": missing,
        "override": view.override is not None,
        "source_run": (view.override or {}).get("source_run_id"),
        "lineage_id": view.lineages.get(card_id) or lineage_root(view, card_id),
        "parent_ids": list(card["parent_ids"]),
        "operator": created.get("op"), "arm": card.get("arm"), "archive_cell": card.get("archive_cell"),
        "lens": card.get("lens"), "title": card["title"], "core_idea": card.get("core_idea"),
        "summary": card["summary"], "mechanism_signature": card_signature(card), "difference_statement": difference,
        "non_reduction_argument": card.get("non_reduction_argument"), "premises": premises,
        "definitions": definitions,
        "stage_verdicts": {s: {"verdict": v["verdict"], "reasons": [clip(r, REASON_CHARS) for r in v["reasons"][:6]],
                               "by_calls": list(v["by_calls"])}
                           for s, v in card["status"]["verdicts"].items()},
        "terminal_stage": stage, "outcome": outcome, "kill_codes": codes,
        "kill_detail": death.detail if death else None, "causes": causes, "evidence": evidence,
        "cost": cost(view, card_id), "budget": lineage.card_budget(view.events, card_id),
    }
    coupled = target_coupled(view, card_id)
    if coupled is not None:
        # FIX-17; left out when there is none, so records of earlier runs rebuild unchanged.
        record["target_coupled"] = coupled
    return record


def target_coupled(view: RunView, card_id: str) -> dict | None:
    """FIX-17: the skill scale at target from the card's stage M math_verdict (None without one, or before FIX-17)."""
    found = view.math_verdict(card_id, "M")
    block = found[0].get("target_coupled") if found else None
    if not block:
        return None
    return {"status": block["status"], "applied": bool(block.get("applied")),
            "slots": {slot: {"growth": entry["growth"], "b": entry.get("b"), "scale": entry["scale"]}
                      for slot, entry in (block.get("slots") or {}).items()}}


@dataclass
class BackfillReport:
    runs: list[str] = field(default_factory=list)
    cards: int = 0
    added: int = 0
    skipped: int = 0
    records: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        from collections import Counter

        by_outcome = Counter(r["outcome"] for r in self.records)
        by_stage = Counter(f"{r['terminal_stage']} {r['outcome']}" for r in self.records)
        by_family = Counter(kill_family(code) for r in self.records for code in r["kill_codes"])
        by_code = Counter(code for r in self.records for code in r["kill_codes"])
        return {"runs": self.runs, "cards": self.cards, "added": self.added, "skipped": self.skipped,
                "legacy": sum(1 for r in self.records if r["legacy"]),
                "override": sum(1 for r in self.records if r["override"]),
                "by_outcome": dict(by_outcome), "by_stage": dict(sorted(by_stage.items())),
                "by_kill_family": dict(by_family.most_common()), "by_kill_code": dict(by_code.most_common())}


def run_dirs(runs_root: Path) -> list[Path]:
    """Run directories that hold cards: a ledger.jsonl and a cards/ directory."""
    return [d for d in sorted(Path(runs_root).iterdir())
            if d.is_dir() and (d / "ledger.jsonl").is_file() and (d / "cards").is_dir()]


def backfill(runs_root: Path, store: OutcomeStore) -> BackfillReport:
    """Import every card of every run under runs_root, read-only. Records already stored (same run, card and card
    file hash) are skipped, so the backfill can run again after a run moves on."""
    report = BackfillReport()
    for run_dir in run_dirs(runs_root):
        view = RunView(run_dir)
        on_disk = sorted(p.name for p in (run_dir / "cards").iterdir() if p.is_dir())
        if on_disk != view.card_ids():
            raise ValueError(f"{run_dir.name}: card directories {on_disk} differ from the ledger's {view.card_ids()}")
        report.runs.append(view.run_id)
        for card_id in view.card_ids():
            record = build_record(view, card_id, "backfill")
            report.cards += 1
            report.records.append(record)
            if store.append(record) is None:
                report.skipped += 1
            else:
                report.added += 1
    return report


def record_for_live_card(run_dir: Path, card_id: str, store: OutcomeStore) -> dict | None:
    """The orchestrator's write at a card's terminal verdict: the record for one card of the run it is writing."""
    return store.append(build_record(RunView(run_dir), card_id, "orchestrator"))
