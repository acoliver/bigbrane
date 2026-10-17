"""Family exclusion and the difference statement at S2 (DISCOVERY-PIPELINE-PLAN.md sections 1.2, 1.3 and 8; Phase 7).

Family exclusion (`s2.family_exclusion`). hybrid/config/families.toml lists the out-of-scope families (decision D8:
transformer, state-space, linear-attention and delta-rule) and the known families (Hopfield-style associative
memories, exact-match caches). The reduction critic is asked, in a section appended to its user message
(prompt_sections/family_exclusion.md) and two fields added to its schema in code, which family the card's core belongs
to (`family`, `family_why`). A verified reduction into an out-of-scope family fails the card with
`excluded_family:<id>` whatever its `difference_if_any` says: the `LEARNING_RULE:` prefix does not apply.
KD4: the family is decided from the comparators the host counted (the ones that verified the reduction, each with the
family its declaration stated), never from the critic's `family` label, which stays recorded as information. The
families.toml `reduces_to` lists map a counted comparator's family to the out-of-scope families that contain it; a
counted family in none of them (other_known, rnn) gives no excluded kill and the card takes the ordinary
verified-reduction path. When two out-of-scope families share the counted family (linear attention and the delta rule
both list `linear_attention`), the critic's `family` picks between them when it is one of the two, and otherwise the
one whose `keywords` occur most often in the counted comparators and the critic's mapping and argument is chosen (the
first listed on a tie). A reduction verified on the critic's confidence alone has no counted comparator, so the
claimed `reduces_to` stands as the counted family. So a renamed linear-attention variant that the critic reduces to
`linear_attention` dies even when the critic, misled by the new names, answers `family: none`, and a Minsky counter
machine counted as other_known does not die because the critic also argued an uncounted ssm simulation.

Difference statement (`s2.difference_statement`). Written by the proposer (the card's explorer role, prompt
`difference`, schema schemas/difference.schema.json) after the other S2 gates pass, then checked:
- on the host: every `closest_prior` entry names a source id of this card's prior-art search (`literature`), a record
  of its own-history matches (`outcome`) or a family of families.toml (`family`); at least min(2, available)
  literature and outcome entries; `changed_equation` points at an update or readout equation of the card or at its
  learning rule; with `outcomes.enforce_r3`, every R3 target (a dead record with the card's exact signature or a
  signature within Jaccard `outcomes.jaccard_min`) is listed and all its cause ids are in `cause_ids_addressed`; when
  the critic named a family (out of scope or known), it is listed as a `family` entry;
- by the challenger role (`refine_challenger`, or `refine_challenger_alt` for a card a GPT model drafted, so the
  checker is not the proposer's family; prompt `difference_check`), shown every recorded cause in full (each cause's
  own deciding explanation, bounded by outcome_records.CAUSE_CHARS; FIX-10): whether the statement is accepted, for
  each R3 cause whether the changed equation is the one the cause names, whether the difference from a named family changes
  an equation of its form, and the specific difference a send-back must ask for.
A statement with any failure gets one send-back (step 4) through `BudgetBook.after_round` with loop `send_back`, which
also takes an idea slot of the card's lineage. The proposer answers once, with the required difference quoted, and
the answer is judged the same way. A second failure, or a send-back the budget refuses, drops the card: with an R3
failure (a cause not cited or not addressed) the code is `not_radical:R3`, otherwise `no_difference`. The last
statement is stored on the card (`difference_statement`), so the outcome record carries it either way.

Artifacts: `difference` (each attempt: statement, host problems, the check, failures, accepted), `difference_check`
inside it. Events: `difference_judged`.
"""
from __future__ import annotations

import copy
import json
import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from hybrid.research import outcomes as outcome_store
from hybrid.research.archive import Families, Family, load_archive_config, load_families
from hybrid.research.schemas import load_schema, validate
from hybrid.research.stages.common import StageContext, card_for_prompt, load_prompt, load_section

ROOT = Path(__file__).resolve().parents[3]
LOOP = "send_back"
WRITER_PROMPT = "difference"
CHECK_PROMPT = "difference_check"
R3_PREFIX = "r3"


# ----- families ------------------------------------------------------------------------------------------------------


def _path(value: str) -> Path:
    return Path(value) if Path(value).is_absolute() else ROOT / value


@lru_cache(maxsize=None)
def _families(families_path: str, archive_path: str) -> tuple[Families, dict[str, tuple[str, ...]]]:
    families = load_families(_path(families_path), load_archive_config(_path(archive_path)))
    with open(_path(families_path), "rb") as handle:
        data = tomllib.load(handle)
    keywords = {entry["id"]: tuple(k.lower() for k in entry.get("keywords", ()))
                for entry in data.get("excluded", []) + data.get("known", [])}
    return families, keywords


def families_for(ctx: StageContext) -> tuple[Families, dict[str, tuple[str, ...]]]:
    """The families of hybrid/config/families.toml (generation.families_config), with each family's keywords."""
    g = ctx.cfg["generation"]
    return _families(g["families_config"], g["archive_config"])


def critic_schema(families: Families) -> dict:
    """The reduction critic's schema with the family answer added (the stored schema file is unchanged)."""
    schema = copy.deepcopy(load_schema("reduction_critic"))
    schema.pop("$id", None)
    schema["properties"]["family"] = {"enum": ["none", *families.ids]}
    schema["properties"]["family_why"] = {"type": "string", "minLength": 1}
    schema["required"] = [*schema["required"], "family", "family_why"]
    return schema


def critic_section(families: Families) -> str:
    lines = [load_section("family_exclusion"), "OUT OF SCOPE FAMILIES:"]
    lines += [f"- {f.id}: {f.name}. Form: {f.form}" for f in families.excluded]
    lines.append("KNOWN FAMILIES:")
    lines += [f"- {f.id}: {f.name}. Form: {f.form}" for f in families.known]
    return "\n\n" + "\n".join(lines)


def excluded_reduction(critic: dict, reduction: str, families: Families, keywords: dict[str, tuple[str, ...]],
                       counted: list[dict]) -> Family | None:
    """The out-of-scope family a verified reduction falls in, or None (no verified reduction, or no counted comparator
    in an out-of-scope family).

    KD4: the family comes from the comparators the host counted (`counted`, from s2_prior_art.counted_comparisons: each
    has the `family` its declaration stated and the `comparator` text), never from the critic's free `family` label.
    The candidates are the out-of-scope families whose `reduces_to` lists a counted comparator's family, in the order
    of families.toml, so any counted comparator in an out-of-scope family decides and a counted family outside every
    one of them (other_known, rnn, ...) gives no kill whatever the critic says.

    When several candidates share the counted family (linear attention and the delta rule both list
    `linear_attention`), the critic's `family` picks between them when it is one of the candidates; the label cannot
    add a family. Otherwise the candidate whose `keywords` occur most often in the counted comparators and the critic's
    mapping and argument is chosen, the first listed on a tie."""
    if reduction != "verified":
        return None
    counted_families = {c["family"] for c in counted}
    candidates = [f for f in families.excluded if counted_families.intersection(f.reduces_to)]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    named = next((f for f in candidates if f.id == critic.get("family")), None)
    if named is not None:
        return named
    text = " ".join([*(f"{c.get('declared_id') or ''} {c['comparator']}".replace("_", " ") for c in counted),
                     critic.get("mapping") or "", critic["why"], critic["difference_if_any"]]).lower()
    scored = [(keyword_hits(text, keywords.get(f.id, ())), -index, f) for index, f in enumerate(candidates)]
    return max(scored, key=lambda item: item[:2])[2]


def keyword_hits(text: str, words: tuple[str, ...]) -> int:
    """Occurrences of the keywords as whole words or phrases (so "rls" does not count inside "worlds")."""
    return sum(len(re.findall(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text)) for word in words)


def named_family(critic: dict, families: Families) -> Family | None:
    """The family the critic placed the core in (out of scope or known), when it named one."""
    named = critic.get("family")
    return families.get(named) if named and named != "none" else None


# ----- requirements and host checks ----------------------------------------------------------------------------------


@dataclass
class Requirements:
    literature: dict[str, str]                 # accepted literature ref -> title
    outcomes: dict[str, dict]                  # record id -> match dict
    r3: list[dict]                             # {"record_id", "card_id", "exact", "jaccard", "causes": [{id, detail}]}
    family: Family | None
    need: int
    families: Families

    def cause_ids(self) -> list[str]:
        return [cause["id"] for target in self.r3 for cause in target["causes"]]


def literature_refs(record: dict, pa_result: dict) -> dict[str, str]:
    """Source ids the prior-art search returned for this card (evidence ids, candidate ids with and without their
    source prefix), each with a title when known."""
    refs: dict[str, str] = {}
    for candidate in pa_result.get("candidates", []):
        refs[candidate] = ""
        if ":" in candidate:
            refs[candidate.split(":", 1)[1]] = ""
    for item in record["evidence"]:
        refs[item["source_id"]] = item["title"]
    return refs


def requirements(ctx: StageContext, record: dict, pa_result: dict, matches: list, critic: dict,
                 families: Families) -> Requirements:
    literature = literature_refs(record, pa_result)
    outcomes = {m.stored.record_id: m.as_dict() for m in matches}
    r3 = []
    if ctx.outcomes is not None and ctx.cfg["outcomes"]["enforce_r3"]:
        for match in outcome_store.r3_targets(matches, ctx.cfg["outcomes"]["jaccard_min"]):
            r3.append({"record_id": match.stored.record_id, "card_id": match.stored.record["card_id"],
                       "exact": match.exact, "jaccard": round(match.jaccard, 4),
                       "causes": [{"id": c["id"], "detail": c["detail"]} for c in match.stored.causes()]})
    need = min(2, len({ref.split(":", 1)[-1] for ref in literature}) + len(outcomes))
    return Requirements(literature, outcomes, r3, named_family(critic, families), need, families)


def resolves(card: dict, ref: str) -> bool:
    if ref == "learning":
        return True
    kind, _, rest = ref.partition("[")
    index = rest.rstrip("]")
    return kind in ("update", "readout") and index.isdigit() and int(index) < len(card["mechanism"][kind])


def host_problems(statement: dict, card: dict, req: Requirements) -> dict[str, str]:
    """Failures the host can see without a model, keyed for stall and send-back records."""
    problems: dict[str, str] = {}
    cited_records, families = set(), set()
    counted = set()
    for item in statement["closest_prior"]:
        kind, ref = item["kind"], item["ref"]
        if kind == "literature" and ref in req.literature:
            counted.add(("literature", ref.split(":", 1)[-1]))
        elif kind == "outcome" and ref in req.outcomes:
            counted.add(("outcome", ref))
            cited_records.add(ref)
        elif kind == "family" and req.families.get(ref) is not None:
            families.add(ref)
        else:
            problems[f"unknown_ref:{kind}:{ref}"] = (f"closest_prior names {kind} {ref!r}, which is not a source of "
                                                    f"this card's prior-art search, one of its own-history records or "
                                                    f"a family of families.toml")
    if len(counted) < req.need:
        problems["closest_prior:count"] = (f"{len(counted)} literature or outcome entries with receipts; at least "
                                           f"{req.need} are required")
    if not resolves(card, statement["changed_equation"]):
        problems["changed_equation"] = (f"changed_equation {statement['changed_equation']} is not an equation of the "
                                        f"card's mechanism")
    addressed = set(statement["cause_ids_addressed"])
    for target in req.r3:
        if target["record_id"] not in cited_records:
            problems[f"{R3_PREFIX}_prior:{target['record_id']}"] = (
                f"dead idea {target['card_id']} ({target['record_id']}) matches this card's signature and is not "
                f"listed in closest_prior")
        for cause in target["causes"]:
            if cause["id"] not in addressed:
                problems[f"{R3_PREFIX}:{cause['id']}"] = (f"recorded cause {cause['id']} of {target['card_id']} is not "
                                                          f"in cause_ids_addressed: {cause['detail'][:300]}")
    if req.family is not None and req.family.id not in families:
        problems[f"family:{req.family.id}"] = (f"the reduction critic places the core in {req.family.id} "
                                               f"({req.family.name}); list it as a family entry and state what your "
                                               f"core changes in its form")
    return problems


# ----- model calls ---------------------------------------------------------------------------------------------------


def writer_schema() -> dict:
    schema = copy.deepcopy(load_schema("difference"))
    schema.pop("$id", None)
    return schema


def check_schema(cause_ids: list[str], family: Family | None) -> dict:
    causes = ({"type": "array", "minItems": len(cause_ids), "maxItems": len(cause_ids),
               "items": {"type": "object", "additionalProperties": False, "required": ["cause_id", "addressed", "why"],
                         "properties": {"cause_id": {"enum": cause_ids}, "addressed": {"type": "boolean"},
                                        "why": {"type": "string", "minLength": 1}}}}
              if cause_ids else {"type": "array", "maxItems": 0})
    real = {"type": "boolean"} if family is not None else {"type": "null"}
    return {"type": "object", "additionalProperties": False,
            "required": ["accepted", "problems", "causes", "family_difference_real", "family_why",
                         "required_difference"],
            "properties": {"accepted": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}},
                           "causes": causes, "family_difference_real": real, "family_why": {"type": "string"},
                           "required_difference": {"type": "string"}}}


def context_block(record: dict, req: Requirements, critic: dict) -> str:
    lit = [f"- {ref}: {title or '(title not recorded)'}" for ref, title in req.literature.items() if ":" not in ref]
    lit += [f"- {ref}" for ref in req.literature if ":" in ref]
    parts = ["PRIOR-ART SEARCH (cite literature entries by these source ids):\n"
             + (f"label {record['label']}\n" + ("\n".join(lit) or "(no source returned)")),
             "OWN HISTORY (outcome-store records; cite them by record id, and causes by cause id):\n"
             + ("\n".join(f"- {rid}: {m['card_id']} {m['title'][:80]} ({'exact' if m['exact_signature'] else 'near'}"
                          f" signature, Jaccard {m['jaccard']}), {m['outcome']} at {m['terminal_stage']}, kill codes "
                          f"{', '.join(m['kill_codes']) or 'none'}"
                          + "".join(f"\n    cause {c['id']}: {c['detail']}" for c in m["causes"])
                          for rid, m in req.outcomes.items()) or "(no match)"),
             "REDUCTION CRITIC:\n" + json.dumps({key: critic.get(key) for key in
                                                 ("reduces_to", "family", "family_why", "mapping", "difference_if_any",
                                                  "why")}, indent=2, ensure_ascii=False)]
    rules = [f"- at least {req.need} closest_prior entries of kind literature or outcome, with the ids above"]
    if req.r3:
        rules.append("- RULE R3: these dead ideas match the card's mechanism signature. List each as an outcome entry, "
                     "put every one of its cause ids in cause_ids_addressed, and make changed_equation the equation "
                     "that answers the cause:")
        rules += [f"  - {t['record_id']} ({t['card_id']}): causes {', '.join(c['id'] for c in t['causes'])}"
                  for t in req.r3]
    if req.family is not None:
        rules.append(f"- the critic places the core in the family {req.family.id} ({req.family.name}; form: "
                     f"{req.family.form}). List it as a family entry and change an equation of that form, not its "
                     f"vocabulary, parameters or application.")
    parts.append("REQUIREMENTS:\n" + "\n".join(rules))
    return "\n\n".join(parts)


async def write(ctx: StageContext, card: dict, role: str, context: str, send_back: str | None) -> tuple[dict, str]:
    parts = ["CARD:\n" + card_for_prompt(card), context]
    if send_back is not None:
        parts.append("SEND-BACK (your first statement was rejected; this is your one chance to answer it):\n"
                     + send_back)
    parts.append("Return the difference statement as JSON.")
    reply = await ctx.call(role, load_prompt(WRITER_PROMPT), "\n\n".join(parts), writer_schema())
    statement = reply.parsed
    validate("difference", statement)
    return statement, reply.call_id


async def check(ctx: StageContext, card: dict, role: str, context: str, statement: dict, req: Requirements,
                problems: dict[str, str]) -> tuple[dict, str]:
    user = "\n\n".join(["CARD:\n" + card_for_prompt(card), context,
                        "DIFFERENCE STATEMENT:\n" + json.dumps(statement, indent=2, ensure_ascii=False),
                        "HOST CHECK PROBLEMS:\n" + ("\n".join(f"- {v}" for v in problems.values()) or "none"),
                        "Return your judgment as JSON."])
    reply = await ctx.call(role, load_prompt(CHECK_PROMPT), user, check_schema(req.cause_ids(), req.family))
    return reply.parsed, reply.call_id


def failures_of(problems: dict[str, str], verdict: dict, req: Requirements) -> dict[str, str]:
    failures = dict(problems)
    for cause in verdict["causes"]:
        if not cause["addressed"]:
            failures.setdefault(f"{R3_PREFIX}:{cause['cause_id']}", f"the challenger finds the cause not addressed: "
                                                                    f"{cause['why'][:300]}")
    if req.family is not None and verdict["family_difference_real"] is not True:
        failures.setdefault(f"family:{req.family.id}", f"no real difference from {req.family.id}: "
                                                       f"{verdict['family_why'][:300]}")
    if not verdict["accepted"] and not failures:
        failures["difference_check"] = "; ".join(verdict["problems"])[:600] or "rejected without a stated problem"
    return failures


@dataclass
class Attempt:
    statement: dict
    failures: dict[str, str]
    verdict: dict
    calls: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return not self.failures and self.verdict["accepted"]

    @property
    def r3_failed(self) -> bool:
        return any(key.split(":", 1)[0] in (R3_PREFIX, f"{R3_PREFIX}_prior") for key in self.failures)

    def send_back_text(self) -> str:
        lines = [self.verdict["required_difference"] or "state precisely what your core changes"]
        lines += [f"- {detail}" for detail in self.failures.values()]
        return "\n".join(lines)


async def attempt(ctx: StageContext, card: dict, number: int, roles: tuple[str, str], context: str,
                  req: Requirements, send_back: str | None) -> Attempt:
    statement, write_call = await write(ctx, card, roles[0], context, send_back)
    problems = host_problems(statement, card, req)
    verdict, check_call = await check(ctx, card, roles[1], context, statement, req, problems)
    result = Attempt(statement, failures_of(problems, verdict, req), verdict, [write_call, check_call])
    ctx.put_json_artifact("difference", ctx.artifact_name(card["id"]),
                          {"card_id": card["id"], "attempt": number, "send_back": send_back, "statement": statement,
                           "host_problems": problems, "check": verdict, "failures": result.failures,
                           "accepted": result.accepted, "r3_targets": req.r3,
                           "family": req.family.id if req.family else None, "calls": result.calls}, card["id"])
    ctx.emit("difference_judged", {"card_id": card["id"], "attempt": number, "accepted": result.accepted,
                                   "failures": sorted(result.failures), "r3_targets": [t["record_id"] for t in req.r3],
                                   "family": req.family.id if req.family else None, "calls": result.calls})
    return result


def store_statement(ctx: StageContext, card_id: str, statement: dict, call_id: str) -> None:
    card = ctx.store.get_card(card_id)
    updated = dict(card, difference_statement=statement, version=ctx.store.next_version(card_id),
                   created_by_call=call_id)
    validate("theory_card", updated)
    ctx.store.put_card(updated)


async def gate(ctx: StageContext, card: dict, *, record: dict, pa_result: dict, matches: list, critic: dict,
               families: Families, roles: tuple[str, str]) -> tuple[str | None, list[str], list[str]]:
    """Steps 3 and 4: the difference statement, its checks and one send-back. Returns (kill code or None, reasons,
    call ids)."""
    req = requirements(ctx, record, pa_result, matches, critic, families)
    context = context_block(record, req, critic)
    first = await attempt(ctx, card, 1, roles, context, req, None)
    calls, reasons = list(first.calls), []
    final = first
    if not first.accepted:
        stop = ctx.book.after_round(card_id=card["id"], stage="S2", loop=LOOP, round_id="difference-1",
                                    failures=first.failures)
        reasons.append("difference statement rejected: " + "; ".join(first.failures.values())[:800])
        if stop is None:
            reasons.append("sent back once with the required difference: " + first.send_back_text()[:600])
            final = await attempt(ctx, ctx.card(card["id"]), 2, roles, context, req, first.send_back_text())
            calls += final.calls
        else:
            reasons.append(stop.line())
    store_statement(ctx, card["id"], final.statement, final.calls[0])
    if final.accepted:
        reasons.append(f"difference statement accepted{' after the send-back' if final is not first else ''}: "
                       f"{final.statement['precise_change'][:400]} (changed {final.statement['changed_equation']}; "
                       f"causes addressed: {', '.join(final.statement['cause_ids_addressed']) or 'none required'})")
        return None, reasons, calls
    code = "not_radical:R3" if final.r3_failed else "no_difference"
    if final is not first:
        reasons.append("send-back answer rejected: " + "; ".join(final.failures.values())[:800])
    return code, reasons, calls
