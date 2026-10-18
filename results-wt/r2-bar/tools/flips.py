"""Lists, for each S2 flip, what the reduction critic answered about known work, the sources it weighed and the
difference statement the checker accepted. Reads runs/*/artifacts; prints Markdown (written to FLIPS.md by the caller)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLIPS = ["card-3485a13f", "card-f5ee7edb", "card-8154d3b4", "card-a4f684fb", "card-89248ee5"]


def artifacts(card_id, kind):
    found = []
    for path in (ROOT / "runs").glob(f"*/artifacts/{kind}/{card_id}-*.json"):
        found.append((path.stat().st_mtime, path))
    return [json.loads(p.read_text()) | {"_path": str(p.relative_to(ROOT))} for _, p in sorted(found)]


def cut(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


out = []
for card_id in FLIPS:
    critic = artifacts(card_id, "reduction_critic")[-1]
    accepted = [d for d in artifacts(card_id, "difference") if d["accepted"]][-1]
    prior = artifacts(card_id, "prior_art")[-1]
    statement = accepted["statement"]
    out += [f"### {card_id}", "",
            f"- Prior-art label: {prior['label']}. Critic: reduces_to {critic['reduces_to']}, known_work "
            f"{critic['known_work']}, surprise {critic['surprise']}, textbook_transplant {critic['textbook_transplant']}.",
            f"- Critic's known_work answer: {cut(critic['why'], 700)}"]
    if critic["known_work_sources"]:
        out.append("- Sources it weighed:")
        for s in critic["known_work_sources"]:
            out.append(f"  - {s['kind']} {cut(s['ref'], 160)}: {cut(s['covers'], 330)}")
    else:
        out.append("- Sources it weighed: none listed.")
    if critic["fixed_learned_parts"]:
        out.append("- Resemblances that held only with a learned part fixed: " + "; ".join(
            f"{cut(p['part'], 140)} (capability: {cut(p['capability'], 120)})" for p in critic["fixed_learned_parts"]))
    out.append(f"- Difference statement accepted (attempt {accepted['attempt']}, changed equation "
               f"{statement['changed_equation']}): {cut(statement['precise_change'], 800)}")
    out.append("- Closest prior entries in the statement: " + "; ".join(
        f"{e['kind']} {cut(e['ref'], 60)}" for e in statement["closest_prior"]))
    if accepted["check"].get("excerpt_gaps"):
        out.append("- Claims the checker scoped as unsupported by the excerpts: " + " | ".join(
            cut(g, 200) for g in accepted["check"]["excerpt_gaps"]))
    out += [f"- Artifacts: {critic['_path']}; {accepted['_path']}", ""]
print("\n".join(out))
