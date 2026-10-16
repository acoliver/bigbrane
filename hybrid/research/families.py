"""Model families and the separation rules of the math roles (DISCOVERY-PIPELINE-PLAN.md section 16).

A family is the lab whose model answered: anthropic (Opus, Sonnet, Haiku), openai (the GPT and ChatGPT routes,
the llxprt profiles gpt-* and terra), qwen and google (local LM Studio models). An id the table does not know is
its own family, so an unknown model never matches another role by accident.

Rules for one card (section 5 and the routing table of section 16):
- the statement fixer and the prover are in different families;
- the challenger and the prover are in different families;
- the statement-match checker differs from both the statement fixer and the prover.
FIX-7: the statement-match cross-check (gateway Sonnet) reads the matcher's anchored objections. It must differ from
the matcher (a second reading from the same family is not a second reading) and from the fixer that wrote the
statements. It may share the prover's family: it is supplementary and never replaces the third-family matcher.
FIX-17 (decision XC): the statement tiebreak settles a dispute between the matcher and the cross-check, so it differs
from both. It is the gateway ChatGPT model, which may share the fixer's family: the matcher, which must differ from
the fixer, has already read the statement.
The challenger should also differ from the card's proposer; with three families that is not always possible, so it
is reported, not enforced.
"""
from __future__ import annotations

ANTHROPIC_TOKENS = ("claude", "opus", "sonnet", "haiku")
OPENAI_TOKENS = ("gpt", "chatgpt", "openai", "terra", "codex")

# (role a, role b): the two roles must not share a family.
SEPARATED = (("statement_fixer", "prover"), ("challenger", "prover"), ("statement_match", "statement_fixer"),
             ("statement_match", "prover"), ("statement_crosscheck", "statement_match"),
             ("statement_crosscheck", "statement_fixer"), ("statement_tiebreak", "statement_match"),
             ("statement_tiebreak", "statement_crosscheck"))


def family(model: str | None) -> str:
    """The family of a model id as reported by a backend or written in models.toml."""
    if not model:
        return "unknown"
    name = model.lower()
    if any(token in name for token in ANTHROPIC_TOKENS):
        return "anthropic"
    if any(token in name for token in OPENAI_TOKENS):
        return "openai"
    if "qwen" in name:
        return "qwen"
    if "gemma" in name or "gemini" in name:
        return "google"
    return name


def violations(models: dict[str, set[str] | list[str] | str]) -> list[str]:
    """Breaches of SEPARATED among the models each role actually used ({role: model ids})."""
    families = {role: {family(m) for m in ([used] if isinstance(used, str) else used)}
                for role, used in models.items()}
    found = []
    for a, b in SEPARATED:
        shared = families.get(a, set()) & families.get(b, set())
        if shared:
            found.append(f"{a} and {b} share the {', '.join(sorted(shared))} family")
    return found
