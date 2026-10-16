"""Tool tiers of DISCOVERY-PIPELINE-PLAN.md section 4.2 and the wording rule for verdicts.

A tier says how far a claim was checked. Only ``lean`` and ``sympy_exact`` may be reported as proven,
``smt_bounded`` as proven up to its bound, and ``executed`` as observed on the listed sizes. ``argued`` never
satisfies an obligation. Self-contained (stdlib only) so it runs in the tools image as part of ``bbmath``.
"""
from __future__ import annotations

LEAN = "lean"
SMT_BOUNDED = "smt_bounded"
SYMPY_EXACT = "sympy_exact"
EXECUTED = "executed"
ARGUED = "argued"
TIERS = (LEAN, SMT_BOUNDED, SYMPY_EXACT, EXECUTED, ARGUED)
PROVEN_TIERS = frozenset({LEAN, SYMPY_EXACT})


def satisfies_obligation(tier: str) -> bool:
    """Whether a result at this tier can discharge a profile obligation (section 4.2: argued never does)."""
    _require(tier)
    return tier != ARGUED


def verdict_phrase(tier: str, *, bound: str | None = None, sizes: list | tuple | None = None) -> str:
    """The words a verdict may use for a claim checked at ``tier``."""
    _require(tier)
    if tier in PROVEN_TIERS:
        return "proven"
    if tier == SMT_BOUNDED:
        if not bound:
            raise ValueError("an smt_bounded result must state its bound")
        return f"proven up to bound {bound}"
    if tier == EXECUTED:
        if not sizes:
            raise ValueError("an executed result must list the sizes it was observed on")
        return "observed on the listed sizes: " + ", ".join(str(s) for s in sizes)
    return "argued in prose only; does not satisfy an obligation"


def _require(tier: str) -> None:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
