"""Pure parts of the Lean check: statement hashing, the source scan, file assembly and the verdict.

Contract (DISCOVERY-PIPELINE-PLAN.md sections 4.1 and 4.2):

* The statement file holds imports and ``def Claim : Prop := ...``. It is frozen before any prover call; its
  whitespace-normalized sha256 is fixed in advance and must match, so a restated statement file fails.
* The proof file supplies ``theorem claim_holds : Claim := ...`` and may add helper lemmas. It has no imports.
* The checker builds one file: ``import Lean``, the statement, the proof, then a trailer it writes itself with a
  fresh nonce. The trailer re-checks ``claim_holds`` against ``Claim`` in the kernel under a name the prover
  cannot know, prints ``#print axioms claim_holds``, and prints the facts the verdict needs (declaration kind,
  whether the elaborated type is exactly the constant ``Claim``, the axioms) tagged with the nonce.
* Pass only when Lean exits 0 with no error, ``claim_holds`` is a theorem whose type is exactly ``Claim``, and
  every axiom is one of propext, Classical.choice, Quot.sound. sorryAx, native evaluation (Lean.ofReduceBool,
  Lean.trustCompiler, native_decide's auxiliary axioms) and any user axiom reject.

The source scan rejects, before anything runs, constructs whose effect ``#print axioms`` cannot see: unsafe code,
implemented_by and extern, metaprogramming commands (#eval, run_cmd, elab, macro, syntax, notation and the like),
scope changes (namespace, section, end) and set_option outside a short allowlist.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

THEOREM = "claim_holds"
CLAIM = "Claim"
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
NATIVE_AXIOMS = frozenset({"Lean.ofReduceBool", "Lean.trustCompiler"})

# Commands and attributes whose effect is invisible to #print axioms, or that change how names in the trailer
# resolve. Checked on tokens after comments and string literals are removed.
FORBIDDEN_TOKENS = frozenset({
    "unsafe", "implemented_by", "extern", "export",
    "#eval", "#exit", "#reduce", "eval%", "run_cmd", "run_elab", "run_meta", "run_tac",
    "elab", "elab_rules", "macro", "macro_rules", "syntax", "declare_syntax_cat",
    "notation", "infix", "infixl", "infixr", "prefix", "postfix",
    "initialize", "builtin_initialize", "namespace", "section", "end", "import",
    "debug.skipKernelTC",
})
SAFE_OPTIONS = re.compile(
    r"(maxHeartbeats|maxRecDepth|synthInstance\.maxHeartbeats|synthInstance\.maxSize|autoImplicit|"
    r"relaxedAutoImplicit|linter\.[\w.]+|pp\.[\w.]+)")
TOKEN = re.compile(r"#?[A-Za-z_][\w.'!?]*%?")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# FIX-18: Lean 4 keywords (reserved tokens of the core grammar and of the commands above) that cannot name a bound
# variable. A statement that binds one does not parse, so the statement lint names it with a rename hint.
LEAN_KEYWORDS = frozenset({
    "fun", "let", "have", "show", "from", "at", "by", "do", "then", "else", "if", "match", "with", "in", "where",
    "deriving", "open", "end", "section", "namespace", "prefix", "infix", "infixl", "infixr", "postfix", "notation",
    "macro", "syntax", "variable", "universe", "theorem", "lemma", "def", "example", "instance", "structure",
    "class", "inductive", "axiom", "import", "export", "mutual", "local", "private", "protected", "noncomputable",
    "partial", "unsafe", "calc", "suffices", "nomatch", "nofun", "forall", "abbrev", "opaque", "attribute",
    "extends", "return", "for", "unless", "mut", "Type", "Sort", "Prop",
})
# Binder positions: the names after a quantifier or lambda up to the binder's ':' ',' or arrow, the names before
# ':' in a bracketed binder, and a let-bound name.
_QUANTIFIED = re.compile(r"(?:∀|∃!|∃|∑|∏|\bfun\b|λ|Σ|Π)\s+([^,:=↦]*)")
_BRACKETED = re.compile(r"[(\{⦃\[]\s*((?:[A-Za-z_][\w'!?]*\s+)*[A-Za-z_][\w'!?]*)\s*:(?!=)")
_LET = re.compile(r"\blet\s+([A-Za-z_][\w'!?]*)")


def keyword_binders(source: str) -> list[str]:
    """The Lean 4 keywords a source uses as bound names, in order of first use (comments and strings removed)."""
    code = strip_comments_and_strings(source)
    found: list[str] = []
    for pattern in (_QUANTIFIED, _BRACKETED, _LET):
        for match in pattern.finditer(code):
            for name in IDENT.findall(match.group(1)):
                if name in LEAN_KEYWORDS and name not in found:
                    found.append(name)
    return found


def normalize_statement(text: str) -> str:
    """Collapse every run of whitespace to one space and trim, so only layout changes keep the hash."""
    return " ".join(text.split())


def statement_hash(text: str) -> str:
    return hashlib.sha256(normalize_statement(text).encode("utf-8")).hexdigest()


def strip_comments_and_strings(source: str) -> str:
    """Replace Lean comments (``--`` and nested ``/- -/``) and string literals with spaces."""
    out, i, n = [], 0, len(source)
    while i < n:
        if source.startswith("--", i):
            end = source.find("\n", i)
            i = n if end == -1 else end
        elif source.startswith("/-", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if source.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif source.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif source[i] == '"':
            i += 1
            while i < n and source[i] != '"':
                i += 2 if source[i] == "\\" else 1
            i += 1
            out.append(' "" ')
        else:
            out.append(source[i])
            i += 1
    return "".join(out)


def scan(source: str, *, allow_imports: bool) -> list[str]:
    """Problems with a statement or proof source, in order of appearance. Empty means it may run."""
    code = strip_comments_and_strings(source)
    problems = []
    tokens = TOKEN.findall(code)
    for index, token in enumerate(tokens):
        if token == "import" and allow_imports:
            continue
        if token in FORBIDDEN_TOKENS:
            problems.append(f"forbidden construct {token!r}")
        if token == "set_option":
            option = tokens[index + 1] if index + 1 < len(tokens) else ""
            if not SAFE_OPTIONS.fullmatch(option):
                problems.append(f"set_option {option!r} is not on the allowlist")
    return list(dict.fromkeys(problems))


@dataclass(frozen=True)
class LeanCheck:
    """One Lean obligation: the frozen statement, its hash fixed in advance, and the prover's proof."""
    statement: str
    statement_sha256: str
    proof: str

    def problems(self) -> list[str]:
        """Reasons to reject before running anything: statement hash mismatch, then the source scans."""
        found = []
        actual = statement_hash(self.statement)
        if actual != self.statement_sha256:
            found.append(f"statement hash mismatch: expected {self.statement_sha256}, got {actual}")
        found += ["statement: " + p for p in scan(self.statement, allow_imports=True)]
        found += ["proof: " + p for p in scan(self.proof, allow_imports=False)]
        return found


def trailer(nonce: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16,64}", nonce):
        raise ValueError(f"nonce must be lowercase hex, got {nonce!r}")
    return f"""
-- Bigbrane checker trailer (written by the checker, not the prover).
theorem bigbrane_check_{nonce} : {CLAIM} := {THEOREM}

#print axioms {THEOREM}

open Lean Elab Command in
#eval show CommandElabM Unit from do
  let env ← getEnv
  let thm := env.find? `{THEOREM}
  let kind := match thm with
    | some (.thmInfo _) => "theorem"
    | some (.axiomInfo _) => "axiom"
    | some (.defnInfo _) => "def"
    | some _ => "other"
    | none => "missing"
  let exact := match thm with
    | some info => info.type == mkConst `{CLAIM}
    | none => false
  let claimProp := match env.find? `{CLAIM} with
    | some (.defnInfo d) => d.type == mkSort Level.zero
    | _ => false
  let axs ← if thm.isSome then collectAxioms `{THEOREM} else pure #[]
  logInfo s!"BIGBRANE_FACTS {nonce} kind={{kind}} exact={{exact}} claim_prop={{claimProp}} axioms={{axs.toList}}"
"""


def assemble(check: LeanCheck, nonce: str) -> str:
    """The file Lean elaborates. ``import Lean`` comes first so the trailer can read the environment."""
    return ("import Lean\n" + check.statement.rstrip() + "\n\n-- proof\n" + check.proof.rstrip() + "\n"
            + trailer(nonce))


_FACTS = re.compile(r"BIGBRANE_FACTS (?P<nonce>[0-9a-f]+) kind=(?P<kind>\w+) exact=(?P<exact>true|false) "
                    r"claim_prop=(?P<claim_prop>true|false) axioms=\[(?P<axioms>[^\]]*)\]")
_PRINT_AXIOMS = re.compile(r"'" + THEOREM + r"' (?:depends on axioms: \[(?P<axioms>[^\]]*)\]"
                           r"|(?P<none>does not depend on any axioms))", re.S)
_ERROR = re.compile(r"^\S*:\d+:\d+: error", re.M)


def _names(text: str) -> list[str]:
    return [name.strip() for name in text.replace("\n", " ").split(",") if name.strip()]


def parse_output(stdout: str, nonce: str) -> dict:
    """Facts from Lean's output. Only a facts line carrying this run's nonce counts."""
    facts = None
    for match in _FACTS.finditer(stdout):
        if match.group("nonce") == nonce:
            facts = match
    printed = None
    match = _PRINT_AXIOMS.search(stdout)
    if match:
        printed = [] if match.group("none") else _names(match.group("axioms"))
    return {
        "errors": len(_ERROR.findall(stdout)),
        "facts_found": facts is not None,
        "kind": facts.group("kind") if facts else None,
        "exact_type": facts.group("exact") == "true" if facts else None,
        "claim_is_prop": facts.group("claim_prop") == "true" if facts else None,
        "axioms": sorted(_names(facts.group("axioms"))) if facts else None,
        "printed_axioms": sorted(printed) if printed is not None else None,
    }


def classify_axioms(axioms: list[str]) -> list[str]:
    """Rejection reasons for an axiom list; empty when every axiom is standard."""
    reasons = []
    for name in axioms:
        if name in STANDARD_AXIOMS:
            continue
        if name == "sorryAx":
            reasons.append("uses sorry (sorryAx)")
        elif name in NATIVE_AXIOMS or "native" in name.lower():
            reasons.append(f"uses native evaluation ({name})")
        else:
            reasons.append(f"depends on a non-standard axiom ({name})")
    return reasons


def verdict(exit_code: int, parsed: dict) -> tuple[str, list[str]]:
    """("pass" | "reject", reasons) for one Lean run."""
    reasons = []
    if exit_code != 0 or parsed["errors"]:
        reasons.append(f"Lean reported errors (exit {exit_code}, {parsed['errors']} error messages)")
    if not parsed["facts_found"]:
        reasons.append("the checker trailer produced no facts")
        return "reject", reasons
    if parsed["kind"] != "theorem":
        reasons.append(f"{THEOREM} is {parsed['kind']}, not a theorem")
    if not parsed["claim_is_prop"]:
        reasons.append(f"{CLAIM} is not a definition of type Prop")
    if not parsed["exact_type"]:
        reasons.append(f"the type of {THEOREM} is not exactly the constant {CLAIM} (restated claim)")
    if parsed["printed_axioms"] is not None and parsed["printed_axioms"] != parsed["axioms"]:
        reasons.append(f"#print axioms {parsed['printed_axioms']} disagrees with the environment {parsed['axioms']}")
    if parsed["printed_axioms"] is None:
        reasons.append("no #print axioms output")
    reasons += classify_axioms(parsed["axioms"])
    return ("reject" if reasons else "pass"), reasons
