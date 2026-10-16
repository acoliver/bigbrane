"""Pure parts of the Lean check. Lean output samples are copied from the Phase 1 smoke run in Podman
(results/phase1-20261012T172040Z/smoke1/cases.json)."""
import pytest

from hybrid.math import lean_check as lc

NONCE = "0123456789abcdef0123456789abcdef"
STATEMENT = "def Claim : Prop := ∀ n : Nat, n + 0 = n\n"


def facts(kind="theorem", exact="true", axioms="", nonce=NONCE):
    return f"BIGBRANE_FACTS {nonce} kind={kind} exact={exact} claim_prop=true axioms=[{axioms}]"


def printed(axioms):
    if not axioms:
        return "'claim_holds' does not depend on any axioms"
    return f"'claim_holds' depends on axioms: [{axioms}]"


# ----- statement hash ------------------------------------------------------------------------

def test_hash_ignores_whitespace_layout_only():
    reflowed = "def  Claim :\n  Prop :=\n\t∀ n : Nat,   n + 0 = n"
    assert lc.statement_hash(reflowed) == lc.statement_hash(STATEMENT)
    assert lc.statement_hash("def Claim : Prop := ∀ n : Nat, 0 + n = n") != lc.statement_hash(STATEMENT)


def test_restated_statement_file_is_rejected_before_running():
    check = lc.LeanCheck("def Claim : Prop := ∀ n : Nat, n + 0 = n ∨ True", lc.statement_hash(STATEMENT),
                         "theorem claim_holds : Claim := fun _ => Or.inr trivial")
    [problem] = check.problems()
    assert problem.startswith("statement hash mismatch")


def test_clean_check_has_no_problems():
    check = lc.LeanCheck(STATEMENT, lc.statement_hash(STATEMENT), "theorem claim_holds : Claim := fun _ => rfl")
    assert check.problems() == []


# ----- source scan ---------------------------------------------------------------------------

@pytest.mark.parametrize("proof, needle", [
    ("unsafe def f : Nat := 0\ntheorem claim_holds : Claim := sorry", "'unsafe'"),
    ("@[implemented_by g] def f : Nat := 0", "'implemented_by'"),
    ("@[extern \"c_f\"] opaque f : Nat", "'extern'"),
    ("#eval 1", "'#eval'"),
    ("run_cmd pure ()", "'run_cmd'"),
    ("elab \"x\" : term => pure default", "'elab'"),
    ("macro \"x\" : term => `(1)", "'macro'"),
    ("notation \"Claim\" => True", "'notation'"),
    ("infixl:65 \" +++ \" => Nat.add", "'infixl'"),
    ("namespace Evil\ndef Claim : Prop := True\nend Evil", "'namespace'"),
    ("section\nend", "'section'"),
    ("import Mathlib", "'import'"),
    ("export Nat (succ)", "'export'"),
    ("set_option debug.skipKernelTC true in\ntheorem claim_holds : Claim := sorry", "debug.skipKernelTC"),
    ("set_option compiler.extract_closed false", "'compiler.extract_closed' is not on the allowlist"),
])
def test_scan_rejects_constructs_axioms_cannot_see(proof, needle):
    problems = lc.scan(proof, allow_imports=False)
    assert any(needle in p for p in problems), problems


@pytest.mark.parametrize("proof", [
    "theorem claim_holds : Claim := by\n  intro n\n  simp",
    "set_option maxHeartbeats 400000 in\ntheorem claim_holds : Claim := by decide",
    "set_option linter.unusedVariables false\ntheorem claim_holds : Claim := fun _ => rfl",
    "-- unsafe namespace #eval in a comment\n/- nested /- unsafe -/ elab -/\ntheorem claim_holds : Claim := fun _ => rfl",
    "theorem claim_holds : Claim := by\n  have h : \"unsafe #eval\".length = 12 := rfl\n  exact fun _ => rfl",
    "open Nat in\ntheorem claim_holds : Claim := fun n => Nat.add_zero n",
    "theorem claim_holds : Claim := by\n  sorry",                 # rejected later by #print axioms
    "axiom magic : Claim\ntheorem claim_holds : Claim := magic",  # rejected later by #print axioms
])
def test_scan_allows_ordinary_proofs(proof):
    assert lc.scan(proof, allow_imports=False) == []


def test_statement_may_import():
    assert lc.scan("import Mathlib\n\ndef Claim : Prop := True", allow_imports=True) == []


# ----- assembly ------------------------------------------------------------------------------

def test_assembled_file_has_import_statement_proof_and_nonce_trailer():
    check = lc.LeanCheck("import Mathlib\n" + STATEMENT, "x", "theorem claim_holds : Claim := fun _ => rfl")
    text = lc.assemble(check, NONCE)
    assert text.startswith("import Lean\nimport Mathlib\n")
    assert text.index("def Claim") < text.index("theorem claim_holds") < text.index("bigbrane_check_" + NONCE)
    assert f"theorem bigbrane_check_{NONCE} : Claim := claim_holds" in text
    assert "#print axioms claim_holds" in text and f"BIGBRANE_FACTS {NONCE}" in text


def test_trailer_rejects_bad_nonce():
    with pytest.raises(ValueError):
        lc.trailer("not-hex")


# ----- parsing and verdicts ------------------------------------------------------------------

def run(stdout, exit_code=0):
    parsed = lc.parse_output(stdout, NONCE)
    return lc.verdict(exit_code, parsed), parsed


def test_small_lemma_passes_with_no_axioms():
    (status, reasons), parsed = run(printed("") + "\n" + facts())
    assert (status, reasons) == ("pass", []) and parsed["axioms"] == []


def test_mathlib_lemma_passes_with_standard_axioms():
    axioms = "propext, Classical.choice, Quot.sound"
    (status, reasons), parsed = run(printed(axioms) + "\n" + facts(axioms=axioms))
    assert status == "pass" and parsed["axioms"] == ["Classical.choice", "Quot.sound", "propext"]


@pytest.mark.parametrize("axioms, reason", [
    ("sorryAx", "uses sorry (sorryAx)"),
    ("bigbrane_magic", "non-standard axiom (bigbrane_magic)"),
    ("claim_holds._native.native_decide.ax_1_1", "native evaluation"),
    ("propext, Lean.ofReduceBool", "native evaluation (Lean.ofReduceBool)"),
    ("Lean.trustCompiler", "native evaluation (Lean.trustCompiler)"),
])
def test_non_standard_axioms_reject(axioms, reason):
    out = "/in/Check.lean:5:8: warning: declaration uses `sorry`\n" + printed(axioms) + "\n" + facts(axioms=axioms)
    (status, reasons), _ = run(out)
    assert status == "reject" and any(reason in r for r in reasons), reasons


def test_restated_claim_in_proof_rejects():
    out = ("/in/Check.lean:8:67: error: Type mismatch\n  claim_holds\nhas type\n  ∀ (n : Nat), n + 0 = n ∨ True\n"
           "but is expected to have type\n  Claim\n" + printed("") + "\n" + facts(exact="false"))
    (status, reasons), parsed = run(out, exit_code=1)
    assert status == "reject" and parsed["errors"] == 1
    assert any("not exactly the constant Claim" in r for r in reasons)


def test_facts_with_another_nonce_are_ignored():
    forged = facts(nonce="f" * 32)
    (status, reasons), parsed = run(printed("") + "\n" + forged)
    assert status == "reject" and not parsed["facts_found"]
    assert "the checker trailer produced no facts" in reasons


def test_print_axioms_must_agree_with_environment():
    (status, reasons), _ = run(printed("propext") + "\n" + facts(axioms=""))
    assert status == "reject" and any("disagrees" in r for r in reasons)


def test_non_theorem_rejects():
    (status, reasons), _ = run(printed("") + "\n" + facts(kind="axiom"))
    assert status == "reject" and "claim_holds is axiom, not a theorem" in reasons


def test_compile_error_rejects_even_with_clean_facts():
    out = "/in/Check.lean:3:0: error: unknown identifier 'x'\n" + printed("") + "\n" + facts()
    (status, reasons), _ = run(out, exit_code=1)
    assert status == "reject" and reasons[0].startswith("Lean reported errors")


def test_multiline_print_axioms_is_parsed():
    out = "'claim_holds' depends on axioms: [propext,\n Classical.choice,\n Quot.sound]\n" + facts(
        axioms="propext, Classical.choice, Quot.sound")
    (status, _), parsed = run(out)
    assert status == "pass" and parsed["printed_axioms"] == parsed["axioms"]
