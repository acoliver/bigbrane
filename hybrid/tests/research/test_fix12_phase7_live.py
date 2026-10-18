"""FIX-12: the S2 reduction check is bound to the critic's declared repair, and evidence defects F1 to F3 of the second
Phase 7 live check (results/phase7-live-20261014T2051Z/REPORT.md).

Labelling. The check prompt counted only published forms, so the matching repaired forms of card-1184f7a7 (linear
attention with a per-channel phase normalizer, difference 0) and card-537e161c (MesaNet with a token-selected mixture
over powers, 1.97e-13) were labelled `rewrite` and their family exclusions became not_radical:R2. The critic now
declares its comparators (id, base family and form, repaired equations, transformation types from the repair policy,
parameter mapping, limit-only status); the check reports against those ids; the host refuses omitted, undeclared or
repeated ids and metadata disagreements, and derives eligibility from the declared types.

The recorded check programs (runs/phase7-live-20261014T2051Z/artifacts/executor, copied to
fixtures/phase7_live_20261014T2051Z/ with sha256 in SHA256SUMS) are rerun with their last statement replaced by a
report of their own measured numbers against the declaration the critic's mapping states.

F1. The critic prompt clipped every own-history cause at 400 characters.
F2. Outcome records rebuilt S3 disproof causes from verdict reasons that S3 cut at 200 and 400 characters.
F3. The critic could say no equation changed although stage R changed one; it now sees the before and after equations
with statuses."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.research.outcome_records import RunView, build_record
from hybrid.research.outcomes import OutcomeStore, own_history, render_history
from hybrid.research.schemas import SchemaViolation, load_schema, validate
from hybrid.research.signature import card_signature
from hybrid.research.stages import difference, s2_prior_art
from hybrid.research.stages.common import load_section
from hybrid.research.stages.contracts import REPAIR_TYPES, UNRESOLVED_TYPE, repair_policy_text
from hybrid.research.stages.s2_prior_art import (ATTEMPTED, UNCHANGED, VALIDATED, eligibility, equation_changes,
                                                 judge_comparisons, render_changes, report_problem)
from hybrid.research.stages.s3_feasibility import disproof_reason
from hybrid.tests.research.critic_answers import novelty_fields
from hybrid.tests.research.test_orchestrator_fakes import (comparison, declared, make_orchestrator, orch_config,
                                                           program)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase7_live_20261014T2051Z"
FIX10_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase7_live_20261014T1752Z"
LIVE_RUN = "phase7-live-20261014T2051Z"
CRITIC_KEYS = ("reduces_to", "mapping", "confidence", "difference_if_any", "why",
               "textbook_transplant", "textbook_why", "family", "family_why")


def load(name, base=FIXTURES):
    return json.loads((base / name).read_text(encoding="utf-8"))


def recorded_critic(name: str, comparators: list[dict], **changes) -> dict:
    record = load(name)
    return {key: record[key] for key in CRITIC_KEYS} | novelty_fields(record) | {"comparators": comparators} | changes


def recorded_program(job: str, marker: str, tail: str, base=FIXTURES) -> str:
    code = load(job, base)["files"]["main.py"]
    assert code.count(marker) == 1, marker
    return code.split(marker, 1)[0] + tail


def report_tail(rows: list[tuple], notes: str) -> str:
    """Program text that prints a comparison report. Each row: (declared_id or None, family, types, limit_only,
    comparator, inputs expression, difference expression)."""
    items = ",\n".join(
        f"    {{'declared_id': {d!r}, 'comparator': {c!r}, 'family': {f!r}, 'transformations': {list(t)!r}, "
        f"'limit_only': {lim!r}, 'reference': {c!r}, 'inputs_tested': {n}, 'max_abs_diff': {x}}}"
        for d, f, t, lim, c, n, x in rows)
    return f"\nprint(json.dumps({{'comparisons': [\n{items}], 'notes': {notes!r}}}))\n"


def run_program(code: str, tmp_path: Path) -> dict:
    path = tmp_path / "main.py"
    path.write_text(code, encoding="utf-8")
    done = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=100, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def transform(kind: str, what: str) -> dict:
    return {"type": kind, "what": what}


# ----- the declarations the recorded critics' mappings state ---------------------------------------------------------

LA_MAP = ("phi(v) = E_v = exp(i Theta_v); k_t = conj(E_{x_{t-1}}); v_t = E_{x_t}; q_t = E_{x_t}; state = diagonal of "
          "S; logits = u + beta Re(conj(E) y_t)/d")
LA_PUBLISHED = {"id": "la_published", "base_family": "linear_attention",
                "base_form": "causal linear attention (Katharopoulos et al. 2020), diagonal state, count normalizer",
                "update": "S_t = S_{t-1} + conj(E_{x_{t-1}}) * E_{x_t}", "read": "y_t = (S_t * E_{x_t}) / (t - 1)",
                "transformations": [], "parameter_mapping": LA_MAP, "limit_only": False}
LA_PHASE = {"id": "la_phase_normalized", "base_family": "linear_attention",
            "base_form": "causal linear attention (Katharopoulos et al. 2020), diagonal state",
            "update": "S_t = S_{t-1} + conj(E_{x_{t-1}}) * E_{x_t}",
            "read": "y_t = N(S_t) * E_{x_t}, N(z)_w = z_w / |z_w| (0 below tau)",
            "transformations": [transform("normalizer", "per-channel modulus normalization of the comparator's own "
                                                        "diagonal state before the read")],
            "parameter_mapping": LA_MAP, "limit_only": False}

MESA_MAP = ("k_t = Phi[x_{t-1}], v_t = Phi[x_t], beta_t = sigmoid(w_beta^T [k_t; v_t]), q_t = Phi[x_t], "
            "S_t = C_t^T (lambda I + G_t)^{-1}, alpha = softmax(A[x_{t-1}]), W_o the output projection")
MESA_SINGLE = {"id": "mesanet_single", "base_family": "linear_attention",
               "base_form": "MesaNet (von Oswald et al. 2025), closed-form in-context least squares",
               "update": "G_t = G_{t-1} + beta_t k_t k_t^T, C_t = C_{t-1} + beta_t k_t v_t^T, "
                         "S_t = C_t^T (lambda I + G_t)^{-1}",
               "read": "y_t = S_t q_t", "transformations": [], "parameter_mapping": MESA_MAP, "limit_only": False}
MESA_HOPS = dict(MESA_SINGLE, id="mesanet_hop_mixture", read="y_t = sum_{m=0..M} alpha_m(x_{t-1}) S_t^m q_t",
                 transformations=[transform("multi_read", "repeated reads of the same MesaNet state, mixed by "
                                                          "weights selected by the previous token, including the "
                                                          "zero-hop identity term")])

BIMAP_MAP = "key w_{t-1} = i, value w_t = j, entry weight C_ij, cap P_max, fallback over values not yet owned"
BIMAP_HARD = {"id": "hard_bimap", "base_family": "other_known",
              "base_form": "bidirectional map (bimap) insert with inverse uniqueness, as a capped exact-match cache",
              "update": "map[i] := j, delete other keys of j and other values of i; skip when both are owned",
              "read": "p(v) = 1 for the value bound to q, leftover mass over unowned values",
              "transformations": [], "parameter_mapping": BIMAP_MAP, "limit_only": True}
BIMAP_WEIGHTED = dict(BIMAP_HARD, id="weighted_bimap", limit_only=False,
                      update="insert with entry weight rho_ij, conflicting entries deleted, min-weight eviction",
                      transformations=[transform("gate", "per-entry confidence weight rho_ij on insert")])

# The recorded programs end by printing their report; everything before computes the numbers.
LA_MARK, MESA_MARK, BIMAP_MARK = "out = {", "comps = [{", "res = {"


def la_tail(normalized_id="la_phase_normalized", normalized_types=("normalizer",), omit_normalized=False):
    rows = [("la_published", "linear_attention", (), False, "linear attention, count normalizer", "n_seq",
             "diff_claimed")]
    if not omit_normalized:
        rows.append((normalized_id, "linear_attention", normalized_types, False,
                     "linear attention with the per-channel phase normalizer", "n_seq", "diff_rewrite"))
    return report_tail(rows, "recorded job-e284e0a16e43 reporting its own numbers")


def mesa_tail(hop_types=("multi_read",)):
    rows = [("mesanet_single", "linear_attention", (), False, "MesaNet single read", "T", "max_claimed"),
            (None, "linear_attention", ("multi_read",), False, "MesaNet fixed 2-hop read", "T", "max_hops[2]"),
            (None, "linear_attention", ("multi_read",), False, "MesaNet fixed 3-hop read", "T", "max_hops[3]"),
            (None, "linear_attention", ("multi_read",), False, "MesaNet fixed 4-hop read", "T", "max_hops[4]"),
            ("mesanet_hop_mixture", "linear_attention", hop_types, False,
             "MesaNet state with the token-selected mixture over powers", "T", "max_rewrite")]
    return report_tail(rows, "recorded job-720f18f6a55a reporting its own numbers")


BIMAP_TAIL = report_tail([("hard_bimap", "other_known", (), True, "hard capped bimap cache", "N_SEQ", "d_hard"),
                          ("weighted_bimap", "other_known", ("gate",), False, "confidence-weighted bimap cache",
                           "N_SEQ", "d_soft")], "recorded job-bf095b15bbd6 reporting its own numbers")


def critic_1184(**changes):
    return recorded_critic("reduction_critic-card-1184f7a7-v7-ca6d21.json", [LA_PUBLISHED, LA_PHASE], **changes)


def critic_537(hops=MESA_HOPS, **changes):
    return recorded_critic("reduction_critic-card-537e161c-v7-c566e6.json", [MESA_SINGLE, hops], **changes)


def critic_c96(**changes):
    return recorded_critic("reduction_critic-card-c9615069-v7-7c4e1f.json", [BIMAP_HARD, BIMAP_WEIGHTED], **changes)


# ----- schema and policy ---------------------------------------------------------------------------------------------


def test_the_declarations_validate_and_the_schema_vocabulary_is_the_policy():
    schema = load_schema("reduction_critic")
    types = schema["properties"]["comparators"]["items"]["properties"]["transformations"]["items"]["properties"]
    assert types["type"]["enum"] == [*REPAIR_TYPES, UNRESOLVED_TYPE]
    for critic in (critic_1184(), critic_537(), critic_c96()):
        validate("reduction_critic", {k: v for k, v in critic.items() if k not in ("family", "family_why")})
    # a reduction without a declaration is refused, and so is a declaration under reduces_to "none"
    bare = {k: v for k, v in critic_1184().items() if k not in ("family", "family_why", "comparators")}
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", bare)
    with pytest.raises(SchemaViolation):
        validate("reduction_critic", dict(bare, reduces_to="none", mapping=None, comparators=[LA_PUBLISHED]))
    text = repair_policy_text()
    assert all(f"- {name}:" in text for name in [*REPAIR_TYPES, UNRESOLVED_TYPE])
    assert "normalizer" in load_section("family_exclusion") and "REPAIR POLICY" in load_section("family_exclusion")


@pytest.mark.parametrize("comparator, status", [
    (LA_PHASE, "counted"), (LA_PUBLISHED, "counted"), (MESA_HOPS, "counted"), (BIMAP_HARD, "limit_only"),
    (dict(MESA_HOPS, transformations=[transform("other", "keys snapped to the nearest stored row")]), "unresolved"),
    (dict(LA_PHASE, transformations=[*LA_PHASE["transformations"], transform("other", "x")]), "unresolved"),
    (dict(LA_PHASE, base_family="ssm"), "another_family"),
])
def test_eligibility_comes_from_the_declared_types(comparator, status):
    assert eligibility(comparator, "linear_attention" if comparator["base_family"] != "other_known"
                       else "other_known") == status


def test_the_check_request_carries_the_family_policy():
    ctx = SimpleNamespace(cfg=orch_config())
    block = s2_prior_art.family_policy_block(ctx, "linear_attention")
    assert "- linear_attention:" in block and "- delta_rule:" in block and "with or without decay or a normalizer" in block
    assert "REPAIR POLICY" in block and "- multi_read:" in block


# ----- the recorded checks, reported against the declarations ---------------------------------------------------------


def test_1184f7a7_the_declared_phase_normalizer_verifies_linear_attention(tmp_path):
    printed = run_program(recorded_program("job-e284e0a16e43.json", LA_MARK, la_tail()), tmp_path)
    critic = critic_1184()
    assert report_problem(printed, critic["comparators"]) is None
    published, normalized = printed["comparisons"]
    assert 0.76 < published["max_abs_diff"] < 0.77 and normalized["max_abs_diff"] == 0.0     # as recorded
    outcome, reasons, judged = judge_comparisons(printed["comparisons"], critic["comparators"], "linear_attention")
    assert outcome == "verified" and "la_phase_normalized" in reasons[0]
    assert [c["eligibility"] for c in judged] == ["counted", "counted"]


def test_537e161c_the_declared_hop_mixture_verifies_the_mesanet_reduction(tmp_path):
    printed = run_program(recorded_program("job-720f18f6a55a.json", MESA_MARK, mesa_tail()), tmp_path)
    critic = critic_537()
    assert report_problem(printed, critic["comparators"]) is None
    mixture = printed["comparisons"][-1]
    assert mixture["declared_id"] == "mesanet_hop_mixture" and mixture["max_abs_diff"] < 1e-11      # 1.97e-13 recorded
    outcome, reasons, judged = judge_comparisons(printed["comparisons"], critic["comparators"], "linear_attention")
    assert outcome == "verified" and "mesanet_hop_mixture" in reasons[0]
    assert [c["eligibility"] for c in judged] == ["counted", "undeclared", "undeclared", "undeclared", "counted"]


def test_c9615069_the_hard_and_weighted_bimap_stay_refuted(tmp_path):
    printed = run_program(recorded_program("job-bf095b15bbd6.json", BIMAP_MARK, BIMAP_TAIL), tmp_path)
    critic = critic_c96()
    assert report_problem(printed, critic["comparators"]) is None
    hard, weighted = printed["comparisons"]
    assert 0.08 < hard["max_abs_diff"] < 0.095 and 0.07 < weighted["max_abs_diff"] < 0.085      # 0.087 and 0.079
    outcome, reasons, judged = judge_comparisons(printed["comparisons"], critic["comparators"], "other_known")
    assert outcome == "refuted" and reasons[0].startswith("numeric check rejects the claimed reduction")
    assert [c["eligibility"] for c in judged] == ["limit_only", "counted"]


def test_the_fix10_correction_term_still_cannot_verify_the_hard_bimap(tmp_path):
    """The first live check's soft rewrite with a bespoke -rho*x*y correction matched to 1e-16. Reported under no
    declared id it never counts; declared as a repair it can only have type `other`, so it stays unresolved."""
    tail = report_tail([(None, "other_known", (), False, "soft delta rule with the -rho*x*y correction", 300, "maxd"),
                        ("hard_bimap", "other_known", (), False, "hard bimap", 300, "bd")], "job-137fb29ee378")
    code = recorded_program("job-137fb29ee378.json", "equiv = maxd < 1e-6", tail, FIX10_FIXTURES)
    printed = run_program(code, tmp_path)
    hard = dict(BIMAP_HARD, limit_only=False)
    assert judge_comparisons(printed["comparisons"], [hard], "other_known")[0] == "refuted"
    corrected = dict(hard, id="soft_corrected",
                     transformations=[transform("other", "a -rho*x*y term at (i, j) that cancels the difference")])
    relabelled = [dict(printed["comparisons"][0], declared_id="soft_corrected", transformations=["other"]),
                  printed["comparisons"][1]]
    assert report_problem({"comparisons": relabelled, "notes": ""}, [corrected, hard]) is None
    outcome, reasons, _ = judge_comparisons(relabelled, [corrected, hard], "other_known")
    assert outcome == "unverified" and "unresolved" in " ".join(reasons)


@pytest.mark.parametrize("tail, phrase", [
    (la_tail(normalized_id="la_phase_v2"), "undeclared id(s) la_phase_v2"),            # substituted id
    (la_tail(omit_normalized=True), "la_phase_normalized not reported"),                 # the declared repair omitted
    (la_tail(normalized_types=()), "disagrees with its declaration on transformations"),  # repair reported as bare form
])
def test_an_undeclared_omitted_or_substituted_comparison_is_refused(tmp_path, tail, phrase):
    printed = run_program(recorded_program("job-e284e0a16e43.json", LA_MARK, tail), tmp_path)
    assert phrase in report_problem(printed, critic_1184()["comparators"])


def test_the_report_must_repeat_family_and_limit_only():
    both = [BIMAP_HARD, BIMAP_WEIGHTED]
    only_hard = {"comparisons": [comparison("other_known", 0.1, "hard_bimap", limit_only=True)], "notes": ""}
    assert report_problem(only_hard, [BIMAP_HARD]) is None
    assert "weighted_bimap not reported" in report_problem(only_hard, both)
    rows = [comparison("other_known", 0.1, "hard_bimap", limit_only=True),
            comparison("other_known", 0.1, "weighted_bimap", transformations=("gate",))]
    assert report_problem({"comparisons": rows, "notes": ""}, both) is None
    assert "limit_only" in report_problem({"comparisons": [dict(rows[0], limit_only=False), rows[1]], "notes": ""},
                                          both)
    assert "family" in report_problem({"comparisons": [dict(rows[0], family="ssm"), rows[1]], "notes": ""}, both)


def test_a_limit_only_match_does_not_verify():
    outcome, reasons, _ = judge_comparisons([comparison("other_known", 0.0, "hard_bimap", limit_only=True)],
                                            [BIMAP_HARD], "other_known")
    assert outcome == "unverified" and "limit-only" in reasons[0]


def test_a_disallowed_transformation_type_stays_unresolved(tmp_path):
    snapped = dict(MESA_HOPS, transformations=[transform("other", "keys snapped to the nearest codebook row of "
                                                                  "the state")])
    printed = run_program(recorded_program("job-720f18f6a55a.json", MESA_MARK, mesa_tail(("other",))), tmp_path)
    declared_list = [MESA_SINGLE, snapped]
    assert report_problem(printed, declared_list) is None
    outcome, reasons, judged = judge_comparisons(printed["comparisons"], declared_list, "linear_attention")
    assert outcome == "unverified" and judged[-1]["eligibility"] == "unresolved" and judged[-1]["within_tolerance"]
    assert any(r.startswith("unresolved") for r in reasons)


# ----- through the orchestrator: the kill codes ------------------------------------------------------------------------


def exclusion_config():
    cfg = orch_config()
    cfg["s2"]["family_exclusion"] = True
    return cfg


def s2_run(tmp_path, name, critic, code):
    run_dir = tmp_path / f"run-20261014T23{name}Z"
    orch, llm = make_orchestrator(run_dir, cfg=exclusion_config(), script={
        "reduction_critic": critic, "reduction_check": {"check_code": code, "explanation": "recorded check"}})
    asyncio.run(orch.run())
    card_id = orch.store.list_cards()[0]
    s2 = orch.store.get_card(card_id)["status"]["verdicts"]["S2"]
    record = build_record(RunView(run_dir), card_id, "orchestrator")
    return s2, record, orch.ctx.latest_json_artifact(card_id, "reduction_critic"), llm


LEARNING = "LEARNING_RULE: the feature table is learned by a local Hebbian rule, not by backpropagation"


@pytest.mark.parametrize("learning_rule", [False, True])
def test_1184f7a7_dies_with_excluded_family_linear_attention(tmp_path, learning_rule):
    changes = {"difference_if_any": LEARNING} if learning_rule else {}
    code = recorded_program("job-e284e0a16e43.json", LA_MARK, la_tail())
    s2, record, artifact, llm = s2_run(tmp_path, "0000" if learning_rule else "0001", critic_1184(**changes), code)
    assert s2["verdict"] == "reduces_to_known"
    assert s2["reasons"][-1].startswith("gate: excluded_family:linear_attention:")
    assert record["kill_codes"] == ["excluded_family:linear_attention"]
    assert artifact["reduction_outcome"] == "verified"
    assert [d["eligibility"] for d in artifact["reduction_check"]["declared"]] == ["counted", "counted"]
    check_user = [c for c in llm.calls if c["prompt"] == "reduction_check"][0]["user"]
    assert "FAMILY POLICY" in check_user and "REPAIR POLICY" in check_user and '"la_phase_normalized"' in check_user


@pytest.mark.parametrize("learning_rule", [False, True])
def test_537e161c_dies_with_excluded_family_delta_rule(tmp_path, learning_rule):
    changes = {"difference_if_any": LEARNING} if learning_rule else {}
    code = recorded_program("job-720f18f6a55a.json", MESA_MARK, mesa_tail())
    s2, record, artifact, _ = s2_run(tmp_path, "0002" if learning_rule else "0003", critic_537(**changes), code)
    assert s2["verdict"] == "reduces_to_known" and s2["reasons"][-1].startswith("gate: excluded_family:delta_rule:")
    assert record["kill_codes"] == ["excluded_family:delta_rule"]


def test_c9615069_is_refuted_and_dies_not_radical(tmp_path):
    code = recorded_program("job-bf095b15bbd6.json", BIMAP_MARK, BIMAP_TAIL)
    s2, record, artifact, _ = s2_run(tmp_path, "0004", critic_c96(), code)
    assert artifact["reduction_outcome"] == "refuted" and s2["verdict"] == "fail"
    assert record["kill_codes"] == ["not_radical:R2"]


def test_a_substituted_repair_does_not_kill_by_family(tmp_path):
    code = recorded_program("job-e284e0a16e43.json", LA_MARK, la_tail(normalized_types=()))
    s2, record, artifact, _ = s2_run(tmp_path, "0005", critic_1184(), code)
    assert artifact["reduction_outcome"] == "unverified" and "disagrees with its declaration" in \
        artifact["reduction_check"]["problem"]
    assert record["kill_codes"] == ["not_radical:R2"]


def test_a_disallowed_type_does_not_kill_by_family(tmp_path):
    snapped = dict(MESA_HOPS, transformations=[transform("other", "a search over the state")])
    code = recorded_program("job-720f18f6a55a.json", MESA_MARK, mesa_tail(("other",)))
    s2, record, artifact, _ = s2_run(tmp_path, "0006", critic_537(hops=snapped), code)
    assert artifact["reduction_outcome"] == "unverified"
    assert record["kill_codes"] == ["not_radical:R2"]


def test_excluded_reduction_still_needs_the_verified_outcome():
    families, keywords = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
    counted = [{"declared_id": "mesanet", "family": "linear_attention", "comparator": "MesaNet"}]
    assert difference.excluded_reduction(critic_537(), "unverified", families, keywords, counted) is None
    assert difference.excluded_reduction(critic_537(), "verified", families, keywords, counted).id == "delta_rule"


# ----- F1: the critic sees each own-history cause whole ---------------------------------------------------------------


def test_f1_the_critic_history_block_shows_the_1411_character_cause_whole(tmp_path):
    extract = load("outcomes-card-46dbb349-matches.json")
    store = OutcomeStore(tmp_path / "outcomes")
    for entry in sorted(extract["records"], key=lambda e: e["seq"]):
        store.append(entry["record"])
    card = load("run-46dbb349/cards/card-46dbb349/v6.json")
    matches = own_history(store, card, card_signature(card), current_run=LIVE_RUN, limit=8, jaccard_min=0.6,
                          tfidf_min=0.3)
    causes = [c["detail"] for m in matches for c in m.stored.causes()]
    long = [c for c in causes if len(c) == 1411]
    assert len(matches) == 8 and len(long) == 1 and any(len(c) == 554 for c in causes)
    block = render_history(matches)
    assert all(c in block for c in causes)
    # the recorded critic call (call-be2acff2f744) received the same cause cut at 400 characters
    recorded_user = load("call-be2acff2f744.json")["user"]
    assert long[0][:390] in recorded_user and long[0] not in recorded_user


# ----- F2: S3 disproof causes carry the whole confirmed attack ---------------------------------------------------------


def test_f2_the_disproof_cause_is_recovered_whole_from_the_artifact():
    run_dir = FIXTURES / "s3checks-card-c9615069-20261011-r2"
    view = RunView(run_dir)
    record = build_record(view, "card-c9615069", "backfill")
    [cause] = [c for c in record["causes"] if c["code"] == "infeasible:disproof"]
    disproof, _ = view.latest_artifact("card-c9615069", "disproof")
    confirmed = [a for a in disproof["attacks"] if a["confirmed"]]
    assert len(confirmed) == 2
    for attack in confirmed:
        assert len(attack["counterexample"]) > 400 or len(attack["target_claim"]) > 200
        assert " ".join(disproof_reason(attack).split()) in cause["detail"]
    assert "verdict reason as recorded" not in cause["detail"]
    # the verdict reasons S3 wrote before FIX-12 are the cut ones
    reasons = view.latest_card("card-c9615069")[0]["status"]["verdicts"]["S3"]["reasons"]
    assert all(len(r) <= len("confirmed disproof of '': ") + 600 for r in reasons if r.startswith("confirmed"))


def test_f2_s3_now_writes_the_whole_attack_in_its_verdict(tmp_path):
    counterexample = "K > d pairs collide; " + "the stored pairs interfere at every step. " * 30 + "END"
    script = {"disprover": {"attacks": [{"target_claim": "capacity " + "of the bound memory " * 15,
                                         "counterexample": counterexample, "status": "succeeds",
                                         "check_code": program({"disproved": True, "detail": "collision"})}]}}
    orch, _ = make_orchestrator(tmp_path / "run-20261014T230100Z", script=script)
    asyncio.run(orch.run())
    card = orch.store.get_card(orch.store.list_cards()[0])
    [reason] = [r for r in card["status"]["verdicts"]["S3"]["reasons"] if r.startswith("confirmed disproof")]
    assert reason.endswith("END") and len(reason) > 1200
    record = build_record(RunView(tmp_path / "run-20261014T230100Z"), card["id"], "orchestrator")
    assert record["causes"][0]["detail"].endswith("END")


# ----- F3: the critic sees what stage R changed -------------------------------------------------------------------------


def run_events(kind: str, card_id: str, drop_rounds_after: int | None = None) -> list[dict]:
    out = []
    for line in (FIXTURES / "run-46dbb349" / "ledger.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["type"] == kind and event["data"].get("card_id") == card_id:
            if drop_rounds_after is not None and kind == "refine_round" and event["data"]["round"] > drop_rounds_after:
                continue
            out.append(event["data"])
    return out


def ctx_46(drop_rounds_after=None):
    cards = FIXTURES / "run-46dbb349" / "cards"
    return SimpleNamespace(
        card_events=lambda card_id, kind: run_events(kind, card_id, drop_rounds_after),
        store=SimpleNamespace(get_card=lambda card_id, version: json.loads(
            (cards / card_id / f"v{version}.json").read_text())))


def test_f3_the_changed_key_equation_is_shown_with_its_status():
    card = load("run-46dbb349/cards/card-46dbb349/v6.json")        # the version the recorded critic judged
    changes = equation_changes(ctx_46(), card)
    rows = {r["where"]: r for r in changes["equations"]}
    assert changes["before_version"] == 3 and changes["now_version"] == 6
    key = rows["mechanism.update[3]"]
    assert key["status"] == VALIDATED and key["changed_in_round"] == 1 and key["examined_in_round"] == 2
    assert "zs[e-1]" in key["before"] and "argmin" in key["now"]
    assert rows["mechanism.update[4]"]["status"] == UNCHANGED         # only its name changed
    block = render_changes(changes)
    assert "mechanism.update[3]: validated_repair (changed in round 1, examined in round 2)" in block
    # the recorded critic said "Nothing in the equations was changed to address those causes"
    critic = load("own_history-card-46dbb349-v6-987c68.json")
    assert critic["card_id"] == "card-46dbb349"
    assert "Do not say an equation is unchanged when the block marks it changed" in load_section("own_history")


def test_f3_a_change_no_later_round_examined_is_an_attempted_repair():
    card = load("run-46dbb349/cards/card-46dbb349/v6.json")
    rows = {r["where"]: r for r in equation_changes(ctx_46(drop_rounds_after=1), card)["equations"]}
    assert rows["mechanism.update[3]"]["status"] == ATTEMPTED


def test_f3_the_critic_history_block_carries_the_changes(tmp_path):
    card = load("run-46dbb349/cards/card-46dbb349/v6.json")
    stored = []
    ctx = SimpleNamespace(run_id=LIVE_RUN, outcomes=OutcomeStore(tmp_path / "outcomes"),
                          cfg={"outcomes": {"history_max": 8, "jaccard_min": 0.6, "tfidf_min": 0.3}},
                          put_json_artifact=lambda kind, name, value, card_id: stored.append(value),
                          artifact_name=lambda card_id: "a", emit=lambda kind, data: None, **vars(ctx_46()))
    _, block = s2_prior_art.own_history(ctx, card)
    assert "EQUATION CHANGES IN STAGE R (card version 3 before the dialogue, version 6 now)" in block
    assert stored[0]["equation_changes"]["equations"]
