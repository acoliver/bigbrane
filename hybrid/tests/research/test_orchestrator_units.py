"""Pure helpers of the orchestrator: TF-IDF, toy comparison, exponents, static checks, configs, prompts."""
import tomllib

import pytest

from hybrid.literature.prior_art import CLASSIFY_SYSTEM, QUERY_SYSTEM
from hybrid.literature.seed import LENSES
from hybrid.research.orchestrator import ROOT, load_lenses, load_run_config
from hybrid.research.schemas import validate
from hybrid.research.stages.common import (PROMPT_DIR, StageError, claimed_exponent, compare_expected, draft_schema,
                                           ground_draft, load_prompt)
from hybrid.research.stages.diversity import most_similar, tfidf_matrix
from hybrid.research.stages.s4_construct import attention_allowed, static_problems
from hybrid.research.stages.s5_evaluate import summarize
from hybrid.tests.research.builders import load_example
from hybrid.tests.research.test_orchestrator_fakes import TEMPLATE, draft, fake_bench_raw

import jsonschema


def test_tfidf_identical_and_disjoint():
    assert most_similar("phase binding memory", {}) == (None, 0.0)
    nearest, cos = most_similar("holographic phase binding memory", {"a": "holographic phase binding memory",
                                                                      "b": "chemical reaction kinetics"})
    assert nearest == "a" and cos == pytest.approx(1.0)
    _, cos = most_similar("tensor network contraction", {"b": "chemical reaction kinetics"})
    assert cos == 0.0
    matrix = tfidf_matrix(["alpha beta", "beta gamma"])
    assert matrix.shape == (2, 3)


def test_compare_expected():
    assert compare_expected({"a": 1.0, "b": [1, 2]}, {"a": 1.0 + 1e-12, "b": [1, 2], "extra": 3}, 1e-6, 1e-9) == []
    problems = compare_expected({"a": 1.0, "c": True}, {"a": 1.1, "c": 1}, 1e-6, 1e-9)
    assert len(problems) == 2
    assert compare_expected({"m": 1}, {}, 1e-6, 1e-9) == ["/m: missing from printed output"]
    assert compare_expected({"s": "x"}, {"s": "y"}, 0, 0)
    assert compare_expected({"v": 1.0}, {"v": float("nan")}, 1e-6, 1e-9)


@pytest.mark.parametrize("expr,exponent", [
    ("O(n)", 1.0), ("O(n log n)", 1.0), ("O(n^2 d)", 2.0), ("Theta(n**2)", 2.0), ("O(1)", 0.0),
    ("O(d^2)", 0.0), ("O(n d^2)", 1.0), ("O(n^{1.5})", 1.5), ("O(k)", 0.0),
])
def test_claimed_exponent(expr, exponent):
    assert claimed_exponent(expr) == exponent


def test_static_checks():
    assert static_problems(TEMPLATE, allow_attention=False) == []
    assert any("forbidden import os" in p for p in static_problems("import os\n" + TEMPLATE, False))
    assert any("subprocess" in p for p in static_problems("from subprocess import run\n" + TEMPLATE, False))
    attention = TEMPLATE + "\nimport torch\nlayer = torch.nn.MultiheadAttention(8, 2)\n"
    assert any("attention" in p for p in static_problems(attention, False))
    assert static_problems(attention, True) == []
    assert any("open()" in p for p in static_problems(TEMPLATE + "\nopen('x')\n", False))
    assert any("no class named Candidate" in p for p in static_problems("x = 1\n", False))
    assert any("syntax error" in p for p in static_problems("def (:\n", False))
    card = load_example()
    assert attention_allowed(card) is False
    card["claims"]["complexity"]["total_sequence"] = "O(n^2 d)"
    assert attention_allowed(card) is True


EXPLORER_USER = "PAPERS:\n- source_id: 2401.00001 (arxiv, 2024)\n  title: A paper"


def test_draft_schema_accepts_explorer_output_and_round_trips_to_a_card():
    schema = draft_schema()
    jsonschema.validate(draft(0, user=EXPLORER_USER), schema)
    for name in ("id", "version", "status", "lens", "parent_ids", "created_by_call"):
        assert name not in schema["properties"] and name not in schema["required"]
    card = dict(draft(0, user=EXPLORER_USER), id="card-0000abcd", version=1, parent_ids=[],
                created_by_call="call-0123456789ab", lens="reservoir_computing", status={"stage": "S0", "verdicts": {}})
    validate("theory_card", card)


@pytest.mark.parametrize("field", ["core_idea", "source_transplant"])
def test_draft_schema_requires_core_idea_and_transplant_but_cards_do_not(field):
    explorer_output = draft(0, user=EXPLORER_USER)
    del explorer_output[field]
    with pytest.raises(jsonschema.ValidationError, match=field):
        jsonschema.validate(explorer_output, draft_schema())
    validate("theory_card", load_example())          # a card written before these fields existed


def test_ground_draft_filters_transplant_sources():
    explorer_output = draft(0, user=EXPLORER_USER)
    explorer_output["source_transplant"]["source_ids"] = ["2401.00001", "made-up"]
    grounded, dropped, dropped_sources = ground_draft(explorer_output, {"2401.00001"})
    assert grounded["source_transplant"]["source_ids"] == ["2401.00001"] and dropped_sources == ["made-up"]
    assert explorer_output["source_transplant"]["source_ids"] == ["2401.00001", "made-up"]   # input untouched
    with pytest.raises(StageError, match="cites no paper"):
        ground_draft(explorer_output, {"other"})
    assert ground_draft(load_example(), set())[2] == []                                       # no transplant field


def test_bench_summary_matches_research_schema():
    summary = summarize(fake_bench_raw(), "card-0000abcd")
    assert summary["per_task"]["hmm"]["n32_kl"] == 0.1
    assert summary["scaling_slopes"] == {"candidate": 1.02, "ngram": 0.1}
    assert summarize(fake_bench_raw("job_failed"), "card-0000abcd")["per_task"] == {}


def test_lenses_cover_every_seed_lens_with_pairings():
    lenses, order = load_lenses(ROOT / "hybrid" / "config" / "lenses.toml")
    assert set(LENSES) <= set(lenses)
    assert "known_llm_alternatives_to_avoid_reinventing" not in order
    pairs = [name for name in order if name.startswith("pair:")]
    assert len(pairs) >= 3 and all(len(lenses[p]["sources"]) == 2 for p in pairs)
    assert len(order) == len(LENSES) - 1 + len(pairs)


def test_run_config_overrides_and_validation(tmp_path):
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml", {"budgets.max_cards": 4, "s5.mode": "local"})
    assert cfg["budgets"]["max_cards"] == 4 and cfg["s5"]["mode"] == "local"
    with pytest.raises(KeyError):
        load_run_config(ROOT / "hybrid" / "config" / "run.toml", {"budgets.nope": 1})
    with open(ROOT / "hybrid" / "config" / "bench_quick.toml", "rb") as handle:
        bench = tomllib.load(handle)
    from hybrid.bench.config import load_config
    assert load_config(bench)["baselines"] == ["ngram", "gru", "transformer", "ssm", "linear_attention"]


def test_s1_max_revisions_defaults_to_two_and_is_validated(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    assert load_run_config(shipped)["s1"]["max_revisions"] == 2
    text = shipped.read_text(encoding="utf-8")
    for replacement, message in (("max_revisions = -1", "max_revisions must be an integer"),
                                 ("max_revisions = 1.5", "max_revisions must be an integer"),
                                 ("revisions = 1", "s1.revisions is now s1.max_revisions")):
        path = tmp_path / "run.toml"
        path.write_text(text.replace("max_revisions = 2", replacement), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_run_config(path)


def test_shipped_run_config_has_the_s1_gate_option_worker_counts_and_no_fixed_seed():
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    cfg = load_run_config(shipped)
    assert cfg["s1"]["pass_with_residual_fixable"] is True
    assert cfg["workers"] == {"S1": 4, "S2": 3, "S3": 2, "M": 2, "C": 1, "S4": 1, "S5": 1, "S6": 1}
    with open(shipped, "rb") as handle:
        assert "seed" not in tomllib.load(handle)["run"]
    assert "# seed = 0" in shipped.read_text(encoding="utf-8")   # documented as an opt-in


def test_pass_with_residual_fixable_defaults_to_true_and_is_validated(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    text = shipped.read_text(encoding="utf-8")
    path = tmp_path / "run.toml"
    path.write_text(text.replace("pass_with_residual_fixable = true\n", ""), encoding="utf-8")
    assert load_run_config(path)["s1"]["pass_with_residual_fixable"] is True
    path.write_text(text.replace("pass_with_residual_fixable = true", "pass_with_residual_fixable = false"),
                    encoding="utf-8")
    assert load_run_config(path)["s1"]["pass_with_residual_fixable"] is False
    path.write_text(text.replace("pass_with_residual_fixable = true", "pass_with_residual_fixable = 1"),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="pass_with_residual_fixable must be true or false"):
        load_run_config(path)


def test_empty_search_counts_as_novel_defaults_to_false_and_is_validated(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    assert load_run_config(shipped)["s2"]["empty_search_counts_as_novel"] is False
    path = tmp_path / "run.toml"
    path.write_text(shipped.read_text(encoding="utf-8").replace('empty_search_counts_as_novel = false',
                                                                'empty_search_counts_as_novel = "false"'),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="empty_search_counts_as_novel must be true or false"):
        load_run_config(path)


def test_run_seed_is_optional_validated_and_overridable(tmp_path):
    shipped = ROOT / "hybrid" / "config" / "run.toml"
    assert load_run_config(shipped, {"run.seed": 7})["run"]["seed"] == 7
    with pytest.raises(ValueError, match="run.seed must be an integer"):
        load_run_config(shipped, {"run.seed": "0"})
    with pytest.raises(ValueError, match="run.seed must be an integer"):
        load_run_config(shipped, {"run.seed": True})


ROLES_WITH_PROMPTS = ["explorer", "explorer_alt", "salad_critic", "reduction_critic", "prior_art",
                      "feasibility_judge_a", "feasibility_judge_b", "formalizer", "disprover", "builder", "analyst",
                      "falsifier_designer", "obligation_checker",
                      # stages M and C (s3.mode = "math"); the statement_match role also uses statement_backtranslate
                      "statement_fixer", "prover", "challenger", "statement_match",
                      # FIX-7: anchored statement-match objections go to a separately recorded cross-check
                      "statement_crosscheck",
                      # FIX-17: a dispute the cross-check leaves open goes to a separately recorded tiebreak
                      "statement_tiebreak",
                      # generation stage G (generation.mode = "arms", Phase 6)
                      "arm_classifier",
                      # refinement dialogue R and the S2 difference check (Phase 7); the proposer uses
                      # refine_proposer and difference, the check difference_check
                      "refine_challenger", "refine_challenger_alt"]


def test_every_role_in_models_toml_has_a_prompt():
    with open(ROOT / "hybrid" / "config" / "models.toml", "rb") as handle:
        routes = tomllib.load(handle)["routes"]
    assert {"llxprt", "gateway"} <= set(routes)
    roles = set(routes["llxprt"]) - {"local"}
    for name, table in routes.items():
        assert set(table) - {"local"} == roles, name
    assert roles == set(ROLES_WITH_PROMPTS)
    for role in roles:
        assert load_prompt(role).startswith("# prompt: ")
    assert load_prompt("statement_backtranslate").startswith("# prompt: statement_backtranslate")
    for name in ("refine_proposer", "difference", "difference_check"):
        assert load_prompt(name).startswith(f"# prompt: {name}")


def test_prior_art_prompt_copy_matches_the_literature_module():
    text = (PROMPT_DIR / "prior_art.md").read_text(encoding="utf-8")
    assert QUERY_SYSTEM in text and CLASSIFY_SYSTEM in text


def test_explorer_prompt_states_the_bans_and_requirements():
    text = load_prompt("explorer")
    for phrase in ("a better transformer", "Chinchilla", "BitNet", "partitioning or windowing", "LCM-style",
                   "renamed RNN", "source_id", "toy_example", "few_shot_argument", "context length n"):
        assert phrase in text, phrase
    assert "undischarged obligation is not feasible" in load_prompt("feasibility_judge")
    assert "adversarial and specific" in load_prompt("salad_critic")
    assert "adversarial and specific" in load_prompt("reduction_critic")
    for phrase in ("core_idea", "source_transplant", "extensions", "at most about 6 state objects",
                   "at most about 6 update equations", "simplest correct form", "Be radical"):
        assert phrase in text, phrase


def test_critic_prompts_describe_the_new_fields():
    salad = load_prompt("salad_critic")
    for phrase in ("fatal", "fixable", "dimension_error", "contradictory_definition", "unfalsifiable_central_claim",
                   "missing_definition", "notation_clash", "at most 8 fixable", "Ignore the extensions"):
        assert phrase in salad, phrase
    reduction = load_prompt("reduction_critic")
    assert "textbook construction from its source field applied without modification" in reduction
    assert "textbook_why" in reduction and "the one answer of this kind that gates" in reduction


def test_prompts_and_docs_avoid_banned_words():
    paths = list(PROMPT_DIR.glob("*.md")) + [ROOT / "hybrid" / "config" / "run.toml",
                                             ROOT / "hybrid" / "config" / "lenses.toml"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "\u2014" not in text, path
        assert "canonical" not in text.lower() and "load-bearing" not in text.lower(), path
