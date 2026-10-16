from pathlib import Path

import pytest

from hybrid.llm import ConfigError, load_config
from hybrid.llm.config import RoleSettings

MODELS_TOML = Path(__file__).resolve().parents[2] / "config" / "models.toml"

SPEC_ROUTING = {
    "explorer": ("opus", "gpt-sol-high"),
    "explorer_alt": ("gpt-sol-high", "opus"),
    "formalizer": ("sonnet", "opus"),
    "salad_critic": ("haiku", "sonnet"),
    "reduction_critic": ("opus", "sonnet"),
    "prior_art": ("haiku", "sonnet"),
    "feasibility_judge_a": ("opus", "sonnet"),
    "feasibility_judge_b": ("gpt-sol-high", "terra"),
    "disprover": ("sonnet", "opus"),
    "builder": ("sonnet", "haiku"),
    "analyst": ("haiku", "sonnet"),
    "obligation_checker": ("haiku", "sonnet"),
}

OPUS, SONNET, HAIKU, SOL = ("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5",
                            "anthropic/claude-haiku-5-5", "chatgpt/gpt-6.1-sol")
GATEWAY_ROUTING = {
    "explorer": (OPUS, SOL),
    "explorer_alt": (SOL, OPUS),
    "formalizer": (SONNET, OPUS),
    "salad_critic": (HAIKU, SONNET),
    "reduction_critic": (OPUS, SONNET),
    "prior_art": (HAIKU, SONNET),
    "feasibility_judge_a": (OPUS, SONNET),
    "feasibility_judge_b": (SOL,),
    "disprover": (SONNET, OPUS),
    "builder": (SONNET, HAIKU),
    "analyst": (HAIKU, SONNET),
    "falsifier_designer": (SOL,),
    "obligation_checker": (HAIKU, SONNET),
}

# Trailing local fallbacks allowed after the gateway entries, and only for these roles.
LMSTUDIO_FALLBACK = ("lmstudio", "qwen3.6-35b-a3b-mlx")
LMSTUDIO_FALLBACK_ROLES = {"feasibility_judge_b", "falsifier_designer"}

EFFORTS = {
    "explorer": "high", "explorer_alt": "high", "formalizer": "medium", "salad_critic": "low",
    "reduction_critic": "medium", "prior_art": "low", "feasibility_judge_a": "high", "feasibility_judge_b": "high",
    "disprover": "medium", "builder": "medium", "analyst": "low", "obligation_checker": "low",
    "statement_fixer": "medium", "prover": "medium", "challenger": "medium", "statement_match": "low",
}


def test_shipped_models_toml_defaults_to_llxprt_and_matches_spec_routing_table():
    config = load_config(MODELS_TOML)
    assert config.routing == "llxprt" and config.routing_source == "file"
    for role, (first, fallback) in SPEC_ROUTING.items():
        entries = config.roles[role]
        assert [(e.backend, e.model) for e in entries] == [("llxprt", first), ("llxprt", fallback)], role


def test_shipped_gateway_table_uses_gateway_model_ids_in_the_same_role_pattern():
    config = load_config(MODELS_TOML, routing="gateway")
    assert config.routing == "gateway" and config.routing_source == "argument"
    for role, models in GATEWAY_ROUTING.items():
        entries = [(e.backend, e.model) for e in config.roles[role]]
        gateway_entries, extra_entries = entries[:len(models)], entries[len(models):]
        assert gateway_entries == [("gateway", m) for m in models], role
        if extra_entries:
            assert role in LMSTUDIO_FALLBACK_ROLES and extra_entries == [LMSTUDIO_FALLBACK], role
    # judge B's primary stays in the OpenAI family, judge A in the Anthropic family
    assert config.roles["feasibility_judge_b"][0].model.startswith("chatgpt/")
    assert all(e.model.startswith("anthropic/") for e in config.roles["feasibility_judge_a"])
    assert config.roles["local"] == load_config(MODELS_TOML).roles["local"]
    assert config.active_backends() == {"gateway", "lmstudio"}


def test_shipped_role_settings():
    config = load_config(MODELS_TOML)
    for role, effort in EFFORTS.items():
        assert config.settings_for(role).effort == effort, role
    assert config.settings_for("salad_critic").max_output_tokens < config.settings_for("explorer").max_output_tokens
    assert config.settings_for("local") == RoleSettings()


def test_routing_env_overrides_file_and_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("BIGBRANE_ROUTING", "gateway")
    from_env = load_config(MODELS_TOML)
    assert from_env.routing == "gateway" and from_env.routing_source == "env"
    from_arg = load_config(MODELS_TOML, routing="llxprt")
    assert from_arg.routing == "llxprt" and from_arg.routing_source == "argument"


def test_unknown_routing_fails_fast(monkeypatch):
    with pytest.raises(ConfigError, match="--routing = 'nope' names no routing table"):
        load_config(MODELS_TOML, routing="nope")
    monkeypatch.setenv("BIGBRANE_ROUTING", "typo")
    with pytest.raises(ConfigError, match="BIGBRANE_ROUTING = 'typo'"):
        load_config(MODELS_TOML)


def test_shipped_models_toml_has_backends_budgets_and_timeouts():
    config = load_config(MODELS_TOML)
    assert {b.kind for b in config.backends.values()} == {"llxprt", "openai_compat", "anthropic_messages", "gateway"}
    assert all(b.concurrency >= 1 and b.timeout_s > 0 for b in config.backends.values())
    assert config.budgets.max_calls > 0 and config.budgets.max_tokens > 0
    assert config.backends["lmstudio"].base_url == "http://127.0.0.1:1234/v1"
    assert config.backends["lmstudio"].api_key_env is None
    assert config.backends["gateway"].base_url == "http://127.0.0.1:8765/v1"


def write(tmp_path, text):
    path = tmp_path / "m.toml"
    path.write_text(text)
    return path


GOOD_BACKEND = '[backends.x]\nkind = "openai_compat"\nbase_url = "http://h/v1"\nconcurrency = 1\ntimeout_s = 5\n'
ANTH_BACKEND = ('[backends.an]\nkind = "anthropic_messages"\nbase_url = "http://h"\napi_key_env = "K"\n'
                'concurrency = 1\ntimeout_s = 5\n')
GOOD_BUDGET = 'routing = "t"\n[budgets]\nmax_calls = 1\nmax_tokens = 1\n'
ONE_ROLE = '[routes.t]\nr = [{ backend = "x", model = "m" }]\n'


@pytest.mark.parametrize("text, message", [
    (GOOD_BUDGET + GOOD_BACKEND + '[routes.t]\nr = [{ backend = "nope", model = "m" }]\n', "unknown backend"),
    (GOOD_BUDGET + GOOD_BACKEND + "[routes.t]\nr = []\n", "at least one entry"),
    (GOOD_BUDGET + '[backends.x]\nkind = "weird"\nconcurrency = 1\ntimeout_s = 5\n' + ONE_ROLE, "kind must be"),
    (GOOD_BUDGET + '[backends.x]\nkind = "openai_compat"\nconcurrency = 1\ntimeout_s = 5\n' + ONE_ROLE, "needs base_url"),
    (GOOD_BUDGET + GOOD_BACKEND.replace("concurrency = 1", "concurrency = 0") + ONE_ROLE, "positive"),
    (GOOD_BUDGET + GOOD_BACKEND + "typo = 1\n" + ONE_ROLE, "unknown keys"),
    ('routing = "t"\n' + GOOD_BACKEND + ONE_ROLE, "missing required key 'budgets'"),
    (GOOD_BUDGET + GOOD_BACKEND + "[roles]\nr = [{ backend = \"x\", model = \"m\" }]\n", "root: unknown keys \\['roles'\\]"),
    (GOOD_BUDGET + GOOD_BACKEND, "missing required key 'routes'"),
    (GOOD_BUDGET.replace('routing = "t"\n', "") + GOOD_BACKEND + ONE_ROLE, "missing required key 'routing'"),
    (GOOD_BUDGET + GOOD_BACKEND + ONE_ROLE + '[routes.u]\nother = [{ backend = "x", model = "m" }]\n',
     "role set differs"),
    (GOOD_BUDGET + GOOD_BACKEND + '[routes.t]\nr = [{ backend = "x", model = "m", effort = "high" }]\n',
     "unknown keys \\['effort'\\]"),
    (GOOD_BUDGET + GOOD_BACKEND + '[role_settings.r]\neffort = "max"\n' + ONE_ROLE, "effort: must be one of"),
    (GOOD_BUDGET + GOOD_BACKEND + '[role_settings.r]\nmax_output_tokens = 0\n' + ONE_ROLE, "positive integer"),
    (GOOD_BUDGET + GOOD_BACKEND + '[role_settings.r]\nmax_output_tokens = 1.5\n' + ONE_ROLE, "positive integer"),
    (GOOD_BUDGET + GOOD_BACKEND + '[role_settings.r]\ntemperature = 0.2\n' + ONE_ROLE, "unknown keys"),
    (GOOD_BUDGET + GOOD_BACKEND + '[role_settings.ghost]\neffort = "low"\n' + ONE_ROLE, "no routing table has a role"),
    (GOOD_BUDGET + ANTH_BACKEND + '[role_settings.r]\neffort = "low"\n[routes.t]\nr = [{ backend = "an", model = "c" }]\n',
     "cannot apply role_settings.r.effort"),
])
def test_invalid_configs_fail_fast(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_role_settings_parse(tmp_path):
    text = GOOD_BUDGET + GOOD_BACKEND + '[role_settings.r]\neffort = "medium"\nmax_output_tokens = 900\n' + ONE_ROLE
    config = load_config(write(tmp_path, text))
    assert config.settings_for("r") == RoleSettings(effort="medium", max_output_tokens=900)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


MATH_ROLES = ("statement_fixer", "prover", "challenger", "statement_match")


@pytest.mark.parametrize("table", ["llxprt", "gateway", "gateway_claude_local"])
def test_every_table_routes_the_math_roles_with_separated_families(table):
    """Plan section 16: the prover is in a family other than the statement fixer, the challenger and the statement
    match, in every entry of every table (fallbacks included), and the primaries of the fixer and the statement
    match differ, so no card has its prover and a checker in one family. A fallback that puts the fixer in the
    matcher's family (both on the local Qwen model) is caught per card by hybrid.research.families."""
    from hybrid.research import families

    config = load_config(MODELS_TOML, routing=table)
    for role in MATH_ROLES:
        assert config.roles[role], (table, role)
        assert config.settings_for(role).max_output_tokens, role
    fam = {role: [families.family(e.model) for e in config.roles[role]] for role in MATH_ROLES}
    assert set(fam["prover"]) == {"anthropic"}
    for role in ("statement_fixer", "challenger", "statement_match"):
        assert "anthropic" not in fam[role], (table, role)
    primaries = {role: config.roles[role][0].model for role in MATH_ROLES}
    assert families.violations(primaries) == [], (table, primaries)


def test_the_tables_route_the_same_roles():
    roles = {table: set(load_config(MODELS_TOML, routing=table).roles)
             for table in ("llxprt", "gateway", "gateway_claude_local")}
    assert roles["llxprt"] == roles["gateway"] == roles["gateway_claude_local"]
    assert set(MATH_ROLES) <= roles["llxprt"]
