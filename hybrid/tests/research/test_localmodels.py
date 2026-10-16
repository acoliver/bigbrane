"""Local-model preflight: a routed LM Studio model must be loaded with the context models.toml requires, or `run`
and `plumb` refuse to start (exit 2) with the measured and required values. Every LM Studio answer here is a fake."""
import httpx
import pytest

from hybrid import localmodels
from hybrid.llm import ConfigError, load_config
from hybrid.research.__main__ import main
from hybrid.research.orchestrator import ROOT
from hybrid.research.stages.common import MATH_ORDER, STAGE_ORDER

MODELS = ROOT / "hybrid" / "config" / "models.toml"
QWEN = "qwen3.6-35b-a3b-mlx"
# conftest.py replaces fetch_models for every research test; these tests exercise the real one with a fake `get`.
REAL_FETCH = localmodels.fetch_models


def listing(**contexts):
    """Fake /api/v0/models data: model id -> loaded context (None: listed but not loaded)."""
    out = []
    for model, context in contexts.items():
        entry = {"id": model.replace("__", "/"), "state": "loaded" if context else "not-loaded",
                 "max_context_length": 262144}
        if context:
            entry["loaded_context_length"] = context
        out.append(entry)
    return out


def fake_fetch(data):
    seen = []

    def fetch(server, **kwargs):
        seen.append(server)
        return data
    fetch.seen = seen
    return fetch


def test_the_shipped_config_requires_131072_tokens_for_the_local_backend():
    models = load_config(MODELS, "gateway")
    assert models.backends["lmstudio"].min_loaded_context == 131072 == localmodels.DEFAULT_MIN_CONTEXT
    reqs = localmodels.requirements(models, localmodels.roles_for(MATH_ORDER))
    assert [(r.model, r.min_context, r.server) for r in reqs] == [(QWEN, 131072, "http://127.0.0.1:1234")]
    # Phase 7: the refinement challengers (stage R and the S2 difference check) fall back to the local Qwen model
    assert set(reqs[0].roles) == {"statement_fixer", "challenger", "statement_match", "falsifier_designer",
                                  "refine_challenger", "refine_challenger_alt"}


def test_only_roles_of_the_stages_that_run_count():
    llxprt = load_config(MODELS, "llxprt")
    assert localmodels.requirements(llxprt, localmodels.roles_for(STAGE_ORDER)) == []
    assert [r.model for r in localmodels.requirements(llxprt, localmodels.roles_for(MATH_ORDER))] == [QWEN]
    assert localmodels.requirements(load_config(MODELS, "gateway"), localmodels.roles_for(("S4",))) == []
    # The `local` role (model = "auto") is in every table but no stage calls it.
    assert "local" not in localmodels.roles_for(MATH_ORDER)


def test_pass_when_the_model_is_loaded_with_enough_context():
    reqs = localmodels.requirements(load_config(MODELS, "gateway"), localmodels.roles_for(MATH_ORDER))
    fetch = fake_fetch(listing(**{QWEN: 131072, "gemma-4-e4b-it": None}))
    assert localmodels.check(reqs, fetch=fetch) == [f"{QWEN}: loaded, context 131072 (required 131072)"]
    assert fetch.seen == ["http://127.0.0.1:1234"]


def test_refuses_a_model_loaded_with_too_little_context_naming_both_numbers():
    reqs = localmodels.requirements(load_config(MODELS, "gateway"), localmodels.roles_for(MATH_ORDER))
    with pytest.raises(localmodels.LocalModelUnavailable) as raised:
        localmodels.check(reqs, fetch=fake_fetch(listing(**{QWEN: 8192})))
    assert "loaded with 8192 tokens of context" in str(raised.value) and "required: at least 131072" in str(raised.value)


def test_refuses_a_model_that_is_not_loaded_or_not_known():
    reqs = localmodels.requirements(load_config(MODELS, "gateway_claude_local"), localmodels.roles_for(MATH_ORDER))
    with pytest.raises(localmodels.LocalModelUnavailable) as raised:
        localmodels.check(reqs, fetch=fake_fetch(listing(**{QWEN: 131072, "gemma-4-e4b-it": None})))
    assert raised.value.problems == ["gemma-4-e4b-it (roles statement_match) is not-loaded at http://127.0.0.1:1234 "
                                     "(measured loaded context: none); required: loaded with at least 131072 tokens "
                                     "of context"]
    with pytest.raises(localmodels.LocalModelUnavailable, match="is not known to LM Studio"):
        localmodels.check(reqs, fetch=fake_fetch(listing(**{QWEN: 131072})))


def test_refuses_when_lm_studio_does_not_answer():
    def down(url, timeout):
        raise httpx.ConnectError("connection refused")
    with pytest.raises(localmodels.LocalModelUnavailable, match="does not answer at http://127.0.0.1:1234/api/v0/models"):
        REAL_FETCH("http://127.0.0.1:1234", get=down)


def test_fetch_reads_the_lm_studio_model_list():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": listing(**{QWEN: 131072}), "object": "list"}
    urls = []
    data = REAL_FETCH("http://127.0.0.1:1234", get=lambda url, timeout: urls.append(url) or Response())
    assert urls == ["http://127.0.0.1:1234/api/v0/models"] and data[0]["loaded_context_length"] == 131072


def test_min_loaded_context_is_configurable_and_only_for_local_backends(tmp_path):
    text = MODELS.read_text().replace("min_loaded_context = 131072", "min_loaded_context = 32768")
    (tmp_path / "models.toml").write_text(text)
    models = load_config(tmp_path / "models.toml", "gateway")
    reqs = localmodels.requirements(models, localmodels.roles_for(MATH_ORDER))
    assert reqs[0].min_context == 32768
    assert localmodels.check(reqs, fetch=fake_fetch(listing(**{QWEN: 32768})))
    without = MODELS.read_text().replace("min_loaded_context = 131072\n", "")
    (tmp_path / "models.toml").write_text(without)
    assert localmodels.requirements(load_config(tmp_path / "models.toml", "gateway"),
                                    ["statement_match"])[0].min_context == 131072
    bad = MODELS.read_text().replace('kind = "gateway"\n', 'kind = "gateway"\nmin_loaded_context = 4\n')
    (tmp_path / "models.toml").write_text(bad)
    with pytest.raises(ConfigError, match="only for kind openai_compat"):
        load_config(tmp_path / "models.toml")


def small_context(monkeypatch):
    monkeypatch.setattr(localmodels, "fetch_models", lambda server, **kwargs: listing(**{QWEN: 8192}))


def test_run_refuses_before_building_anything(monkeypatch, tmp_path, capsys):
    import hybrid.research.orchestrator as orchestrator
    small_context(monkeypatch)
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    monkeypatch.setattr(orchestrator, "build_orchestrator", lambda *a, **k: pytest.fail("orchestrator built"))
    config = tmp_path / "run.toml"
    config.write_text((ROOT / "hybrid" / "config" / "run.toml").read_text().replace('mode = "judges"', 'mode = "math"'))
    assert main(["run", "--config", str(config), "--run-dir", str(tmp_path / "run")]) == 2
    err = capsys.readouterr().err
    assert "refusing to start the research run: the local model check failed (routing table 'llxprt')" in err
    assert f"{QWEN} (roles statement_fixer, challenger, statement_match) is loaded with 8192 tokens" in err
    assert "required: at least 131072" in err and "never loads or unloads" in err
    assert not (tmp_path / "run").exists()


def test_run_starts_when_no_routed_role_is_local(monkeypatch, tmp_path):
    import hybrid.research.orchestrator as orchestrator
    small_context(monkeypatch)
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)

    class Built(Exception):
        pass

    def build(*args, **kwargs):
        raise Built()
    monkeypatch.setattr(orchestrator, "build_orchestrator", build)
    with pytest.raises(Built):          # judges mode on the llxprt table routes nothing to LM Studio
        main(["run", "--run-dir", str(tmp_path / "run")])


def test_plumb_refuses_before_reading_the_source(monkeypatch, tmp_path, capsys):
    import hybrid.research.plumb as plumb
    small_context(monkeypatch)
    monkeypatch.setattr(plumb, "plumb_command", lambda args: pytest.fail("plumb started"))
    code = main(["plumb", "--source-run", str(tmp_path / "src"), "--card", "card-00000000", "--stages", "M,C",
                 "--routing", "gateway", "--run-dir", str(tmp_path / "out"), "--reason", "test"])
    assert code == 2
    err = capsys.readouterr().err
    assert "refusing to start the research plumb" in err and "8192" in err and "131072" in err


def test_plumb_of_stages_without_local_roles_is_not_blocked(monkeypatch, tmp_path):
    import hybrid.research.plumb as plumb
    small_context(monkeypatch)
    monkeypatch.setattr(plumb, "plumb_command", lambda args: 7)
    code = main(["plumb", "--source-run", str(tmp_path / "src"), "--card", "card-00000000", "--stages", "S4",
                 "--routing", "gateway", "--run-dir", str(tmp_path / "out"), "--reason", "test"])
    assert code == 7
