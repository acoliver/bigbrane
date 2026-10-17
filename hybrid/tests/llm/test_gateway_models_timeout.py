"""The /v1/models preflight waits longer than the gateway's own upstream model-list timeout."""

from hybrid.gateway.paths import ModelsConfig
from hybrid.llm import gateway_check


def test_default_wait_exceeds_the_gateway_upstream_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    assert gateway_check.models_timeout_s() == ModelsConfig().upstream_timeout_s + gateway_check.MODELS_MARGIN_S


def test_wait_follows_a_configured_upstream_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text("[models]\nupstream_timeout_s = 30\n")
    assert gateway_check.models_timeout_s() == 30 + gateway_check.MODELS_MARGIN_S
