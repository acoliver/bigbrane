"""The Claude model registry and per-model rules ported from LLxprt Code (anthropic_models.py)."""
import pytest

from hybrid.gateway import anthropic_models as models
from hybrid.gateway.backends import AnthropicBackend
from .support import run

LLXPRT_CLAUDECODE_MODELS = [
    "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-opus-4-5-20251101", "claude-opus-4-5", "claude-sonnet-5-5", "claude-sonnet-5",
    "claude-sonnet-4-6", "claude-sonnet-4-5-20250929", "claude-sonnet-4-5", "claude-sonnet-4-20250514",
    "claude-sonnet-4", "claude-haiku-5-5", "claude-haiku-4-5-20251001", "claude-haiku-4-5"]


def test_model_list_is_the_one_llxprts_claudecode_alias_serves_in_the_same_order():
    assert models.model_ids() == LLXPRT_CLAUDECODE_MODELS


def test_backend_lists_the_registry():
    backend = AnthropicBackend(refresher=None)
    assert run(backend.models()) == LLXPRT_CLAUDECODE_MODELS


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", 128000), ("claude-sonnet-5-5", 128000), ("claude-haiku-5-5", 128000), ("claude-opus-5", 128000),
    ("claude-fable-5-1", 128000), ("claude-opus-4-6", 128000), ("claude-sonnet-4-6", 128000),
    ("claude-opus-4-5", 32000), ("claude-sonnet-4-5", 64000), ("claude-haiku-4-5", 16000),
    ("claude-haiku-4-5-20251001", 16000),
])
def test_default_max_tokens_is_the_registry_limit_so_older_models_are_not_asked_for_more_than_they_allow(model, expected):
    assert models.default_max_tokens(model) == expected


@pytest.mark.parametrize("model,expected", [
    ("claude-sonnet-5-20260101", 128000), ("claude-opus-5-5-20261001", 32000), ("claude-opus-4-latest", 32000),
    ("claude-fable-5-latest", 40000), ("claude-3-5-sonnet-20241022", 8192), ("claude-3-opus-20240229", 4096),
    ("claude-3-haiku-20240307", 4096), ("claude-3-7-sonnet-20250219", 64000), ("something-else", 4096),
])
def test_ids_outside_the_registry_follow_llxprts_getMaxTokensForModel(model, expected):
    assert models.default_max_tokens(model) == expected


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", True), ("CLAUDE-OPUS-5-5", True), ("claude-opus-4-6", True), ("claude-opus-4-latest", True),
    ("claude-opus-4-5", False), ("claude-opus-4-1", False), ("claude-opus-5-5-mini", False),
    ("claude-sonnet-5", True), ("claude-sonnet-5-latest", True), ("claude-sonnet-5-20260101", True),
    ("claude-sonnet-5-5", True), ("claude-sonnet-50", False), ("claude-sonnet-4-6", False),
    ("claude-haiku-5-5", True), ("claude-haiku-5", False), ("claude-haiku-4-5", False),
    ("claude-fable-5", True), ("claude-fable-5-1", True), ("claude-fable-5-1-latest", True), ("claude-fable-50", False),
    ("vendor/claude-sonnet-5", False),
])
def test_adaptive_thinking_boundaries(model, expected):
    assert models.supports_adaptive_thinking(model) is expected


@pytest.mark.parametrize("model,mode", [
    ("claude-opus-5", "disabled"), ("claude-opus-5-20260101", "disabled"), ("claude-haiku-5-5", "disabled"),
    ("claude-sonnet-5-5", "between_tools"),
    ("claude-opus-5-5", None), ("claude-sonnet-5", None), ("claude-fable-5", None), ("claude-opus-4-6", None),
])
def test_thinking_off_mode_per_model(model, mode):
    assert models.thinking_off_mode(model) == mode


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", False), ("claude-sonnet-5-5", False), ("claude-haiku-5-5", False), ("claude-fable-5", False),
    ("claude-fable-5-1", False), ("claude-opus-5", True), ("claude-sonnet-5", True), ("claude-opus-4-6", True),
    ("claude-haiku-4-5", True), ("claude-sonnet-4-5", True),
])
def test_prefill_is_refused_by_the_5_5_models_and_fable(model, expected):
    assert models.supports_prefill(model) is expected


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5-5", True), ("claude-sonnet-5-5-20261001", True), ("claude-sonnet-5-5-mini", False), ("claude-fable-5-1", True),
    ("claude-fable-5-1-latest", True), ("claude-fable-5", False), ("claude-opus-5", False), ("claude-haiku-5-5", True),
])
def test_preserved_thinking_prefix_check_models(model, expected):
    assert models.enforces_preserved_thinking_prefix(model) is expected


@pytest.mark.parametrize("model,allowed", [
    ("claude-opus-5-5", False), ("claude-sonnet-5-5", False), ("claude-haiku-5-5", False),    # claudecode.config
    ("claude-opus-5", False), ("claude-sonnet-5", False), ("claude-fable-5", False), ("claude-fable-5-1", False),
    ("claude-opus-4-8", False), ("claude-opus-4-7", False),                                     # refused by the API
    ("claude-opus-4-6", True), ("claude-opus-4-5", True), ("claude-sonnet-4-6", True), ("claude-haiku-4-5", True),
])
def test_sampling_parameters_per_model(model, allowed):
    assert models.allows_sampling_parameters(model) is allowed


@pytest.mark.parametrize("model,refused", [
    ("claude-opus-5-5", True), ("claude-sonnet-5-5", True), ("claude-fable-5-1", True),
    ("claude-haiku-5-5", False), ("claude-opus-5", False), ("claude-sonnet-5", False), ("claude-opus-4-6", False),
])
def test_forced_tool_choice_is_refused_by_three_models(model, refused):
    assert models.rejects_forced_tool_choice(model) is refused


@pytest.mark.parametrize("model,takes_effort", [
    ("claude-opus-5-5", True), ("claude-opus-4-5", True), ("claude-sonnet-4-6", True),
    ("claude-sonnet-4-5", False), ("claude-sonnet-4-5-20250929", False), ("claude-sonnet-4", False),
    ("claude-haiku-4-5", False), ("claude-3-5-sonnet-20241022", False),
])
def test_effort_parameter_support(model, takes_effort):
    assert models.supports_effort_parameter(model) is takes_effort
