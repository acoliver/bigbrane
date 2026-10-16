"""Claude model registry and per-model rules, ported from LLxprt Code.

Sources (llxprt-code origin/main, 2026-10-09):
- packages/providers/src/composition/aliases/claudecode.config: the model list and limits the OAuth ("claudecode")
  alias serves, and the sampling parameters it forbids.
- packages/providers/src/anthropic/AnthropicModelData.ts: id predicates (isOpus46Plus, isSonnet5, isSonnet55,
  isHaiku55, isFable5), adaptive-thinking support, thinking-off modes, prefill support, getMaxTokensForModel.

Ids are matched with ``fullmatch`` so near-misses such as ``claude-sonnet-50`` or ``claude-opus-5-5-mini`` do not
inherit a neighbour's rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    context_window: int
    max_output_tokens: int


# claudecode.config staticModels, in the order LLxprt lists them.
CLAUDE_CODE_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("claude-opus-5-5", "Claude Opus 5.5", 1_000_000, 128_000),
    ModelInfo("claude-opus-5", "Claude Opus 5", 1_000_000, 128_000),
    ModelInfo("claude-fable-5-1", "Claude Fable 5.1", 1_000_000, 128_000),
    ModelInfo("claude-fable-5", "Claude Fable 5", 1_000_000, 128_000),
    ModelInfo("claude-opus-4-8", "Claude Opus 4.8", 1_000_000, 128_000),
    ModelInfo("claude-opus-4-7", "Claude Opus 4.7", 1_000_000, 128_000),
    ModelInfo("claude-opus-4-6", "Claude Opus 4.6", 1_000_000, 128_000),
    ModelInfo("claude-opus-4-5-20251101", "Claude Opus 4.5", 500_000, 32_000),
    ModelInfo("claude-opus-4-5", "Claude Opus 4.5", 500_000, 32_000),
    ModelInfo("claude-sonnet-5-5", "Claude Sonnet 5.5", 1_000_000, 128_000),
    ModelInfo("claude-sonnet-5", "Claude Sonnet 5", 1_000_000, 128_000),
    ModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6", 1_000_000, 128_000),
    ModelInfo("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", 400_000, 64_000),
    ModelInfo("claude-sonnet-4-5", "Claude Sonnet 4.5", 400_000, 64_000),
    ModelInfo("claude-sonnet-4-20250514", "Claude Sonnet 4", 400_000, 64_000),
    ModelInfo("claude-sonnet-4", "Claude Sonnet 4", 400_000, 64_000),
    ModelInfo("claude-haiku-5-5", "Claude Haiku 5.5", 1_000_000, 128_000),
    ModelInfo("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 500_000, 16_000),
    ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5", 500_000, 16_000),
)
_BY_ID = {m.id: m for m in CLAUDE_CODE_MODELS}


def model_ids() -> list[str]:
    return [m.id for m in CLAUDE_CODE_MODELS]


def _matcher(pattern: str):
    compiled = re.compile(pattern, re.IGNORECASE)
    return lambda model: compiled.fullmatch(model) is not None


is_opus_46_plus = _matcher(r"claude-opus-(?:4-latest|5-latest|4-6|4-7|4-8|5|5-5)(?:-[0-9]{8})?")
is_sonnet_5 = _matcher(r"claude-sonnet-5(?:-latest|-[0-9]{8})?")
is_sonnet_55 = _matcher(r"claude-sonnet-5-5")
is_haiku_55 = _matcher(r"claude-haiku-5-5")
is_fable_5 = _matcher(r"claude-fable-5(?:-1)?(?:-latest|-[0-9]{8})?")
_is_opus_5 = _matcher(r"claude-opus-5(?:-latest|-[0-9]{8})?")
_is_claude_5_5 = _matcher(r"claude-(?:opus|sonnet|haiku)-5-5")
_enforces_preserved_thinking_prefix = _matcher(
    r"(?:claude-fable-5-1|claude-opus-5-5|claude-sonnet-5-5|claude-haiku-5-5)(?:-latest|-[0-9]{8})?")


def supports_adaptive_thinking(model: str) -> bool:
    return any(p(model) for p in (is_opus_46_plus, is_sonnet_5, is_sonnet_55, is_haiku_55, is_fable_5))


def is_claude(model: str) -> bool:
    return model.lower().startswith("claude-")


def thinking_off_mode(model: str) -> str | None:
    """The ``thinking.type`` that turns thinking off on this model, or None when the model offers none.

    Sonnet 5.5 takes ``between_tools``; Opus 5 and Haiku 5.5 take ``disabled``; every other model keeps its default
    (LLxprt omits the field and warns).
    """
    if is_sonnet_55(model):
        return "between_tools"
    if _is_opus_5(model) or is_haiku_55(model):
        return "disabled"
    return None


def enforces_preserved_thinking_prefix(model: str) -> bool:
    """Models whose adaptive or enabled thinking carries ``block_binding`` (and needs its beta header)."""
    return _enforces_preserved_thinking_prefix(model)


def supports_prefill(model: str) -> bool:
    """False when a request must not end with an assistant message (Fable 5 and the 5.5 models reject prefill)."""
    return not (is_fable_5(model) or _is_claude_5_5(model))


# Observed against the Claude subscription endpoint on 2026-10-10 (results/gateway-20261011/probe-matrix-*.log);
# LLxprt Code does not encode these, because it never forces a tool and sends effort only where its alias defaults do.
_rejects_forced_tool_choice = _matcher(r"claude-(?:opus-5-5|sonnet-5-5|fable-5-1)(?:-latest|-[0-9]{8})?")
_lacks_effort_parameter = _matcher(r"claude-(?:sonnet-4(?:-5)?|haiku-4-5|3-.+)(?:-[0-9]{8})?")


def rejects_forced_tool_choice(model: str) -> bool:
    """tool_choice of type ``tool`` or ``any`` is refused with a 400 (Opus 5.5, Sonnet 5.5, Fable 5.1)."""
    return _rejects_forced_tool_choice(model)


def supports_effort_parameter(model: str) -> bool:
    """``output_config.effort`` is refused on Sonnet 4/4.5, Haiku 4.5 and Claude 3.x."""
    return not _lacks_effort_parameter(model)


# claudecode.config forbids temperature, top_p and top_k on the 5.5 models only. The API (probed live 2026-10-10,
# results/gateway-20261011/probe-temp.log) refuses them on every model below as well: "`temperature` is deprecated for
# this model". Opus 4.6, Opus 4.5, Sonnet 4.6 and Haiku 4.5 still take them.
_rejects_sampling = _matcher(r"claude-(?:opus-(?:4-7|4-8|4-latest|5-latest|5-5|5)|sonnet-5(?:-5)?|haiku-5-5|fable-5(?:-1)?)"
                             r"(?:-latest|-[0-9]{8})?")


def allows_sampling_parameters(model: str) -> bool:
    return not _rejects_sampling(model)


# AnthropicModelData.MODEL_TOKEN_PATTERNS: [required substrings], tokens. First full match wins.
_TOKEN_PATTERNS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("opus", "4"), 32000), (("sonnet", "4"), 64000), (("haiku", "4"), 200000), (("3", "7", "sonnet"), 64000),
    (("3", "5", "sonnet"), 8192), (("3", "5", "haiku"), 8192), (("3", "opus"), 4096), (("3", "haiku"), 4096),
)


def _strip_date_segment(model_id: str) -> str:
    head, sep, tail = model_id.rpartition("-")
    return head if sep and re.fullmatch(r"[0-9]{8}", tail) else model_id


def llxprt_max_tokens_for_model(model: str) -> int:
    """AnthropicModelData.getMaxTokensForModel, for ids the registry does not list."""
    if re.fullmatch(r"claude-opus-5-5", model, re.IGNORECASE):
        return 128000
    if "claude-opus-4" in model or is_opus_46_plus(model):
        return 32000
    if "claude-sonnet-4" in model:
        return 64000
    if is_sonnet_5(model) or is_sonnet_55(model) or is_haiku_55(model):
        return 128000
    if is_fable_5(model):
        return 40000
    normalized = _strip_date_segment(model.lower())
    for parts, tokens in _TOKEN_PATTERNS:
        if all(part in normalized for part in parts):
            return tokens
    return 4096


def default_max_tokens(model: str) -> int:
    """``max_tokens`` when the caller sets none.

    LLxprt's claudecode alias sets ``maxOutputTokens`` 128000 for every model, which is also the registry limit of
    every 4.6+ and 5.x model. Older models have a lower real limit in the same alias (Opus 4.5 32000, Sonnet 4.x 64000,
    Haiku 4.5 16000), and the API refuses a larger ``max_tokens``, so the registry value is used for them. Ids outside
    the registry (dated snapshots, ``-latest`` pointers) follow getMaxTokensForModel.
    """
    info = _BY_ID.get(model.lower())
    return info.max_output_tokens if info else llxprt_max_tokens_for_model(model)
