"""OpenAI chat completions <-> Anthropic Messages, shaped the way LLxprt Code shapes OAuth requests.

Ported behavior (packages/providers/src/anthropic/, origin/main 2026-10-09):
- AnthropicRequestBuilder.buildAnthropicSystemPrompt: under OAuth the ``system`` field holds only the Claude
  Code string.
- AnthropicRequestPreparation.placeSystemInstruction + utils/systemPromptPlacement.formatContextPrefix: the
  real system prompt goes at the top of the conversation as a user message wrapped in ``<system>`` tags
  followed by a boundary line.
- schemaConverter.convertToolsToAnthropic: tool names get the ``llxprt_`` prefix under OAuth; response tool
  names are unprefixed (AnthropicProvider.unprefixToolName). Tool-use blocks replayed from history keep
  their unprefixed names, as LLxprt does. Tools are sent sorted by name with sorted property keys, so the
  request prefix is stable for prompt caching.
- AnthropicMessageNormalizer / toolIdNormalization.normalizeToAnthropicToolId: ``toolu_`` ids.
- AnthropicMessageValidator (see anthropic_sequence.py): orphaned tool results removed, missing results
  synthesized, sequence starts with a user turn, empty turns get placeholders, no trailing assistant turn when
  thinking is on or the model rejects prefill.
- AnthropicRequestBuilder.attachPromptCaching + the 1h ``prompt-caching`` default of the claudecode alias:
  ``cache_control`` (ephemeral, ttl 1h) on the context-prefix block and on the last cacheable block of the last
  message, with the extended-cache-ttl beta header.
- AnthropicResponseParser: prompt tokens are input + cache read + cache creation.
- anthropic-reasoning-config.ts (legacy auto path) and AnthropicModelData.ts: see ``reasoning_plan``.
- claudecode.config unallowedParameters: temperature, top_p and top_k are not sent to the 5.5 models.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .anthropic_models import (allows_sampling_parameters, default_max_tokens, enforces_preserved_thinking_prefix,
                               is_claude, is_fable_5, rejects_forced_tool_choice, supports_adaptive_thinking,
                               supports_effort_parameter, thinking_off_mode)
from .anthropic_schema import to_structured_output_schema
from .anthropic_sequence import repair
from .chat import COMMON_FIELDS, chat_response, check_single_choice, json_schema_format, text_of, tool_arguments
from .errors import BadRequest, UpstreamError

TOOL_PREFIX = "llxprt_"
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."
CONTEXT_BOUNDARY = "User provided conversation begins here:"
JSON_TOOL = f"{TOOL_PREFIX}emit_json"

PROMPT_CACHE_TTL = "1h"
CACHE_BETA = "extended-cache-ttl-2025-04-11"
BINDING_BETA = "thinking-binding-controls-2026-08-01"
LEGACY_THINKING_BUDGET = 10000     # ANTHROPIC_LEGACY_DEFAULT_BUDGET_TOKENS

_CONSUMED = COMMON_FIELDS | {"max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k", "stop", "n",
                             "reasoning_effort"}
_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
           "refusal": "content_filter", "pause_turn": "stop"}


@dataclass(frozen=True)
class RequestContext:
    json_tool: bool            # the reply arrives as the input of a forced tool and is returned as content
    dropped: tuple[str, ...]   # request fields Anthropic does not take


def normalize_tool_id(tool_id: str) -> str:
    if not tool_id:
        return "toolu_empty"
    if tool_id.startswith("toolu_"):
        return tool_id
    for prefix in ("hist_tool_", "call_"):
        if tool_id.startswith(prefix):
            return "toolu_" + re.sub(r"[^a-zA-Z0-9_-]", "-", tool_id[len(prefix):])
    return "toolu_" + re.sub(r"[^a-zA-Z0-9_-]", "-", tool_id)


def unprefix_tool_name(name: str) -> str:
    return name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name


def format_context_prefix(system_prompt: str) -> str:
    return f"<system>\n{system_prompt}\n</system>\n\n{CONTEXT_BOUNDARY}"


def cache_control() -> dict:
    return {"type": "ephemeral", "ttl": PROMPT_CACHE_TTL}


# ---- reasoning ---------------------------------------------------------------------------------------------------

# reasoning_effort values the gateway accepts. "none" turns thinking off (OpenAI's own spelling); the rest follow
# LLxprt's effort vocabulary (anthropic-reasoning-config.ts normalizeEffort).
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def reasoning_effort_of(body: dict) -> str | None:
    effort = body.get("reasoning_effort")
    if effort is None:
        return None
    if effort not in REASONING_EFFORTS:
        raise BadRequest(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)} on the anthropic backend, "
                         f"not {effort!r}")
    return effort


def normalize_effort(model: str, effort: str) -> str:
    """minimal and low are low. xhigh and max are max on adaptive-capable models and high elsewhere."""
    if effort in ("minimal", "low"):
        return "low"
    if effort in ("xhigh", "max"):
        return "max" if supports_adaptive_thinking(model) or not is_claude(model) else "high"
    return effort


@dataclass(frozen=True)
class ReasoningPlan:
    thinking: dict | None = None
    output_config: dict | None = None
    unsupported: bool = False       # an effort was requested that this model cannot carry

    @property
    def thinking_on(self) -> bool:
        return self.thinking is not None and self.thinking["type"] in ("adaptive", "enabled")


def reasoning_plan(model: str, effort: str | None) -> ReasoningPlan:
    """The ``thinking`` and ``output_config`` fields LLxprt sends for ``reasoning.enabled`` true (or false for
    "none") with this model and effort.

    - Adaptive-capable models (Opus 4.6+, Sonnet 5/5.5, Haiku 5.5, Fable 5): ``thinking`` adaptive with
      display summarized, plus ``output_config.effort``. Fable 5 is adaptive-only.
    - Other Claude models: manual budgeted thinking (10000 tokens), plus ``output_config.effort`` where the API
      takes it (it refuses the parameter on Sonnet 4/4.5 and Haiku 4.5, so those get the budget only).
    - "none": the model's thinking-off mode (Opus 5 and Haiku 5.5 "disabled", Sonnet 5.5 "between_tools"). Other
      models have none, so the request omits ``thinking`` and the effort is reported as dropped.
    - Models that enforce the preserved-thinking prefix check (Fable 5.1, Opus/Sonnet/Haiku 5.5) get
      ``block_binding`` on an adaptive or enabled ``thinking``.
    """
    if effort is None:
        return ReasoningPlan()
    if effort == "none":
        mode = thinking_off_mode(model)
        return ReasoningPlan(thinking={"type": mode}) if mode else ReasoningPlan(unsupported=True)
    if not is_claude(model):
        return ReasoningPlan(unsupported=True)
    if is_fable_5(model) or supports_adaptive_thinking(model):
        thinking: dict = {"type": "adaptive", "display": "summarized"}
    else:
        thinking = {"type": "enabled", "budget_tokens": LEGACY_THINKING_BUDGET}
    if enforces_preserved_thinking_prefix(model):
        thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
    output_config = {"effort": normalize_effort(model, effort)} if supports_effort_parameter(model) else None
    return ReasoningPlan(thinking=thinking, output_config=output_config)


# ---- messages ----------------------------------------------------------------------------------------------------

def _user_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise BadRequest("user message content must be a string or a list of parts")
    blocks = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            match = re.fullmatch(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
            if match:
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": match[1], "data": match[2]}})
            elif url.startswith(("http://", "https://")):
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                raise BadRequest("image_url must be a data: URL or an http(s) URL")
        else:
            raise BadRequest(f"unsupported user content part {kind!r}")
    return blocks


def _assistant_blocks(message: dict) -> list[dict]:
    blocks = []
    text = text_of(message.get("content"))
    if text.strip():
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        blocks.append({"type": "tool_use", "id": normalize_tool_id(call["id"]),
                       "name": unprefix_tool_name(call["function"]["name"]), "input": tool_arguments(call)})
    return blocks


def convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """(system text, Anthropic messages with adjacent same-role turns merged, tool results first).

    Turns that came out empty are kept so ``anthropic_sequence.repair`` can give them LLxprt's placeholders.
    """
    system_parts: list[str] = []
    turns: list[tuple[str, list[dict]]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = text_of(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            turns.append(("user", [b for b in _user_blocks(message.get("content")) if b["type"] != "text" or b["text"].strip()]))
        elif role == "assistant":
            turns.append(("assistant", _assistant_blocks(message)))
        elif role == "tool":
            turns.append(("user", [{"type": "tool_result", "tool_use_id": normalize_tool_id(message["tool_call_id"]),
                                    "content": text_of(message.get("content"))}]))
        else:
            raise BadRequest(f"unsupported message role {role!r}")
    merged: list[dict] = []
    for role, blocks in turns:
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"].extend(blocks)
        else:
            merged.append({"role": role, "content": list(blocks)})
    for turn in merged:
        if turn["role"] == "user":
            turn["content"].sort(key=lambda b: b["type"] != "tool_result")     # stable: tool results first
    return "\n\n".join(system_parts), merged


def _last_cacheable_index(content: list[dict]) -> int:
    """AnthropicRequestBuilder.selectLastCacheableBlockIndex: skip thinking blocks and empty text or results."""
    for index in range(len(content) - 1, -1, -1):
        block = content[index]
        if block["type"] in ("thinking", "redacted_thinking"):
            continue
        if block["type"] == "text" and not block["text"].strip():
            continue
        if block["type"] == "tool_result" and isinstance(block["content"], str) and not block["content"].strip():
            continue
        return index
    return -1


def attach_prompt_caching(messages: list[dict]) -> None:
    """Mark the last cacheable block of the last message (AnthropicRequestBuilder.attachPromptCaching)."""
    if not messages:
        return
    content = messages[-1]["content"]
    index = _last_cacheable_index(content)
    if index >= 0:
        content[index] = {**content[index], "cache_control": cache_control()}


# ---- tools -------------------------------------------------------------------------------------------------------

def convert_tool(tool: dict) -> dict:
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        raise BadRequest("only function tools are supported")
    fn = tool["function"]
    schema = dict(fn.get("parameters") or {})
    schema.setdefault("type", "object")
    if schema["type"] != "object":
        raise BadRequest(f"tool {fn.get('name')!r} parameters must be an object schema")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return {"name": TOOL_PREFIX + fn["name"], "description": fn.get("description") or "", "input_schema": schema}


def stabilize_tools(tools: list[dict]) -> list[dict]:
    """Sort by name and sort each schema's top-level property keys (AnthropicRequestPreparation)."""
    ordered = []
    for tool in sorted(tools, key=lambda t: (t["name"].casefold(), t["name"])):
        schema = tool["input_schema"]
        if schema.get("properties"):
            tool = {**tool, "input_schema": {**schema, "properties": dict(sorted(schema["properties"].items()))}}
        ordered.append(tool)
    return ordered


def _tool_choice(choice) -> dict | None:
    if choice in (None, "auto"):
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": TOOL_PREFIX + choice["function"]["name"]}
    raise BadRequest(f"unsupported tool_choice {choice!r}")


# ---- request -----------------------------------------------------------------------------------------------------

def build_request(body: dict, model: str) -> tuple[dict, RequestContext]:
    """Anthropic Messages payload (streaming) for a chat completion request, plus response-side context."""
    check_single_choice(body)
    system_text, messages = convert_messages(body.get("messages") or [])
    effort = reasoning_effort_of(body)
    plan = reasoning_plan(model, effort)
    messages = repair(messages, thinking=plan.thinking_on, model=model)
    if system_text:
        # A user turn at the top may now be adjacent to the first real user turn; the API merges them.
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": format_context_prefix(system_text),
                                                          "cache_control": cache_control()}]})
    attach_prompt_caching(messages)
    payload: dict = {"model": model, "messages": messages, "stream": True, "system": CLAUDE_CODE_SYSTEM,
                     "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens(model)}
    dropped = set(body) - _CONSUMED
    for key in ("temperature", "top_p", "top_k"):
        if body.get(key) is not None:
            if allows_sampling_parameters(model):
                payload[key] = body[key]
            else:
                dropped.add(key)
    stop = body.get("stop")
    if stop:
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    tools = [convert_tool(t) for t in body.get("tools") or []]
    spec = json_schema_format(body)
    native_json = False
    if spec is not None:
        if tools:
            raise BadRequest("response_format json_schema cannot be combined with tools on the anthropic backend")
        schema = dict(spec["schema"])
        if schema.get("type") != "object":
            raise BadRequest("the anthropic backend needs a json_schema whose root type is object")
        schema.setdefault("properties", {})
        schema.setdefault("required", [])
        native_json = rejects_forced_tool_choice(model)
        if native_json:
            # These models refuse a forced tool, so the reply is native structured output, returned as text.
            output_config = {**(plan.output_config or {}), "format": {"type": "json_schema",
                                                                         "schema": to_structured_output_schema(schema)}}
            plan = ReasoningPlan(plan.thinking, output_config, plan.unsupported)
        else:
            payload["tools"] = [{"name": JSON_TOOL, "input_schema": schema,
                                 "description": spec.get("description") or "Return the final answer by calling this tool with an object matching the schema."}]
            payload["tool_choice"] = {"type": "tool", "name": JSON_TOOL}
    elif tools and body.get("tool_choice") != "none":
        payload["tools"] = stabilize_tools(tools)
        payload["tool_choice"] = _tool_choice(body.get("tool_choice"))
    if plan.thinking is not None:
        payload["thinking"] = plan.thinking
    if plan.output_config is not None:
        payload["output_config"] = plan.output_config
    if plan.unsupported:
        dropped.add("reasoning_effort")
    return payload, RequestContext(json_tool=spec is not None and not native_json, dropped=tuple(sorted(dropped)))


def extra_betas(payload: dict) -> list[str]:
    """anthropic-beta values this payload needs beyond the OAuth base set: the extended cache TTL when a block asks
    for ttl 1h, and thinking binding controls when ``thinking`` carries ``block_binding``."""
    betas = []
    if any(isinstance(b, dict) and (b.get("cache_control") or {}).get("ttl") == "1h"
           for message in payload.get("messages", []) for b in message["content"]):
        betas.append(CACHE_BETA)
    if "block_binding" in (payload.get("thinking") or {}):
        betas.append(BINDING_BETA)
    return betas


class MessageAccumulator:
    """Folds Anthropic stream events into one Messages-API-shaped dict."""

    def __init__(self) -> None:
        self.message: dict = {"content": [], "usage": {}}
        self._json: dict[int, str] = {}
        self.finished = False

    def feed(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "message_start":
            msg = data["message"]
            self.message.update({k: msg.get(k) for k in ("id", "model", "role")})
            self.message["usage"].update(msg.get("usage") or {})
        elif kind == "content_block_start":
            block = dict(data["content_block"])
            if block["type"] == "tool_use":
                self._json[data["index"]] = ""
            self.message["content"].append(block)
        elif kind == "content_block_delta":
            block = self.message["content"][data["index"]]
            delta = data["delta"]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "input_json_delta":
                self._json[data["index"]] += delta["partial_json"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif kind == "content_block_stop":
            index = data["index"]
            if index in self._json:
                try:
                    self.message["content"][index]["input"] = json.loads(self._json[index] or "{}")
                except ValueError as exc:
                    raise UpstreamError("anthropic sent tool input that is not valid JSON", code="stream_invalid") from exc
        elif kind == "message_delta":
            self.message["stop_reason"] = data.get("delta", {}).get("stop_reason", self.message.get("stop_reason"))
            self.message["usage"].update({k: v for k, v in (data.get("usage") or {}).items() if v is not None})
        elif kind == "message_stop":
            self.finished = True
        elif kind == "error":
            err = data.get("error") or {}
            raise UpstreamError(f"anthropic stream error: {err.get('type', 'unknown')}: {err.get('message', '')}"[:500],
                                code=err.get("type") or "upstream_error")


def usage_of(message: dict) -> dict:
    raw = message.get("usage") or {}
    if "input_tokens" not in raw or "output_tokens" not in raw:
        raise UpstreamError("anthropic response carried no usage; refusing to return an unmetered call",
                            code="usage_missing")
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    return {"input": int(raw["input_tokens"]) + cache_read + cache_write, "output": int(raw["output_tokens"]),
            "cached": cache_read}


def to_chat(message: dict, context: RequestContext, now: float) -> tuple[dict, dict]:
    """(OpenAI chat.completion body, usage {"input","output","cached"})."""
    usage = usage_of(message)
    texts, calls, json_text = [], [], None
    # thinking and redacted_thinking blocks are skipped. The chat format has no field that could carry their
    # signatures back on a later tool-use turn, so they are not kept.
    for block in message.get("content") or []:
        if block["type"] == "text":
            texts.append(block["text"])
        elif block["type"] == "tool_use":
            if context.json_tool and block["name"] == JSON_TOOL:
                json_text = json.dumps(block.get("input", {}), ensure_ascii=False)
            else:
                calls.append({"id": block["id"], "type": "function",
                              "function": {"name": unprefix_tool_name(block["name"]),
                                           "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}})
    text = json_text if json_text is not None else ("".join(texts) or None)
    finish = "stop" if json_text is not None else _FINISH.get(message.get("stop_reason") or "", "stop")
    return chat_response(response_id=message.get("id"), model=message.get("model") or "", now=now, text=text,
                         tool_calls=calls, finish_reason=finish, usage=usage), usage
