"""OpenAI chat completions <-> OpenAI Responses API for the Sign in with ChatGPT endpoint.

Request shaping follows LLxprt Code's openai-responses provider where it applies to
``POST https://api.openai.com/v1/responses`` (the documented ChatGPT plan usage route, not the Codex
``backend-api``): ``instructions`` for the system prompt, ``function_call`` / ``function_call_output`` items
with ``call_`` ids (toolIdNormalization.normalizeToOpenAIToolId), function tools (shipped differently, see
below), ``tool_choice`` when the caller sets it, ``parallel_tool_calls`` when tools are present, and dropping
function calls and outputs that have no partner (OpenAIResponsesInputBuilder). The preview limits in OpenAI's
docs add: ``store: false`` and
``stream: true`` on every request, no ``max_output_tokens``, ``temperature``, ``top_p``, ``user`` or
``metadata``, and no ``system`` role items in ``input``.

Tools: the preview-limitations page says to "group function/custom tools in namespaces or supply them through
``additional_tools`` input items", so a flat top-level ``tools`` array is not sent. Function tools travel as one
``{"type": "additional_tools", "role": "developer", "tools": [...]}`` item at the head of ``input`` (the shape in
OpenAI's tool-search guide and SDK types). That choice keeps ``function_call`` items unchanged in both directions;
a namespace would add a ``namespace`` field to every call and replayed call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .chat import COMMON_FIELDS, chat_response, check_single_choice, json_schema_format, text_of
from .chatgpt_plan import lead_with_manage_usage
from .errors import BadRequest, UpstreamError

_CONSUMED = COMMON_FIELDS | {"reasoning_effort", "n"}
_ERROR_STATUS = {
    "subscription_sharing_usage_limit_exceeded": 429,
    "subscription_sharing_usage_unavailable": 503,
    "subscription_sharing_user_not_eligible": 403,
    "subscription_sharing_unsupported_capability": 400,
    "subscription_sharing_route_not_supported": 403,
    "subscription_sharing_invalid_user": 401,
    "subscription_sharing_user_unavailable": 503,
}


@dataclass(frozen=True)
class RequestContext:
    dropped: tuple[str, ...]


def normalize_call_id(call_id: str) -> str:
    if not call_id:
        return "call_"
    for prefix in ("call_", "hist_tool_", "toolu_"):
        if call_id.startswith(prefix):
            call_id = call_id[len(prefix):]
            break
    return "call_" + re.sub(r"[^a-zA-Z0-9_-]", "", call_id)


def _user_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest("user message content must be a string or a list of parts")
    parts = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            parts.append({"type": "input_text", "text": part.get("text", "")})
        elif kind == "image_url":
            parts.append({"type": "input_image", "image_url": (part.get("image_url") or {}).get("url", "")})
        else:
            raise BadRequest(f"unsupported user content part {kind!r}")
    return parts


def convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    answered = {normalize_call_id(m["tool_call_id"]) for m in messages if m.get("role") == "tool"}
    issued = {normalize_call_id(c["id"]) for m in messages if m.get("role") == "assistant"
              for c in m.get("tool_calls") or []}
    instructions: list[str] = []
    items: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = text_of(message.get("content"))
            if text:
                instructions.append(text)
        elif role == "user":
            items.append({"role": "user", "content": _user_content(message.get("content"))})
        elif role == "assistant":
            text = text_of(message.get("content"))
            if text:
                items.append({"role": "assistant", "content": text})
            for call in message.get("tool_calls") or []:
                call_id = normalize_call_id(call["id"])
                if call_id in answered:        # a call with no output would be rejected upstream
                    items.append({"type": "function_call", "call_id": call_id, "name": call["function"]["name"],
                                  "arguments": call["function"].get("arguments") or "{}"})
        elif role == "tool":
            call_id = normalize_call_id(message["tool_call_id"])
            if call_id in issued:
                items.append({"type": "function_call_output", "call_id": call_id,
                              "output": text_of(message.get("content"))})
        else:
            raise BadRequest(f"unsupported message role {role!r}")
    return "\n\n".join(instructions), items


def convert_tool(tool: dict) -> dict:
    if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
        raise BadRequest("only function tools are supported")
    fn = tool["function"]
    parameters = dict(fn.get("parameters") or {})
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return {"type": "function", "name": fn["name"], "description": fn.get("description"),
            "parameters": parameters, "strict": None}


def _text_format(body: dict) -> dict | None:
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_object":
        return {"type": "json_object"}
    spec = json_schema_format(body)
    if spec is None:
        return None
    out = {"type": "json_schema", "name": spec.get("name") or "response", "schema": spec["schema"],
           "strict": bool(spec.get("strict", False))}
    if spec.get("description"):
        out["description"] = spec["description"]
    return out


def build_request(body: dict, model: str) -> tuple[dict, RequestContext]:
    check_single_choice(body)
    instructions, items = convert_messages(body.get("messages") or [])
    payload: dict = {"model": model, "input": items, "store": False, "stream": True}
    if instructions:
        payload["instructions"] = instructions
    tools = [convert_tool(t) for t in body.get("tools") or []]
    choice = body.get("tool_choice")
    if tools and choice != "none":
        items.insert(0, {"type": "additional_tools", "role": "developer", "tools": tools})
        if isinstance(choice, dict) and choice.get("type") == "function":
            payload["tool_choice"] = {"type": "function", "name": choice["function"]["name"]}
        elif choice in ("auto", "required"):
            payload["tool_choice"] = choice
        elif choice is not None:
            raise BadRequest(f"unsupported tool_choice {choice!r}")
        payload["parallel_tool_calls"] = True
    fmt = _text_format(body)
    if fmt:
        payload["text"] = {"format": fmt}
    if body.get("reasoning_effort"):
        payload["reasoning"] = {"effort": body["reasoning_effort"]}
    return payload, RequestContext(dropped=tuple(sorted(set(body) - _CONSUMED)))


class ResponseAccumulator:
    """Folds Responses stream events. ``result`` is set by the terminal event."""

    def __init__(self) -> None:
        self._done_items: dict[int, dict] = {}
        self.result: dict | None = None

    def feed(self, event: str | None, data: dict) -> None:
        kind = data.get("type") or event
        if kind == "response.output_item.done":
            self._done_items[data.get("output_index", len(self._done_items))] = data["item"]
        elif kind in ("response.completed", "response.incomplete"):
            response = dict(data["response"])
            if not response.get("output"):
                response["output"] = [self._done_items[i] for i in sorted(self._done_items)]
            self.result = response
        elif kind == "response.failed":
            err = (data.get("response") or {}).get("error") or {}
            self._fail(err.get("code"), err.get("message"))
        elif kind == "error":
            self._fail(data.get("code"), data.get("message"))

    @staticmethod
    def _fail(code: str | None, message: str | None):
        status = _ERROR_STATUS.get(code or "", 502)
        text = f"chatgpt response failed: {code or 'unknown_error'}: {message or ''}"
        if status in (403, 429):
            text = lead_with_manage_usage(text)
        raise UpstreamError(text[:500], status=status, code=code or "upstream_error")


def usage_of(response: dict) -> dict:
    raw = response.get("usage")
    if not isinstance(raw, dict) or "input_tokens" not in raw or "output_tokens" not in raw:
        raise UpstreamError("chatgpt response carried no usage; refusing to return an unmetered call",
                            code="usage_missing")
    cached = int((raw.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    return {"input": int(raw["input_tokens"]), "output": int(raw["output_tokens"]), "cached": cached}


def to_chat(response: dict, requested_model: str, now: float) -> tuple[dict, dict]:
    usage = usage_of(response)
    texts, calls = [], []
    for item in response.get("output") or []:
        if item["type"] == "message":
            for part in item.get("content") or []:
                if part["type"] == "output_text":
                    texts.append(part["text"])
                elif part["type"] == "refusal":
                    texts.append(part["refusal"])
        elif item["type"] == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}})
    if calls:
        finish = "tool_calls"
    elif response.get("status") == "incomplete":
        finish = "length" if (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens" else "content_filter"
    else:
        finish = "stop"
    return chat_response(response_id=response.get("id"), model=response.get("model") or requested_model, now=now,
                         text="".join(texts) or None, tool_calls=calls, finish_reason=finish, usage=usage), usage
