"""Helpers shared by both translators: reading OpenAI chat requests and building chat responses."""
from __future__ import annotations

import json
import uuid

from .errors import BadRequest

# Chat request fields each translator consumes; everything else the client sent is reported as dropped.
COMMON_FIELDS = {"model", "messages", "tools", "tool_choice", "response_format", "stream"}


def text_of(content) -> str:
    """Flatten message content (string or list of parts) to text. Non-text parts are rejected."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                raise BadRequest("only text content parts are supported here")
        return "".join(parts)
    raise BadRequest("message content must be a string or a list of parts")


def tool_arguments(call: dict) -> dict:
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except (KeyError, ValueError) as exc:
        raise BadRequest("tool call arguments are not valid JSON") from exc
    if not isinstance(args, dict):
        raise BadRequest("tool call arguments must be a JSON object")
    return args


def json_schema_format(body: dict) -> dict | None:
    """The ``json_schema`` object of response_format, or None. Other formats besides text are refused."""
    fmt = body.get("response_format")
    if fmt is None or fmt.get("type") == "text":
        return None
    if fmt.get("type") != "json_schema":
        raise BadRequest(f"response_format type {fmt.get('type')!r} is not supported by this backend")
    spec = fmt.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise BadRequest("response_format.json_schema.schema must be an object")
    return spec


def check_single_choice(body: dict) -> None:
    if body.get("n", 1) != 1:
        raise BadRequest("n greater than 1 is not supported")


def chat_response(*, response_id: str | None, model: str, now: float, text: str | None, tool_calls: list[dict],
                  finish_reason: str, usage: dict) -> dict:
    """OpenAI chat.completion body. ``usage`` is {"input", "output", "cached"} with input counting cached tokens."""
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(now),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                  "total_tokens": usage["input"] + usage["output"],
                  "prompt_tokens_details": {"cached_tokens": usage["cached"]}},
    }
