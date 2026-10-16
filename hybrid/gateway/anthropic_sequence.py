"""Message-sequence repair for the Anthropic Messages API, ported from LLxprt Code.

Source: packages/providers/src/anthropic/AnthropicMessageValidator.ts (validateToolResults, enforceToolResultAdjacency,
ensureValidMessageSequence, stripEmptyTextBlocks), applied in the order convertToAnthropicMessages applies them.

Messages are ``{"role", "content": [blocks]}`` with the content always a list of blocks.
"""
from __future__ import annotations

from .anthropic_models import supports_prefill

INTERRUPTED_RESULT = "[tool execution interrupted]"
CONTINUE_TEXT = "Continue the conversation"
EMPTY_PLACEHOLDER = {"assistant": "[No content generated]", "user": "[Empty message]"}


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _blocks(message: dict, kind: str) -> list[dict]:
    return [b for b in message["content"] if b["type"] == kind]


def _is_blank_text(block: dict) -> bool:
    return block["type"] == "text" and not block.get("text", "").strip()


def validate_tool_results(messages: list[dict]) -> list[dict]:
    """Drop tool_result blocks whose tool_use is not in the conversation, and turns left empty by that."""
    tool_use_ids = {b["id"] for m in messages if m["role"] == "assistant" for b in _blocks(m, "tool_use")}
    result = []
    for message in messages:
        if message["role"] == "user" and any(b["type"] == "tool_result" for b in message["content"]):
            kept = [b for b in message["content"] if b["type"] != "tool_result" or b["tool_use_id"] in tool_use_ids]
            if not kept:
                continue
            message = {**message, "content": kept}
        result.append(message)
    return result


def _interrupted(tool_use_id: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": INTERRUPTED_RESULT, "is_error": True}


def _take_later_result(messages: list[dict], start: int, tool_use_id: str) -> dict | None:
    """Remove and return the tool_result for ``tool_use_id`` from a user turn after ``start``. A turn this empties is
    removed too."""
    for j in range(start + 1, len(messages)):
        later = messages[j]
        if later["role"] != "user":
            continue
        for k, block in enumerate(later["content"]):
            if block["type"] == "tool_result" and block["tool_use_id"] == tool_use_id:
                del later["content"][k]
                if not later["content"]:
                    del messages[j]
                return block
    return None


def enforce_tool_result_adjacency(messages: list[dict]) -> list[dict]:
    """Each tool_use turn is followed by a user turn holding its results; missing results are synthesized."""
    result = [{**m, "content": list(m["content"])} for m in messages]
    i = 0
    while i < len(result):
        message = result[i]
        if message["role"] == "assistant":
            use_ids = [b["id"] for b in _blocks(message, "tool_use")]
            if use_ids:
                _enforce_for(result, i, use_ids)
        i += 1
    return result


def _enforce_for(messages: list[dict], index: int, use_ids: list[str]) -> None:
    nxt = index + 1
    if nxt < len(messages) and messages[nxt]["role"] == "user":
        present = {b["tool_use_id"] for b in _blocks(messages[nxt], "tool_result")}
        collected = []
        for tool_use_id in (i for i in use_ids if i not in present):
            collected.append(_take_later_result(messages, nxt, tool_use_id) or _interrupted(tool_use_id))
        messages[nxt]["content"][:0] = collected
        return
    answered = {b["tool_use_id"] for m in messages if m["role"] == "user" for b in _blocks(m, "tool_result")}
    missing = [i for i in use_ids if i not in answered]
    if missing:
        messages.insert(nxt, {"role": "user", "content": [_interrupted(i) for i in missing]})


def _merge_consecutive(messages: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            combined = merged[-1]["content"] + message["content"]
            if message["role"] == "user":      # tool results first, as the API requires
                combined = [b for b in combined if b["type"] == "tool_result"] + \
                           [b for b in combined if b["type"] != "tool_result"]
            merged[-1] = {**merged[-1], "content": combined}
        else:
            merged.append({**message, "content": list(message["content"])})
    return merged


def _is_empty(message: dict) -> bool:
    content = message["content"]
    return not content or all(_is_blank_text(b) for b in content)


def ensure_valid_message_sequence(messages: list[dict], *, thinking: bool, model: str) -> list[dict]:
    """Merge adjacent turns, start with a user turn, never be empty, and do not end on an assistant turn when
    thinking is on or the model rejects prefill."""
    result = _merge_consecutive(messages)
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": [_text(CONTINUE_TEXT)]})
    if not result:
        result = [{"role": "user", "content": [_text("Hello")]}]
    last = len(result) - 1
    result = [{**m, "content": [_text(EMPTY_PLACEHOLDER[m["role"]])]} if i != last and _is_empty(m) else m
              for i, m in enumerate(result)]
    if result[-1]["role"] == "assistant" and (thinking or not supports_prefill(model)):
        result.append({"role": "user", "content": [_text(CONTINUE_TEXT)]})
    return result


def strip_empty_text_blocks(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        kept = [b for b in message["content"] if not _is_blank_text(b)]
        result.append(message if len(kept) == len(message["content"]) and kept
                      else {**message, "content": kept or [_text(EMPTY_PLACEHOLDER[message["role"]])]})
    return result


def repair(messages: list[dict], *, thinking: bool, model: str) -> list[dict]:
    messages = validate_tool_results(messages)
    messages = enforce_tool_result_adjacency(messages)
    messages = ensure_valid_message_sequence(messages, thinking=thinking, model=model)
    return strip_empty_text_blocks(messages)
