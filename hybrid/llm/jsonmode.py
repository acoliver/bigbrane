"""JSON schema mode helpers: instruct, extract the top-level JSON objects of a reply, validate.

A reply may hold more than one top-level object (a fragment and then the full answer, as an A2 schema retry did in
the Phase 6 draft run); the first object that validates is used. A reply cut off inside its object (as the k1 prover
reply of the Phase 4 calibration was) is reported as never closed, never by the keys of an object nested inside it.
A rejected reply's message names every failing field by its dotted path (``field_problems``), so the retry can fix
them all at once."""
from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator

_DECODER = json.JSONDecoder()
# Schema errors quoted to the model on a retry, so it learns every wrong key at once rather than one per retry.
MAX_REPORTED_ERRORS = 5
# Field problems quoted on a retry; enough for any real draft, bounded for a reply that is not the object at all.
MAX_REPORTED_FIELDS = 20


class UnclosedJSON(ValueError):
    """The output opens a JSON object and ends inside it (a reply cut off by the output limit)."""


def check_schema(schema: dict) -> None:
    Draft202012Validator.check_schema(schema)


def schema_instruction(schema: dict) -> str:
    return ("\n\n# OUTPUT FORMAT\n"
            "Reply with exactly one JSON object that validates against the JSON Schema below. "
            "Do not add commentary before or after the object.\n\n"
            f"```json\n{json.dumps(schema, indent=2, sort_keys=True)}\n```\n")


def retry_instruction(previous_output: str, error: str, *, truncated: bool = False) -> str:
    advice = ("Your previous reply was cut off by the output limit before the JSON object was complete. Reply again "
              "with the complete object and keep it shorter: brief notes, no commentary, less deliberation.\n"
              if truncated else "")
    return ("\n\n# PREVIOUS ATTEMPT FAILED\n"
            f"Your previous reply was:\n<<<\n{previous_output}\n>>>\n"
            f"It was rejected: {error}\n" + advice
            + "Reply again with exactly one complete JSON object that validates against the schema above: every "
              "required field filled in, no field the schema does not list, no placeholder keys or values, and no "
              "partial object or commentary before or after it.\n")


def _group_end(text: str, start: int) -> int | None:
    """Index just past the brace group that opens at ``start`` (braces inside JSON strings ignored), or None when the
    text ends inside the group."""
    depth = 0
    in_string = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _scan(text: str) -> tuple[list[dict], str | None]:
    """Every top-level JSON object in the text, in order, and the message for a brace group still open at the end of
    the text (None when there is none). Objects nested inside a brace group that does not parse are never returned:
    when a reply is cut off inside ``{"attempts": [{...}, {...``, the first inner attempt object is valid JSON, and
    returning it would report a schema error about the wrong object."""
    objects: list[dict] = []
    index = text.find("{")
    while index != -1:
        try:
            value, end = _DECODER.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            end = _group_end(text, index)
            if end is None:
                return objects, (f"the JSON object that starts at character {index} is never closed: the output "
                                 f"ends inside it ({exc.msg}: character {exc.pos} of {len(text)})")
            index = text.find("{", end)
            continue
        objects.append(value)
        index = text.find("{", end)
    return objects, None


def json_objects(text: str) -> list[dict]:
    """The top-level JSON objects embedded in text (plain, fenced, or surrounded by prose), in order."""
    return _scan(text)[0]


def extract_first_json_object(text: str) -> dict:
    """Return the first top-level JSON object embedded in text. A text with no complete object raises UnclosedJSON
    when a group is still open at its end, ValueError otherwise."""
    objects, unclosed = _scan(text)
    if objects:
        return objects[0]
    if unclosed:
        raise UnclosedJSON(unclosed)
    raise ValueError("no JSON object found in model output")


def _errors(parsed: dict, schema: dict) -> list:
    return sorted(Draft202012Validator(schema).iter_errors(parsed), key=lambda e: list(map(str, e.absolute_path)))


def _one_each(error) -> list[str]:
    """For an array whose items carry a required enum id with exactly one item per value (``minItems`` equal to the
    number of values), the values no item has and the values more than one item has."""
    items = error.schema.get("items") or {}
    notes = []
    for name in items.get("required", []):
        values = (items.get("properties", {}).get(name) or {}).get("enum")
        if not values or error.schema.get("minItems") != len(values):
            continue
        seen = [item.get(name) for item in error.instance if isinstance(item, dict)]
        absent = [v for v in values if v not in seen]
        repeated = [v for v in values if seen.count(v) > 1]
        if absent:
            notes.append(f"no item has {name} " + ", ".join(map(str, absent)))
        if repeated:
            notes.append(f"more than one item has {name} " + ", ".join(map(str, repeated)))
    return notes


def _message(error) -> str:
    """The validator's message, except that an array of the wrong length is described by its count and, when its items
    carry one id each, by the ids that are missing or repeated. The default message quotes the whole array ("[...] is
    too short"): in FIX-7 item 6 the local matcher, shown ten thousand characters of its own translations and "is too
    short", left the same statement out again on the retry."""
    if error.validator in ("minItems", "maxItems") and isinstance(error.instance, list):
        exact = error.schema.get("minItems") == error.schema.get("maxItems")
        bound = "exactly" if exact else "at least" if error.validator == "minItems" else "at most"
        notes = _one_each(error)
        return (f"has {len(error.instance)} items; the schema requires {bound} {error.validator_value}"
                + "".join("; " + note for note in notes))
    return error.message


def validation_errors(parsed: dict, schema: dict) -> list[str]:
    return [f"at {'/'.join(str(part) for part in e.absolute_path) or '(root)'}: {_message(e)}"
            for e in _errors(parsed, schema)]


def dotted_path(path) -> str:
    """``claims.complexity.constants_notes``, ``items[1].task``; the empty string for the root."""
    text = ""
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else ("." if text else "") + str(part)
    return text


def _extra_keys(error) -> list[str]:
    properties = error.schema.get("properties", {})
    patterns = list(error.schema.get("patternProperties", {}))
    return [key for key in error.instance
            if key not in properties and not any(re.search(pattern, key) for pattern in patterns)]


def field_problems(parsed: dict, schema: dict) -> list[str]:
    """One line per failing field, by dotted path: ``a is missing (required)``, ``b is not allowed (remove it)``,
    ``items[1].task: 3 is not of type 'string'``."""
    problems: list[str] = []
    for error in _errors(parsed, schema):
        where = dotted_path(error.absolute_path)
        prefix = f"{where}." if where else ""
        if error.validator == "required" and isinstance(error.instance, dict):
            lines = [f"{prefix}{name} is missing (required)" for name in error.validator_value
                     if name not in error.instance]
        elif error.validator == "additionalProperties" and error.validator_value is False \
                and isinstance(error.instance, dict):
            lines = [f"{prefix}{key} is not allowed (remove it)" for key in _extra_keys(error)]
        else:
            lines = [f"{where or '(root)'}: {_message(error)}"]
        problems.extend(line for line in lines if line not in problems)
    return problems


def _rejection(parsed: dict, schema: dict) -> str:
    errors = validation_errors(parsed, schema)
    shown = errors[:MAX_REPORTED_ERRORS]
    more = f" (and {len(errors) - len(shown)} more)" if len(errors) > len(shown) else ""
    problems = field_problems(parsed, schema)
    listed = problems[:MAX_REPORTED_FIELDS]
    more_fields = (f" (and {len(problems) - len(listed)} more)" if len(problems) > len(listed) else "")
    return ("schema validation failed " + "; ".join(shown) + more
            + ". Fields to fix: " + "; ".join(listed) + more_fields)


def parse_and_validate(text: str, schema: dict) -> tuple[dict | None, str | None]:
    """Return (parsed, None) for the first top-level object that validates, or (None, error message). The message
    names up to MAX_REPORTED_ERRORS schema errors with their paths and then every failing field
    (``field_problems``). When the reply holds several objects and none validates, the one with the fewest field
    problems is reported and the message says how many objects there were."""
    objects, unclosed = _scan(text)
    if not objects:
        return None, unclosed or "no JSON object found in model output"
    best = None
    for position, parsed in enumerate(objects):
        count = len(field_problems(parsed, schema))
        if count == 0 and not validation_errors(parsed, schema):
            return parsed, None
        if best is None or count < best[0]:
            best = (count, position, parsed)
    _, position, parsed = best
    message = _rejection(parsed, schema)
    if len(objects) > 1:
        message += (f". The reply held {len(objects)} JSON objects and none validates; the problems above are those "
                    f"of object {position + 1}, the one with the fewest")
    if unclosed:
        message += f". After them, {unclosed}"
    return None, message
