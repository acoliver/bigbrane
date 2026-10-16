"""JSON schemas for Anthropic native structured outputs (``output_config.format``).

Some models (Opus 5.5, Sonnet 5.5, Fable 5.1) refuse the forced ``tool_choice`` the gateway otherwise uses for
``response_format: json_schema``, so they get native structured outputs. That feature is stricter than JSON Schema:
every object must say ``additionalProperties: false`` and a set of validation keywords is refused (verified live on
2026-10-10: "'additionalProperties' must be explicitly set to false", "property 'minimum' is not supported").
This module rewrites a caller's schema into the accepted subset. A keyword it removes is no longer enforced by the
model, so the removed constraint is appended to the property's description where the model can still read it.
"""
from __future__ import annotations

import copy

# Validation keywords structured outputs refuse. Structural keywords (type, properties, items, required, enum,
# const, anyOf, allOf, $ref, $defs) and annotations (description, title, default) are kept.
_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
                "maxItems", "uniqueItems", "minProperties", "maxProperties", "patternProperties", "propertyNames",
                "contains", "minContains", "maxContains", "not", "if", "then", "else", "dependentSchemas",
                "dependentRequired", "unevaluatedProperties", "unevaluatedItems")
_SCHEMA_VALUED = ("items", "additionalItems")
_SCHEMA_MAPS = ("properties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")


def to_structured_output_schema(schema: dict) -> dict:
    return _rewrite(copy.deepcopy(schema))


def _rewrite(node):
    if not isinstance(node, dict):
        return node
    notes = []
    for key in _UNSUPPORTED:
        if key in node:
            notes.append(f"{key}: {node.pop(key)}")
    if isinstance(node.get("minItems"), int) and node["minItems"] > 1:
        notes.append(f"minItems: {node.pop('minItems')}")
    if notes:
        note = "Constraint not enforced by the API (" + "; ".join(notes) + ")."
        node["description"] = f"{node['description']} {note}" if node.get("description") else note
    kind = node.get("type")
    if kind == "object" or (isinstance(kind, list) and "object" in kind) or "properties" in node:
        node["additionalProperties"] = False
    for key in _SCHEMA_VALUED:
        if key in node:
            node[key] = _rewrite(node[key])
    for key in _SCHEMA_MAPS:
        if isinstance(node.get(key), dict):
            node[key] = {name: _rewrite(child) for name, child in node[key].items()}
    for key in _SCHEMA_LISTS:
        if isinstance(node.get(key), list):
            node[key] = [_rewrite(child) for child in node[key]]
    return node
