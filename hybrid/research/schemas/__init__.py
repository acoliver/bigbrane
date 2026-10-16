"""JSON Schemas for research records and stage critic outputs.

Each schema lives in its own file as ``<name>.schema.json``; ``load_schema`` addresses them by name.
"""
import json
from functools import lru_cache
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parent


class SchemaViolation(ValueError):
    """An instance does not satisfy the named schema."""


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    path = SCHEMA_DIR / f"{name}.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


@lru_cache(maxsize=None)
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name))


def validate(name: str, instance: dict) -> dict:
    error = jsonschema.exceptions.best_match(_validator(name).iter_errors(instance))
    if error is not None:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise SchemaViolation(f"{name}: {location}: {error.message}")
    return instance
