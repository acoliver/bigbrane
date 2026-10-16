"""JSON schemas for the two model calls made by classify_prior_art."""
from __future__ import annotations

PRIOR_ART_LABELS = ("novel", "prior_art_tested", "prior_art_undertested", "known_disproven")
EVIDENCE_RELATIONS = (
    "same_mechanism",
    "closest_prior_work",
    "tests_similar_claim",
    "refutes_claim",
    "reduction_source",
)

QUERY_PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 3, "maxLength": 200},
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

CLASSIFICATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {"enum": list(PRIOR_ART_LABELS)},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"enum": ["arxiv", "openalex", "s2"]},
                    "source_id": {"type": "string", "minLength": 1},
                    "relation": {"enum": list(EVIDENCE_RELATIONS)},
                },
                "required": ["source", "source_id", "relation"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["label", "evidence", "notes"],
    "additionalProperties": False,
}
