"""JSON schemas for the three model calls made by classify_prior_art."""
from __future__ import annotations

PRIOR_ART_LABELS = ("novel", "prior_art_tested", "prior_art_undertested", "known_disproven")
EVIDENCE_RELATIONS = (
    "same_mechanism",
    "closest_prior_work",
    "tests_similar_claim",
    "refutes_claim",
    "reduction_source",
)

# PA2: round 1 restates the mechanism in the vocabulary of several fields; the referee's closest works for the flip
# cards sat in program synthesis, factorization machines and sparse associative memory, not in sequence learning.
QUERY_FIELDS = (
    "automata_formal_languages",
    "compression_coding",
    "cognitive_bayesian",
    "physics_dynamical_systems",
    "optimization",
    "program_synthesis",
    "associative_memory",
    "signal_processing",
)

QUERY_PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 10,
            "maxItems": 14,
            "uniqueItems": True,
            "description": "Distinct queries using at least four different fields; case and whitespace do not distinguish queries.",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"enum": list(QUERY_FIELDS)},
                    "query": {"type": "string", "minLength": 3, "maxLength": 200},
                },
                "required": ["field", "query"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

# Round 2: queries the model writes after it has seen the titles round 1 returned.
REFINE_PROPOSAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 0,
            "maxItems": 4,
            "uniqueItems": True,
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
        "coverage_assessment": {
            "type": "object",
            "properties": {
                "adequate_for_novelty": {"type": "boolean"},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["adequate_for_novelty", "reason"],
            "additionalProperties": False,
        },
    },
    "required": ["label", "evidence", "notes", "coverage_assessment"],
    "additionalProperties": False,
}
