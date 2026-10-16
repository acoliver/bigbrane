"""JSON schemas for stage outputs that have no record schema under hybrid/research/schemas/.

The record schemas (theory_card, salad_critic, reduction_critic, disprover, feasibility_judge,
analyst, prior_art) come from `hybrid.research.schemas.load_schema`.
"""

PRIOR_ART_DELTA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["adds_untested_element", "element", "why"],
    "properties": {
        "adds_untested_element": {"type": "boolean"},
        "element": {"type": "string"},
        "why": {"type": "string", "minLength": 1},
    },
}

REDUCTION_CHECK = {
    "type": "object",
    "additionalProperties": False,
    "required": ["check_code", "explanation"],
    "properties": {
        "check_code": {"type": ["string", "null"]},
        "explanation": {"type": "string", "minLength": 1},
    },
}

# FIX-12: the repair policy of the family definition (prompt_sections/family_exclusion.md, families.toml forms and the
# reduction critic prompts). A comparator the reduction critic declares (schemas/reduction_critic.schema.json,
# `comparators`) types each change it makes to a published base form with one of these ids. A declared comparator
# whose types are all listed here stays in its base family, so its numeric comparison counts for or against the
# reduction; a comparator with a type outside the list (`other`) is unresolved and never verifies one.
REPAIR_TYPES = {
    "normalizer": "rescales the state or the read by a quantity the comparator computes from its own state or read "
                  "(a count, a norm, a per-channel modulus), elementwise or global",
    "gate": "a fixed, learned or input-dependent gate or decay on the write, the state or the read",
    "feature_map": "a fixed or learned function applied to each token's key, value or query, computed from the input "
                   "tokens only (not from the memory state and not by a search over it)",
    "change_of_basis": "an invertible linear map applied to the state, the keys, the values or the queries",
    "lifted_state": "a larger or lifted state of the same kind: more channels, a state that grows with n or stores "
                    "every token, or a second memory of the same family",
    "sparsity": "a fixed or input-selected sparsity pattern on the state, the write or the read (top-k, slots, a mask)",
    "transformed_input": "the base form run over a transformed token stream (shifted, paired or windowed tokens), "
                         "computed from the input only",
    "multi_read": "repeated reads of the same memory with its write rule unchanged (each read's output is the next "
                  "query), combined by fixed or input-selected weights, including the zero-hop identity term",
}
UNRESOLVED_TYPE = "other"
UNRESOLVED_NOTE = ("any change the list does not name, such as keys or queries generated from the memory state, a "
                   "discrete search over the state, or a new write rule; a comparator with it is unresolved")


def repair_policy_text() -> str:
    """The repair vocabulary as the critic and the check program see it."""
    lines = ["REPAIR POLICY (comparator transformation types; a repair built only from these stays in its base "
             "family):"]
    lines += [f"- {name}: {text}" for name, text in REPAIR_TYPES.items()]
    lines.append(f"- {UNRESOLVED_TYPE}: {UNRESOLVED_NOTE}")
    lines.append("No type covers a term added only to cancel the card's difference from the base form; such a term "
                 "is not a repair, and a comparator with it is not declared.")
    return "\n".join(lines)


# FIX-10, FIX-12: the last line the reduction check program prints. The program reports each comparison it ran as
# data, each under the id of the comparator the critic declared (`declared_id`, repeating that declaration's family,
# transformation types and limit_only), or with declared_id null for anything else it computed. The host
# (stages/s2_prior_art.report_problem and judge_comparisons) checks the report against the declaration and decides
# equivalence from the numbers. Any extra key, such as an `equivalent` flag the program computed itself, is refused,
# so a program cannot hand the host a verdict, and no label the program chooses decides which comparison counts.
REDUCTION_CHECK_REPORT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["comparisons", "notes"],
    "properties": {
        "comparisons": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["declared_id", "comparator", "family", "transformations", "limit_only", "reference",
                             "inputs_tested", "max_abs_diff"],
                "properties": {
                    "declared_id": {"type": ["string", "null"]},
                    "comparator": {"type": "string", "minLength": 1},
                    "family": {"type": "string", "minLength": 1},
                    "transformations": {"type": "array", "maxItems": 8,
                                        "items": {"enum": [*REPAIR_TYPES, UNRESOLVED_TYPE]}},
                    "limit_only": {"type": "boolean"},
                    "reference": {"type": "string"},
                    "inputs_tested": {"type": "integer", "minimum": 1},
                    "max_abs_diff": {"type": "number"},
                },
            },
        },
        "notes": {"type": "string"},
    },
}

FORMALIZER = {
    "type": "object",
    "additionalProperties": False,
    "required": ["obligations"],
    "properties": {
        "obligations": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "claim", "statement", "argument", "known_bounds", "optional"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "kind": {"enum": ["complexity", "capacity", "information", "learnability", "other"]},
                    "claim": {"type": "string", "minLength": 1},
                    "statement": {"type": "string", "minLength": 1},
                    "argument": {"type": "string", "minLength": 1},
                    "known_bounds": {"type": "string"},
                    # true only for an obligation no card claim rests on (s3 obligation checks, OPTIONAL_NOTE)
                    "optional": {"type": "boolean"},
                },
            },
        },
    },
}

BUILDER = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_py", "notes"],
    "properties": {
        "candidate_py": {"type": "string", "minLength": 50},
        "notes": {"type": "string"},
    },
}


def falsifier_design_schema(claim_ids: list[str], entrants: list[str]) -> dict:
    """Output of the falsifier_designer role. claim_ids are the card's claim ids (stages.falsify.claim_items);
    entrants are "candidate", the bench baselines and the trivial references "uniform" and "unigram"."""
    condition = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entrant", "minus", "metric", "group", "op", "value"],
        "properties": {
            "entrant": {"enum": list(entrants)},
            "minus": {"enum": [*entrants, None]},
            "metric": {"enum": ["acc", "kl"]},
            "group": {"type": ["string", "null"]},
            "op": {"enum": [">=", "<=", ">", "<"]},
            "value": {"type": "number"},
        },
    }
    rule = {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims", "quote", "conditions", "if_met"],
        "properties": {
            "claims": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": list(claim_ids)}},
            "quote": {"type": "string", "minLength": 15},
            "conditions": {"type": "array", "minItems": 1, "maxItems": 4, "items": condition},
            "if_met": {"enum": ["supported", "refuted"]},
            # index of a rule of the same task that the candidate must meet first (falsify.control_problems)
            "control_rule": {"type": "integer", "minimum": 0, "maximum": 2},
        },
    }
    protocol_param = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "card", "task"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "card": {"type": "number"},
            "task": {"type": "number"},
            # true for a size of the model itself (d, M), checked against describe()["hyper"]
            "model_side": {"type": "boolean"},
        },
    }
    task = {
        "type": "object",
        "additionalProperties": False,
        "required": ["task_id", "title", "protocol", "deviations", "program", "rules", "protocol_params"],
        "properties": {
            "protocol_params": {"type": "array", "maxItems": 8, "items": protocol_param},
            "solvable_by": {"enum": ["lookup"]},
            # the card's own pair-memory mechanism, scored by the validation job (falsifier_reference.py)
            "reference": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "d", "min_token"],
                "properties": {"kind": {"enum": ["pair_memory"]},
                               "d": {"type": "integer", "minimum": 2, "maximum": 512},
                               "min_token": {"type": "integer", "minimum": 0}},
            },
            "task_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{2,39}$"},
            "title": {"type": "string", "minLength": 1},
            "protocol": {"type": "string", "minLength": 1},
            "deviations": {"type": "string"},
            "program": {"type": "string", "minLength": 100},
            "rules": {"type": "array", "minItems": 1, "maxItems": 3, "items": rule},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tasks", "untestable"],
        "properties": {
            "tasks": {"type": "array", "minItems": 1, "maxItems": 3, "items": task},
            "untestable": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["claim_id", "why"],
                    "properties": {"claim_id": {"enum": list(claim_ids)}, "why": {"type": "string", "minLength": 1}},
                },
            },
        },
    }


def analyst_schema(claim_ids: list[str], task_ids: list[str]) -> dict:
    """The analyst's model output for one card: the record schema `analyst` with claim_id required and limited
    to the card's claim ids, and falsifier_tasks required and limited to the falsifier task ids S5 ran (empty
    when it ran none). falsifier_tasks links an entry to the tasks whose numbers it uses (s6_verdict.merge)."""
    import copy

    from hybrid.research.schemas import load_schema

    schema = copy.deepcopy(load_schema("analyst"))
    schema.pop("$id", None)
    item = schema["properties"]["claims"]["items"]
    item["properties"]["claim_id"] = {"enum": list(claim_ids)}
    tasks = item["properties"]["falsifier_tasks"]
    tasks["items"] = {"enum": list(task_ids)} if task_ids else {"type": "string"}
    if not task_ids:
        tasks["maxItems"] = 0
    item["required"] = ["claim_id", *item["required"], "falsifier_tasks"]
    return schema
