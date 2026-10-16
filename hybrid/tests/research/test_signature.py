"""Mechanism signature (plan section 9): unchanged by renaming variables, roles and titles; different for different
mechanisms; dimension names dropped; operators from a fixed vocabulary."""
import copy
import itertools

import pytest

from hybrid.research.signature import (LEARN_OPS, WRITE_OPS, card_signature, extract_skeleton, jaccard, mask,
                                       normalize, signature, tokens)
from hybrid.tests.research.builders import load_example
from hybrid.tests.research.outcome_fixtures import KNOWN_MECHANISMS, REAL_CARDS, real_card, renamed

REAL = sorted(path.stem for path in REAL_CARDS.glob("card-*.json"))


@pytest.mark.parametrize("card_id", REAL)
def test_real_cards_keep_their_signature_under_renaming(card_id):
    card = real_card(card_id)
    other = renamed(card, extra={"x_t": "tok_now", "k": "kk9", "q": "qq9"})
    assert other["title"] != card["title"] and other["mechanism"] != card["mechanism"]
    assert card_signature(other)["signature"] == card_signature(card)["signature"]
    assert card_signature(other)["tokens"] == card_signature(card)["tokens"]


@pytest.mark.parametrize("name", sorted(KNOWN_MECHANISMS))
def test_known_mechanisms_keep_their_signature_under_renaming(name):
    card = KNOWN_MECHANISMS[name]
    other = renamed(card, extra={"x_t": "w9", "q_t": "probe", "k_t": "kk", "v_t": "vv", "x_prev": "before"})
    assert card_signature(other)["signature"] == card_signature(card)["signature"]


def test_the_example_card_keeps_its_signature_and_reads_its_running_sum():
    card = load_example()
    other = renamed(card, extra={"k_t": "key9", "v_t": "val9", "theta": "ang"})
    assert card_signature(other)["signature"] == card_signature(card)["signature"]
    skeleton = extract_skeleton(card)
    assert "additive_accumulate" in skeleton["write"] and "exponential_weighting" in skeleton["read"]


def test_title_summary_lens_roles_and_meanings_are_never_read():
    card = KNOWN_MECHANISMS["linear_attention"]
    other = copy.deepcopy(card)
    other["title"] = "softmax argmax union stack posterior"     # operator words in prose fields only
    other["summary"] = other["core_idea"] = "softmax sort push pop hash sample"
    other["lens"] = "stack_machines"
    for item in other["mechanism"]["state"]:
        item["meaning"] = "a softmax-sorted stack that is hashed and sampled"
    other["mechanism"]["update"][0]["name"] = "softmax push"
    assert card_signature(other)["signature"] == card_signature(card)["signature"]


def test_a_state_name_that_is_an_operator_word_is_still_masked():
    card = KNOWN_MECHANISMS["ngram_counts"]
    other = renamed(card)
    other = renamed(other, extra={"zq0w": "table"})   # "table" would read as a lookup if it were not a state name
    other["mechanism"]["state"][0]["name"] = "table"
    assert card_signature(other)["signature"] == card_signature(card)["signature"]


def test_distinct_mechanisms_get_distinct_signatures():
    sigs = {name: card_signature(card) for name, card in KNOWN_MECHANISMS.items()}
    assert len({s["signature"] for s in sigs.values()}) == len(KNOWN_MECHANISMS)
    for (a, sa), (b, sb) in itertools.combinations(sigs.items(), 2):
        assert jaccard(sa["tokens"], sb["tokens"]) < 1.0, (a, b)


def test_the_extractor_names_the_defining_operator_of_each_family():
    ops = {name: extract_skeleton(card) for name, card in KNOWN_MECHANISMS.items()}
    assert {"outer_product_write", "additive_accumulate"} <= set(ops["linear_attention"]["write"])
    assert "error_correction" in ops["delta_rule"]["write"]
    assert "error_correction" not in ops["linear_attention"]["write"]
    assert "softmax_read" in ops["softmax_attention"]["read"]
    assert "merge_equivalence_classes" in ops["union_find"]["write"]
    assert "stack_push_pop" in ops["stack_machine"]["write"]
    assert "count_increment" in ops["ngram_counts"]["write"]
    assert "bayesian_update" in ops["hmm_forward"]["learn"]
    assert "outer_product_write" not in ops["ssm"]["write"]


def test_dimension_names_are_dropped_but_rank_counts():
    base = {"write": ["outer_product_write"], "read": [], "learn": [], "state": [["d", "d"]]}
    assert signature(base) == signature(dict(base, state=[["k", "m"]]))
    assert signature(base) == signature(dict(base, state=[["d", "d"], ["n", "n"]]))
    assert signature(base) != signature(dict(base, state=[["d"]]))
    assert normalize(dict(base, state=[[1], ["V", 1], ["a", "b", "c", "d"]]))["state"] == ["rank0", "rank1", "rank3+"]


def test_operators_are_sorted_and_deduplicated_and_must_come_from_the_vocabulary():
    a = {"write": ["outer_product_write", "additive_accumulate", "outer_product_write"], "read": [], "learn": []}
    b = {"write": ["additive_accumulate", "outer_product_write"], "read": [], "learn": []}
    assert signature(a) == signature(b)
    assert signature(a) != signature({"write": [], "read": b["write"], "learn": []})
    with pytest.raises(ValueError, match="outside the vocabulary"):
        normalize({"write": ["attention_but_renamed"]})
    with pytest.raises(ValueError, match="unknown keys"):
        normalize({"write": [], "title": "x"})
    assert set(WRITE_OPS).isdisjoint({"gradient_training"}) and "gradient_training" in LEARN_OPS


def test_commitments_enter_the_signature_only_when_all_five_axes_are_given():
    base = {"write": ["argmax_select"], "read": [], "learn": []}
    full = {"substrate": "program", "learning_rule": "search", "unit": "word", "objective": "description length",
            "structure_source": "search at inference"}
    assert signature(dict(base, commitments=full)) != signature(base)
    assert "c:unit=word" in tokens(dict(base, commitments=full))
    with pytest.raises(ValueError, match="commitments"):
        normalize(dict(base, commitments={"unit": "word"}))


def test_mask_numbers_identifiers_by_first_appearance_and_keeps_operator_words():
    assert mask("R = R + conj(E[x])", frozenset({"R", "E"})) == "§1 = §1 + conj(§2[§3])"
    assert mask("Mem = Mem + conj(Emb[y])", frozenset({"Mem", "Emb"})) == "§1 = §1 + conj(§2[§3])"
    assert mask(r"S_t = S_{t-1} + v_t k_t^T") == r"§1[t] = §1[t-1] + §2[t] §3[t]^\top "
    assert mask("v-Sk") == "§1-§2"
