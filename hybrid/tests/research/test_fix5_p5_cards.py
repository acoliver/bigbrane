"""FIX-5 item 2: every calibration card defines a target-rule class for each P4 task, justifies its size and states
log2 |H|, prior bits and bits per whole instance (results/phase4-20261013T073748Z/FIX-5.md; the P5 rule of FIX-4:
a sample entry needs hypothesis_class {definition, size_justification, log2_size}). The cards replaced one uniform
"log2 |H| = 8" class with no definition and "2 bits of answer" per example, which p4comp-diag/REPORT.md found
unsupported. The example counts m are unchanged."""
from __future__ import annotations

import math
import re

from hybrid.math import constructions
from hybrid.research import calibration

SIZES = {"k1": (4, 4, 2, 4), "k2": (4, 4, 2, 4), "k3": (2, 2, 2, 2), "k4": (2, 1, 2, 1)}
COUNTS = {"k1": (3000, 3000, 1500, 4000), "k2": (1500, 2500, 1200, 3000), "k3": (400, 400, 300, 400),
          "k4": (300, 300, 300, 300)}


def account(card: dict) -> str:
    return card["claims"]["sample_efficiency"]["mechanism"]


def section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_each_card_keeps_its_counts_and_drops_the_unsupported_uniform_account():
    for key, card in calibration.load_cards().items():          # schema and anonymization run here
        text = account(card)
        L, K, _, T = SIZES[key]
        for count, task in zip(COUNTS[key], (f"copying at L = {L}", f"recall at K = {K}",
                                             "bracket nesting at depth 2", f"two-hop lookup at table size {T}")):
            assert f"{count} for {task}" in text, (key, task)
        assert "log2 |H| = 8 for the hypothesis class" not in text and "2 bits of answer" not in text


def test_each_class_size_matches_its_definition():
    for key, card in calibration.load_cards().items():
        text = account(card)
        L, K, _, T = SIZES[key]
        copy = section(text, f"Copying at L = {L}.", "Recall at")
        recall = section(text, f"Recall at K = {K}.", "Bracket nesting")
        dyck = section(text, "Bracket nesting at depth 2.", "Two-hop lookup")
        comp = section(text, f"Two-hop lookup at table size {T}.", "Prior bits")
        for part in (copy, recall, dyck, comp):
            assert "Class definition:" in part and "Size justification:" in part
        assert f"|H| = {L}^{L} = {L ** L}" in copy and f"log2 |H| = {L * math.log2(L):g}" in copy
        if K > 1:
            assert f"|H| = {K}^{K} = {K ** K}" in recall and f"log2 |H| = {K * math.log2(K):g}" in recall
        else:
            assert "|H| = 1" in recall and "log2 |H| = 0" in recall
        assert "|H| = 2^6 = 64" in dyck and "log2 |H| = 6" in dyck
        if T > 1:
            assert f"|H| = {T}^{T} = {T ** T}" in comp and f"log2 |H| = {T * math.log2(T):g}" in comp
        else:
            assert "|H| = 1" in comp and "log2 |H| = 0" in comp


def test_the_bits_per_instance_follow_the_generator():
    """Upper bounds on one whole instance: scored answers times log2 of the answer alphabet, measured here."""
    for key, card in calibration.load_cards().items():
        text = section(account(card), "Bits per example (one whole instance):", "The account assumes")
        L, K, depth, T = SIZES[key]
        expected = []
        for task, size in (("copy", L), ("recall", K), ("dyck", depth), ("composition", T)):
            instance = constructions.generate(task, size, 1, 0, None)[0]
            alphabet = {"copy": 4, "recall": 16, "dyck": 2, "composition": T}[task]
            expected.append(len(instance.queries) * math.log2(alphabet))
        stated = [float(x) for x in re.findall(r"(?:at most |, |and )(\d+(?:\.\d+)?) for ", text)]
        assert stated[:3] == expected[:3], (key, stated, expected)
        assert stated[3] == (expected[3] if T > 1 else 1.0), (key, stated)      # 1 is the loose bound at T = 1


def test_prior_bits_are_stated_relative_to_the_declared_class():
    for card in calibration.load_cards().values():
        text = account(card)
        assert "Prior bits: 0, relative to these declared classes" in text
        assert "not a claim that learning the parsing or lookup operation is free" in text
