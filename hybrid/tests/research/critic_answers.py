"""Reduction critic answers for the novelty fields (decisions R2A, R2C), shared by the S2 tests. KNOWN is a synthetic routing declaration, not recorded source proof."""
KNOWN = {"known_work": True, "known_work_sources": [{"kind": "textbook", "ref": "Hopcroft, Motwani and Ullman, section 6.2",
                                                    "covers": "the update, the readout and the learning rule"}],
         "fixed_learned_parts": [], "surprise": False}
NOT_KNOWN = {"known_work": False, "known_work_sources": [], "fixed_learned_parts": [], "surprise": True}


def novelty_fields(record: dict) -> dict:
    """Legacy resemblance judgments contain no checked source or executable reference.

    Keep them as unclaimed known work for routing. The source fixture retains radical_enough;
    translating it into a textbook citation would manufacture evidence the recording never had.
    """
    return dict(NOT_KNOWN)
