"""Reduction critic answers for the novelty fields (decisions R2A, R2C), shared by the S2 tests."""
KNOWN = {"known_work": True, "known_work_sources": [{"kind": "textbook", "ref": "Hopcroft, Motwani and Ullman, section 6.2",
                                                    "covers": "the update, the readout and the learning rule"}],
         "fixed_learned_parts": [], "surprise": False}
NOT_KNOWN = {"known_work": False, "known_work_sources": [], "fixed_learned_parts": [], "surprise": True}


def novelty_fields(record: dict) -> dict:
    """The novelty fields for a critic artifact recorded before decision R2C: its radical_enough answer, read as
    known work when false (with a textbook source) and as not known when true."""
    return dict(NOT_KNOWN if record["radical_enough"] else KNOWN)
