"""KD7: the difference writer and checker are shown the prior-art papers with their titles and excerpts, and the host
accepts the refs of exactly the papers that are listed."""
import asyncio
from types import SimpleNamespace

from hybrid.research.stages import difference
from hybrid.tests.research.builders import make_card

FAMILIES, _ = difference._families("hybrid/config/families.toml", "hybrid/config/archive.toml")
SDM = "s2:712c23c8bee6077b726a211dfb3fea42cd102e81"


def paper(key, title, excerpt="An abstract.", year=2020):
    source, source_id = key.split(":", 1)
    return {"key": key, "source": source, "source_id": source_id, "title": title, "year": year, "url": "",
            "excerpt": excerpt}


def search(count=3):
    """The adaptive SDM paper (no abstract) among arXiv papers, in the order a search returns them."""
    papers = [paper(f"arxiv:2401.{n:05d}", f"Paper number {n}", f"Excerpt of paper {n}.") for n in range(count)]
    papers.insert(1, paper(SDM, "An adaptive sparse distributed memory", "(no abstract)", 2003))
    return papers


def evidence_of(key, relation="closest_prior_work"):
    return {"source_id": key.split(":", 1)[1], "title": "ignored", "url": "", "relation": relation}


def requirements(candidates, evidence=()):
    record = {"label": "novel", "evidence": list(evidence)}
    ctx = SimpleNamespace(outcomes=None, cfg={"outcomes": {"enforce_r3": False}})
    critic = {"family": "none", "known_work_sources": []}
    return record, difference.requirements(ctx, record, {"candidates": candidates}, [], critic, FAMILIES)


def test_the_cited_paper_comes_first_with_its_title_then_the_others_in_search_order():
    record, req = requirements(search(), [evidence_of(SDM)])

    block = difference.context_block(record, req, {"reduces_to": "none"})

    assert "title not recorded" not in block
    lines = [line for line in block.split("OWN HISTORY")[0].splitlines() if line.startswith("- ")]
    assert lines == [
        f"- {SDM} (2003) An adaptive sparse distributed memory [cited by the prior-art search as closest_prior_work]",
        "- arxiv:2401.00000 (2020) Paper number 0",
        "- arxiv:2401.00001 (2020) Paper number 1",
        "- arxiv:2401.00002 (2020) Paper number 2",
    ]
    assert "Excerpt of paper 1." in block and "4 of the 4 papers" in block


def test_a_ref_is_accepted_as_the_key_or_the_bare_source_id():
    _, req = requirements(search(), [evidence_of(SDM), evidence_of("arxiv:2401.00000")])
    card = make_card("card-0000a001")
    for ref in (SDM, SDM.split(":", 1)[1]):
        statement = {"closest_prior": [{"kind": "literature", "ref": ref},
                                       {"kind": "literature", "ref": "arxiv:2401.00000"}],
                     "changed_equation": "learning", "cause_ids_addressed": []}
        assert difference.host_problems(statement, card, req) == {}


def test_a_paper_beyond_the_listing_bound_is_not_shown_and_not_accepted():
    candidates = [paper(f"arxiv:2402.{n:05d}", f"Many {n}") for n in range(difference.LISTED_LITERATURE + 5)]
    last = candidates[-1]["key"]
    record, req = requirements(candidates, [evidence_of(last)])

    block = difference.context_block(record, req, {"reduces_to": "none"})

    assert len({entry["key"] for entry in req.literature.values()}) == difference.LISTED_LITERATURE
    assert last in req.literature and candidates[difference.LISTED_LITERATURE - 1]["key"] not in req.literature
    assert block.index(last) < block.index(candidates[0]["key"])
    assert f"{difference.LISTED_LITERATURE} of the {len(candidates)} papers" in block
    statement = {"closest_prior": [{"kind": "literature", "ref": candidates[-2]["key"]}],
                 "changed_equation": "learning", "cause_ids_addressed": []}
    assert "unknown_ref:literature:" + candidates[-2]["key"] in difference.host_problems(
        statement, make_card("card-0000a002"), req)


def test_the_required_entry_count_follows_the_relevant_sources_not_the_listing():
    ctx = SimpleNamespace(outcomes=None, cfg={"outcomes": {"enforce_r3": False}})
    critic = {"family": "none", "known_work_sources": []}

    def need(candidates, cited=0):
        evidence = [{"source_id": c["source_id"], "relation": "related"} for c in candidates[:cited]]
        record = {"label": "novel", "evidence": evidence}
        return difference.requirements(ctx, record, {"candidates": candidates}, [], critic, FAMILIES).need

    many = [paper(f"arxiv:2403.{n:05d}", f"Many {n}") for n in range(difference.LISTED_LITERATURE + 40)]
    # D: papers the classifier did not cite as evidence are far relatives and do not raise the requirement
    assert (need([]), need(many), need(many, 1), need(many, 2), need(many, 5)) == (0, 0, 1, 2, 2)


class RecordingCtx:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def call(self, role, system, user, schema):
        self.calls.append(user)
        return SimpleNamespace(parsed=self.reply, call_id="call-0")


STATEMENT = {"closest_prior": [{"kind": "literature", "ref": SDM, "title": "t", "what_was_tried": "w",
                                "why_it_failed": "f"},
                               {"kind": "literature", "ref": "arxiv:2401.00000", "title": "Paper number 0",
                                "what_was_tried": "w", "why_it_failed": "f"}],
             "what_was_tried": "w", "why_it_failed": "f", "precise_change": "c", "changed_equation": "learning",
             "cause_ids_addressed": []}
VERDICT = {"accepted": True, "problems": [], "excerpt_gaps": [], "causes": [], "family_difference_real": None, "family_why": "",
           "required_difference": ""}


def test_the_writer_and_the_checker_prompts_list_the_same_titles_and_excerpts():
    record, req = requirements(search(), [evidence_of(SDM)])
    context = difference.context_block(record, req, {"reduces_to": "none"})
    card = make_card("card-0000a003")
    writer, checker = RecordingCtx(STATEMENT), RecordingCtx(VERDICT)

    asyncio.run(difference.write(writer, card, "explorer", context, None))
    asyncio.run(difference.check(checker, card, "refine_challenger", context, STATEMENT, req, {}))

    for prompt in (writer.calls[0], checker.calls[0]):
        assert f"- {SDM} (2003) An adaptive sparse distributed memory" in prompt
        assert "Paper number 1" in prompt and "Excerpt of paper 1." in prompt
        assert "title not recorded" not in prompt
