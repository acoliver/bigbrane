p="hybrid/research/stages/difference.py"
s=open(p).read()
def rep(a,b,count=1):
    global s
    assert s.count(a)==count,(a,s.count(a))
    s=s.replace(a,b)
rep("from hybrid.research.stages.common import StageContext, card_for_prompt, load_prompt, load_section\n",
    "from hybrid.research.stages import known_work\nfrom hybrid.research.stages.common import StageContext, card_for_prompt, load_prompt, load_section\n")
rep('''    need: int
    families: Families
    searched: int = 0                          # papers the prior-art search returned, of which `literature` lists some
''','''    need: int                                  # literature or outcome entries required (relevant sources, at most 2)
    families: Families
    searched: int = 0                          # papers the prior-art search returned, of which `literature` lists some
    cited_papers: tuple[str, ...] = ()         # bare ids the reduction critic cited: each is a required entry (R2C)
    cited_names: tuple[dict, ...] = ()         # textbook sections and standard algorithms it cited (checker's job)
''')
rep('''def literature_refs(record: dict, pa_result: dict) -> dict[str, dict]:''','''def literature_refs(record: dict, pa_result: dict, cited: tuple[str, ...] = ()) -> dict[str, dict]:''')
rep('''    writer and the one for the checker are built from this one listing."""
    candidates = pa_result["candidates"]
    by_source_id = {candidate["source_id"]: candidate for candidate in candidates}
    listed: dict[str, dict] = {}
''','''    writer and the one for the checker are built from this one listing. Papers the reduction critic cited as known
    work come before both, so a required entry is always listed."""
    candidates = pa_result["candidates"]
    by_source_id = {candidate["source_id"]: candidate for candidate in candidates}
    listed: dict[str, dict] = {}
    for candidate in candidates:
        if known_work.bare(candidate["source_id"]) in cited:
            listed.setdefault(candidate["key"], dict(candidate, relation="cited by the reduction critic as known work"))
''')
rep('''    literature = literature_refs(record, pa_result)
''','''    cited = tuple(dict.fromkeys(known_work.cited_papers(critic, pa_result["candidates"])))
    literature = literature_refs(record, pa_result, cited)
''')
rep('''    need = min(2, len({ref.split(":", 1)[-1] for ref in literature}) + len(outcomes))
    return Requirements(literature, outcomes, r3, named_family(critic, families), need, families,
                        len(pa_result["candidates"]))''','''    # D: only sources the classifier judged relevant (its cited evidence) or the critic cited count toward the two
    # required entries; when fewer than two are relevant, family entries fill the rest (see host_problems).
    relevant = {known_work.bare(item["source_id"]) for item in record["evidence"]} | set(cited)
    need = min(2, len(relevant) + len(outcomes))
    return Requirements(literature, outcomes, r3, named_family(critic, families), need, families,
                        len(pa_result["candidates"]), cited,
                        tuple(known_work.cited_names(critic, pa_result["candidates"])))''')
rep('''    if len(counted) < req.need:
        problems["closest_prior:count"] = (f"{len(counted)} literature or outcome entries with receipts; at least "
                                           f"{req.need} are required")
''','''    if len(counted) < req.need:
        problems["closest_prior:count"] = (f"{len(counted)} literature or outcome entries with receipts; at least "
                                           f"{req.need} are required")
    elif len(counted) + len(families) < 2:
        problems["closest_prior:count"] = (f"{len(counted)} literature or outcome entries and {len(families)} family "
                                           f"entries; at least 2 entries are required, and a family entry may fill a "
                                           f"slot the search has no relevant source for")
    listed = {ref for kind, ref in counted if kind == "literature"}
    for paper in req.cited_papers:
        if paper not in listed:
            problems[f"cited_source:{paper}"] = (f"the reduction critic cites {paper} as the closest known work; it "
                                                 f"must be a literature entry of closest_prior, with what it did and "
                                                 f"what your card changes")
''')
rep('''    rules = [f"- at least {req.need} closest_prior entries of kind literature or outcome, with the ids above"]
''','''    rules = [f"- at least {req.need} closest_prior entries of kind literature or outcome, with the ids above; if the "
             f"search has fewer than two relevant sources, add family entries so that closest_prior has at least two"]
    if req.cited_papers:
        rules.append(f"- the reduction critic cites these papers as the closest known work. List each as a literature "
                     f"entry: {', '.join(req.cited_papers)}")
    for source in req.cited_names:
        rules.append(f"- the reduction critic cites the {source['kind'].replace('_', ' ')} {source['ref']!r}, covering: "
                     f"{source['covers'][:300]}. Name it in a closest_prior entry (a family entry if it is one) and "
                     f"say what your card changes in it")
''')
rep('''                                                 ("reduces_to", "family", "family_why", "mapping", "difference_if_any",
                                                  "why")}''','''                                                 ("reduces_to", "family", "family_why", "mapping", "difference_if_any",
                                                  "why", "known_work_sources")}''')
rep('''            "required": ["accepted", "problems", "causes", "family_difference_real", "family_why",
                         "required_difference"],
            "properties": {"accepted": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}},''','''            "required": ["accepted", "problems", "excerpt_gaps", "causes", "family_difference_real", "family_why",
                         "required_difference"],
            "allOf": [{"if": {"properties": {"accepted": {"const": True}}},
                       "then": {"properties": {"problems": {"maxItems": 0}}},
                       "else": {"properties": {"problems": {"minItems": 1}}}}],
            "properties": {"accepted": {"type": "boolean"}, "problems": {"type": "array", "items": {"type": "string"}},
                           "excerpt_gaps": {"type": "array", "items": {"type": "string"}},''')
rep('''        lines = [self.verdict["required_difference"] or "state precisely what your core changes"]
        lines += [f"- {detail}" for detail in self.failures.values()]
        return "\\n".join(lines)''','''        lines = [self.verdict["required_difference"] or "state precisely what your core changes"]
        lines += [f"- {detail}" for detail in self.failures.values()]
        lines += [f"- scope this claim, do not assert it: {gap}" for gap in self.verdict["excerpt_gaps"]]
        return "\\n".join(lines)''')
rep('''        reasons.append(f"difference statement accepted{' after the send-back' if final is not first else ''}: "''','''        if final.verdict["excerpt_gaps"]:
            reasons.append("claims about prior papers the supplied excerpts do not support (scoped, not a kill): "
                           + "; ".join(final.verdict["excerpt_gaps"])[:600])
        reasons.append(f"difference statement accepted{' after the send-back' if final is not first else ''}: "''')
open(p,"w").write(s)
