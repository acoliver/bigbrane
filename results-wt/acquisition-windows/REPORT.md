# Acquisition is insufficient; successful HTTP responses do not resolve the recall loss

The saved response bodies now distinguish empty matching pools from truncated windows. Three first-query arXiv responses explicitly report zero matches, while every first-query OpenAlex and S2 response reports more records than the six returned. Parsing drops none of their fifty records. No unintended adapter transformation is proved. The evidence identifies bounded acquisition and concentrated seed selection, but cannot locate missing works inside unseen provider results.

The delivery keeps [compact measured receipts](DELIVERY-RECEIPTS.json), the [counts](SUMMARY.json), plan, tools and synthetic tests. Receipt sections retain 72 query strings/hashes, 288 historical window classifications, twelve first-window totals, raw-envelope hashes and ordered IDs, selected seed support, and the eleven registered traces. Full per-record lexical comparisons, repeated query ASTs, full candidate orderings, raw bodies and duplicate before/after input histories remain local. Each omitted structured derivation has its exact original SHA256 and byte count in the receipts. These receipts permit checking reported totals without a raw cache; they cannot regenerate or independently authenticate provider bodies. Full extraction and its determinism check require the preserved local inputs and manifests. The source-export tests exercise synthetic inputs without them.

**Original PA2 remains FAIL, 2/4 cards against 3/4. Original PM2 failed.** This offline diagnostic changes neither result, query text, ranking, prompts, retries nor gates. The historical initial 429 cause remains unknown. No live requests, including documentation requests, were made.

## Actual first-query matching pools

[First-window evidence](DELIVERY-RECEIPTS.json) joins each raw body to its pinned card, query SHA, source, request parameters, ordered receipt IDs and saved transport operation. Values below come from Atom OpenSearch `totalResults`, OpenAlex `meta.count`, and S2 `total`, not from counts inferred from returned records.

| Card | arXiv returned / total | OpenAlex returned / total | S2 returned / total |
|---|---:|---:|---:|
| 3485a13f | 0 / 0 | 6 / 839 | 6 / 8 |
| f5ee7edb | 0 / 0 | 6 / 1,973 | 6 / 16,505 |
| a4f684fb | 2 / 2 | 6 / 4,703 | 6 / 21,761 |
| 89248ee5 | 0 / 0 | 6 / 160 | 6 / 49 |

All start at zero. Atom `itemsPerPage=6` is the requested size, including the zero/two-result feeds. OpenAlex reports page 1 and per_page 6. Every S2 response reports offset 0 and next 6. Thus eight windows are truncated, three are explicit empty pools, and one exhausts a two-record pool. S2's 8-match query leaves only two unseen records; its 21,761-match query leaves 21,755. Neither count reveals whether a registered work is among those unseen records.

These are the previously saved twelve operations and thirteen exchanges, not new calls. Their fifty source-result positions and zero registered matches belong to four first queries. The historical 3/11 works across 2/4 cards uses all saved queries and is a different population.

## All 72 original contexts

[Historical windows](DELIVERY-RECEIPTS.json) account for all 288 source contexts:

| Source | Explicit zero pools | Exhausted nonempty pools | More unseen records | Unobserved because of source error | Local only |
|---|---:|---:|---:|---:|---:|
| arXiv | 24 | 2 | 0 | 46 | 0 |
| OpenAlex | 0 | 2 | 70 | 0 | 0 |
| S2 | 13 | 7 | 52 | 0 | 0 |
| Local arXiv | 0 | 0 | 0 | 0 | 72 |

The 46 arXiv errors are 44 cooldown skips and two declared exhausted-429 operations for these four cards, not 46 observed HTTP 429s. Historical cache bodies do not establish whether their owning calls used cache or network. Local matching-pool totals were not recomputed.

All 170 answered remote-search contexts have their original raw envelope; none is missing. They contain 761 raw and 761 parsed records with exact ordered-ID agreement. The 432 historical local positions bring the total to 1,193 source positions before deduplication. There are zero parser drops in the remote search bodies. This does not claim a new audit of every bibliography/get parser.

## Exact query and window policy

[Query contexts](DELIVERY-RECEIPTS.json) retain all strings, hashes, rounds, source order, field/lens tags, request ASTs and local expressions. The four first queries remain:

1. `exact match key value cache induction head copying`
2. `successor automaton learning from positive examples`
3. `permutation automata group-structured state transitions`
4. `sparse distributed memory hard location activation Kanerva`

The [arXiv builder](../../hybrid/literature/sources/arxiv.py) extracts ASCII alphanumeric runs, keeps stopwords and builds an AND of `all:` terms, with start 0, max_results 6 and relevance sort. Consequently `group-structured` becomes two conjuncts. Saved Atom links independently show the server's nested conjunctions. Historical HTTP cache keys and fresh HTTPS cache keys are kept separate.

[OpenAlex](../../hybrid/literature/sources/openalex.py) and [S2](../../hybrid/literature/sources/semantic_scholar.py) pass the original string unchanged with per-page/limit 6; neither adapter specifies an explicit relevance sort or advances pagination. The [client](../../hybrid/literature/client.py) fetches one request per source search and parses that response, without following the returned next page. Provider-internal boolean interpretation, default ranking and field scope are not established by the local source. External protocol documentation was not fetched.

[Local arXiv](../../hybrid/literature/sources/arxiv_local.py) removes declared stopwords, preserves quoted phrases, deduplicates terms and searches stemmed title/abstract FTS. It fills fewer-than-six all-term results from any-term BM25 results. Its semantics cannot be substituted for remote `all:` matching.

## Generic ordering evidence, without registered labels

[Ordering evidence](DELIVERY-RECEIPTS.json) reproduces every historical seed choice from initial-query results, within-query deduplication, distinct query support and discovery-order ties. No target labels enter those metrics.

| Card | Initial candidates | Single-query candidates | Seeds present in first query | Lenses represented in seed support |
|---|---:|---:|---:|---:|
| 3485a13f | 214 | 213 | 4/5 | 2 |
| f5ee7edb | 212 | 212 | 5/5 | 1 |
| a4f684fb | 214 | 212 | 4/5 | 2 |
| 89248ee5 | 208 | 208 | 5/5 | 1 |

Only three of 848 initial candidate keys have multiple-query support; the other 845 tie at one. All twenty selected seeds are retained arXiv-source identities, following local-first acquisition. Eighteen appear in the first query. Ten seeds return no references or citations. This establishes how source/query ordering constrains expansion, without demonstrating that another ordering would recover a missing work.

Queries contain 6 to 11 ASCII tokens. Among 761 remote returned records, 618 have abstracts, 15 contain every literal query token in title/abstract, and 732 contain at least one. These label-free lexical diagnostics are not provider eligibility tests. Raw OpenAlex bibliography metrics include exact-ID overlap with initial acquisitions and selected seeds; they are lower bounds because unresolved aliases are not guessed.

First-query literal overlap with the entire saved card input is 7/8, 6/6, 5/6 and 7/7 tokens. High overlap does not establish mechanism coverage. The measurements do not support labeling planner queries generically unrelated to their cards or choosing particular words to remove.

## Earliest explicit registered-work trace

[Registered trace](DELIVERY-RECEIPTS.json) separately projects existing direct-hit, bibliography, local eligibility, rank and exposure evidence for all eleven works. Hwang is a raw bibliography identity without a fetched work record: two distinct returned works point to it, but seed ranks 65/66 are outside five. Srinivas, Silver, Rendle, SciPy QAP 2-opt, Rehn, Jiang and Stoklosa have no saved direct or selected-seed edge record. Rossi is acquired, then remains rank 128; Zhou is acquired, then rank 258 historically/256 in existing replay. Neither is exposed at 35/60/80. Sacramento is acquired and exposed, historically rank 27 and existing replay rank 25. No new ranking was run.

Preexisting diagnostic/local metadata supports literal comparisons for nine works; exact SciPy/Rehn paper metadata is unavailable in those sources. Every available metadata version lacks at least one literal token for every original query on its card. The existing local FTS proof is retained rather than recomputed. Missing title/abstract terms cannot prove remote arXiv `all:` exclusion, including any unsaved full-text fields. Absence from a window never establishes that a work is unindexed.

## Preservation and next test

[Input hashes](DELIVERY-RECEIPTS.json) pin main `09dd7568` and sibling `a7bb1e0`, all 274 originals, twelve raw envelopes in main, existing diagnostics and adapter source. [After hashes](DELIVERY-RECEIPTS.json) verify unchanged bytes. Raw bodies remain in place and were not copied. The plan was pinned before numerical extraction. Eighteen new scoped tests and all seven unchanged prior transport tests pass with socket connections blocked. Seven repeated structured outputs are byte-identical; see [original validation receipt](DELIVERY-RECEIPTS.json). Output is restricted to this directory.

The next test should use an independent held-out mechanism corpus, with query policy and window/seed budgets declared before inspecting relevance labels. Compare a generic all-term versus source-aware query policy and balanced seed allocation under equal budgets, measuring recall separately from transport success and seed coverage. Keep these eleven registrations out of policy selection. No particular token removal, larger window, retry change or acceptance rerun is justified by the present evidence.
