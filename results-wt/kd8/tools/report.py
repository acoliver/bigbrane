"""Build a bounded saved-evidence report. Does not run experiments or revise preregistration."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from extract_live import INCREMENTAL, ROOT


def main():
    live = json.loads((ROOT / "live.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "census.jsonl").read_text().splitlines()]
    lines = ["# PA2 retrieval failed; the published-mechanism guard passed", "",
             "The saved PA2 experiment retrieves at least one registered work for 2/4 incremental cards. The registered "
             "requirement is 3/4, so retrieval FAILS even under this generous interpretation. Four published-mechanism "
             "controls die at S2. These results describe historical snapshots, not the final repaired tree. No live "
             "searches, retries or model calls were made to prepare this report.", "",
             "## Criteria", "",
             "| Criterion | Status | Evidence and limits |", "|---|---|---|",
             "| PA2 registered retrieval, at least 3/4 incremental cards | FAIL | 2/4 using any registered work per card; exact id or full normalized title only |",
             "| PA2 registered guard, all four selected published mechanisms die at S2 | PASS | 4/4; Rule 110 is other_known, not a nonstandard RNN activation |",
             "| PA2 not tuned to referee outcomes | PASS for this implementation | No new live runs or attempts; historical missing works remain missing |",
             "| KD8 saved annotated label-interpreter replay | PASS within its synthetic scope | Historical replay.json has 14 records, including Jeffrey/IPF; declarations were supplied by hand or mechanically |",
             "| KD8 live Jeffrey/IPF preservation after repair | NOT TESTED | No new live Jeffrey run in the saved PA2 experiment |",
             "| KD8 executable declarations bound to published operations | NOT TESTED by these historical runs | Other worker owns the production fix and its targeted execution tests |",
             "| Census fully classified for restatements | NOT TESTED | Four identified comparator restatements below; remaining population is not exhaustively adjudicated |",
             "| Historical suite3 all green | FAIL | 1589 passed, three skipped, one prompt-copy failure |",
             "| Full combined suite on final repaired tree | NOT TESTED | Driver will delegate after both implementation workers finish |",
             "| R2-bar original Rule 1 and Rule 2 named-gate guards | FAIL | Historical 9/10 and 3/6; see ../r2-bar/SCORE.md |",
             "| Novel theory, subquadratic learning and sample efficiency | NOT TESTED | Runs stop at S2, with no mathematical or learning evidence |", "",
             "## All saved live runs", "",
             "Complete means run_end reason=complete and no queued unfinished cards. Source failures are shown separately; "
             "completion is not scientific acceptance.", "",
             "| Run | Completion | Terminal/stage error events | Failed model calls | Saved/ledger calls | Run wall s |",
             "|---|---|---:|---:|---:|---:|"]
    for run, record in live["runs"].items():
        lines.append(f"| {run} | {'complete' if record['complete'] else 'INCOMPLETE'} | {len(record['errors'])} | "
                     f"{record['failed_model_calls']} | {record['saved_calls']}/{record['recorded_calls']} | {record['elapsed_s']} |")
    lines += ["", "All saved call transcripts were attributable to an explicit primary card id. A history reference to "
              "another card is not an attribution. Attempt spans are paired only inside the same card.", "",
              "## Per-card literature cost and outcomes", "",
              "Wall covers query proposal start through classification result. The parenthesized value starts at the "
              "query-proposal receipt, after that proposal model call. Search receipts count source-level searches, "
              "including the local index, cache hits and skips. Actual HTTP request/attempt counts are NOT AVAILABLE. "
              "Related lookups can perform multiple HTTP requests and are counted separately.", "",
              "| Card | Run | Search / related | Source receipts | Literature model calls / requests | Wall s (after proposal) | Source errors | Related errors | Candidates / fields | Saved S2 |",
              "|---|---|---:|---:|---:|---:|---|---|---:|---|"]
    for cid, records in live["cards"].items():
        for record in records:
            for attempt in record["attempts"]:
                c = attempt["cost"]
                errors = ", ".join(f"{s}:{n}" for s, n in attempt["source_errors"].items()) or "none"
                related = ", ".join(f"{s}:{n}" for s, n in attempt["related_errors"].items()) or "none"
                verdict = record["s2"][-1]["verdict"] if record["s2"] else "MISSING"
                lines.append(f"| {cid} | {record['run']} | {c['searches']} / {c['related_lookups']} | {c['source_search_receipts']} | "
                             f"{c['model_calls']} / {c['model_request_attempts']} | {c['proposal_to_result_wall_s']} "
                             f"({c['queries_to_result_wall_s']}) | {errors} | {related} | {attempt['candidates']} / "
                             f"{len(set(attempt['fields'] or []))} | {verdict} |")
    lines += ["", "Each classification has three prior-art model calls and three recorded request attempts: query proposal, "
              "refinement and classification. The old flow had two model calls, so the increase is one, not two. "
              "This corrects the accounting claim in DECISIONS.md without editing that document. "
              "flipA has 51/52 arXiv search failures; flipB has 11/36. The other sources, including a complete local "
              "arXiv snapshot, sometimes provide usable coverage; a failed source is neither evidence of absence nor "
              "by itself a reason to discard all results.", "",
              "## Exact registered retrieval and actual decision exposure", "",
              "Any named work counts toward the generous card-level retrieval score. Missing works are retained. "
              "Decision positions come from saved prompt listings, not the candidate rank or today's selection rule.", "",
              "| Card | Work | Candidate rank(s) | Actual exposed decision positions |", "|---|---|---|---|"]
    for cid in INCREMENTAL:
        for card in live["cards"].get(cid, []):
            for attempt in card["attempts"]:
                for work, record in attempt["referee"].items():
                    ranks = ", ".join(str(h["rank"]) for h in record["hits"]) or "MISSING"
                    shown = "; ".join(f"{e['kind']}:{e['position']}" for h in record["hits"] for e in h["exposed"]) or "not shown"
                    lines.append(f"| {cid} | {work} | {ranks} | {shown} |")
    lines += ["", "Rossi is stored at rank 128 but not exposed to the classifier, critic or difference listing. "
              "Zhou is stored at rank 258 and likewise not shown. Sacramento at rank 27 is shown to the classifier "
              "and critic at position 27 and to the difference writer/checker at position 3. The underlying per-call "
              "ids and exact matches are in live.json. Nothing here reclassifies a missing source as retrieved.", "",
              "## Census and annotation limits", "",
              f"The saved census has {len(rows)} counted comparison rows in {len({r['path'] for r in rows})} artifact paths: "
              f"{dict(Counter(r['family'] for r in rows))}. Its recorded artifact-level outcomes are "
              f"{dict(Counter(r['outcome'] for r in rows))}. The prior collection log reports 327 examined artifacts; "
              "the census only stores matching rows and cannot independently reconstruct that denominator. "
              "It omits per-row within_tolerance and executable verification data, so artifact-level verified must "
              "not be read as every row being verified.", "",
              "Four identified counted-and-verified restatements are fixed_quadratic_rnn on card-8f2e71cf, "
              "elman_rnn_identity_w and rnn_diff_sum_basis on card-136a1e9b, and block_sequential_ca_moore on "
              "card-383bd914. They insert the card's nonlinear operation into a purported published template. "
              "Jeffrey/IPF on card-ea50345f is a named published comparator in the historical record. "
              "The remaining census is not a reviewed population-wide classification, and these examples cannot "
              "establish a prevalence estimate. The historical 14-record annotated replay verifies how labels are "
              "interpreted, not that a live critic chooses or executes the correct published operation.", "",
              "## Recovered historical suite3", "",
              "| Log | Passed | Skipped | Failed | SHA-256 |", "|---|---:|---:|---:|---|"]
    totals = Counter()
    for log in sorted((ROOT / "suite3").glob("g*.log")):
        text = log.read_text()
        summary = next((line for line in reversed(text.splitlines()) if re.search(r"\d+ passed", line)), "")
        counts = {kind: int(match.group(1)) if (match := re.search(rf"(\d+) {kind}", summary)) else 0
                  for kind in ("passed", "skipped", "failed")}
        totals.update(counts)
        lines.append(f"| suite3/{log.name} | {counts['passed']} | {counts['skipped']} | {counts['failed']} | "
                     f"{hashlib.sha256(log.read_bytes()).hexdigest()} |")
    lines += ["", f"Total: {totals['passed']} passed, {totals['skipped']} skipped, {totals['failed']} failed. "
              "g3 failed test_orchestrator_units.py::test_prior_art_prompt_copy_matches_the_literature_module. "
              "These logs predate the repaired tree; no per-shard code hash was captured. The review observed "
              "that all three current prompt strings already matched the current document at fa08b93. A "
              "current targeted check and a historical suite failure can both be true. Neither is a final full-suite success.", "",
              "## Provenance and remaining work", "",
              "Inputs are the six saved runs under results-wt/kd8/runs, census.jsonl, replay.json and suite3/g0.log "
              "through g6.log. extract_live.py reads card-tagged ledger events and saved transcripts; report.py reads "
              "its derived live.json. Raw runs, stores, prompts and logs were not rewritten. Historical flip artifacts "
              "lack query_fields/round2_queries added in later snapshots, while the query events retain the fields. "
              "live.json lists missing artifact fields separately rather than claiming no tagged queries ran. "
              "No saved run contains the new coverage/listing payload from this implementation.", "",
              "The final-tree retrieval target is NOT TESTED, and the saved 2/4 remains FAIL. Do not tune or retry "
              "to force 3/4. The distinct-query and bounded-exposure defects are addressed in source and targeted "
              "tests, but their effect on retrieval has no new live measurement. Final combined verification is "
              "delegated by the driver after the two workers finish. No result establishes complete novelty."]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ROOT / 'REPORT.md'}; suite3 {dict(totals)}")


if __name__ == "__main__":
    main()
