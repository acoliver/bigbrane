# KD8 and PA2 status

- [ ] Part 1 KD8: locate comparator code
- [ ] quantify KD8 over recorded runs
- [ ] host-checkable rule + prompts + tests
- [ ] suites
- [ ] Part 2 PA2

## Log
- Step 1 (done): comparators are declared in schemas/reduction_critic.schema.json (`comparators`), typed by contracts.REPAIR_TYPES, bound/counted in stages/s2_prior_art.py (eligibility, judge_comparisons, counted_comparisons). Census script tools/quantify.py writes census.jsonl (64 counted rnn/lstm_gru/other_known comparison rows across 327 reduction_critic artifacts).
- Step 2 (done): census 64 counted rows (tools/quantify.py, census.jsonl). Counted-and-verified restatements: see REPORT.md.
- Step 3 (done, uncommitted until suite): component rule (contracts.py ALLOWED_ACTIVATIONS etc, s2_prior_art.component_problem, eligibility `restatement`), schema `components`, prompts. test_kd8_restatement.py 27 pass; replay.json: 3 KD8 records no longer counted, Jeffrey + 7 probe + 2 controls still count, P08 Rule 110 (rnn) no longer counted.
- Step 4: full research+literature suites running (results-wt/kd8/suite1.log)
- Step 3 committed (028889e). Suites re-running with fixture fixes (results-wt/kd8/suite2/).

## Part 2 PA2 pre-registration (written before any live run)
- Guard: the 4 novelty-probe published mechanisms must still die at S2 (reduces_to_known, known work, or not_radical/no_untested_element). P08 (Rule 110) is among them and is the case KD8 changed, so it will be reported separately.
- Retrieval: the referee's closest works must be retrieved (present in the card's candidate set) for at least 3 of the 4 INCREMENTAL cards (card-3485a13f, card-f5ee7edb, card-a4f684fb, card-89248ee5). Named works: 3485a13f Hwang/Stuhlmuller/Goodman 1110.5667 (also Srinivas 2204.09837, Silver 1904.06317); f5ee7edb Rendle FM 2010 and Rossi 1906.05059; a4f684fb SciPy 2-opt QAP, Rehn thesis; 89248ee5 Jiang 1409.0334, Zhou 0901.0553, Sacramento 1207.7196, Stoklosa 2411.14480. Retrieval counts as matched by title or arXiv id.
- Not tuned: no prompt or rule is adjusted toward the referee's verdicts; only retrieval is measured, and S2 verdicts are reported as they fall.
