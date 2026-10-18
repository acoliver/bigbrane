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
