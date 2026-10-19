# STATUS r2-bar (worktree bigbrane-wt-r2, branch r2-bar)

## Pre-registered acceptance rules (written before any replay run)

1. Novelty probe guard. All 8 published mechanisms of results/novelty-probe-20261017 (cards P01 to P08, card-17100001
   to card-17100008) end at S2 with verdict `reduces_to_known` (kill code reduces_to_known:*), and both controls
   (C01 card-17100009, C02 card-1710000a) end as `excluded_family:*` kills. Any other outcome means the change leaks
   and must be fixed.
2. Counted reductions stay dead. Every replayed card whose S2 in the earlier run had a host-counted reduction
   (reduction check outcome `verified` and the kill `reduces_to_known:*` or `excluded_family:*`) still dies at S2 with
   the reduction gate, not at a later gate and not as a pass.
3. Flips get inspected. Every card that goes from kill to S2 pass is listed with its reasons and gets the radicality
   question (is its learned general mechanism already published or textbook?) answered in plain words in REPORT.md.

Replay cards: the 11 recheck cards (cards.txt of results/defect-recheck-20261017), 8 batch cards whose binding kill
was R2 (card-0fe3943c, 3485a13f, a775f78d, ebf0e99d, f5ee7edb, 8154d3b4 = the six batch kills that froze or zeroed a
learned part; card-084a2510 and card-909ad773 = a moved-field kill and a refuted-claim kill), and the 10 novelty
probe cards as singleton runs. Store: copy of results/discovery-live-20261017/outcomes-before for batch and recheck
cards, the probe's empty store for probe cards. Configs: recheck.toml and probe.toml copied unchanged.
Budget about 350 calls.

## Steps

- [x] audit and DECISIONS rows read
- [x] code: A + C (critic prompt, schema, host evidence rule) e4a50d9
- [x] code: D (difference gate) e4a50d9
- [ ] unit tests
- [x] DECISIONS rows R2A, R2C, R2D
- [x] replay launched (12 live_check runs)
- [ ] replay scored
- [ ] REPORT.md

## Update
- Code for A, C, D committed (e4a50d9). Unit tests not yet updated.
- Replay launched (4 nohup groups: recheck, batch8, probeA, probeB) via results-wt/r2-bar/tools/run_replay.sh;
  logs in results-wt/r2-bar/logs, runs in results-wt/r2-bar/runs. data/arxiv-index is a symlink to the main
  workspace index (gitignored) so the literature search matches the earlier runs.
- unit tests: fakes updated, hybrid/tests/research/test_known_work.py added (14 pass); full research suite run in logs/tests-2.log
- tools/finish.sh waits for the 12 runs, runs tools/score.py and commits SCORE.md; REPORT.md is written from it
- Step: session started; reading state, rerunning suites
- Step: classified failures. Stale fixtures (critic fakes without known_work_sources, textbook_transplant default True, test asserting textbook 'does not decide the gate') updated to the new contract; env-only failures (results/, runs/ absent from worktree) pending
- Step: phase7_replay fixture's radical_enough read as known_work fields (phase7_replay.py); 4 replay tests pass
- Step: tools/flips.py extracts per-flip artifacts to logs/flips.md
- Step: research suite final run started (logs/tests-10-research-final.log); other suites (literature, llm, gateway, bench, math, top-level): 1827 passed + 4 env failures that pass once the worktree has .venv-hybrid and the seed build log (31 passed)
- Step: score.py rewritten (intent dead/alive plus code); flips.py done; writing REPORT.md next
- Step: research suite relaunched as 4 shards (logs/tests-11-shard*.log)
- Step: REPORT.md written; final suites pass
