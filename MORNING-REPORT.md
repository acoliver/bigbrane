# Morning report, 2026-10-11

**Two theory cards passed every gate overnight, were built as small models and were benchmarked against the five baselines. Neither beat them. The research loop has not yet found a new theory.**

The loop ran from 22:09 to 08:33 on Claude, on GPT through our own Sign in with ChatGPT login (the plan account), and on local Qwen as a fallback. Across runs 3, 4 and 5 it produced 188 cards:
- 93 were matched to known methods by the prior-art critic (S2).
- 15 were refuted in the feasibility stage (S3), 2 of them by executed math checks.
- 2 reached the end of the pipeline, both in run 4.

## 1. The two cards that reached the end

### card-03fa1d38, "Two-Sheet Residual Rewriting"

**Origin.** Run 4. A cross of two earlier cards (lenses: cellular automata and sheaves). A demonstration writes a sparse rewrite location on one of two "sheets" of token codes, and rewrites compose by packing them without overlap.

| Stage | Outcome |
|---|---|
| S2 prior art | Passed as "prior art under-tested". The nearest work is ripple-down rules (arXiv 1412.4021, 1709.06307). |
| S3 feasibility | Both judges voted feasible: Opus at 0.85, GPT at 0.97. All 8 obligations were discharged, the complexity check by an executed and replayed program. |
| S4 build | 42 parameters, 176 state floats per sequence. |
| S5 bench | Time grows linearly with length: fitted slope 0.997 against the claimed 1.0 (transformer 1.42). Fingerprint: `no_capacity_advantage`, `no_fewshot_gain`. Nearest baselines: transformer by output ratio, linear attention by JS distance. |
| S6 verdict (as recorded) | 1 claim supported, 1 refuted, 3 inconclusive, 11 untested. |

On all three card-specific falsifier tasks the model did not train: it never beat the uniform and unigram references. The one "refuted" entry came from the analyst model, which read 0.008 accuracy from a task the rule had already called inconclusive for that reason. I fixed that rule this morning (section 4). Under the fixed rule the card has 1 supported, 0 refuted, 4 inconclusive and 11 untested claims.

**Reading.** This is a hand-designed symbolic rewriting system. The bench feeds it consecutive token pairs as demonstrations, and on the bench tasks it learned nothing beyond the references. Its linear-time claim holds. Its rule-learning claims were not shown, because the model never got into a state where they could be tested.

### card-779795a5, "Conjugate Sewing"

**Origin.** Run 4. An escape card: its parent had been reduced to an order-2 affine projection algorithm. Its argument was that conjugating the rule by data-dependent polynomial shears changes both input and output coordinates, so the parent's fixed linear readout cannot express it.

| Stage | Outcome |
|---|---|
| S3 feasibility | Both judges voted feasible: Opus at 0.88, GPT at 0.98. |
| S4 build | 15 parameters, 64 state floats per sequence. |
| S5 bench | Time grows faster than claimed: tail slope 1.59 against a claimed 1.0. Fingerprint: `quadratic_or_worse`, `no_fewshot_gain`. Its falsifier task returned non-finite values and was marked invalid. |
| S6 verdict | 0 supported, 2 refuted (both complexity claims), 1 inconclusive, 8 untested. The S6 fix does not change this. |

## 2. What the gates did

| | Run 3 (22:09 to 01:22, with resumes) | Run 4 (01:23 to 06:28) | Run 5 (06:48 to 08:33) |
|---|---|---|---|
| Cards | 47 | 103 | 38 |
| Model calls | 438 | 1046 | 205 |
| Model errors | 0, plus 2 calls where every backend failed | 0, plus 1 call where every backend failed | 0 |
| S2 prior art | 6 pass, 23 known method, 12 fail | 12 pass, 58 known method, 30 fail | 1 pass, 12 known method, 4 fail |
| S3 feasibility | 5 refuted, 1 undecided | 9 refuted, 2 pass, 1 undecided | 1 refuted |
| S4 to S6 | none | 2 cards | none |
| Escape cards | 19 created; best result: 2 refuted at S3 | 40 created; best result: 1 reached S6 | 10 created |
| Textbook transplants, by S2's count | Opus 16 of 36, GPT 0 of 5 | Opus 7 of 51, GPT 0 of 49 | Opus 2 of 9, GPT 0 of 8 |

**Run 4's reductions by family:** other known 38, linear attention 12, RNN 6, SSM 1, transformer variant 1.

**How run 4's S3 refutations were decided:**
- 7 by a judge's disproof that the other side confirmed;
- 2 by an executed check program that the replay confirmed. On card-c8ef549e it was a capacity check, at d=64. On card-b2761f4a it was a precision check.

The executed checks are the first S3 decisions that do not rest on a judge's opinion alone.

**Two trends between runs:**
- Opus copied textbook constructions less in run 4 than in run 3. Run 4 used a larger graveyard and the families list.
- GPT, as the second explorer, produced no textbook transplants in any run.

Most cards still reduce to something known. Most reductions now go to "other known" methods, such as data structures, symbolic learners and classical algorithms. Linear attention accounts for 12 of run 4's 58 reductions and none of run 5's 12.

## 3. Run 5 (06:48 to 08:33)

Graveyards of runs 1 to 4 (up to 40 entries per prompt), the fixed S6 rule and the patched gateway.

- 38 cards and 205 calls in 1 h 45 min, with no model errors.
- S2: 1 pass, 12 known methods (other known 10, Hopfield or kernel memory 1, LSTM or GRU 1), 4 fail.
- The one card that passed S2, card-ffcfe0e5, was refuted at S3. It is a per-slot drift-diffusion race with a sticky upper bound and an absorbing lower bound. One judge disproved its sample-efficiency claim, that one confirmation commits a rule, and the other judge confirmed the disproof.
- When the wall budget ran out, 12 cards had passed S0 and were waiting for S1; 5 had failed S0.
- All 85 literature searches returned results, with 510 hits from the local index.

The fixed S6 rule did not get exercised, because no card in run 5 reached S6.

## 4. Changes and fixes overnight

**Changes**
- **GPT access.** Our Sign in with ChatGPT login for the plan account went in at 23:45. It refreshed 8 times on schedule, from 00:40 to 07:05, every one returning 200. GPT handled 209 calls in runs 3 to 5 with no errors. Pacing: one request at a time, at least 10 s between starts, at most 60 per hour, effort capped at medium.
- **Claude access.** The Anthropic login refreshed by itself at 07:00 and is valid until 15:00.
- **Local arXiv index.** 1,419,455 records. The harvest finished at 08:00 with all 15 sets complete, including cs at 1,029,985 records. Sets deferred after timeouts were picked up on the retry pass. In run 4, all 493 literature searches returned results: 2958 hits from the local index and 243 from live arXiv. Semantic Scholar and OpenAlex spent most of the night in rate-limit cooldown (461 and 457 skipped searches).
- **S3 obligation checks.** These were added earlier in the night and decided 2 cards in run 4.

**Fixes**
- **Harvest deferral.** The arXiv harvest stopped at 01:15 after one cs page timed out 8 times. It now defers a stuck set, continues with the others and retries deferred sets at the end, keeping their position. 110 literature tests pass.
- **Second call budget.** Run 4 stopped at 04:26 on a 600-call limit in `models.toml`, separate from the run's own `--max-calls 2500`. I raised it to 3000 calls and 120M tokens.
- **Gateway model list.** The 04:29 resume failed its preflight, because OpenAI's model list took 3.5 minutes and the gateway waited on it. Each provider's list now has a 10 s timeout and a 10-minute cache. 525 gateway tests pass. The gateway was restarted at 06:47.
- **S6 rule over analyst.** A rule's outcome now overrides the analyst for the same falsifier or claim. A task the candidate did not train on cannot refute anything. The analyst's opposite view is kept as a note. 11 new tests.
- **Full suite.** 1182 passed, 9 skipped. This covers every hybrid test except the P0 container probes.

## 5. Limits

- **The bench cannot train hand-built mechanisms well.** Both cards that reached S6 failed to train on their own falsifier tasks. One rule-based model did no better than unigram, and one produced non-finite outputs. Until the bench gives these models a protocol they can follow (explicit demonstration and query events), most of their claims will stay untested.
- **S3 still leans on judges.** It now has executed checks, but 7 of 9 refutations in run 4 came from a judge's disproof, and both passes came from two judges agreeing.
- **The fingerprint still separates linear attention and the transformer poorly.** card-03fa1d38's nearest baseline was the transformer by one measure and linear attention by the other.

## 6. Decisions for you

1. **S6 when a falsifier fires.** Should S6 turn into `fail` when a validated falsifier rule fires? Today it records the refutation and still passes the card.
2. **Literature keys.** A Semantic Scholar API key and an OpenAlex mailto would end most of the rate-limit skips.
3. **Next bench work.** Give built cards an explicit demonstration and query interface, so rule-learning claims can be tested instead of left untested.
4. **SIWC refresh risk.** It worked all night, but a client can still lose its refresh grant; devkit#5 reports `invalid_grant` for many clients. If that happens, the gateway stops using ChatGPT, the second judge and the falsifier designer fall back to local Qwen, and a new browser login is needed.

## 7. State at handoff

I left everything running. Tell me if you want any of it stopped.

- **Research runs:** none running. Run directories are `runs/run-20261011T011300Z` (run 3), `runs/run-20261011T042400Z` (run 4) and `runs/run-20261011T095000Z` (run 5). `.venv-hybrid/bin/python -m hybrid.research inspect <run>` prints a summary and builds the HTML report.
- **Gateway:** running as PID 9118 on port 8765, logging to `results/gateway-20261011/serve4.log`. Both logins refresh by themselves. Stop it with `kill -INT 9118`.
- **arXiv harvest:** finished; no process running.
- **Podman:** VM `podman-machine-default` running, with no bigbrane containers. Stop it with `scripts/podman-teardown.sh` (dry run), then `--apply --stop-machine`.
- **LM Studio:** qwen3.6-35b-a3b-mlx loaded with a 131,072-token context.

Detailed numbers are in `results/morning-20261011/` (inspect and literature summaries per run, full test log).
