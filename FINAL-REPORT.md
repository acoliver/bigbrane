# Bigbrane research system: prototype report

Date: 2026-10-10. Written from the run records, status files and logs listed in each section. No code was changed and no model was called for this report.

## 1. Verdict

The research loop works end to end and every decision it makes can be inspected. Over two live runs it generated 42 theory cards with parallel explorers, checked them for well-formedness, searched the literature, tried to reduce each one to a known architecture with executed numeric checks, attacked the survivors mathematically with two judges from different model families, and kept a hash-chained ledger of every call, check and verdict. S4 to S6 (build a candidate, benchmark it against baselines, compare its claims to the numbers) ran on real models and the Podman sandbox, but only through an operator override, because no card has passed S3.

It has not found the model Andrew asked for. No card has passed the feasibility gate (S3), so no card has reached construction through the gates. Every card so far was a known architecture under new vocabulary, mathematically refuted, left undecided, or stopped earlier as a duplicate or a malformed card.

The strongest part is the adversarial reduction gate at S2. Across both runs:

- 38 S2 verdicts. 36 of them carried a claimed reduction with an executed numeric check.
- 32 checks confirmed that the card's update rule equals a known form. The largest difference among them was max_abs_diff 3.5e-11; 8 printed exactly 0.0.
- 4 checks refuted the claimed reduction (max_abs_diff 0.060, 0.445, 0.716, 0.889), so the gate did not accept a reduction just because the critic asserted one.
- 29 cards ended as `reduces_to_known`, and 3 more with confirmed reductions failed as not radical enough. Families of the 29: other_known 23, linear_attention 3, hopfield_or_kernel_memory 2, transformer_variant 1. "other_known" covers textbook algorithms such as parity union-find, worklist closure, orthogonal matching pursuit, and an exact-match cache paired with a most-specific lattice learner.
- 2 cards passed S2, both in run 1, and reached S3. Both were children of the escape operator. One was refuted at S3 by three executed disproofs; the other ended `undetermined` with a split vote.

Main gaps:

- Literature access is degraded. Semantic Scholar failed on 102 of 108 searches in run 1, and in run 2 OpenAlex failed on all 99 searches, so 69 of 99 came back empty. Prior-art checks in run 2 rest mostly on arXiv.
- Explorers are now one model family. The ChatGPT route was denied and then lost its grant. The GPT explorer had the higher share of modified (non-textbook) ideas, 4 of 5.
- The bench does not test what the cards claim. In both plumbing checks the analyst marked 10 to 12 of the card's 12 to 13 claims `untested`, because the generic five-task bench does not run the card's own protocols.
- S3 has no proof assistant or symbolic check beyond up to three executed attack scripts.
- The behavioral fingerprint cannot separate linear attention from a transformer on the current tasks, and the default S5 bench is the small quick configuration.

## 2. What exists

### Components

| Component | What it does | Where |
| --- | --- | --- |
| Orchestrator | asyncio scheduler: 4 explorers plus worker pools per stage (S1 4, S2 3, S3 2, S4 to S6 1). Enforces that a card reaches Sk only after passing S(k-1), and S4 only after an S3 pass. Budgets, resume, graveyard, escape operator. | `hybrid/research/orchestrator.py`, `stages/`, [ORCH-STATUS.md](hybrid/research/ORCH-STATUS.md) |
| Stage agents | 11 prompt files for explorer, salad critic, prior art, reduction critic and check, formalizer, disprover, two feasibility judges, builder, analyst. JSON schemas for every output. | `hybrid/research/prompts/`, schemas |
| LLM client | Role-to-model routing tables (`llxprt`, `gateway`, `gateway_claude_local`), per-backend concurrency, schema validation and retry, per-run budgets. | `hybrid/llm/`, `hybrid/config/models.toml` |
| Gateway | Local OpenAI-compatible server for Anthropic and ChatGPT OAuth logins. Tokens in private files, no OS keychain. Single refresher, per-run bearer token. | `hybrid/gateway/`, [STATUS.md](hybrid/gateway/STATUS.md) |
| Literature | arXiv, OpenAlex and Semantic Scholar search with per-source receipts; evidence must match retrieved IDs. | `hybrid/literature/` |
| Executor | Podman jobs with no network, read-only root, unprivileged user, dropped capabilities, resource caps, labeled resources. | `hybrid/executor/`, [P0-RESULT.md](P0-RESULT.md), [PODMAN-LIFECYCLE.md](PODMAN-LIFECYCLE.md) |
| Bench | Five exact-oracle tasks (HMM, variable-order Markov, associative recall, hierarchical composition, few-shot rule induction), five baselines (n-gram, GRU, transformer, diagonal SSM, linear attention), scaling probe, mechanism fingerprint with `behaves_like`, `no_capacity_advantage`, `quadratic_or_worse`, `no_fewshot_gain` flags. | `hybrid/bench/`, [STATUS.md](hybrid/bench/STATUS.md), [BASELINES.md](hybrid/bench/BASELINES.md) |
| Store and report | Append-only hash-chained ledger, file hashes, `inspect` command that verifies the ledger and builds an HTML report with card dossiers, lineage and a creativity block. | `hybrid/research/store.py`, `inspect.py` |
| Plumbing override | Runs real S4 to S6 on one copied card; writes an `operator_override` as ledger event 1 and a red banner on every report page. | `hybrid/research/plumb.py` |

### Tests

- Full suite on 2026-10-10: 631 passed, 8 skipped, run without the P0 probe file ([results/final-20261010/pytest-all.log](results/final-20261010/pytest-all.log)). The skips are live-model tests.
- The research suite grew afterwards with the empty-search change: `hybrid/tests/research` 209 passed. The full suite was not re-run after that change.
- P0 executor probes: 44 of 45 pass. Probe 8 (bit-identical fixture inside the container) fails by 10 ulp on one HMM KL value (1.39e-16) because Accelerate and OpenBLAS round one matrix-vector product differently. All scoring therefore runs in the container only.

### Trust boundaries

- Every executed program (toy examples, reduction checks, disproof checks, construct checks, candidates) runs in Podman with `--network=none`, read-only root, uid 10001, no capabilities, pid, memory and wall caps, and only staged input and output volumes.
- Candidates see training data and test prefixes. The oracle stays on the host and a separate evaluator job scores the prediction arrays.
- Known holes: a candidate shares a process with the runner, so it can read test prefixes during fitting and tamper with runner-side timing; scaling evidence is weaker than scoring evidence. `/out` has no disk quota. The capacity ratio uses the candidate's self-reported state size. `--local` S5 runs are marked `trusted: false`.
- Gateway OAuth records and the run token are private files under `~/.bigbrane-gateway/`, never keychain items. The gateway listens on localhost only.
- Literature results and model outputs are treated as external input. Grounding against retrieved IDs stops invented citations. Search coverage limits what any novelty label can show.

## 3. What the runs showed

Runs and their state:

| Run | Routing | State |
| --- | --- | --- |
| `runs/run-20261010T000322Z` | llxprt bridge | First attempt, last night. 4 cards, 12 calls. Ended after S1 with no `run_stop`; nothing reached S2. |
| `runs/run-20261010T181300Z` (run 1) | `gateway` for the first invocation, `gateway_claude_local` for the 5 resumes | Complete. 6 invocations (start plus resumes), wall 2:16:48 in total. |
| `runs/run-20261010T205500Z` (run 2) | gateway_claude_local, run 1 as graveyard | Complete. One invocation, 17:43 to 18:37 -03, stopped with reason `complete` after using its 20 card slots. |
| `runs/plumb-20261010T201437Z`, `runs/plumb-20261010T210700Z` | gateway_claude_local | Operator-override plumbing checks of S4 to S6. Not gate passes. |

### 3.1 Cards per stage and verdict counts

| | Run 1 | Run 2 |
| --- | --- | --- |
| Cards | 23 | 19 |
| S0 | 23 pass | 17 pass, 2 fail (duplicates) |
| S1 | 21 pass, 2 fail, 45 revise | 17 pass, 0 fail, 34 revise |
| S2 | 16 reduces_to_known, 3 fail, 2 pass | 13 reduces_to_known, 2 fail, 2 undetermined |
| S3 | 1 infeasible_proved, 1 undetermined | none |
| S4 to S6 | none | none |

Where the cards stopped in run 1: S1 2, S2 19, S3 2. In run 2: S0 2, S2 17. Run 2's 20th card slot went to an explorer task that failed before drafting (only 2 papers for the lens pair crn+predictive_coding).

S1 almost never passes clean. Of 21 S1 passes in run 1, 20 carried residual fixable findings and 1 was clean; in run 2 all 17 passes carried residual findings. The two S1 failures in run 1 happened under the earlier gate (fixable findings after two revisions), before residual pass-through was turned on. In practice S1 now enforces "no fatal finding and the toy example reproduces in the executor", and the residual findings are handed to the S3 formalizer and disprover.

### 3.2 Reductions by family

| | Run 1 | Run 2 |
| --- | --- | --- |
| reduces_to_known | 16 | 13 |
| other_known | 12 | 11 |
| linear_attention | 3 | 0 |
| hopfield_or_kernel_memory | 1 | 1 |
| transformer_variant | 0 | 1 |
| Numeric checks confirming equivalence | 18 (max_abs_diff 0.0 to 3.5e-11) | 14 (max_abs_diff 0.0 to 1.0e-11) |
| Numeric checks refuting the claimed reduction | 1 (0.445) | 3 (0.060, 0.716, 0.889) |

Examples, each with the executed check's result:

- **card-537e161c, Koopman Power Readout** (in-context EDMD whose operator is iterated by the semigroup law). The critic mapped the RLS update of P_t and K_t to the preconditioned delta rule with a token-shifted key, closed form K_t = (lambda I + sum beta k k^T)^-1 sum beta k v^T, which is MesaNet's in-context least squares. Reduced to linear_attention, max_abs_diff 1.8e-11.
- **card-1184f7a7, Holographic Displacement Interrogation.** The state is a diagonal write R_t = R_{t-1} + k_t ⊙ v_t, which is diagonal, no-decay linear attention (an SSM with A = I). Reduced to linear_attention, max_abs_diff 2.7e-15.
- **card-582f8abf, Gauge-Quotient Constraint Reservoir.** Parent pointers, parity edge labels and union by size: Z2-labeled union-find, the standard online solver for XOR-pair equations. Reduced to other_known, max_abs_diff 0.0.

Two more show the gate rejecting a reduction and then still deciding on other grounds:

- **card-ddb071bd, Self-Growing Holographic Clean-up Memory** (an escape child). The claimed reduction failed its own check (max_abs_diff 0.445), and the critic confirmed that the routing and eviction depend on the stored state. The card then failed as "not radical enough".
- **card-f7c87f65 (run 2).** The claimed reduction failed (max_abs_diff 0.060), and the critic then identified the mechanism as a class-bigram language model with a token cache gated by the predicted class, a 1990s construction, and failed it as not radical.

### 3.3 Escape operator

When S2 reduces a card, the escape operator queues a task that gives an explorer the reduction mapping and the executed check, and requires it to break the property the mapping relies on and to argue why no repair recovers the family.

Run 1: 21 escapes requested, 12 skipped because no card slot was left, 9 created. Of the 9 children, 6 were reduced again, 1 failed S2 as not radical (card-ddb071bd), and 2 passed S2. Every card that reached S3 in run 1 came from the escape operator. Lineage examples:

- card-8dc4db49 (Birch-Point Binding Chemistry, reduced: a gated bigram count cache over a linear-attention outer-product state, max_abs_diff 5.6e-16) produced card-c9615069 (Strand-Exchange Binding Kinetics), whose writes rescale other cells by state-dependent factors. Its reduction critic found no reduction, re-ran the toy prefix by hand and matched it. It passed S2 and was refuted at S3.
- card-537e161c (reduced to linear_attention, 1.8e-11) produced card-36725a7c (Switched Koopman Repertoire), which was reduced again (other_known, 1.3e-12). Its escape child card-46dbb349 (Unknown-Duration Koopman Flow), a second-generation escape, makes the written key depend on the inferred duration through powers of one operator. It passed S2 and ended `undetermined` at S3.
- card-14429dbc (exact-match hard-attention cache plus the most-specific lattice learner, 0.0) produced card-68612a2a (Explaining-Away Episodic Store), reduced again to generic OMP over an SDM store (1.4e-14).

Run 2: 5 escapes requested, 8 skipped for lack of a card slot, 5 created. 3 were reduced again (for example card-9e17a842 to card-5206e5ed, Max-Plus Skeleton Memory, reduced to other_known at 3.2e-13), and 2 died at S0. Those two died as duplicates of their own parents (TF-IDF cosine 0.609 and 0.686 against a threshold of 0.6). An escape child shares much of its parent's vocabulary, so the diversity filter can remove the cards the escape operator exists to produce.

### 3.4 Creativity metric

The S2 reduction critic answers whether each card's core idea is a textbook construction applied without modification. It is recorded and does not gate, and it does not measure novelty. In run 1, 10 of the 14 cards rated "modified" were still reduced.

| Explorer model | Run 1 transplant | Run 1 modified | Run 2 transplant | Run 2 modified |
| --- | --- | --- | --- | --- |
| claude-opus-5-5 | 6 | 10 | 5 | 12 |
| gpt-6.1-sol | 1 | 4 | (route down) | (route down) |
| All | 7 of 21 | 14 of 21 | 5 of 17 | 12 of 17 |

Core size stayed small after the parsimony change: mean 5.2 state objects and 4.2 update equations in run 1 (max 8 and 6), and mean 5.7 and 4.4 in run 2 (max 7 and 6). The first run last night had 23 to 54 state objects per card.

### 3.5 S3 results

**card-c9615069, Strand-Exchange Binding Kinetics: `infeasible_proved`.** Three attacks were confirmed by executed checks:

1. The card claimed that with K = 32 and capacity 16, eviction removes the oldest record, so the last 16 pairs score at least 0.9 and the first 16 at most 0.15. Eviction is by smallest concentration, and trained binding strengths vary across pairs (for example 0.93 against 0.99), so the weakest record is evicted regardless of age. The check printed `disproved: true` with survival 0.50 in both halves.
2. The card claimed one-shot recall reaches 0.95 within 3000 pretraining sequences. At batch 32 that is about 94 Adam steps at learning rate 1e-2, which moves each raw parameter by at most about 0.94 in total. With softplus parameters starting near k = 1, closing the gates needs raw parameters near -7.
3. An ablation threshold of 0.35 depends on an unconstrained parameter: at eta of about 0.05 the held-out share is 1/(1 + 0.7 + 2.8) = 0.22.

Both judges voted infeasible: claude-opus-5-5 at confidence 0.8, qwen3.6-35b-a3b-mlx at 0.95.

**card-46dbb349, Unknown-Duration Koopman Flow: `undetermined`.** Two rounds, split both times. Judge A (claude-opus-5-5) voted undetermined (confidence 0.6, then 0.68): complexity and the one-shot capacity obligations discharged, learnability and numerical stability left open. Judge B (qwen3.6-35b-a3b-mlx) voted infeasible (0.9, then 0.85), citing an executed float32 check in which the RLS state overflowed with P_0 = 1e6 and accuracy fell to 0.58. Judge A marked the same stability obligation open. This card passed S2 on an empty literature search, a gap fixed afterwards (3.9); under the current rule it would have stopped at S2 as `undetermined`.

Run 2 sent no card to S3. Its two cards with refuted reductions and no literature support (3.6) stopped at S2 as `undetermined`.

### 3.6 S2 passes

Run 1 had 2 S2 passes out of 21 S2 verdicts, both described above. card-c9615069 had literature hits and was labeled novel against them. card-46dbb349 was labeled novel because the search returned 0 papers ("the label is novel by definition"). Run 2 had no S2 pass out of 17 S2 verdicts. Two cards came closest: card-d29c6933 and card-6259da73 both had their claimed reduction refuted by the check (max_abs_diff 0.889 and 0.716; the second had been claimed as linear attention), but the literature search returned nothing even after the broadened retry, so both ended `undetermined` with "novelty unassessed". With working literature sources they would have been judged on prior art and could have reached S3.

### 3.7 Plumbing checks (operator override, not gate passes)

Both checks took card-537e161c, which S2 had reduced to linear attention, and ran the real S4, S5 and S6 stages on it with the Podman executor. The ledger's first event is the override, and every report page carries the banner. These results show that the stages run end to end on real models and the sandbox, and are no evidence that the card passed S2 or S3.

| | Check 1, `plumb-20261010T201437Z` | Check 2, `plumb-20261010T210700Z` |
| --- | --- | --- |
| Baselines | n-gram, GRU | n-gram, GRU, SSM, linear attention |
| S4 build | pass | pass |
| Model calls, wall | 2 calls, 1:25 | 3 calls, 1:56 |
| Flags | no_capacity_advantage, no_fewshot_gain | no_capacity_advantage, no_fewshot_gain |
| Nearest baseline by mean ratio | gru (0.506), 0 of 5 tasks close | linear_attention (0.428), close on 1 of 5 tasks (hmm: JS 0.016, ratio 0.40) |
| Nearest baseline by mean JS | gru (0.118) | ssm (0.121); linear attention 0.127 |
| `behaves_like` | not flagged | not flagged (needs more than half the tasks close) |
| Few-shot accuracy at 2, 4, 8 shots | 0.0, 0.0, 0.0 (chance 0.077) | 0.0, 0.0, 0.0 |
| Time slope (claimed 1.0) | 1.008 tail, match | 1.042 tail, match; memory flat |
| Analyst claims ledger | 1 supported, 0 refuted, 12 untested | 2 supported, 0 refuted, 10 untested |

Adding the two baselines moved the nearest neighbor from GRU to linear attention by ratio, which agrees with S2's verdict, but the fingerprint still did not flag the card. The `behaves_like` rule needs closeness on more than half of the tasks, and a 1-of-5 match is weak. The supported claims were only the complexity claims. Every capability and falsifier claim was untested because the bench has no iterated-rule task, no pointer chasing, no k* protocol and no recall at K = 256.

### 3.8 Throughput and model usage

| | Run 1 | Run 2 |
| --- | --- | --- |
| Wall | 2:16:48 over 6 invocations | 0:53:51, one invocation |
| Cards | 23 | 19 |
| Model calls | 233 (7 failed on every backend) | 178 (0 failed) |
| Input plus output tokens | 5.77M | 4.87M |

Calls per model in run 1:

| Model | Calls | Input | Output | Mean seconds |
| --- | --- | --- | --- | --- |
| claude-haiku-5-5 (salad critic 68, prior art 43) | 111 | 1.50M | 0.54M | 54 |
| claude-opus-5-5 (explorers 52, reduction critic 40, judge A 4) | 96 | 1.79M | 1.57M | 170 |
| gpt-6.1-sol (explorers 9, before the 403) | 9 | 0.06M | 0.07M | 273 |
| claude-sonnet-5-5 (formalizer 3, disprover 3, salad critic fallback 1) | 7 | 0.14M | 0.04M | 53 |
| qwen3.6-35b-a3b-mlx, local (judge B) | 3 | 0.06M | 0.02M | 96 |
| all backends failed (explorers 6, judge B 1) | 7 | | | |

Run 2 used only two models: claude-haiku-5-5, 91 calls (mean 22 s), and claude-opus-5-5, 87 calls (mean 192 s). No card reached S3, so the local judge was never called.

The operator guide describes the local judge as Qwen 3.8 27B. The ledger records `qwen3.6-35b-a3b-mlx` because judge B was moved to it during run 1 (see the local judge entry in section 4).

Most cards ended at S2, so S2 throughput set the pace. Opus calls average about three minutes, and the explorer, reduction critic and judge A all run on Opus.

### 3.9 Live tuning during the runs

- **Salad critic too strict.** In the first run the critic found new fixable gaps on every pass and revisions made cards larger (43 to 54 and 28 to 47 state objects). Changes: fatal versus fixable findings, two revisions with a "tighten, do not grow" instruction, parsimony fields in the card schema, then residual pass-through (no fatal finding and a reproduced toy pass after the revisions, with the remaining findings handed to S3). Result: 20 of 21 S1 passes in run 1 used it.
- **Explorer output cap.** Six explorer calls in run 1 came back empty with `finish_reason=length` at the 40,000-token cap, five of them escape tasks, whose user message carries the parent card, mapping and check (about 54 KB for card-537e161c). The explorer cap is now 96,000.
- **Gateway concurrency** raised from 4 to 10 requests in flight. With Opus calls averaging about three minutes, 4 slots were shared by 4 explorers and the S1 to S3 workers.
- **Lens seed per run.** `run.seed` is no longer fixed; the lens order comes from a hash of the run id, so a new run visits lenses in a new order, and a resume keeps the original order.
- **Empty search counted as novel, then fixed.** card-46dbb349 passed S2 on a search that returned 0 papers. S2 now retries with broadened queries, and if that is also empty records `novelty_unassessed` and an `undetermined` verdict. `s2.empty_search_counts_as_novel` defaults to false. In run 2 the retry ran 3 times and 2 cards ended `undetermined` for this reason.

## 4. Incidents

**Keychain prompts from the llxprt bridge.** The first run routed every role through LLxprt profiles, whose OAuth credentials live in the macOS keychain. Its calls produced keychain prompts that needed a person at the machine. That run ended after S1 with no recorded stop. The replacement is the Bigbrane gateway, which holds its own Anthropic and ChatGPT logins as private files under `~/.bigbrane-gateway/oauth/` and has a test (`test_no_keychain.py`) that fails if any module under `hybrid/` uses keychain APIs.

**ChatGPT access denied, then grant lost.** The ChatGPT route made 10 successful calls between 15:11 and 15:23 -03, then every request returned HTTP 403 `access_denied`, including `GET /v1/models`, which uses no plan quota. The token was unexpired and held all six scopes. A later token refresh was rejected with `invalid_grant` (seen during the session and not recorded in `serve2.log`), which means the grant itself is gone and a new login is needed. Diagnosis and request IDs: [results/gateway-20261010/chatgpt-403.md](results/gateway-20261010/chatgpt-403.md). Separately, the gateway's error parser did not read `detail.error_code`, so the first log lines showed an empty message. Since the 403, both explorer roles run on Opus, and the only non-Claude model in the loop is the local judge B. This costs the most at S0: in run 1 the GPT explorer had 4 of 5 cards rated modified, against 10 of 16 for Opus.

**Literature APIs rate-limited.** From `tmp/peek_lit.py` and the search receipts:

| | Run 1 | Run 2 |
| --- | --- | --- |
| Searches | 108 | 99 |
| Semantic Scholar failures | 102 (92 cooldown skips, 10 HTTP 429) | 95 (91 cooldown skips, 4 HTTP 429) |
| OpenAlex failures | 8 (7 cooldown, 1 HTTP 429) | 99, every search (93 cooldown, 6 HTTP 429) |
| IDs returned (arXiv / OpenAlex / S2) | 42 / 600 / 31 | 88 / 0 / 15 |
| Searches returning nothing | 7 | 69 |

In run 1 OpenAlex carried prior-art checking. In run 2 it returned nothing, so arXiv was the only working source. The reduction gate does not depend on literature, so it kept working, but novelty labels in run 2 rest on thin evidence, and cards whose reductions were refuted could only be marked `undetermined`.

**Local judge context limit.** On card-c9615069, judge B returned HTTP 400 because the prompt exceeded the loaded context length. At that point judge B was qwen/qwen3.8-27b, which LM Studio had loaded with an 8,192-token context. The card recorded a `stage_error`. I loaded qwen3.6-35b-a3b-mlx with a 131,072-token context, pointed judge B at it in `models.toml`, and resumed; the card was then judged. I unloaded that model at teardown, so a new run needs it loaded again with the long context (`POST http://127.0.0.1:1234/api/v1/models/load` with `context_length` 131072, or from the LM Studio UI).

## 5. What would make it more useful, in priority order

1. **Reliable literature access.** Add an OpenAlex `mailto` (polite pool) and a Semantic Scholar API key, and build a local paper index (arXiv plus OpenAlex metadata for the lens fields) that S2 queries first. Without it, prior-art labels in run 2 rest on arXiv alone.
2. **A second model family for explorers and judges.** Restore ChatGPT through a new login, or add another non-Claude route. The GPT explorer produced the higher modified share, and judge independence currently depends on one local model with a short context.
3. **Escape-slot reservation.** In run 1, 12 of 21 escape requests were skipped for lack of a card slot, and the escape operator produced every S3 card. Reserve a share of `max_cards` for escapes, and compare an escape child to cards other than its own parent in the diversity filter (two run 2 children died as duplicates of their parents).
4. **Bench tasks tied to card falsifiers.** The analyst marked most claims untested because the generic bench does not run each card's protocol. Let S4 emit a falsifier task (data generator plus exact oracle) that runs in the evaluator next to the five standard tasks, so S6 can mark claims refuted or supported.
5. **Lean or sympy proof obligations in S3.** Today S3 rests on model judges plus up to three executed attack scripts. Symbolic checks of the complexity and capacity obligations would give the judges executed results for those obligations, and a split vote could then be settled by a check.
6. **Fingerprint separation of linear attention and transformer.** On the default bench the two are flagged `behaves_like` each other (3 of 5 tasks, ratio 0.30). Add tasks where they differ in principle, such as recall well beyond the linear-attention state capacity and exact copy over long gaps.
7. **A larger default bench.** S5 uses the quick configuration (train sizes 32 and 128). Its baselines were n-gram and GRU until 18:06 today; `hybrid/config/bench_quick.toml` now also lists SSM and linear attention, because check 2 showed they change the nearest-neighbor answer. The transformer baseline is still absent from the quick configuration. Use the default configuration (train sizes up to 512, all five baselines) for any card that reaches S5 through the gates.

## 6. How to run and inspect

Full instructions are in [RESEARCH-SYSTEM.md](RESEARCH-SYSTEM.md). The short version:

```sh
podman machine start podman-machine-default
.venv-hybrid/bin/python -m hybrid.gateway status
.venv-hybrid/bin/python -m hybrid.gateway serve --port 8765        # separate terminal
.venv-hybrid/bin/python -m hybrid.research run --config hybrid/config/run.toml \
    --run-dir runs/run-<UTC-id> --routing gateway_claude_local
.venv-hybrid/bin/python -m hybrid.research run --run-dir runs/<run-id> --resume --routing gateway_claude_local
.venv-hybrid/bin/python -m hybrid.research inspect runs/<run-id>    # then open runs/<run-id>/report/index.html
```

State at handoff (18:45 -03): no research process is running; the gateway is stopped; the Podman VM is stopped; run 1's leftover container and volumes were removed with `scripts/podman-teardown.sh --apply --run run-20261010T181300Z --stop-machine`; the labeled executor image is kept; qwen3.6-35b-a3b-mlx is unloaded from LM Studio. The Anthropic login in the gateway expires around 23:10 -03 and refreshes on the next request.

To feed a new run the reductions of earlier ones, list them in `explore.graveyard_runs` (as `hybrid/config/run2.toml` does). The `gateway_claude_local` table needs LM Studio serving the judge B model on port 1234. The report's overview has stage counts, model usage, the creativity block, reduction families and escape lineage; each card dossier links its versions, critic transcripts, literature receipts, obligations, attacks, executor receipts and bench results.

Andrew's open decisions:

1. **GPT route.** Either sign in again with `.venv-hybrid/bin/python -m hybrid.gateway login chatgpt` after checking the Bigbrane app at chatgpt.com/settings/usage, or choose another GPT or non-Claude route for `explorer_alt` and judge B.
2. **Literature API keys.** A Semantic Scholar API key and an OpenAlex contact address (or key) for the polite pool, plus whether to build the local paper index.
