# Bigbrane research system: operator guide

## Verdict

Bigbrane automates a staged search for sequence-learning theories, from paper-grounded proposals through mathematical checks and, only after feasibility passes, sandboxed model construction and benchmark evaluation. Its records make decisions inspectable, while its model judgments and finite benchmark remain fallible evidence. The live run has already rejected several candidates as known architectures with numeric reduction checks; it has not established a new theory.

## The pipeline

Cards move through S0 to S6 in order. A failed gate ends that card's path. A card reaches construction only after S3 passes.

| Stage | Check and outcome | Model role |
|---|---|---|
| S0 Explore | Draft must match the theory-card schema. Similarity against existing cards uses TF-IDF; cosine at or above 0.6 stores and marks a duplicate as failed. | `explorer` and `explorer_alt` propose theories, mutations, or crosses using a lens, 3 to 6 papers, and existing-card summaries. |
| S1 Well-formed | Salad critic checks definitions, dimensions, falsifiability, and numeric claims. Executor runs the toy example and compares expected values (rtol 1e-6, atol 1e-9). Fatal findings or a final toy mismatch fail. Two revisions are allowed. With no fatal issue and a reproduced toy, remaining fixable findings pass as recorded residuals after revisions. | `salad_critic`; it does not run the toy itself. |
| S2 Prior art and reduction | Literature searches and grounded evidence classify prior art. A reduction critic tries to map the update to known families. A claimed reduction's check code runs in the executor when supplied. `known_disproven`, verified reduction, tested prior art without a specific untested addition, or failure of the radical-enough rule kills the card. A learning-rule difference explicitly marked `LEARNING_RULE:` can exempt a reduction. | `prior_art`, `reduction_critic`; a second `reduction_critic` call writes check code when needed. |
| S3 Feasibility | Formalizer writes complexity, capacity/information, and learnability obligations. Disprover attacks them; up to three attack checks run in the executor. Two independent judges decide. Confirmed disproof records `infeasible_proved`; both feasible with no surviving disproof passes. A split gets one extra round, then `undetermined`. | `formalizer`, `disprover`, `feasibility_judge_a`, `feasibility_judge_b`. Judges get identical inputs and do not see each other's verdict. |
| S4 Construct | Builder writes a minimal candidate. Static AST checks restrict imports and dangerous operations and block attention building blocks for claims below quadratic cost. Executor checks runtime, causality, output shape, and parameter-count consistency. Two repairs are allowed; failure to pass ends the card. | `builder`. |
| S5 Evaluate | Trusted bench compares candidate and baselines on exact-oracle tasks and measures scaling. This stage records results; it is not itself a scientific pass/fail gate. A job failure prevents usable results and records failure. | No model role. |
| S6 Verdict | Analyst compares card claims with bench numbers, records each as supported, refuted, or untested, and queues mutations for cards with supported claims while card budget remains. | `analyst`. |

## Run it

Use Python from `.venv-hybrid`. A new run needs an empty or absent run directory; resume requires its existing `ledger.jsonl`.

```sh
podman machine start podman-machine-default
.venv-hybrid/bin/python -m hybrid.gateway status
.venv-hybrid/bin/python -m hybrid.gateway serve --port 8765
```

For gateway routing, log in in advance from another terminal:

```sh
.venv-hybrid/bin/python -m hybrid.gateway login anthropic
.venv-hybrid/bin/python -m hybrid.gateway login chatgpt
```

The ChatGPT login uses a loopback listener on port 1455; avoid simultaneous Codex or LLxprt ChatGPT login. Anthropic login asks you to paste the `code#state`. Gateway credentials are private files under `~/.bigbrane-gateway/oauth/`, not the OS keychain. `status` reports login state without printing tokens. See [gateway login details](hybrid/gateway/LOGIN.md).

Start a new run, resume one, or inspect it:

```sh
.venv-hybrid/bin/python -m hybrid.research run --config hybrid/config/run.toml --run-dir runs/run-<UTC-id> --routing llxprt
.venv-hybrid/bin/python -m hybrid.research run --run-dir runs/<run-id> --resume --routing gateway_claude_local
.venv-hybrid/bin/python -m hybrid.research inspect runs/<run-id>
```

The current routing tables are in `hybrid/config/models.toml`:

| Table | Routing |
|---|---|
| `llxprt` | Explorers alternate Opus and GPT Sol profiles. Other roles use Claude profiles, with GPT/ Terra as the second judge family. |
| `gateway` | Claude roles use Anthropic through the gateway; explorer diversity and judge B use ChatGPT GPT-6.1 Sol. Judge B has no fallback. |
| `gateway_claude_local` | Claude roles use the gateway. Both explorer roles use Opus; judge B uses local LM Studio Qwen 3.8 27B to retain a separate family. |

`--routing` overrides `BIGBRANE_ROUTING`, which overrides the config's `routing` setting. The run checks gateway availability, bearer token, and all routed model IDs before creating run state. The gateway is a separate host process. The local Qwen route requires its LM Studio server to be available.

Run defaults are in `hybrid/config/run.toml`: 4 explorers, 12 S0 card attempts, 300 orchestrator calls, 20,000,000 tokens, and a 14,400-second wall budget per invocation. `hybrid/config/models.toml` adds per-run LLM limits of 600 backend requests and 40,000,000 tokens; schema retries and fallback attempts count as backend requests there. The run-level and LLM-level call counters have different definitions. Current workers: S1 4, S2 3, S3 2, S4-S6 1 each. Quick bench is `hybrid/config/bench_quick.toml`; S5 defaults to Podman. Executor caps and role efforts/output caps are in the two TOML files. CLI flags override matching run config keys. On resume, budgets are rebuilt from records, but wall time is fresh for that invocation.

## Inspect a run

`inspect` prints a summary and builds `runs/<run-id>/report/index.html`. Open the overview for run totals, card stages and reasons, model/token usage, lineage, and ledger verification. Each card dossier links versions, critic transcripts, literature receipts, feasibility obligations and attacks, executor receipts, and bench results. The overview's creativity block reports textbook-transplant answers versus modified ideas by explorer model, lenses receiving verdicts at each stage, and mean/maximum core size. These are diagnostic summaries, not novelty measures.

For S5, read the bench result and its fingerprint details. Flags include `behaves_like:<baseline>`, `no_capacity_advantage`, `quadratic_or_worse`, and `no_fewshot_gain`. Inspect the measured values and thresholds in `fingerprint.config` and `flag_details`, plus nearest-baseline similarity. The fingerprint uses predictions and timings, but relies on the candidate's self-reported state-floats count for one capacity ratio. Timing and memory flags are noisy, and linear attention and transformers can resemble each other on these tasks. See [bench details](hybrid/bench/README.md) and [baseline results](hybrid/bench/BASELINES.md).

The live run is still in progress. A read-only summary is available with:

```sh
.venv-hybrid/bin/python tmp/peek_run.py runs/run-20261010T181300Z
```

## Trust boundaries

- Candidate, toy-check, reduction-check, disproof-check, and construct jobs use Podman with `--network=none`, read-only root filesystem, unprivileged uid, dropped capabilities, resource caps, and only staged input/output volumes. Candidate jobs receive training data and test prefixes; oracle files stay outside the candidate job and are read by a separate evaluator job.
- Candidates share a process with the runner. They can inspect test prefixes during fitting and can tamper with runner-side timing. Evaluator scores are computed separately from prediction arrays, but scaling measurements can be affected.
- Podman is not a perfect resource boundary: `/out` has no disk quota, and a job can fill VM storage before its wall cap. The executor's local mode runs on the host without isolation.
- S5 `--local` is explicitly untrusted and marks results `trusted: false`. Do not use it for evidence presented as trusted evaluation.
- Gateway OAuth records and the per-start bearer token are private filesystem files, never keychain entries. Keep `~/.bigbrane-gateway` private, do not copy tokens into logs or run artifacts, and do not expose the gateway listener beyond localhost without a deliberate network/security review.
- Literature and model outputs are external inputs. Evidence references are grounded against retrieved hits, but search coverage and model classification cannot prove novelty.

## Where it can fool itself

| Failure mode | Safeguard | Remaining limit |
|---|---|---|
| Known work is missed or model calls retrieval a discovery | S2 records search receipts from arXiv, OpenAlex, and Semantic Scholar; evidence must match retrieved source IDs. | Search and databases are incomplete. A model label cannot establish novelty. Review closest statements and assumptions independently. |
| Mathematical prose looks coherent while definitions or claims conflict | S1 critic, schema, numeric toy execution, revisions, fatal/fixable categories, and residual findings passed to the S3 formalizer and disprover. | Residuals still pass S1 when the toy reproduces and there is no fatal finding. S3 judges do not receive the residual block, so follow whether formalizer/disprover treat the issues seriously. |
| A new name hides a known architecture | S2 requires an explicit reduction mapping or numeric check; S5 fingerprints behavior against baselines. | The reduction critic is a model judgment. Similarity fingerprints are task- and threshold-dependent; linear attention and transformers overlap, and noisy slopes can cross thresholds. |
| A feasibility argument misses a counterexample or both judges share a blind spot | Disprover and two judges from different model families; executor checks selected attacks; a card cannot reach construction without S3 pass. | Model-family diversity does not guarantee independent reasoning. Finite checks cannot prove general claims. |
| Candidate leaks evaluation information or distorts measurements | No network, staged inputs, separate evaluator, output hashing, and host-held oracle. | Candidate can read test prefixes during fit and tamper with runner timing. Prediction scoring remains evaluator-computed; scaling evidence is weaker. |
| Exact benchmark performance is mistaken for broad learning or complexity claims | Five exact-oracle tasks, multiple training sizes, baselines, length probes, and recorded scaling slopes. | Tasks and finite lengths are limited. Shared-VM timing varies; memory has a fixed floor; token-to-target interpolation is a finite-data estimate. |
| Formal proof or published claim exceeds what was checked | Keep novelty, feasibility, and measured bench claims distinct; analyst uses bench JSON, and literature/math claim failures are documented in [LIT-MATH-CLAIM-FAILURES.md](LIT-MATH-CLAIM-FAILURES.md). | Bigbrane does not have a proof assistant stage. Model verdicts and a finite benchmark cannot certify a theorem. |
| Run status or evidence appears stronger after interruption | Append-only hash-chained ledger, file hashes, stage artifacts, resume reconstruction, terminal verdicts. | A worker exception stops the run and is re-raised; inspect ledger events and `stage_error` records rather than assuming completion. |

## What the first live runs showed

The earlier run `runs/run-20261010T000322Z` stopped after S1, with no recorded `run_stop` or `run_end`; two cards remained resumable at S1. Its cards were mathematically structured but baroque, and revisions often added definitions without shrinking the mechanism. That prompted the parsimony fields and prompts, explicit fatal versus fixable findings, two revisions, and a critic instruction to leave toy execution to the executor.

The current run `runs/run-20261010T181300Z` shows the residual-pass-through change in use. Cards with no fatal finding and a reproduced toy can pass S1 after two revisions while carrying fixable findings forward. The live peek showed several such passes. Its S2 numeric reduction checks have caught every card that reached S2 as a known architecture: parity union-find, delta-rule fast weights, weighted RLS/EDMD, and related known forms. The recorded findings include mappings and numeric checks, so this is stronger evidence than label-only classification. It still does not establish that every possible disguised form will be detected.

S1's earlier over-strictness was addressed by residual pass-through: repeated fixable nits no longer block a card indefinitely after revisions, while fatal findings and toy failures still block. This changes throughput and what reaches S2/S3; it does not certify residual issues as harmless.

On 2026-10-10 the ChatGPT gateway route began returning HTTP 403 with `access_denied`, including `GET /v1/models`. The stored token was unexpired and held its expected scopes; the evidence points to an authorization/grant or account-side block, though the exact cause is unconfirmed. Request-shape changes will not fix this observed denial. Do not loop retries or logins against the 403. Check ChatGPT Settings > Usage or contact OpenAI with the request IDs in [the incident record](results/gateway-20261010/chatgpt-403.md). The error parser also failed to surface `detail.error_code`, a separate logging defect. Use `gateway_claude_local` only when the local LM Studio endpoint is available and the chosen routing is intentional.

## Operations

Every Bigbrane Podman container, image, volume, and pod must have `bigbrane.owner=bigbrane` and `bigbrane.run=<run-id>` labels. The executor also labels jobs with `bigbrane.job=<job-id>`. Prefer `--rm` for one-shot containers. Inspect labels before cleanup:

```sh
podman ps -a --filter label=bigbrane.owner=bigbrane
podman images --filter label=bigbrane.owner=bigbrane
podman volume ls --filter label=bigbrane.owner=bigbrane
podman pod ls --filter label=bigbrane.owner=bigbrane
scripts/podman-teardown.sh --run <run-id>       # dry run
scripts/podman-teardown.sh --apply --run <run-id>
```

The teardown script defaults to dry-run and only targets labeled resources. It can stop the VM with `--apply --stop-machine` if no container is running. Never use Podman system prune/reset or machine reset for Bigbrane cleanup; those can remove other projects' resources. Stop the gateway with Ctrl-C in the terminal running `serve`; its singleton lock is released on exit. Stop the Podman VM only when no active run or other project needs it: `podman machine stop podman-machine-default`.

Run state, transcripts, receipts, reports, and gateway-independent records live under `runs/<run-id>/`, not `/tmp`. Gateway OAuth and run-token files live under `~/.bigbrane-gateway/`; default metering is `~/.bigbrane-gateway/usage.jsonl`, and gateway serve output goes to its terminal unless redirected. Run-specific gateway metering can be selected with `serve --meter <path>`. Executor stdout/stderr and receipts are under each run's receipts/artifact paths. Preserve the run directory when auditing a result; teardown removes Podman resources, not research records.
