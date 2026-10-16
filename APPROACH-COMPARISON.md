# Keep Bigbrane's evaluator; trial autonomous research around it

Comparison cutoff: 2026-09-30, covering the September 29/30 research workflow. Live source access: 2026-10-07, the date reported by this session's environment. Release records and cutoff-filtered commits distinguish evidence available by the cutoff from current documentation. This is an approach comparison. No framework was installed, no experiment or test suite was rerun, and no new implementation review was conducted.

## User-ready synthesis

**Keep the Bigbrane instrument. Add broader literature search and trial an iterative research agent before replacing its orchestration.** The completed study answers a narrow question about a disclosed four-state prior. It provides a foundation for further experiments, but it has not demonstrated an autonomous discovery process or a new learning mechanism.

The actual workflow was literature investigation, a fixed hypothesis and protocol, delegated implementation, correctness pilots, review and repairs, then a frozen confirmatory run. Its 576 candidate fits compared a learned HMM with tiny RNN and attention controls across eight generators. Exact next-token oracles, independent fixtures, validation-only checkpoint selection, predefined statistical gates, source/data inventories, and recovery controls make the result inspectable. The recorded follow-up passed 29 tests. These checks were performed previously; this comparison reads their evidence. [B1-B6]

The HMM passed the low-data KL gate against both controls, with average advantages of .02292 nats/target over RNN and .06235 over attention. It reached the sustained error target with fewer tokens than both on only 1/8 generators. Both neural controls beat it at the largest matched budget. Its worsening high-data curve leaves fitting adequacy unresolved, and the stress comparison cannot attribute that entire gap to model mismatch. Known K=4, favorable synthetic generators, fixed neural widths, and validation assistance limit the claim. [B3]

| Approach | Main activity | What it could contribute here | Main qualification |
| --- | --- | --- | --- |
| Bigbrane so far | Execute a prespecified controlled study | Exact evaluation, attribution, and confirmatory evidence | Research agenda remains manually fixed between studies |
| DeerFlow 2.1 | General-purpose agent delegation, tools, memory, sandboxing | Durable orchestration and inspectable deliverables | Scientific design and statistical policy need integration |
| GPT Researcher | Web/local search and cited synthesis | Wider related-work search and competing explanations | Report quality does not establish experimental validity |
| Local Deep Research | Adaptive search with local-model/document options | Searchable literature library and private-source workflows | Local inference still incurs compute; web search sends queries outward |
| Microsoft RD-Agent | Hypothesis, code, execution, feedback, refinement | Optimization diagnostics and applied ML engineering | Scenario adaptation and independently protected final evaluation required |
| Sakana AI Scientist v2 | Idea generation, experiment tree search, analysis, manuscript | Open-ended hypothesis exploration and ablations | Selected workshop outcome supplies limited reliability evidence |
| LangChain Open Deep Research | Configurable search/synthesis with report evaluation | Reference implementation and evaluation patterns | Archived; adopting it creates maintenance responsibility |

**RD-Agent and AI Scientist v2 change the research method most directly.** RD-Agent separates performance-driven research from execution-error-driven development and supports multiple exploration traces. Its documented data-science objective is iterative metric improvement. That fits a follow-on investigation of optimization adequacy, or a practical prediction task, provided the original Bigbrane result remains unchanged. The project reports MLE-bench results, which concern ML engineering rather than proof of Bigbrane's scientific hypotheses. Its supported environment is Linux; most scenarios require Docker. [R1-R3]

AI Scientist v2 goes further into hypothesis generation and manuscript production. Its experiment manager advances through feasibility, tuning, research execution, and ablations, retaining checkpoints and replicating selected experiments. Those are documented scientific controls. Its paper also reports human selection among roughly forty ideas and among multiple completed runs: three selected manuscripts were submitted, one received workshop acceptance, and it was withdrawn before publication under the agreed procedure. This demonstrates a possible outcome, without establishing a one-in-three success rate per attempted run. The accepted work used synthetic arithmetic, so synthetic foundations alone do not distinguish Bigbrane from autonomous discovery. [S1-S3]

DeerFlow deserves a separate operational trial. Its 2.1 release documents runtime-stamped tool receipts, parent-side deliverable checks, durable SQL-backed batches, leases/retries, and shared token budgets. These address the kind of orchestration friction observed here: repeated subagent timeouts left recoverable artifacts, and the driver recovered work from files. They do not guarantee that a scientific conclusion is correct. GPT Researcher and Local Deep Research are more immediately useful for expanding the bounded literature search, including novelty checks and contradictory evidence. [D1-D2, G1-G2, L1-L2]

There is no measured cost winner. Bigbrane reports 1,320.79 seconds of candidate-worker wall time, excluding the whole literature, coding, repair, and LLM orchestration effort. Framework dollar figures refer to other models and workloads. Its local provenance has been exercised; platform controls have been documented here. No direct benchmark establishes superiority. [B3]

The practical next step is a bounded trial with separate exploration and confirmation budgets, a fresh protected holdout, and an inventory of every hypothesis, failed fit, retry, and selection. Retain local raw artifacts regardless of framework. Trial RD-Agent for task-specific engineering, AI Scientist v2 for creative hypotheses, and DeerFlow for orchestration recovery. Replace an orchestration layer only if it reduces intervention or improves independently confirmed work within the same declared resource limits.

## What Bigbrane actually established

The study's scientific value is a controlled prior-versus-flexibility baseline and an exercised measurement instrument. It exposes a distinction between lower low-data prediction error and reaching a target with fewer training tokens. The latter was not generally supported. It also exposes an unresolved fitting issue rather than silently extending the search until the HMM wins.

Its confirmation phase contains algorithmic adaptation: early stopping, checkpoint selection, and a conditional stress trigger. Those mechanisms were specified before outcomes. It contains no autonomous cycle in which an agent invents new scientific hypotheses from the measured result, changes the model family, chooses a new domain, and then validates a resulting discovery. LLMs helped research, write, implement, repair, and orchestrate the fixed study. That is different in scope from RD-Agent's continuing feedback loop and AI Scientist v2's hypothesis/experiment search.

The narrow synthetic setting buys exact conditional distributions and interpretable mismatch. An applied ML task would instead test usefulness on a particular data distribution, preprocessing pipeline, scoring rule, and resource budget. Neither setting automatically supplies novel science. An unknown-state-count or predictive-state follow-on would test a different claim; a retrieval/composition study would require its own task construction and held-out combinations. None is implied by the present HMM result.

The workspace contains historical status lines saying the experiment has not run. RESEARCH.md and EXPERIMENT.md preserve the documentation-stage record, while FOLLOWUP.md precedes the full run. The complete manifest and RESULT.md establish the later state. This comparison does not treat those earlier lines as current status.

### Local evidence locations

| ID | Exact location | Evidence used |
| --- | --- | --- |
| B1 | RESEARCH.md, sections “What the first study can establish,” “Decision path,” and “Sources accessed and access limits” | Bounded literature coverage, disclosed prior, unestablished novelty, fixed decision path |
| B2 | EXPERIMENT.md:7-17, 23-48, 69-86, 206-224, 246-278, 303-307, 331-344 | Scope, favorable generators, learner/evaluator separation, fit grid, predefined gates, correctness obligations |
| B3 | RESULT.md:3-11, 29-61, 67-75, 87-97 | Completed result, limits, full search costs, and previous integrity verification |
| B4 | README.md:15-45 | Correctness suite, pilots, frozen selection, artifacts, non-sandbox boundary, unknown interruption counters |
| B5 | REVIEW.md, F1-F3; FOLLOWUP.md:5-13, 25-31 | Earlier ownership/provenance defects and recorded repairs, 29-test follow-up, repeated pilots and unchanged completed resume |
| B6 | bigbrane/protocol.py:9-51; runner.py:117-139, 160-204, 208-240, 255-274; fitting.py:55-120; artifacts.py:60-106, 115-153 | Fixed search coordinates, frozen-tree execution, committed checkpoint inventories, locking, environment and provenance checks |
| B7 | results/experiment-001/confirmatory-v1-followup-c482b1/manifest.json:1-35; report.json:1-74 | Complete non-pilot run, frozen source/environment identities, both-control low-data decisions |

The mission's supplied session history additionally records TIMEOUT outcomes at configured limits of 900, 1,200, 1,800, and 2,400 seconds; an initial partial implementation admitted missing dependencies and failing checks. The driver recovered artifacts through files and later verification/repairs. This history describes observed engineering friction. It was not independently reconstructed from conversation logs during this comparison, and it supplies no cross-framework reliability rate. The recorded review/follow-up corroborates that recovery and provenance engineering needed work.

## Method, evidence, and autonomy

### RD-Agent: direct fit for iterative ML engineering

The Researcher learns from performance feedback; the Developer debugs using execution feedback. Its paper describes debugging on sampled data before full-data runs, multiple traces, checkpoints, and trace fusion. These mechanisms make experimental iteration part of the application instead of requiring the driver to initiate each next study. Its scenario documentation describes adapting a custom dataset and private evaluation files. [R1-R3]

The documented data-science loop optimizes validation or Kaggle leaderboard performance and also uses “test set” terminology in its feedback step. That terminology does not establish leakage in every implementation. It does mean that the adapter must explicitly define which scores may guide search and which data remain unavailable until the final claim is frozen. Repeated feedback on a held-out score turns it into selection information, regardless of the filename.

RD-Agent's MLE-bench evidence is relevant to implementing and optimizing models on applied tasks. The cutoff README reports 30.22 ± 1.5% for its o3/GPT-4.1 configuration; the May 2025 paper's table reports different earlier configurations and results. These are author-reported, version-specific outcomes, not a current universal ranking. No result establishes that it will diagnose Bigbrane's HMM curve correctly or discover a new predictive-state mechanism.

### AI Scientist v2: more exploration, with selection risk

The four-stage manager records checkpoints and replicates selected experiments for means and standard deviations. Configurable workers, per-stage iterations, debug depth, and seed replication bound parts of its search. Its README explicitly says some apparent controls, including k-fold validation, are unused in the current version. Merely setting those fields would not supply cross-validation. [S1-S3]

Selecting a promising node, tuning it, and replicating it helps detect stochastic instability. Repeating the chosen method on the same reused benchmark does not remove adaptive selection bias. A new protected confirmation set and a full attempt ledger remain necessary. The paper's human selection among ideas and complete manuscripts also matters when evaluating reliability or cost per independently useful result. The authors report citation inaccuracies and methodological/presentation problems in generated work.

The public code is designed for Linux, NVIDIA GPUs, CUDA, and PyTorch, and warns that LLM-written code can use dangerous packages, uncontrolled web access, or unintended processes. This is not a verified drop-in replacement for the existing M4 Max CPU experiment. Its license also requires disclosure of AI use in resulting manuscripts. [S1]

### Search tools and orchestration have different jobs

GPT Researcher's planner/executor/publisher architecture, recursive depth/breadth search, local documents, source tracking, and tracing could widen RESEARCH.md's coverage. Version 3.7 adds passage filtering and a local keyword alternative to embeddings. Citations and retrieved passages still require checking against readable primary material; a consensus of copied web pages is weak scientific corroboration. [G1-G2]

Local Deep Research documents adaptive academic/web/document search, a persistent searchable library, local-model endpoints, research history, and usage/cost analytics. It fits literature accumulation and private documents. Its local inference option avoids hosted-model charges but uses hardware and energy; public search services still receive queries. Its own README qualifies community benchmarks with small samples, grader noise, and possible contamination. These benchmarks answer retrieval questions, not experimental-science questions. [L1-L2]

DeerFlow now includes a general agent harness. Its release documents receipts and verifiable deliverable handles, deterministic checks for file existence and recorded test-command status, and explicit UNVERIFIED results for undecidable criteria. Durable batches and caps could reduce manual recovery burden. A recorded successful test command supports execution provenance, while interpretation still needs the experiment's tests and evaluator. Its local sandbox provider executes directly; an isolated provider must be deliberately configured. [D1-D2]

Open Deep Research remains useful as a reference for LangGraph orchestration, role-specific models, and report-quality evaluation. Its README documents Deep Research Bench and LangSmith export paths. The archive status shifts dependency/security maintenance to anyone adopting a fork. No source accessed here identifies a supported successor that can be assumed to replace it. [O1-O3]

## Reproducibility and total cost

Bigbrane's previous checks exercised source/data freezes, repeat pilots, committed checkpoint digests, surviving-worker ownership, and final inventories. This is local execution evidence. DeerFlow receipts and batch leases, RD-Agent traces/checkpoints, AI Scientist replications/journals, and search-tool histories/tracing are documented controls. This comparison has not adversarially tested those platforms or proven Bigbrane's controls superior.

Keep a framework-independent record of protocol versions, code and data hashes, environments, prompts/model settings, all generated hypotheses, search results, attempts, checkpoints, selections, and raw metrics. Archive readable source material and access dates where permitted. A hash establishes byte identity, not the truth of a scientific claim. LLM generation and live-web retrieval may also vary despite recorded settings.

Bigbrane's 576 fits produced measured search-resource totals and retained-model costs. Its 1,320.79-second worker total is not the cost of producing the instrument. Literature calls, LLM token charges, incomplete coding attempts, tests, review, fixtures, driver supervision, and platform setup need separate accounting. Interrupted uncommitted work can have unknown counters, as README.md discloses.

For context only, GPT Researcher's README advertises approximately $0.40 for an o3-mini deep-research run. AI Scientist v2's README gives approximately $15-$20 for experimentation plus about $5 for writing under its stated default models, excluding some ideation and broader selection costs. Neither price estimates this workload. RD-Agent's applied GPU experiments and Open Deep Research's report-benchmark costs are also incomparable to tiny CPU fits. A fair trial must record LLM/search charges, CPU/GPU use, all failed attempts, storage, and intervention counts.

## Retain, augment, and conditionally replace

Retain the existing experiment as an immutable result and retain its evaluator contracts. If the aim is a bounded mathematical comparison, continue using prespecified studies with exact synthetic oracles where available. If the aim is autonomous discovery, add an exploration lane that can propose and reject hypotheses before a separately frozen confirmation lane. If the aim is useful applied ML, use a task-specific dataset and scorer, with RD-Agent as a candidate implementation/search layer.

Trial one literature tool first: GPT Researcher for hosted-model/web integration, or Local Deep Research when local sources and model control matter. Trial DeerFlow separately for delegated work, evidence export, and recovery. Trial RD-Agent and AI Scientist v2 on research tasks matching their documented scope. None has been measured against the actual Bigbrane process here, so replacement remains conditional on evidence.

### Bounded trial design, proposed only

Use two lanes to avoid scoring orchestration and scientific creativity as if they were the same capability.

1. **Fixed-protocol lane:** give the existing driver and DeerFlow the same fresh synthetic study specification, hidden scoring service, correctness contract, and artifact requirements. Measure completion, protocol compliance, artifact completeness, interventions, duplicate work, and recovery after a predetermined interruption. The orchestration layer cannot modify the evaluator or its thresholds.
2. **Exploration lane:** compare the current driver, RD-Agent, and AI Scientist v2 on three fresh bounded research briefs, with two independent agent sessions per brief. Suitable briefs are HMM optimization diagnosis, unknown-state-count learning, and one task-specific sequence prediction problem. These are follow-on studies, not changes to Experiment 1. Supply the same literature packet and public development data. Use the same backing LLM where all platforms support it; otherwise label the comparison as whole systems rather than attributing effects to the framework alone.
3. **Declare budgets first:** cap each exploratory session at five hypotheses and ten attempted candidate fits, counting failed fits, and set common monetary and compute ceilings before launch. Keep setup costs separate but visible. Reserve a distinct confirmation budget with predefined replication counts. A budget exhausted without valid evidence is a reported outcome.
4. **Protect confirmation:** freeze one selected hypothesis, implementation, tuning procedure, metrics, and acceptance rule before opening fresh final data. For synthetic work, include independent generator draws as well as new sequences where the generalization claim requires them. Score through a separate trusted process. No agent gets final labels, oracle parameters, or authority to edit tests, scoring code, inventories, or thresholds.
5. **Record the denominator:** retain every hypothesis, code revision, failed/debugged node, retry, seed, abandonment, and selected result. Blindly assess claims against primary sources and independently recomputed metrics. Novelty needs a separate related-work assessment; paper polish or an LLM review score cannot stand in for it.
6. **Use isolated execution:** mount only allowed development data, restrict network access and runtime credentials, impose process/resource limits, and retain outputs outside the writable agent workspace. Container use alone does not establish isolation if host credentials, broad mounts, or the Docker socket are exposed.

Compare independently confirmed improvement, supported negative findings, protocol violations, citation correctness, attempt completeness, intervention count, and total cost. Adopt a platform when it improves the objective that matters within the declared limits. This outline has not been executed.

## Maintenance fact checks

| Supplied claim | Status from accessed primary source |
| --- | --- |
| DeerFlow 2.1.0 released September 24, 2026 | Verified. GitHub release API gives published_at 2026-09-24T10:40:44Z. |
| DeerFlow had 772 merged PRs since 2.0 | The official 2.1 release says exactly this. Verified as a maintainer statement; an independent PR enumeration/count was not performed. |
| GPT Researcher 3.7.0 released September 26, 2026 | Verified. GitHub release API gives published_at 2026-09-26T17:46:25Z. The same release identifies the pip package as 0.16.0; repository and package version labels differ. |
| Local Deep Research 1.10.7 released August 28, 2026 | Verified. GitHub release API gives published_at 2026-08-28T21:43:07Z. This release is labeled security hardening. |
| Open Deep Research archived August 21, 2026 | Verified. GitHub's repository banner names that date; the API separately reports archived=true. updated_at and pushed_at are not archive timestamps. |

Read-only release lists returned those three versions as the newest non-prerelease releases published by the cutoff. RD-Agent's list returned v1.0.0, published September 23, 2026. AI Scientist v2's release list was empty, and its latest main-branch commit before the cutoff was December 19, 2025. Its API currently reports archived=false. This documents less recent repository activity, without establishing abandonment or a guarantee of future support. Stars were not used to rank maintenance or performance.

Cutoff-filtered default-branch commit records: DeerFlow 1414e9fae75850480a8643d146d825b30b536714 (September 30); GPT Researcher 0957c301ed06c2a5857b834358c7227c739041d4 (September 26); Local Deep Research 75a4592915ae0c2d89317f588b7ae9619f99b82f (September 30); RD-Agent 484776c211e4fbbeef03e0ec00d6bbee7362a4f4 (September 23); AI Scientist v2 96bd51617cfdbb494a9fc283af00fe090edfae48 (December 19, 2025). These show dated repository activity, not contributor breadth or quality.

## Primary sources actually accessed

All web accesses occurred during this comparison. Some long README/API/HTML tool responses were truncated; only visible portions are used. The AI Scientist paper's experiment-manager, evaluation, and limitations sections were separately extracted from primary HTML. No benchmark outputs were reproduced.

| ID | Accessed source URL | Basis |
| --- | --- | --- |
| D1 | https://api.github.com/repos/bytedance/deer-flow/releases/tags/v2.1.0 | Release date, maintainer PR-count statement, receipts, durable batches, budgets, sandbox controls |
| D2 | https://raw.githubusercontent.com/bytedance/deer-flow/v2.1.0/backend/docs/ARCHITECTURE.md | Runtime, tools, state, local versus Docker execution |
| G1 | https://api.github.com/repos/assafelovic/gpt-researcher/releases/tags/v3.7.0 | Release date, context filtering, Python floor, package/repository version distinction |
| G2 | https://raw.githubusercontent.com/assafelovic/gpt-researcher/v3.7.0/README.md | Planner/executor/publisher, recursive search, local documents, tracing, contextual price claim |
| L1 | https://api.github.com/repos/LearningCircuit/local-deep-research/releases/tags/v1.10.7 | Release date and security-release status |
| L2 | https://raw.githubusercontent.com/LearningCircuit/local-deep-research/v1.10.7/README.md | Search strategies, library, local endpoints, analytics, benchmark caveats |
| R1 | https://raw.githubusercontent.com/microsoft/RD-Agent/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/README.md | Cutoff-supported environment and author-reported MLE-bench result |
| R2 | https://raw.githubusercontent.com/microsoft/RD-Agent/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/docs/scens/data_science.rst and https://raw.githubusercontent.com/microsoft/RD-Agent/484776c211e4fbbeef03e0ec00d6bbee7362a4f4/docs/project_framework_introduction.rst | Cutoff scenario objective and hypothesis/implementation/feedback loop |
| R3 | https://arxiv.org/html/2505.14738v1 | Dedicated research/development roles, sampled-data debugging, multiple traces, earlier benchmark settings |
| S1 | https://raw.githubusercontent.com/SakanaAI/AI-Scientist-v2/main/README.md | Scope, supported environment, code-execution warning, unused config fields, price claims and disclosure requirement |
| S2 | https://raw.githubusercontent.com/SakanaAI/AI-Scientist-v2/main/bfts_config.yaml | Workers, stage caps, debug depth, copies, logs and seed replication configuration |
| S3 | https://arxiv.org/html/2504.08066v1, especially sections 3.2.1, 4.1, 4.2 and 5 | Manager stages, checkpoints/replication, idea/run selection, workshop outcome and withdrawal, limitations |
| O1 | https://github.com/langchain-ai/open_deep_research | Dated archive banner |
| O2 | https://api.github.com/repos/langchain-ai/open_deep_research | archived=true, last push metadata |
| O3 | https://raw.githubusercontent.com/langchain-ai/open_deep_research/main/README.md | Configurable search/model roles, benchmark/report evaluation and export |

Additional directly accessed discovery/current-documentation URLs: https://raw.githubusercontent.com/bytedance/deer-flow/main/README.md ; https://raw.githubusercontent.com/assafelovic/gpt-researcher/master/README.md ; https://raw.githubusercontent.com/LearningCircuit/local-deep-research/main/README.md ; https://raw.githubusercontent.com/microsoft/RD-Agent/main/README.md ; https://rdagent.readthedocs.io/en/latest/scens/data_science.html ; https://rdagent.readthedocs.io/en/latest/project_framework_introduction.html . Current-branch details that differ from the cutoff snapshots were not silently attributed to September 30.

For each of bytedance/deer-flow, assafelovic/gpt-researcher, LearningCircuit/local-deep-research, microsoft/RD-Agent, and SakanaAI/AI-Scientist-v2, these public endpoints were actually read: `https://api.github.com/repos/{owner}/{repo}`, `https://api.github.com/repos/{owner}/{repo}/releases?per_page=30`, and `https://api.github.com/repos/{owner}/{repo}/commits?until=2026-09-30T23:59:59Z&per_page=1`. No credentials were used or changed. A focused search returned unrelated papers and supplied no evidence for this report.
