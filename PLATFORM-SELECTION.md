# Platform selection for mathematical discovery

**Select Microsoft RD-Agent as the research-loop foundation, adapted around Bigbrane's existing executor and a separately trusted evaluator.** RD-Agent's inspected design contains the scientific-loop concepts this project needs: scenarios, hypotheses, experiments, developers, experiment-to-feedback mapping, trace ancestry, and iterative search. Start with those abstractions, not its default Qlib/ALPHA20 or competition-scoring application. DeerFlow is the runner/workbench alternative where macOS execution or general agent infrastructure outweighs the extra work of building scientific search policy. GPTResearcher and Local Deep Research can support literature and evidence gathering. Open Deep Research is a lower-priority report workflow. None of these selections is a measured performance ranking.

## Decision and guardrails

The target capability is repeated mathematical learning discovery: state a falsifiable hypothesis, design a discriminating experiment, generate or revise candidate code, execute experiments, analyze outcomes, and retain traceable evidence. Producing a cited research report alone does not satisfy that target.

Adopt RD-Agent selectively. Keep Bigbrane's executor as an external service, outside the agent's writable workspace. Do not rely on RD-Agent's default custom-data workflow for confirmation: its custom evaluator path injects answer CSV data into a writable experiment workspace and writes candidate `scores.csv`; the inspected sources did not establish an adequate independent, protected final evaluator. Build that trust boundary in Bigbrane, with a fresh holdout and evaluator inaccessible to candidate code. Avoid composing the default Qlib ALPHA20 assumptions into the scientific workflow.

RD-Agent's documented support is Linux-only. Treat Linux arm64 compatibility as an initial deployment gate. If it fails or a macOS-native interactive runner becomes the immediate requirement, evaluate DeerFlow as the alternative, while budgeting for a Bigbrane-specific scientific controller. DeerFlow supplies general Python agent factories, LangChain/tool/extension integration, lifecycle handling, SQL-backed batches, leases and retries, and token caps/defaults. The inspected core assessment found that its scientific acceptance and batch-completion evidence are insufficient for a confirmation boundary, so retain the same independent evaluator requirement.

## Candidate findings

| Candidate | Fit for hypothesis-to-experiment discovery | Selection |
| --- | --- | --- |
| **Microsoft RD-Agent** | Directly models research scenarios, hypotheses, experiment execution and feedback, trace ancestry, and iteration. Its default examples are domain-specific and need replacement or adaptation. | **Choose as research-loop foundation**, subject to Linux arm64 adapter proof and external trusted evaluation. |
| **DeerFlow** | General-purpose agent orchestration and execution lifecycle features; it does not supply Bigbrane's scientific search strategy in the inspected assessment. | **Runner/workbench alternative**, particularly if platform flexibility is a priority. Not selected as the discovery loop. |
| **GPTResearcher** | README describes planner/execution agents that collect and summarize sources, then write a cited report; supports web and local-document research and a Python `conduct_research` / `write_report` interface. | Optional literature-research companion. The read sources do not establish a hypothesis-design, candidate-code, experiment-feedback loop. Apache-2.0 in the cached source. |
| **Local Deep Research (LDR)** | README describes configurable research strategies, web/academic/local sources, citations, local and hosted model endpoints, and a Python/API interface. | Optional evidence-gathering companion if a specific source/search workflow is useful. The read sources do not establish mathematical experiment orchestration or trusted scoring. MIT in the cached source. |
| **Open Deep Research (ODR)** | LangGraph research workflow with configurable models and search, synthesis/compression, final report generation, and documented report-oriented evaluation. | Do not adopt as core. Cached application README's latest dated update is August 2025; maintenance screen reports archived August 21, 2026. Lockfile activity on August 10 meets the maintenance cutoff mechanically, but does not establish active application development. MIT in the cached source. |

The literature companions may inform a human or controller. Do not grant them access to hidden answers, candidate scoring, or the final holdout. Their reported research benchmarks, where present, measure research/report tasks and cannot be used as evidence of mathematical discovery performance.

## Maintenance screen

This table records the supplied maintenance-screen findings, not a quality score. The screen's rule is machine date October 7, 2026, with an April 7, 2026 cutoff. A qualifying recent commit is evidence of activity under that rule; it does not establish suitability, reliability, or support quality.

| Candidate | Maintenance finding | Interpretation |
| --- | --- | --- |
| RD-Agent | Active code through September 23, 2026 | Qualifies; compatible with the dated source assessment. |
| DeerFlow | Active through October 7, 2026 | Qualifies. |
| GPTResearcher | Active through September 26, 2026 | Qualifies. |
| Local Deep Research | Active through October 7, 2026 | Qualifies. |
| Open Deep Research | Lockfile activity August 10, 2026; archived August 21, 2026; latest app update August 2025 | Meets the cutoff through lockfile activity, but archival and old application activity lower its priority. |
| AI Scientist v2 | Last main-branch activity December 19, 2025 | Excluded by cutoff. |

## First bounded follow-up

Before installing or adopting a platform, run a narrow compatibility proof for RD-Agent's Linux arm64 controller and a custom scenario adapter. Use one question: whether learned-state complexity predicts when a recurrence model versus a hidden Markov model (HMM) is preferred. Freeze the hypothesis, proposed equations, development-data design, and falsifier before candidate generation. Route execution through Bigbrane's trusted harness. Record each attempt, cost, code, and artifact; evaluate only on a fresh holdout through an evaluator unavailable to the candidate workspace. Stop if the controller cannot preserve the freeze boundary or if platform incompatibility blocks the adapter. This is a proposed test, not a result.

## Evidence, limits, and sources

The detailed source-level comparison of RD-Agent and DeerFlow is in [PLATFORM-CORE-ASSESSMENT.md](PLATFORM-CORE-ASSESSMENT.md). It inspected fixed commits and documents assumptions, execution and evaluation weaknesses, and deployment caveats. This selection adopts its recommendation and does not repeat that code audit. The maintenance findings are from [MAINTENANCE-SCREEN.md](MAINTENANCE-SCREEN.md). Cached README and license evidence for the three literature platforms is in `results/platform-literature-sources-20261007/`; the pinned source identifiers below identify those snapshots.

Primary sources:

- [Microsoft RD-Agent repository](https://github.com/microsoft/RD-Agent/tree/484776c211e4fbbeef03e0ec00d6bbee7362a4f4)
- [DeerFlow repository](https://github.com/bytedance/deer-flow/tree/a4427625cfc0759021736cca51426bfc2759d6c8)
- [GPTResearcher repository](https://github.com/assafelovic/gpt-researcher/tree/0957c301ed06c2a5857b834358c7227c739041d4)
- [Local Deep Research repository](https://github.com/LearningCircuit/local-deep-research/tree/2940c8cd8445fd676b34f0f9291cd3118a5a974d)
- [Open Deep Research repository](https://github.com/langchain-ai/open_deep_research/tree/1b7d2e80db9faa586165c60e09096dbbfd483a64)

No platform was installed or run for this synthesis. There is no head-to-head experiment, independent reproduction of upstream benchmarks, or verified deployment on this machine. For GPTResearcher, LDR, and ODR, selection claims are limited to the cached README/license evidence and supplied maintenance screen; absence of a capability in those sources does not prove the codebase lacks it. Platform and dependency license review beyond the displayed root licenses was not performed.
