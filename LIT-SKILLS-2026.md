# Agent skills literature, August to early October 2026

Machine date checked with `date`: Friday, October 9, 2026 (local -03). Window covered: early August to early October 2026. Items dated before August 2026 appear only under "Background (before the window)". All arXiv dates below come from the arXiv abstract pages linked, which I opened. Figures come from the paper HTML I read, either directly or through search excerpts of that HTML; the one item read only at abstract level is marked.

## Verdict

The recent skills work mostly measures how agents use SKILL.md packages and how to maintain skill libraries. No vendor launched a new skills capability or spec version in the window, and the agentskills.io spec only received wording clarifications in early August. The most useful in-window result (Skill-Use, August 5) shows that agents often fail to open a relevant skill at all, that the best model and harness pair scored 0.613 on a combined skill-use measure, and that a partly followed skill lowered task completion below having no skill. Skill self-evolution papers (WikiSkill, SE-GoS, SkillPivot, ContinualSkillBench) report gains only when every skill edit passes a validation gate, and one benchmark found that plain in-context history matched explicit skill maintenance on average. On model scale, the evidence points both ways. Within one model family, larger models gained more from evolved skills, and skills let smaller models pass larger skill-less ones; very small models fail at choosing skills. Security papers keep finding exploitable skills: SkillSecurer flagged 17.6% of 954 popular public skills, and reviewers confirmed 84 of 100 sampled findings. The evidence is moderate. Most results come from single papers with three runs or fewer per condition, LLM judges, and author-built benchmarks, and no in-window work tests skills on mathematical discovery with an exact oracle.

## In-window items

### 1. Skill-Use: Can LLMs Actually Use Skills in Agentic Harnesses?

- Authors/org: Jinyi Han, Yuanjian Xu, Ying Liao, Xinyi Wang, Zishang Jiang, Zixiang Di, Fanyang Lu, Zhichao Hu, Yanghua Xiao (East China Normal University, HKUST, Fudan University, Tencent Hunyuan).
- Date: arXiv v1, August 5, 2026.
- Link: https://arxiv.org/abs/2608.04828 (full text: https://arxiv.org/html/2608.04828v1)
- What was measured: 79 real skills, filtered from 105,586 GitHub candidates, paired with 177 executable Docker tasks in nine domains. Hidden trajectory rubrics hold 1,314 scored items. Eight models (Claude Opus 4.7, Opus 4.8, GPT-5.5, DeepSeek-V4-Pro, GLM-5.1, MiniMax-M3, Kimi-K2.6, Qwen3.6-Max) ran under two harnesses (Claude Code, Codex). The agent saw only each skill's name, description and path. Metrics: Trigger (did the agent open the skill), Compliance (prescribed steps done), Boundary (forbidden operations avoided), and a gated SU score of Trigger times (0.7 Compliance + 0.3 Boundary). Extra experiments covered preloaded versus native injection, library sizes of 1, 10, 20 and 30 with random distractors, 53 out-of-scope tasks, and a paired comparison against a no-library baseline over 779 triggered Claude Code runs.
- Finding: The best SU was 0.613 (GPT-5.5 under Claude Code). On traces where the skill was opened, the best conditional Compliance was 0.638. Under Claude Code, DeepSeek-V4-Pro, GLM-5.1 and Kimi-K2.6 opened the target skill on about a third of tasks (Trigger 0.324 to 0.337), yet followed it about as well as stronger models once opened. Preloading the whole skill raised Trigger and SU but barely changed execution quality after triggering. Growing the library from one skill to ten caused the main drop, mostly from runs that called no skill; wrong-skill calls were rare and clustered on broad descriptions such as test-driven-development. Models that followed skills best were among the worst at holding back on out-of-scope tasks (negative rank correlation). Paired task-completion gain turned positive only above an SU of about 0.5; below that, using the skill lowered completion compared with no skills. Model rankings changed between harnesses.
- Limits: An LLM judge (GPT-5.4) scored items without a deterministic check; repeated rescoring gave an ICC of 0.53 for SU. Gemini was excluded because its gateway broke the Skill tool. The authors built and validated the rubrics. Task-completion effects are reported by SU bin, without one headline effect size.

### 2. ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?

- Authors/org: Tianyi Guan, Yiding Wang, Haotong Yang, Siyuan Cao, Shirui Liu, Yi Hu, Jiaqi Li, Muhan Zhang.
- Date: arXiv v1, August 4, 2026.
- Link: https://arxiv.org/abs/2608.03874 (HTML: https://arxiv.org/html/2608.03874v1)
- What was measured: Five domains, each with 100 interconnected subtasks ordered by difficulty. Sequential agents that keep and update a skill repository were compared with independent runs from scratch, and on three domains with an in-context-learning condition that keeps prior context and feedback but no explicit skills. Models named in the results include GPT-4o, GPT-5.3-Codex and Claude Opus 4.7.
- Finding: Sequential execution improved normalized reward in 14 of 15 model and domain pairs, an aggregate relative gain of 16.9%. On the three ablation domains, plain in-context learning (0.605) matched explicit skill maintenance (0.602); independent runs scored 0.466. Explicit skills helped on exact-match and programmatic outputs, while in-context learning scored higher on rubric-graded outputs. GPT-4o built 384 skills with an average quality score of 5.68; GPT-5.3-Codex built 205 with an average of 7.94 and reused them more often. The strongest baseline model (Opus 4.7) did not get the largest sequential gain.
- Limits: The skill-versus-ICL ablation covers three of five domains. Skill quality scores come from an LLM grader. Few models tested.

### 3. SkillEval: Decomposing Agent Skill Quality into Interpretable Signals

- Authors/org: Jiahui Han, Qinuo Li, Ziheng Peng, Haotian Wu, Haoze Liu, Danfeng Shan, Guanchu Wang, Huiqi Deng, Ninghao Liu.
- Date: arXiv v1, August 7, 2026.
- Link: https://arxiv.org/abs/2608.06891 (HTML: https://arxiv.org/html/2608.06891v1)
- What was measured: A document-level scorer for SKILL.md files. It has seven metrics in four dimensions (applicability, content quality, execution guidance, robustness). One metric is a rule-based format check. The other six are directions in a frozen model's hidden space learned from controlled good/bad skill pairs, with length and formatting directions projected out. Validation used 518 skills (259 positive, 259 negative). Downstream checks used SkillsBench tasks with Codex and GPT-5.5 at xhigh effort, three runs per condition.
- Finding: Per-metric linear fits to pass-rate uplift had Pearson r of 0.779 to 0.787. On continuous-score SkillsBench tasks, unrevised skills averaged an 18.6% pass rate, LLM-only revision reached 38.3%, and revision guided by SkillEval diagnostics reached 48.1%.
- Limits: One model and harness downstream, three runs per condition, and correlations fitted and reported on the same benchmark. The revision experiment covers a subset of tasks.

### 4. SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem for Real-World LLM Agents (revised in window)

- Authors/org: Yanze Wang, Pengfei Yao, Tianyi Sun, Chuanrui Hu, Yan Xiao, Xiaotian Luo, Yunyun Han, Yifan Chen, Jun Sun, Yafeng Deng (code at github.com/EverMind-AI/SkillCorpus).
- Date: v1 July 17, 2026; v5 August 6 and v6 August 26, 2026. The first version predates the window.
- Link: https://arxiv.org/abs/2607.15557 (HTML: https://arxiv.org/html/2607.15557v6)
- What was measured: About 821,000 crawled community skills filtered to 96,401, served by a fine-tuned retrieval and selection stack. Evaluated on SkillsBench (87 tasks, pass@1 over three runs), GDPVal (220 tasks, LLM-judge reward) and QwenClawBench (100 tasks), with two harnesses (OpenClaw, Raven) and two open backbones (Qwen3.5-27B, Qwen3.5-397B-A17B), plus a Claude Opus 4.7 check.
- Finding: Pooled gains were +7.5 pp on SkillsBench (z = 3.2), +1.51 on GDPVal and +2.79 on QwenClawBench. The SkillsBench baselines sat near 10%. The strongest cell (Raven with the 397B model) went from 9.2% to 22.6%. Opus 4.7 gained +8.0 pp on SkillsBench. Gains were largest where the no-skill baseline was low, and the harness changed how much gain appeared.
- Limits: Low absolute pass rates for open backbones. Per-task noise on GDPVal was 20 to 27 pp, so most single cells cannot be separated from zero. English-only corpus.

### 5. WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

- Authors/org: Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu.
- Date: arXiv v1, August 27, 2026.
- Link: https://arxiv.org/abs/2608.27454 (HTML: https://arxiv.org/html/2608.27454v1)
- What was measured: A skill-evolution loop with four roles. An inference agent runs tasks with the current skills and writes immutable traces. A wiki maintainer distills traces into a persistent knowledge base that also records rejected proposals. A skill proposer edits skills from the wiki. A validation gate accepts an edit or rolls it back. Benchmarks: LiveMathematicianBench, SealQA, SpreadsheetBench, OfficeQA, ALFWorld. Models: Gemini-3.5-Flash, Qwen-3.5-4B and 9B, Qwen-3.6-27B, Gemma-4-31B. Baselines: Trace2Skill, EvoSkill, SkillOpt, no skill. Three independent evolution runs per method, paired bootstrap at p < 0.05.
- Finding: Within the Qwen family, average gains from evolved skills grew with size: +12.3 points for 4B, +17.5 for 9B, +23.9 for 27B. Qwen-3.5-9B with evolved skills (47.4%) beat Qwen-3.6-27B without skills (39.4%). Qwen-3.6-27B went from 39.4% to 63.3% on average and Gemini-3.5-Flash from 49.5% to 68.1%. On LiveMathematicianBench, Gemini-3.5-Flash went from 33.0% to 72.6%. Skills evolved by one model often helped another more than its own; Qwen-3.5-4B's OfficeQA skills lowered its own score (30.2% to 28.5%) but raised Qwen-3.6-27B's (42.1% to 52.9%). Removing the persistent wiki from the proposer dropped Gemini-3.5-Flash's average from 63.7% to 48.7%. Giving the inference agent wiki access during training rollouts hurt.
- Limits: The scale trend comes from one family at three sizes up to 27B. Validation and test splits come from the same benchmark distributions. The authors' own framework, compared against reimplemented baselines.

### 6. SE-GoS: Self-Evolving Graph-of-Skills for Skill Library at Scale

- Authors/org: Dawei Fu, Cheng Jiang, Sitian Qian, Huainan Wang, Zhongkai Hao.
- Date: arXiv v1 September 8, 2026; v2 October 1, 2026.
- Link: https://arxiv.org/abs/2609.08228
- What was measured: Updates to a skill retrieval graph from execution traces, while the retrieval algorithm, skill contents and model stay fixed. Three update types: graph edges added or retracted from execution evidence, edge weights, and retrieval-facing node descriptions. Evaluated on SkillsBench with three backbones.
- Finding (abstract level; full text not read): One evolution round raised average reward from 52.4% to 59.4%, above full-library loading, vector retrieval, static Graph-of-Skills and SkillDAG, with the same ordering on all three backbones. Retrieval used about two-thirds of the input tokens of full loading. A second round did not help. A held-out split went from 52.9% to 58.3%.
- Limits: I read only the abstract page. Backbones and run counts not verified here. Single benchmark.

### 7. SkillSecurer: Detecting and Patching Prompt-Injection Vulnerabilities in AI Agent Skills

- Authors/org: Donato Mecca, Alberto Verna, Youness Bouchari, Nikhil Jha, Marco Mellia (Politecnico di Torino).
- Date: arXiv v1, September 12, 2026.
- Link: https://arxiv.org/abs/2609.14079 (full text: https://arxiv.org/html/2609.14079)
- What was measured: A red agent injects context-fitting attacks across nine threat types and records the exact edit; a blue agent reports findings with quoted evidence and proposes patches; a verifier checks whether the recorded injection was localized. Controlled sets: 180 generated injections over 20 base skills, and 165 Skill-Inject instances. In the wild: 954 popular skills.sh skills (snapshot July 3, 2026). Live exploitation: five flagged skills run in OpenCode in containers with six backend models, before and after patching. Compared against Cisco Skill Scanner, NVIDIA SkillSpector and Tencent AI-Infra-Guard.
- Finding: With Claude Sonnet 5 as backend, the blue agent localized 100% of injections on both controlled sets. Rule-only Cisco Skill Scanner localized 2.2% and 0.6%. On 130 likely-benign skills it flagged 45; reviewers confirmed 74 of 87 findings as latent vulnerabilities. In the wild it flagged 168 of 954 skills (17.6%); three reviewers confirmed 84 of 100 sampled findings. Live exploits fired in 13 of 30 model and skill runs before patching and 3 of 30 after. Claude Sonnet 5 fired none; Kimi K3 and DeepSeek V4 Flash fired 4 of 5. Models often described a payload as unsafe and still ran it. One patch failed because its warning sat in a Troubleshooting section that the executed path never reached.
- Limits: Five skills in the live test. The verifier uses an LLM judge for non-lexical matches. Patch review is static and does not test that legitimate function survives.

### 8. A Wrong Turn Does Not Ruin the Journey: Deviation-Guided Skill Self-Evolution for LLM Agents (SkillPivot)

- Authors/org: Yichun Feng, Jiawei Wang, Haozhe Sun.
- Date: arXiv v1, September 24, 2026.
- Link: https://arxiv.org/abs/2609.29154
- What was measured (abstract and search excerpt of the HTML): The method finds the step where a failed trajectory turns from useful work into error, using execution validity, goal progress and action diversity. A stronger teacher continues from that prefix, and contrasting the two suffixes yields a small skill edit. A regression-gated verifier checks that the edit keeps working rules before it is merged. Evaluated on ToolQA, LogicBench and WildClawBench.
- Finding: The abstract reports that it beats competing skill-evolution methods, improves several agent models, and produces compact, transferable updates. I did not read the result tables, so no numbers are given here.
- Limits: Abstract-level only. Depends on a stronger teacher model.

### 9. Agent Skills specification (agentskills.io): changes in the window

- Org: agentskills/agentskills repository; its README says Anthropic originally developed the format and released it as an open standard (https://github.com/agentskills/agentskills).
- Dates and links: Commit history at https://github.com/agentskills/agentskills/commits/main shows the last commits on main on August 9, 2026 (client showcase entries). Spec-text changes in the window: PR #479, merged August 4, presents the key and value types of `metadata` in the frontmatter overview table (https://github.com/agentskills/agentskills/pull/479); PR #268, merged August 4, states that the listed optional directories are recommendations and a skill may contain any other files (https://github.com/agentskills/agentskills/pull/268). The current spec (https://agentskills.io/specification) still has the fields `name`, `description`, `license`, `compatibility`, `metadata` and experimental `allowed-tools`, with three-tier progressive disclosure.
- Finding: No new fields, no versioning, and no signing landed in the window. A proposal for skill versioning and lockfiles in the `.well-known` distribution index (PR #380, https://github.com/agentskills/agentskills/pull/380) was open, last updated July 16, 2026.
- Limits: I searched for vendor announcements (Anthropic, OpenAI, Google) in the window and found only secondary news pages about the existing anthropics/skills repository. I found no primary vendor post dated August to early October 2026, so none is claimed.

## Model scale: what the evidence says

- Larger models gained more within one family when skills were evolved and validated (WikiSkill, Qwen 4B to 27B). Smaller models with evolved skills beat larger models without them.
- In SkillsBench v4 (background, June), the configuration with the smallest gain (+4.1 pp) used Gemini 3.1 Flash Lite, and Claude Opus 4.8 under OpenHands also gained little (+8.4 pp). Gains do not line up with scale in any simple way.
- Skill-Use shows that mid-tier open-weight models lose mostly at triggering, so harness choices that preload or name the skill matter more for them.
- ContinualSkillBench links weaker models to larger, more fragmented self-built skill libraries.
- Taken together, skills narrow gaps between capable small and large models, very small models lose skill selection first, and frontier-model gains depend on the harness. No in-window study isolates scale cleanly across families.

## Background (before the window)

- SkillsBench, Li et al., arXiv 2602.12670, v1 February 13, v4 June 14, 2026: https://arxiv.org/abs/2602.12670. v4: 87 tasks, 18 model and harness configurations, curated skills raised mean pass rate from 33.9% to 50.5%, configuration gains +4.1 to +25.7 pp, focused skills with at most three modules beat exhaustive bundles. v1: self-generated skills averaged -1.3 pp and 16 of 84 tasks had negative deltas.
- Skill-Inject, Schmotz et al., arXiv 2602.20156 (February 2026), as summarized in SkillSecurer's related work and the skill-inject.com page: 202 injection and task pairs; script-hidden payloads succeeded more often than text injections.
- Earlier pre-window results, read only through search excerpts of their arXiv HTML and listed here as leads: "The Scaling Laws of Skills in LLM Agent Systems" (arXiv 2605.16508) reports routing accuracy falling with the log of library size across 15 models; "More Skills, Worse Agents?" (arXiv 2605.24050) attributes up to a 21% pass-rate drop at 202 skills mainly to wrong or missing skill selection.
- Anthropic's Agent Skills launch (October 16, 2025) and the open standard (December 18, 2025): https://claude.com/blog/skills.

## What this means for Bigbrane

### What to package as skills

Skills fit procedures that tell an agent how to do something whose correctness trusted code checks afterwards. Based on the findings above, each Bigbrane skill should be short and concrete, a few modules at most, and should state its trigger in plain words. Skill-Use and SkillsBench both found that concrete procedures beat broad guidance, and that a half-followed procedure can do worse than none.

Candidate skills:

1. Hypothesis and falsifier drafting: how to fill Bigbrane's hypothesis schema (claim, equations, falsifier, margin, development-data design) and then submit it to the freeze tool. The skill writes the draft; the freeze is done by code.
2. Prior-art search: query plan, sources to try, how to record citations with links and dates, and how to record "not found" explicitly in the ledger record.
3. Experiment design checklist: how to pick a discriminating development experiment, what the executor contract expects, and which outputs (predictions, hashes) a candidate must emit.
4. Candidate code conventions: the executor's input and output format, resource limits, and the no-network rule, so generated code fits the trusted harness.
5. Lean proof-writing workflow: how to state a lemma, structure tactics, read checker errors and iterate. The proof check itself stays trusted code (see below).
6. Oracle-result reading: how to interpret the evaluator's returned record (scientific negative versus invalid run) when writing feedback, matching the distinction PLATFORM-TRIAL.md already requires.
7. Report write-up: results headline first, attribution, and the styleguide bans.

### What stays as trusted code outside agent authority

These must not live in SKILL.md text, because the agent may skip, misread or be steered away from instructions (Skill-Use Compliance at most 0.638; SkillSecurer shows models running payloads they had flagged):

- The exact-oracle evaluator, fresh holdout data and their credentials.
- Protocol freezing: hashing and timestamping the hypothesis record before candidate generation, and refusing edits after the freeze.
- The attempt and provenance ledger writes, including parent IDs, costs, artifact hashes, and the skill versions in force for each attempt.
- The Lean kernel check and any proof-acceptance decision.
- Budget limits, sandbox setup, network policy and filesystem mounts.
- The skill library itself: mount it read-only for the agent and record each skill's content hash per attempt. The agentskills spec has no merged versioning, so Bigbrane has to pin skills by content hash on its own.
- Any skill-evolution acceptance gate. If Bigbrane later evolves skills WikiSkill-style, the gate must score on development data only, keep rejected proposals in the ledger, and never touch the holdout.

### RD-Agent-based loop versus an LLxprt-driven loop

In the pinned RD-Agent loop, the controller decides which stage runs (proposal, coding, running, feedback, record). Skills there work best as stage-specific instruction files that the controller loads directly into each stage's prompt. This sidesteps the triggering failure Skill-Use measured, since preloading raised Trigger and SU, and it keeps the choice of skill deterministic and recordable in the ledger. RD-Agent does not need a skill-discovery mechanism for this; a small adapter that reads the pinned SKILL.md files, inserts the body for the stage, and logs the hash is enough. The evaluator and freeze tool stay outside RD-Agent's writable workspace, as PLATFORM-SELECTION.md already requires.

In an LLxprt or other agent-driven loop, the native progressive-disclosure mechanism decides when a skill is read. That brings the measured failure modes into play: skills never opened, over-invocation on out-of-scope tasks, and losses as the library grows from one to ten skills. Mitigations: keep the Bigbrane skill set small and well separated, name the skill to use in the task prompt for each stage, and expose the evaluator, freeze and ledger only as tools whose outputs the agent cannot write. Third-party skills should not be installed into this environment; the security papers show that registries contain exploitable and malicious skills.

The practical split: use skills for agent procedure in both loops, keep every acceptance decision in trusted code, and prefer controller-injected skills in the RD-Agent loop because that is where the evidence shows the fewest skill-use failures.

## Gaps

No in-window paper evaluates skills on research or mathematical-discovery loops with an exact oracle. No primary vendor announcement in the window was found. SE-GoS and SkillPivot were read at abstract level only. Several pre-window results listed under Background were read only through search excerpts.
