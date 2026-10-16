# Five repositories pass the date rule; AI Scientist v2 does not

Verified today: **2026-10-07**. Six-calendar-month cutoff: **2026-04-07**, inclusive. The machine returned `2026-10-07T16:37:37-0300` and `2026-10-07T19:37:37Z`; `date -v-6m '+%Y-%m-%d'` returned `2026-04-07`. This screen uses the verified current date, rather than the September session date or the older comparison's cutoff.

**Carry DeerFlow, GPT Researcher, Local Deep Research, RD-Agent, and Open Deep Research into the date-eligible set. Exclude AI Scientist v2.** Open Deep Research qualifies through actual dependency changes, but it is archived and its latest application-source change predates the window. Recommend against choosing it as a maintained starting point on those grounds, separately from the user's date exclusion. The other four have verified implementation changes inside the window.

This document screens maintenance eligibility only. It does not choose the best system or perform the detailed architectural comparison. `APPROACH-COMPARISON.md` was read for context; its assertions were not used as proof. All remote evidence below was retrieved from public GitHub sources during this screen on 2026-10-07.

## Rule and evidence handling

The literal rule is repository changes within six calendar months. Documentation, dependency, release-record, and bot-authored changes can satisfy that rule. Implementation changes are reported separately, without silently replacing the rule with a stricter one. Here, implementation means a change to application behavior or executable research tooling. Package/runtime configuration and CI maintenance are identified separately.

Default-branch commit history supplies the comparable baseline. All six repositories currently report `main` as the default branch. Commit timestamps below are Git committer timestamps in UTC, not `pushed_at` or `updated_at`. Author and committer dates agree for all six branch-tip commits. A recent merge is checked through its actual changed files; a merge title alone is insufficient. Stars, issue activity, and repository metadata refreshes are not counted as implementation maintenance. Git timestamps do not independently establish the time a commit was pushed.

The date window is 2026-04-07 through the observation on 2026-10-07. No finding is near the boundary, so a local-midnight versus UTC-midnight interpretation would not alter eligibility. The absence finding for AI Scientist v2 additionally checks its listed public branches and release list. This screen does not audit every ref or unmerged contributor fork for the five already-qualified repositories.

## Eligibility table

Full commit identifiers, titles, and diffs follow this table. The linked abbreviated identifiers refer to the exact commits inspected.

| Repository | Default branch; archived | Latest actual default-branch commit, UTC | Latest inspected implementation change | Latest published GitHub release | Literal date result |
| --- | --- | --- | --- | --- | --- |
| [bytedance/deer-flow](https://github.com/bytedance/deer-flow) | `main`; false | **2026-10-07T13:16:46Z**, [a442762](https://github.com/bytedance/deer-flow/commit/a4427625cfc0759021736cca51426bfc2759d6c8): skill scanner and notification redaction fix | Same commit. Python code adds fine-grained GitHub PAT and Google API key detection and `.zsh` scanning, with tests. **Inside window.** | [v2.1.0](https://github.com/bytedance/deer-flow/releases/tag/v2.1.0), **2026-09-24T10:40:44Z** | **Eligible**, with current implementation changes. |
| [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | `main`; false | **2026-09-26T16:56:07Z**, [0957c30](https://github.com/assafelovic/gpt-researcher/commit/0957c301ed06c2a5857b834358c7227c739041d4): merge restoring documentation homepage layout | Latest inspected implementation landing: **2026-09-26T15:56:53Z**, [af42b2d](https://github.com/assafelovic/gpt-researcher/commit/af42b2d1f218b768d896fa68f382c17e80cd0cd3), integrating BM25 context selection into research and report chat. Underlying package-source commit [2906e63](https://github.com/assafelovic/gpt-researcher/commit/2906e63275788d1252d59194d6e06dc4aace116e), **2026-09-26T13:47:15Z**. **Inside window.** | [v3.7.0](https://github.com/assafelovic/gpt-researcher/releases/tag/v3.7.0), **2026-09-26T17:46:25Z**; release identifies Python package **0.16.0** | **Eligible**. Tip is documentation-site work, but implementation maintenance is independently verified. |
| [LearningCircuit/local-deep-research](https://github.com/LearningCircuit/local-deep-research) | `main`; false | **2026-10-07T06:30:58Z**, [2940c8c](https://github.com/LearningCircuit/local-deep-research/commit/2940c8cd8445fd676b34f0f9291cd3118a5a974d): saved report preferences and error-disclosure fixes | Same commit. Report generation, API metadata sanitization, report assembly, and research-worker changes, with tests. **Inside window.** | [v1.10.7](https://github.com/LearningCircuit/local-deep-research/releases/tag/v1.10.7), **2026-08-28T21:43:07Z**, security hardening | **Eligible**, with current implementation changes. |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | `main`; false | **2026-09-23T11:05:10Z**, [484776c](https://github.com/microsoft/RD-Agent/commit/484776c211e4fbbeef03e0ec00d6bbee7362a4f4): release changelog only | **2026-09-23T10:37:14Z**, [cb8e4b8](https://github.com/microsoft/RD-Agent/commit/cb8e4b81115b4939991d5dac7b6b4b00b65f2ff3): missing imports in RL UI session loading and data-science debugging, with regression tests. **Inside window.** | [v1.0.0](https://github.com/microsoft/RD-Agent/releases/tag/v1.0.0), **2026-09-23T11:05:33Z** | **Eligible**. Tip is a release record, but implementation maintenance is independently verified. |
| [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) | `main`; false | **2025-12-19T07:46:31Z**, [96bd516](https://github.com/SakanaAI/AI-Scientist-v2/commit/96bd51617cfdbb494a9fc283af00fe090edfae48): merge changing license and responsible-use documentation | **2025-10-24T01:19:03Z**, [defddb8](https://github.com/SakanaAI/AI-Scientist-v2/commit/defddb8174905aac3bf4f7de7650e4cbf2ac353c): implementation revert affecting writeup and launcher code. **Outside window; no qualifying implementation change found.** | Public GitHub release list is empty. | **Excluded**: no default-branch commits since cutoff; both listed public branch heads predate cutoff; no GitHub releases. Not archived, but fails the date rule. |
| [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) | `main`; **true**, archived **2026-08-21** | **2026-08-10T18:13:29Z**, [1b7d2e8](https://github.com/langchain-ai/open_deep_research/commit/1b7d2e80db9faa586165c60e09096dbbfd483a64): Dependabot updates `h2` in `uv.lock` | **2025-08-18T03:32:04Z**, [7bbed5e](https://github.com/langchain-ai/open_deep_research/commit/7bbed5e4c5bbc854dcd6a43e5e4d754a739e83b7): fixes `datetime.now()` in application utilities. **Outside window; no application-source change in window.** | Public GitHub release list is empty. | **Eligible under literal rule**, through real bot dependency maintenance. **Recommend against as a maintained starting point**, separately: archived, with older application code. |

## Commit and diff evidence

### DeerFlow

Latest actual commit and latest inspected implementation change:

- SHA: `a4427625cfc0759021736cca51426bfc2759d6c8`.
- UTC timestamp: `2026-10-07T13:16:46Z`.
- Message: `fix(skillscan): detect fine-grained PATs and Google API keys, scan .zsh (#6423)`.
- Actor reported by GitHub: `yu1128-L`, not a bot account.
- [Commit API with files and patches](https://api.github.com/repos/bytedance/deer-flow/commits/a4427625cfc0759021736cca51426bfc2759d6c8).

The patch in `backend/app/scheduler/notification_text.py`, hunk starting at line 48, adds regexes for `github_pat_` and `AIza` token families. `backend/packages/harness/deerflow/skills/skillscan/orchestrator.py`, hunks starting at lines 130 and 1033, adds those token families and `.zsh` to shell-path recognition. `backend/tests/test_notification_text.py` and `backend/tests/test_skillscan_native.py` add regression coverage. This is application security behavior, rather than documentation or dependency-only maintenance.

The [release list](https://api.github.com/repos/bytedance/deer-flow/releases?per_page=3) reports v2.1.0 as its newest published release. The release description covers execution verification, durable batch delegation, memory backends, and sandbox providers. Those are release claims; this screen does not evaluate their quality.

### GPT Researcher

Latest actual default-branch commit:

- SHA: `0957c301ed06c2a5857b834358c7227c739041d4`.
- UTC timestamp: `2026-09-26T16:56:07Z`.
- Message: `Merge pull request #2173 from assafelovic/docs/homepage-restore-hero`.
- Actor: `assafelovic`.
- [Commit API](https://api.github.com/repos/assafelovic/gpt-researcher/commits/0957c301ed06c2a5857b834358c7227c739041d4).

Its diff changes only `docs/src/pages/index.js` and `docs/src/pages/index.module.css`. It restores the documentation site's code card and two-column hero. These are executable website files, but do not establish maintenance of the research engine.

The latest inspected implementation landing is `af42b2d1f218b768d896fa68f382c17e80cd0cd3`, `2026-09-26T15:56:53Z`, message `Merge pull request #2162 from assafelovic/feat/lean-context-filter`. The [commit API](https://api.github.com/repos/assafelovic/gpt-researcher/commits/af42b2d1f218b768d896fa68f382c17e80cd0cd3) shows added `gpt_researcher/context/lexical.py` and `gpt_researcher/context/select.py`, changes to `gpt_researcher/skills/context_manager.py`, and replacement of the report-chat embedding path in `backend/chat/chat.py`. The selection module exposes local BM25 keyword ranking and shared context selection. Tests include `tests/test_lexical_context_filter.py` and `tests/chat/test_chat_agent_rag_path.py`.

The [package-path history](https://api.github.com/repos/assafelovic/gpt-researcher/commits?sha=main&path=gpt_researcher&per_page=1) reports the underlying source commit `2906e63275788d1252d59194d6e06dc4aace116e`, `2026-09-26T13:47:15Z`, message `feat(context): keyword fallback so no filter needs a key or embeddings`. Path-filtered Git history can omit a merge already represented by its ancestry. Both the underlying source commit and its later merge are reported to avoid confusing authorship with integration.

A later configuration change, `039024e24e16eba7420a659b9fe0d50090613af6`, `2026-09-26T16:10:11Z`, message `Raise the supported Python floor to 3.12 and release 0.16.0.`, updates Python runtime and packaging configuration. Its [diff](https://api.github.com/repos/assafelovic/gpt-researcher/commits/039024e24e16eba7420a659b9fe0d50090613af6) includes `backend/Dockerfile`, `backend/runtime.txt`, and `langgraph.json`. This is deployment maintenance, separately from the preceding research-engine feature.

The [v3.7.0 release API](https://api.github.com/repos/assafelovic/gpt-researcher/releases/tags/v3.7.0) gives `2026-09-26T17:46:25Z` and explicitly identifies the installable Python package as `0.16.0`. Those labels must not be conflated. It documents the context-filter change and Python 3.12 requirement.

### Local Deep Research

Latest actual commit and latest inspected implementation change:

- SHA: `2940c8cd8445fd676b34f0f9291cd3118a5a974d`.
- UTC timestamp: `2026-10-07T06:30:58Z`.
- Message: `fix(report): honor legacy preferences and hide raw errors (#7165)`.
- Actor: `LearningCircuit`, not a bot account.
- [Commit API](https://api.github.com/repos/LearningCircuit/local-deep-research/commits/2940c8cd8445fd676b34f0f9291cd3118a5a974d).

`src/local_deep_research/error_handling/report_generator.py`, hunk starting at line 108, replaces raw error-message output with an authored generic message. The patch adds `src/local_deep_research/error_handling/error_messages.py` and changes `security/data_sanitizer.py`, `web/routers/research.py`, `web/services/report_assembly_service.py`, and `web/services/research_service.py`. These alter saved reports, status metadata, and handling of failed synthesis. Regression tests accompany the runtime changes. The recent history also contains test-only commits and a Dependabot `filelock` bump, but eligibility does not rely on them.

The [release list](https://api.github.com/repos/LearningCircuit/local-deep-research/releases?per_page=3) identifies v1.10.7, `2026-08-28T21:43:07Z`, as the newest published release. Its description names notification-URL and news-subscription security hardening. A commit message referring to a future v2 release gate does not establish a published v2 release.

### RD-Agent

Latest actual default-branch commit:

- SHA: `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`.
- UTC timestamp: `2026-09-23T11:05:10Z`.
- Message: `chore(main): release 1.0.0 (#1286)`.
- Actor: `you-n-g`.
- [Commit API](https://api.github.com/repos/microsoft/RD-Agent/commits/484776c211e4fbbeef03e0ec00d6bbee7362a4f4).

The branch-tip diff only adds 33 lines to `CHANGELOG.md`. The preceding implementation commit is `cb8e4b81115b4939991d5dac7b6b4b00b65f2ff3`, `2026-09-23T10:37:14Z`, message `fix: import pickle and re where they are used (#1496)`, actor `David-Wu1119`. Its [commit API](https://api.github.com/repos/microsoft/RD-Agent/commits/cb8e4b81115b4939991d5dac7b6b4b00b65f2ff3) shows `import pickle` added to `rdagent/app/rl/ui/data_loader.py`, hunk starting at line 4, and `import re` added to `rdagent/scenarios/data_science/debug/data.py`, hunk starting at line 1. It adds regression tests in `test/rl/test_ui_data_loader.py` and `test/scenarios/data_science/test_debug_data.py`.

The [release list](https://api.github.com/repos/microsoft/RD-Agent/releases?per_page=3) reports v1.0.0 published at `2026-09-23T11:05:33Z`. The release documents required UI authentication and other breaking changes. A later repository push timestamp does not override the verified default-branch tip date.

### AI Scientist v2

Latest actual default-branch commit:

- SHA: `96bd51617cfdbb494a9fc283af00fe090edfae48`.
- UTC timestamp: `2025-12-19T07:46:31Z`.
- Message: `Merge pull request #78 from conglu1997/main`.
- Actor: `RobertTLange`.
- [Commit API](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/commits/96bd51617cfdbb494a9fc283af00fe090edfae48).

The diff changes `LICENSE` and adds a responsible-use section to `README.md`. Neither file is implementation code. Both changes still count as repository changes for the literal rule, but their dates are outside the window.

The newest inspected implementation landing is `defddb8174905aac3bf4f7de7650e4cbf2ac353c`, `2025-10-24T01:19:03Z`, message `Merge pull request #73 from SakanaAI/revert-68-main`. Its [diff](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/commits/defddb8174905aac3bf4f7de7650e4cbf2ac353c) changes `ai_scientist/perform_writeup.py` and `launch_scientist_bfts.py`, reverting implementation changes. A revert is an actual code change, even though it removes rather than adds functionality. The underlying revert commit is [6cdc32c84a2d0cec15f1ec1bf6eb26d0421cb462](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/commits/6cdc32c84a2d0cec15f1ec1bf6eb26d0421cb462), `2025-10-24T01:18:51Z`, message `Revert "ollama support implantation"`.

The [cutoff-filtered main history](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/commits?sha=main&since=2026-04-07T00%3A00%3A00Z&per_page=1) returns an empty array. The [public branch list](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/branches?per_page=100) lists only `main` and `revert-68-main`, with the old heads above. The [release list](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/releases?per_page=3) is empty. These establish exclusion on observed public repository history. No GitHub release records does not prove the absence of artifacts published elsewhere.

### Open Deep Research

Latest actual default-branch commit:

- SHA: `1b7d2e80db9faa586165c60e09096dbbfd483a64`.
- UTC timestamp: `2026-08-10T18:13:29Z`.
- Message: `Bump h2 from 4.3.0 to 4.4.1 in the uv group across 1 directory (#334)`.
- Actor: `dependabot[bot]`.
- [Commit API](https://api.github.com/repos/langchain-ai/open_deep_research/commits/1b7d2e80db9faa586165c60e09096dbbfd483a64).

The patch changes only `uv.lock`, six additions and six deletions. The hunk starting at line 1511 changes `h2` from `4.3.0` to `4.4.1` and updates artifact metadata. This is a genuine repository dependency change inside the window, sufficient for the literal rule.

The [complete cutoff-filtered history](https://api.github.com/repos/langchain-ai/open_deep_research/commits?sha=main&since=2026-04-07T00%3A00%3A00Z&per_page=100) returns 18 commits, all attributed to Dependabot, with dependency or GitHub Action bump messages. The [source-path history](https://api.github.com/repos/langchain-ai/open_deep_research/commits?sha=main&path=src&per_page=1) identifies `7bbed5e4c5bbc854dcd6a43e5e4d754a739e83b7`, `2025-08-18T03:32:04Z`, message `fix: AttributeError: type object 'datetime.datetime' has no attribute 'datetime' (#178)`, as the latest `src` change. Its [diff](https://api.github.com/repos/langchain-ai/open_deep_research/commits/7bbed5e4c5bbc854dcd6a43e5e4d754a739e83b7) changes `datetime.datetime.now()` to `datetime.now()` in `src/open_deep_research/utils.py`, hunk starting at line 875. No application-source change in the window was found. Bot-authored dependency updates can alter installed behavior, but are not evidence that the project's own research implementation is being developed.

The [repository API](https://api.github.com/repos/langchain-ai/open_deep_research) reports `archived=true`. The [repository page](https://github.com/langchain-ai/open_deep_research) explicitly says the owner archived it on **August 21, 2026**. The archive date comes from that banner, not from `updated_at`. The [release list](https://api.github.com/repos/langchain-ai/open_deep_research/releases?per_page=3) is empty.

## Metadata and history source index

These endpoints were read directly. Release and commit links in the evidence sections supplement this index.

| Project | Repository metadata | Recent main history |
| --- | --- | --- |
| DeerFlow | [API](https://api.github.com/repos/bytedance/deer-flow) | [API](https://api.github.com/repos/bytedance/deer-flow/commits?sha=main&per_page=8) |
| GPT Researcher | [API](https://api.github.com/repos/assafelovic/gpt-researcher) | [API](https://api.github.com/repos/assafelovic/gpt-researcher/commits?sha=main&per_page=60) |
| Local Deep Research | [API](https://api.github.com/repos/LearningCircuit/local-deep-research) | [API](https://api.github.com/repos/LearningCircuit/local-deep-research/commits?sha=main&per_page=8) |
| RD-Agent | [API](https://api.github.com/repos/microsoft/RD-Agent) | [API](https://api.github.com/repos/microsoft/RD-Agent/commits?sha=main&per_page=8) |
| AI Scientist v2 | [API](https://api.github.com/repos/SakanaAI/AI-Scientist-v2) | [API](https://api.github.com/repos/SakanaAI/AI-Scientist-v2/commits?sha=main&per_page=8) |
| Open Deep Research | [API](https://api.github.com/repos/langchain-ai/open_deep_research) | [API](https://api.github.com/repos/langchain-ai/open_deep_research/commits?sha=main&per_page=100) |

## Handoff and limits

There are **five verified date-eligible repositories, one verified exclusion, and no unresolved date-eligibility cases**. There is no docs-only qualifier among these six: GPT Researcher's tip is documentation work and RD-Agent's tip is a changelog, but earlier in-window implementation diffs establish more than documentation activity. Open Deep Research is the dependency-only qualifier and must retain that qualification in the later comparison.

For the maintained-starting-point assessment, prioritize detailed examination of DeerFlow, GPT Researcher, Local Deep Research, and RD-Agent. Keep Open Deep Research visible as literally eligible, with the separate recommendation against adoption as a maintained upstream. Remove AI Scientist v2 from that detailed candidate comparison under the user's date rule.

This is not a guarantee of future support, release quality, or safe dependencies. Only inspected files support the implementation classifications. The screen did not run any framework, evaluate contributors or benchmarks, inspect every open pull request, or select an overall winner. An initial large API summary exceeded the tool output limit and was replaced by compact queries; no conclusions rely on that inaccessible output. No repositories were cloned, no packages were installed, and no GitHub state or authentication settings were changed.
