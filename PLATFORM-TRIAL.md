# Start Bigbrane with RD-Agent's research core; isolated execution is still a prerequisite

**Choose RD-Agent's core interfaces and `LoopBase` for Bigbrane's research loop.** Both pinned platforms installed and ran a shared development task on this Mac, including failed candidates and recovery in a new process. RD-Agent's scenario, experiment, feedback and ancestry interfaces worked without importing Qlib or its finance application. DeerFlow's agent factory, tool handling and SQLite persistence also worked, but the scientific proposal, status and lineage policy remained custom application code.

The trial supports retaining the earlier RD-Agent selection. It does **not** rank autonomous discovery quality: proposals and model responses were scripted, and only trusted fixture code executed. The full RD-Agent application still advertises Linux-only support. Successful native execution of its selected core does not extend that support claim.

**Next step:** make a Linux isolation backend available, then connect these RD-Agent interfaces to a bounded development-only Bigbrane executor and a separately controlled evaluator. The existing Podman VM is stopped or unreachable. An operator can start it with `podman machine start podman-machine-default`, then check `podman info`; neither action was performed here. Test a small pinned arm64 CPU execution image with no network, restricted mounts and explicit resource limits before allowing generated code. Do not mount confirmation data or evaluator credentials. Use DeerFlow instead if a native interactive workbench becomes the immediate objective, or if the RD-Agent Linux/executor integration fails while a matched DeerFlow integration passes.

## What actually ran

Run identifier: `20261008T003451`, begun October 8, 2026 UTC, October 7 local date. Source and environments are under [`platform-trials/20261008T003451/`](platform-trials/20261008T003451/). Raw evidence is under [`results/platform-trial-20261008T003451/`](results/platform-trial-20261008T003451/), abbreviated `E/` below.

| Check | RD-Agent | DeerFlow |
| --- | --- | --- |
| Pinned commit | `484776c211e4fbbeef03e0ec00d6bbee7362a4f4` | `a4427625cfc0759021736cca51426bfc2759d6c8` |
| Installation | Unchanged editable package, declared dependencies, pytest | Unchanged editable harness and matching extension API, declared dependencies, pytest/pytest-asyncio |
| Package inventory | 281; `uv pip check` passed | 198; `uv pip check` passed |
| Measured environment disk use | 1.1 GB | 902 MB |
| Actual runtime | `Scenario`, `ExpGen`, `Developer`, `Experiment2Feedback`, `HypothesisFeedback`, `Trace`, `LoopBase` | `create_deerflow_agent`, feature-assembled middleware, LangChain tool, LangGraph graph, `SqliteSaver` |
| Scientific negatives | Retained as acceptable runs with negative feedback | Retained as successful tool executions with negative scientific status in the result |
| Invalid candidate | `ValueError` routed by native loop to feedback and record; unacceptable feedback | Native tool error middleware produced an error `ToolMessage`; invalid record retained |
| Recovery | Signed snapshot loaded in a new process; unfinished running step reran | SQLite checkpoint opened in a new process; interrupted tool resumed with `Command` |
| Ancestry | Native `Trace` DAG preserved `[[], [0], [0], [1]]` | Parent IDs preserved in custom records and persisted tool results; no native scientific DAG added |
| Custom trial adapter | 159 lines | 110 lines |

Disk values are rounded `du -sh` outputs, not download sizes or additive storage totals. Both environments share a local uv cache measured at 1.9 GB. The package lists include the test tools. Neither a frontend nor a GPU execution image was installed. Exact resolved versions and local source paths are in `E/rd-packages.log` and `E/deer-packages.log`. The RD install resolved in 5.57 seconds and prepared packages in 10.96 seconds; Deer resolved in 3.35 seconds and prepared in 7.02 seconds. These are installer-reported observations, not effort estimates or cold-install benchmarks. Cache reuse and different dependency sets prevent a speed ranking.

The host is Darwin arm64, macOS 26.4. System Python is 3.14.6; both trial environments use the already available Python 3.12.12. Docker's client exists but cannot reach its daemon. Podman lists an existing AppleHV VM but cannot reach its socket. Apple Container is absent. No daemon, VM, OS package or container image was installed or started. SSH authenticated as `acoliver` through the existing `github-acoliver` alias. No account switch, global configuration change, GitHub mutation or push occurred. See `E/environment.log` and `E/clones.log`.

## Shared scientific task and its limits

The trusted fixture creates fresh, seeded two-state HMM data: eight independent length-16 training sequences and sixteen length-16 development sequences. No existing Bigbrane data arrays, frozen candidates or confirmatory results are read. It fits a two-state HMM for eight EM iterations and a scalar tanh recurrence with fixed recurrent weights and a learned logistic output. Prediction is causal and state resets at sequence boundaries.

The candidate record contains an attempt ID, parents, claim, equations, numeric prediction arrays, a falsifier, metrics, failures, scientific status and data/prediction hashes. The equations are:

```text
HMM: p_t = q_t B
     q_(t+1) = (q_t * B[:, x_t]) A / p_t[x_t]
RNN: h_(t+1) = tanh(0.8 h_t + 0.7 (2 x_t - 1))
     p_t(1) = sigmoid(w h_t + b)
Metric: mean_t sum_v oracle_t(v) log(oracle_t(v) / learner_t(v))
Claim falsifier: KL_RNN - KL_HMM < declared margin
```

Both platforms process the same four scripted attempts: `base`, a stronger-margin child `strong`, an invalid child `invalid`, and an interrupted child `resumed`. Parent IDs are respectively `[]`, `[base]`, `[base]`, `[strong]`. The valid attempts reuse the same development design; this intentionally tests retention and execution plumbing rather than adaptive scientific search.

All valid runs produced identical metrics in both environments:

| Metric | Value |
| --- | ---: |
| Training tokens | 128 |
| Development targets | 256 |
| HMM conditional KL | 0.12230803167784307 |
| Restricted RNN conditional KL | 0.005312276268561993 |
| RNN KL minus HMM KL | -0.11699575540928107 |

The `base` and `resumed` claims require a margin of -0.05; `strong` requires 1.0. All are scientific negatives. The invalid attempt has null metrics and a recorded failure. The HMM's poor result is retained, with no attempt to tune the fixture into a favorable outcome. These small learners and budgets differ from Experiment 1, so these numbers neither reproduce nor revise `RESULT.md`.

The shared fixture is 165 lines; its four contract tests were written before the implementation. They cover repeatable development output, numerical integrity, scientific-negative versus invalid classification, and idempotent artifact writing. This is trusted in-process execution without a sandbox. It establishes neither generated-code safety nor the correctness of Bigbrane's full scientific protocol.

## Framework behavior versus custom integration

### RD-Agent

The custom adapter subclasses the actual installed interfaces and invokes the native five-stage loop: proposal, coding, running, feedback, record. The proposal is scripted; the developer marks the trusted fixture ready and does not generate code. Custom analysis maps numeric results to `HypothesisFeedback`, explicitly separating acceptability from scientific support. The native trace method stores ancestry and retains rejected hypotheses.

A controlled interruption raises before the fourth experiment effect. The latest successful checkpoint is `3/1_coding`; there is no completed running snapshot. A second process loads that signed checkpoint with checkout into a new path, reruns the pending step and finishes the record. JSON persistence uses custom attempt-ID idempotency. This does not prove exactly-once effects after an interruption occurring after execution.

Native `logger.log_object` scientific records were reread through `FileStorage` and compared with saved prediction artifacts. The selected imports loaded 531 Python modules, with no `qlib`, `kaggle` or `rdagent.app` module loaded. The package installation remains broad, including unrelated application dependencies; import separation does not make it a small distribution. See `E/import-scope.log`, `E/rd-trial-provenance.log` and `E/rd-verified/native-log-export.json`.

### DeerFlow

The actual `create_deerflow_agent` factory assembled middleware around one custom trusted tool. `RuntimeFeatures(sandbox=False)` deliberately disables sandbox setup because no backend is usable. Other defaults were retained; this was not a full middleware takeover. A `BaseChatModel` extension supplies scripted tool calls and final messages, replacing only model inference. No private framework monkeypatch was used in either trial adapter.

The graph executed each tool, retained the invalid tool's exception as an error message and persisted full state in SQLite. The fourth tool called LangGraph `interrupt` before its experiment effect. A second process rebuilt the graph against the same database and resumed the pending call. A third read-only graph opening verified all four stored threads against the artifact files, with zero model invocations.

Tool execution metadata reported success for valid scientific negatives and error for the invalid candidate. That distinction must remain in Bigbrane's schema. The observed metadata was `deerflow_tool_meta`; this trial does not claim receipt-hash attestation. Parent IDs and scientific decisions came from the custom tool record. The selected factory/runtime imports loaded 2,445 Python modules and no `app.*` gateway modules. See `E/deer-trial-verified.log`, `E/deer-verified/initial-messages.json`, `E/deer-verified/resumed-messages.json` and `E/verification.log`.

The successful Deer graph used eight scripted model invocations: seven before the interruption and one after recovery. RD used scripted proposal/feedback extensions without a chat transport. Both made **zero live model calls**. No endpoint or provider credential was present in the environment; secret files were not searched for a replacement. This is not a matched model-quality comparison.

## Tests, errors and evidence checks

The fixture tests passed in both environments: **4 RD + 4 Deer**. RD's unchanged upstream signed-serialization subset passed **9 tests**. Deer's unchanged upstream SQLite batch-repository and subagent-runtime subset passed **25 tests**. The batch repository tests exercise real SQLite persistence, leasing and recovery paths. The runtime tests include mocked batch-service lifecycle collaborators, and upstream `conftest.py` mocks the executor. These tests do not establish live subagent execution or a complete durable batch worker.

Two initial adapter mistakes are retained in the raw logs. RD's result assertion expected a supported ancestor even though the actual numeric outcome rejected that claim. The assertion was corrected to require no supported ancestor and to check parent ancestry. Deer's first graph budget allowed only twelve graph steps, which was insufficient for the assembled middleware chain; the adapter now permits sixty-four graph steps. The successful transcript contains one experiment tool call per candidate and eight scripted model invocations, rather than an uncontrolled retry loop. No upstream implementation, dependency constraint or upstream test assertion was changed.

Verification recomputed KL from saved predictions, checked normalized distributions and prediction hashes, matched fresh fixture output to both platforms, compared native RD records and Deer tool transcripts to artifact files, and checked exact equality across platforms. Each completed platform has exactly four attempt files, including the loser and invalid attempt. `E/verification.log` records the checks. `E/evidence-inventory.json` hashes stable result artifacts and logs; the actively written verification/import-scope logs are excluded from that inventory.

No main Bigbrane source, protocol document, existing virtual environment or old result artifact was edited. The requested new report is the only substantive output outside the trial trees. **Scope deviation:** the initial fixture pytest commands used pytest's existing project-default cache before later test commands redirected caches into the evidence directory. This can update `.pytest_cache` metadata; that cache was not deleted or restored over potentially concurrent user activity.

## Confidence and selection consequences

| Evidence level | Established here |
| --- | --- |
| Source assessment | Earlier pinned-source findings remain the architectural basis; default finance/competition and evaluator limitations were not removed |
| Actual installation/runtime | Both declared installs passed; selected interfaces executed; failure, native persistence and new-process recovery passed |
| Live LLM capability | Untested: no generation, tool-selection quality, hypothesis revision quality or provider compatibility result |
| Full scientific experiment | Untested: no generated implementation, Linux executor, isolation validation, independent confirmation or Bigbrane rerun |

RD-Agent remains the better starting choice for the stated full scientific-search objective because its usable core already represents the experiment loop and ancestry. Deer's smaller installed environment and shorter tool adapter do not remove the need to build scientific search policy. The absence of an execution backend currently constrains both.

Evidence that would change the choice is concrete: the RD core/executor adapter fails on a usable Linux arm64 setup while Deer passes the same protected development task, or matched live-model trials show Deer producing more valid falsifiable experiments with comparable total cost and retained failures. Neither outcome was measured here. Keep independent evaluation and a frozen confirmation lane outside either agent's writable authority.

## Command and evidence appendix

Commands below run from `/Volumes/XS1000/acoliver/projects/bigbrane`. `T` and `E` abbreviate the paths used by the logged commands. The raw logs contain expanded invocation paths and actual output/errors.

```sh
T="$PWD/platform-trials/20261008T003451"
E="$PWD/results/platform-trial-20261008T003451"

ssh -T -o BatchMode=yes -o ConnectTimeout=10 git@github-acoliver
# Existing authorized alias resolves to github.com. Git operations used SSH only.
git -c core.sshCommand='ssh -o HostName=github.com -o BatchMode=yes -o ConnectTimeout=15' clone --filter=blob:none --no-checkout git@github-acoliver:microsoft/RD-Agent.git "$T/rdagent"
git -C "$T/rdagent" checkout --detach 484776c211e4fbbeef03e0ec00d6bbee7362a4f4
git -c core.sshCommand='ssh -o HostName=github.com -o BatchMode=yes -o ConnectTimeout=15' clone --filter=blob:none --no-checkout git@github-acoliver:bytedance/deer-flow.git "$T/deerflow"
git -C "$T/deerflow" checkout --detach a4427625cfc0759021736cca51426bfc2759d6c8

export HOME="$T/home" UV_CACHE_DIR="$T/cache" XDG_CACHE_HOME="$T/cache"
export PYTHONDONTWRITEBYTECODE=1 LOG_TRACE_PATH="$E/rd-verified-log"
uv venv --python /Users/acoliver/.local/bin/python3.12 "$T/rd-venv"
uv pip install --python "$T/rd-venv/bin/python" -e "$T/rdagent" pytest
uv venv --python /Users/acoliver/.local/bin/python3.12 "$T/deer-venv"
uv pip install --python "$T/deer-venv/bin/python" -e "$T/deerflow/backend/packages/extension-api" -e "$T/deerflow/backend/packages/harness" pytest pytest-asyncio

# Use fresh output directories for a replay. Existing evidence is retained.
"$T/rd-venv/bin/python" "$T/rd_trial.py" initial "$E/rd-verified"
"$T/rd-venv/bin/python" "$T/rd_trial.py" resume "$E/rd-verified"
"$T/deer-venv/bin/python" "$T/deer_trial.py" initial "$E/deer-verified"
"$T/deer-venv/bin/python" "$T/deer_trial.py" resume "$E/deer-verified"
"$T/deer-venv/bin/python" "$T/check_deer_persistence.py" "$E/deer-verified"
"$T/rd-venv/bin/python" "$T/verify_evidence.py" "$E"
```

| Evidence | Contents |
| --- | --- |
| `environment.log`, `clones.log` | OS/Python/runtime availability, SSH identity, exact clone/checkouts and hashes |
| `install-rd.log`, `install-deer.log` | Exact installation commands, resolver/build/install results and dependency lists |
| `imports.log`, `import-scope.log` | Real interface import smoke and observed loaded-module boundaries |
| `rd-trial.log`, `deer-trial.log` | Initial trial outcomes and adapter assertion/graph-budget errors |
| `rd-trial-provenance.log`, `deer-trial-verified.log` | Successful native framework execution and process recovery |
| `rd-verified/`, `deer-verified/` | Final attempt records, summaries, native RD exports/checkpoints and Deer SQLite/transcripts |
| `upstream-rd.log`, `upstream-deer.log` | Unchanged upstream subset test commands and results |
| `rd-packages.log`, `deer-packages.log`, `footprint.log` | Resolved versions, compatibility checks, pinned-source diff checks and code/disk measurements |
| `verification.log`, `evidence-inventory.json` | Prediction/metric/provenance comparison, persisted-state checks and stable artifact hashes |

The existing source assessment is [PLATFORM-CORE-ASSESSMENT.md](PLATFORM-CORE-ASSESSMENT.md); the earlier selection is [PLATFORM-SELECTION.md](PLATFORM-SELECTION.md). Neither was edited. The relevant platform source interfaces are the pinned clones' `rdagent/core/{scenario,proposal,developer,experiment}.py`, `rdagent/utils/workflow/loop.py`, `backend/packages/harness/deerflow/agents/factory.py`, and `deerflow/subagents/runtime.py`. No review agent or OCR service was used.
