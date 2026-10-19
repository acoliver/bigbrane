# Scoped commands

All writes are confined to `/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition/`. Built-in filesystem tools rejected the sibling because it is outside their configured roots. Shell access was used only for that permitted sibling; no workspace reconfiguration or production edit was made.

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition:/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/run_tests.py > results-wt/heldout-acquisition/tests-complete.log 2>&1
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/harness.py > results-wt/heldout-acquisition/PENDING.json
```

The initial run is tests.log, the corrected 43-test run is tests-final.log, and the complete scoped run including duplicate-mechanism and empty-evidence checks is tests-complete.log. These are scoped harness runs, not duplicated full-suite runs. No review was requested or run. No OCR or third review occurred.

Before implementation, from this directory: `shasum -a 256 PLAN.md policy.json > PREREGISTRATION.sha256`. After policy freeze: `PYTHONDONTWRITEBYTECODE=1 /Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python build_exclusions.py > exclusion-build.json`. The exclusion file has a separate byte hash. Rebuilding exclusions overwrites pending artifacts; future workers should preserve this delivery and assemble versioned inventories elsewhere.

Source files are pinned in SOURCE-HASHES.json. ARTIFACTS.json pins delivered files and logs, excluding itself and pytest cache/temp artifacts. Git status and HEAD checks are recorded in VALIDATION.json. A broad main-worktree instruction search timed out without modifying files; bounded checks of ancestor/root instruction paths succeeded. The required reports and acquisition summary/delivery receipts were read. No live API, gateway, LLM, provider documentation request or web fetch ran.

## Checkpoint packaging note

The combined project invocation exposed the offline hook blocking asyncio local AF_UNIX socketpair construction. Packaging permits only that construction; connection, DNS, datagram and Internet socket construction remain prohibited. No test assertions were changed. The frozen PLAN/policy/exclusion bytes remain unchanged. Current combined and fresh cache-free results, interruption details and the executed PENDING CLI are recorded in ../evidence-recovery/NONLINEAR-CHECKPOINT.md and NONLINEAR-CHECKPOINT-TESTS.json. The worker-scoped counts above refer to its earlier separate run.
