# Execution commands

All module execution used sibling cwd, sibling PYTHONPATH, the main virtualenv interpreter, and `PYTHONDONTWRITEBYTECODE=1`. All stdout logs, pytest temp/cache and raw files are in this directory. No full suite or live pytest invocation occurred.

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/preparation/provider-syntax/preflight.py docs
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/preparation/provider-syntax/run_offline.py > results-wt/heldout-acquisition/preparation/provider-syntax/tests.log 2>&1
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/preparation/provider-syntax/preflight.py probe > results-wt/heldout-acquisition/preparation/provider-syntax/runtime.log 2>&1
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/preparation/provider-syntax/run_offline.py > results-wt/heldout-acquisition/preparation/provider-syntax/tests.log 2>&1
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python results-wt/heldout-acquisition/preparation/provider-syntax/finalize_preflight.py
```

The documentation and probe commands each ran once. Their STARTED markers forbid repetition. The probe ran as a managed background process and was monitored to completion, not restarted. The first offline run had 53 passed and six expected failures. The final run added both-policy empty-token checks, fourteen actual OA/S2 mock-Transport comparisons and a read-only lease test; it had 68 passed and nine expected failures. The final log/XML preserve that final run. No frozen source was modified between these runs.

Before the probe, a separate offline Python process imported the frozen harness to write REQUEST-SPEC.json and checked exact Q1 request equality in the focused tests. That process's PYTHONPATH additionally included the harness and preparation directories; its socket prohibition did not carry into the separate live process.

Only the offline test runner is intended to be repeated. Live execution is not a CI test. The frozen plan, raw bytes and exclusion handoff must remain pinned. No artifact was staged or committed.
