# Offline correction verification

The executed command used the sibling checkout and the main virtualenv. All temporary paths and outputs were inside the preparation directory.

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2 \
TMPDIR=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2/results-wt/heldout-acquisition/preparation/provider-syntax/helper-fix-v2/temp \
/Volumes/XS1000/acoliver/projects/bigbrane/.venv/bin/python \
results-wt/heldout-acquisition/preparation/provider-syntax/run_helper_fix.py \
> results-wt/heldout-acquisition/preparation/provider-syntax/helper-fix-v2/tests-v2.log 2>&1
```

The command exited 0. Results: 44 held-out tests, 77 provider-preflight tests, and 123 new focused tests passed. No skips or expected failures remain in this run. The original 68-pass/9-xfail run is retained in the original preparation logs and XML. The exact pytest argument list, interpreter, cwd, and temporary path are in `TEST-SUMMARY-v2.json`.

`REQUEST-PARITY-v2.json` records 72 valid combinations, 144 mock dispatches, and zero live dispatches. The retained old source is hash-checked before it is loaded with the unchanged held-out policy. Both old and new helpers use the same offline audit prohibition. Production Request, Transport, parsing, and provenance components are used with MockTransport and fake time.

No documentation fetch, live probe, study acquisition, acceptance rerun, or independent review was executed. Do not run the original finalizers to overwrite historical receipts. The v2 manifests are separate implementation records.
