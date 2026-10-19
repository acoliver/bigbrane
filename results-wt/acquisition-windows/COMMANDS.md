# Offline reproduction and delivery limits

The delivery separates cache-free tests and count checks from the historical audit. It does not fetch missing evidence. Main's Python environment is reused; imports and cwd resolve to the sibling or exported package.

## Delivered, without raw inputs

```sh
cd /Volumes/XS1000/acoliver/projects/bigbrane-wt-r2
export BIGBRANE_LIVE=0 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/results-wt/evidence-recovery/composites-network-guard:$PWD"
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/acquisition-windows/package_receipts.py --check
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -m pytest \
  --confcutdir="$PWD" --basetemp="$PWD/tmp/acquisition-reproduce" \
  results-wt/acquisition-windows/test_extract_windows.py -q
```

The 18 synthetic tests require the included `extract_windows.py` and `pin_inputs.py`, plus the existing literature modules. They create and remove their own synthetic raw envelopes under this directory's `test-work`, not a system temporary directory. The inherited audit hook prohibits socket connections, datagrams and DNS in the Python process and its Python subprocesses.

`package_receipts.py --check` checks the delivered measurements and exact first-window populations. Its no-argument mode packages the existing full local outputs and fails if they are absent. It performs no provider query.

## Historical extraction, local originals required

The following commands are archival reproduction commands for this workstation. They require the original 274 saved inputs, 39 supplemental inputs including twelve main raw envelopes, `INPUTS-before.json`, and the full diagnostic metadata at the paths recorded during the investigation. Those files and the before/after histories are preserved locally, not duplicated in the delivery.

```sh
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/acquisition-windows/extract_windows.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/acquisition-windows/run_prior_tests.py
/Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python results-wt/acquisition-windows/verify.py
```

`verify.py` additionally requires the original scoped test logs and complete repeated outputs. It records the original 18 new and seven prior synthetic-tool results, not the delivery's combined JUnit results. `pin_inputs.py` ran before numerical extraction and refuses to overwrite its manifest. Do not delete that manifest to rerun it. `PLAN.sha256` retains the original plan and local input-manifest hashes; the latter identifies an intentionally local file.

The delivery's `DELIVERY-RECEIPTS.json` records hashes and byte sizes of the full local derivations. `SUMMARY.json` retains the original structured counts. `REPORT.md` states the measured populations and limits. Full `QUERY-CONTEXTS.json`, `HISTORICAL-WINDOWS.json`, `FIRST-WINDOWS.json`, `ORDERING.json`, `REGISTERED-TRACE.json`, before/after manifests, `ARTIFACTS.json`, `VALIDATION.json`, logs and raw provider bodies remain at their original local paths. Compact receipts preserve measured metadata, not the original provider bodies.

The delivery checkpoint under `../evidence-recovery/COMPOSITES-CHECKPOINT.md` records the separate once-only full research/literature/tool run and the cache-free selected run, their JUnit hashes, exact scope and ordinary Phase7 failures. No network acquisition, model call, acceptance rerun or additional review is part of delivery.

File tools could not address the sibling worktree in this session. Packaging used absolute paths through the shell and wrote nothing into main.
