# Commands and module selection

All diagnostic outputs are under `/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement`. These commands were launched from main. Preparation and live execution are single-use commands, not instructions to repeat the measurement.

Preparation, before any network:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/measure.py prepare > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/prepare.log 2>&1
```

The first offline test invocation used `python -m unittest -v test_measure` from main. Python put main ahead of PYTHONPATH for that invocation and could not import the sibling provenance module. This was a module-selection failure before test execution, not a source failure. Its output remains in tests.log. The corrected invocation runs the test script directly:

```sh
TMPDIR=/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/test-tmp PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/test_measure.py -v > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/tests-final.log 2>&1
```

The live command, launched once as a managed background job:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement nohup /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/measure.py run > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/live.log 2>&1 &
```

The runner validates pinned implementation/tool hashes, all 274 saved inputs, and request settings before constructing the live client. It then creates RUN-ONCE.json exclusively and refuses any repeat. It uses `load_api_keys()` with the existing BIGBRANE_OPENALEX_API_KEY/BIGBRANE_S2_API_KEY environment schema and private files. It does not print key objects or read the OS credential store.

The runner saves a hash-chained real Ledger through CardLiteratureSink after each source invocation, followed by compact public paper metadata in RESULTS.json. Raw successful API envelopes remain local in raw-cache. The existing process-shared pacer writes its ordinary coordination files in the sibling var/slots. No original run or production cache is opened for writing.

The first background launch returned completed immediately with an empty live.log. A subsequent process check found no child, RUN-ONCE.json, or runtime ledger, so it had not reached the runner's network boundary. The actual measurement used the managed job without a trailing shell ampersand:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/measure.py run > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/live-run.log 2>&1
```

This second launch created the single-use sentinel and emitted source records. No completed or failed source measurement was repeated. The shell-launch discrepancy is recorded for the coordinator; it caused no production change.

Offline follow-up, with no source calls:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/analyze.py > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/analysis-final.log 2>&1
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/measure.py finalize > /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/finalize.log 2>&1
```

The first analysis invocation is retained in analysis.log. Its broad hash assertion detected the other worker's seed.py/build_seed.py edits after runtime completion. The offline analyzer records those two hashes and timestamps separately, verifies that their mtimes follow RUN-END, and requires all other pinned implementation hashes to remain equal. It does not change the pinned plan, transport, source results, or tests. A separate stdin inspection also selected main's package; the successful direct-file invocation avoids that import-path issue.

Final artifact hashing uses Python hashlib over files under this directory, excluding bytecode and ARTIFACTS.json itself. ARTIFACTS.json includes raw-cache file hashes and hashes of the saved runtime ledger, plan, fixtures, commands, report, tests, and results. No file is staged or committed. git diff --check is run in both worktrees; other-worker changes are left untouched.

The final offline verification also refreshes the credential scan before hashing finished files:

```sh
PYTHONPATH=/Volumes/XS1000/acoliver/projects/bigbrane-wt-r2:/Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python /Volumes/XS1000/acoliver/projects/bigbrane/results-wt/transport-measurement/verify_final.py
```

analyze.py and verify_final.py are post-run, offline tools. They do not replace the prerun hashes for measure.py and test_measure.py.
