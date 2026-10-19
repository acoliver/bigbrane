# Scoped checks pass; historical v1 validation rejects the corrected harness

The requested local scope ran once with unchanged assertions: 298 cases, 297 passed and one failed. There were no skips or expected failures. The failure is recorded in `local-receipts/tests.log` and XML; no guard or pin was relaxed.

| Local test group | Passed | Failed |
| --- | ---: | ---: |
| Original held-out harness | 44 | 0 |
| Provider preflight | 77 | 0 |
| Focused helper correction | 123 | 0 |
| Inventory v1 | 31 | 1 |
| Inventory v2 | 17 | 0 |
| Corpus synthetic validators | 5 | 0 |

The 244 request/helper cases all pass. The original preflight run had 68 passing cases and nine expected failures. Exactly two diagnostic xfail decorators were removed; their nine generated cases now pass. An AST comparison confirms all assertions are unchanged. The root 44-case source is byte-identical to HEAD before this checkpoint. No Phase 7 or full-suite assertion was changed.

The failing v1 test is `test_named_scopes_validate_but_broader_scope_stays_pending`. Its `validate_stage()` requires the historical harness digest `1546667435c4206f384ded06862da66cfcc2b8105bf0828423f1a2bcaf73b867` at the live harness path. The authorized implementation is `2c1d366af78edd96cb12048b86c4b5656a11f396a91ebb8b2a7094c1c00cc5f0`. The exact old source is retained separately. The frozen v1 stage and validator were not rewritten to accept a new hash. The prior 32/32 result is historical, before that authorized correction. It is not reported as the current result. V2 explicitly records both revisions.

## Cache-free scope

A fresh source/metadata-only copy passed 179 cases: 44 root, 123 focused helper, and 12 compact-package checks. Original checkout reads were prohibited by an additional audit hook, except the Python environments used for installed dependencies. Neither private source bodies nor raw caches were copied. The unchanged network audit hook prohibited connections, DNS and datagrams and permitted AF_UNIX construction for local socket pairs.

The two provider input-hash manifests contain absolute paths. Only their path keys were relocated in the isolated copy; every expected content hash remained unchanged. Their original public files were not edited. No test assertion was transformed. The remaining 77 preflight cases and the full v1/v2 source suites were not claimed as standalone cache-free checks.

The first isolated attempt passed 178 cases and failed the versioned-pin test because the runner replaced an overlapping path prefix twice. Its failure is retained separately. A fresh copy using path-relative relocation passed all 179 cases. This corrected a packaging-runner input path; it did not change the helper or its assertions.

To reproduce from the sibling checkout, select a fresh output name under the repository:

```sh
CHECKPOINT_ISOLATION_NAME=isolated-reproduce PYTHONDONTWRITEBYTECODE=1 \
  /Volumes/XS1000/acoliver/projects/bigbrane/.venv-hybrid/bin/python -B \
  results-wt/heldout-acquisition/preparation/checkpoint/run_cache_free.py
```

The runner refuses an existing destination. Logs, XML and temporary outputs remain under `tmp/preparation-checkpoint/`. The retained successful receipts are in `isolated-receipts/`. Source/metadata validation can also run directly with the sibling PYTHONPATH and `load_checkpoint.py`; rebuilding DELIVERY requires the unshipped original v1 registry. Never rerun the historical live probe or documentation stages to reproduce offline checks.

## Preservation and limits

All 57,604 v2 source hashes matched local originals. The original 274 replay inputs and 75 seed inputs matched their manifests. All 21 v1 stage artifacts, eight unchanged frozen base files and 160 private corpus snapshot artifacts matched their hashes. Local corpus validation rechecked the 11 source records and left VALIDATION bytes unchanged. Those full source checks require local originals and cannot be recreated from this public package alone.

Valid-request parity matched 72 Q0/Q1/source/text combinations through 144 mock dispatches, including encoded requests and request/association identities. There was no live dispatch. `CONTRACT-PRESERVATION.json` records the exact two added require calls, production-file preservation and unchanged test predicates. Historical manifests remain historical; implementation-v2 manifests explicitly pin the corrected helper.

The full suite was not rerun. Its latest retained result at 8cad9a1 remains 1,975 passed, two failed and three skipped. That result and its Phase 7 failures are separate from the scoped helper and cache-free results above. No review, integration, semantic-labeling run, arm comparison or acceptance rerun occurred.

## Staged whitespace diagnostic

The all-artifact `git diff --cached --check` reports trailing whitespace inside retained pytest failure logs/XML and generated historical unified diffs. These evidence files were preserved verbatim. The same check over maintained Python, Markdown, JSON, INI and SHA256 files passes. This diagnostic does not change any lint rule, test assertion or workflow enforcement.
