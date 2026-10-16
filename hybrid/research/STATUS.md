# hybrid/research status

Scope: the run ledger, the card and artifact store, the theory-card and critic schemas, and the
`inspect` command with its static report. The orchestrator, stage code and the `run` subcommand are
not built. Only `hybrid/research/` and `hybrid/tests/research/` were touched.

## What is built

- `ledger.py`: `Ledger(run_dir)` with an append-only, hash-chained `ledger.jsonl`. Methods:
  `append(type, actor, data)`, `verify()`, and `iter_events(type=None)`.
- `store.py`: `Store(run_dir, ledger)` with `put_card`, `get_card`, `versions`, `latest_version`,
  `next_version`, `list_cards`, `put_artifact`, `set_verdict`, `lineage`, and `file_problems`.
- `schemas/`: one JSON Schema file per record type, draft 2020-12, plus `load_schema(name)` and
  `validate(name, instance)` in `schemas/__init__.py`.
- `inspect.py` and `report/`: the text summary and the jinja2 templates for the static report.
- `__main__.py`: `python -m hybrid.research inspect <run-dir>`. A placeholder marks where `run` goes.

## Ledger

- Event fields: `seq`, `ts`, `type`, `actor`, `data`, `prev_hash`, `hash`. `seq` starts at 1.
  The genesis `prev_hash` is 64 zeros.
- `hash` is the SHA-256 of `prev_hash` followed by the sorted-key JSON of the event without `hash`.
  The JSON uses `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`, encoded as UTF-8.
  The function is `sorted_json` in `ledger.py`.
- `append` takes an exclusive `flock` on `ledger.lock`, reads the last line to get the next `seq` and
  `prev_hash`, writes one line, and calls `fsync`. `verify` and `iter_events` take a shared lock.
- Each append opens and closes the lock file. Code running in asyncio should call `append` through
  `asyncio.to_thread`, because `append` blocks on `fsync`.
- `verify()` returns `(True, None)` for an intact chain. Otherwise it returns `(False, seq)` for the
  first event that fails to parse, has the wrong keys, has the wrong `seq` or `prev_hash`, or has a
  wrong hash. An unterminated last line is reported as bad.

## Store

- Card files are `cards/<id>/v<N>.json`, written with `open(..., "x")` so an existing version is never
  overwritten. Each write appends a `card_version` event with the file's SHA-256.
- `put_card` validates the card against `theory_card.schema.json`. The card's `version` must equal the
  next version number, which the caller reads with `next_version`. The store does not stamp versions.
- `put_card` rejects a parent that does not exist, a card that lists itself, and any `parent_ids` that
  would create a cycle.
- `put_artifact(kind, name, content, card_id=None)` accepts bytes or a path. `kind` and `name` must be
  plain file-name parts. Artifacts are immutable. It returns `{"path": <absolute Path>, "sha256": ...}`.
- `set_verdict` validates a verdict record, writes a new card version whose `status.stage` and
  `status.verdicts[stage]` are updated, then appends a `verdict` event. Verdicts cannot move to an
  earlier stage. A second verdict at the same stage is allowed as a revision.
- `by_calls` entries must match `call-<12 hex>` or `job-<12 hex>`. The store does not check that the
  ids appear in the ledger. The report marks unknown call ids.
- Event paths are relative to the run directory. `put_card` returns an absolute `Path`.
- `file_problems()` checks both directions. Every `card_version` or `artifact` event must point at a
  file whose SHA-256 matches. Every file under `cards/` or `artifacts/` must have such an event.

## Schemas

| File | Required fields and key rules |
| --- | --- |
| `theory_card` | `id`, `version`, `parent_ids`, `created_by_call`, `lens`, `title`, `summary`, `mechanism`, `claims`, `known_relatives`, `falsifiers`, `toy_example`, `references`, `status`. Every state object needs `name`, `type`, `shape`, `meaning`. `update` and `readout` are arrays of equations that each need `latex` and `plain`. `complexity` needs `time_per_token`, `memory`, `total_sequence` and `constants_notes`, each in `O(...)`, `Theta(...)` or `Omega(...)` form. `falsifiers` needs at least 2. `toy_example` needs `python` and `expected`. `known_relatives` needs at least 1. `status.stage` and verdict values are enums. Optional: `core_idea` (one sentence), `source_transplant` (`object`, `source_ids` with at least 1, `change`), `extensions` (items with `name`, `description`, optional `equations`); cards written before these existed stay valid, and the explorer's draft schema requires `core_idea` and `source_transplant`. Unknown fields are rejected. |
| `verdict` | `card_id`, `card_version`, `stage`, `verdict` (enum), `reasons` (at least 1), `by_calls` (at least 1). |
| `literature_receipt` | `query`, `source` (arxiv, openalex or semantic_scholar), `ts`, `result_ids`, optional `card_id`. No other fields. |
| `bench_result` | Loose. Required: `per_task` (map of metric maps), `baselines`, `scaling_slopes` (map of numbers). |
| `salad_critic` | `ok`, `fatal`, `fixable`, `notes`. Each finding has `kind`, `where`, `problem`. Fatal kinds: `dimension_error`, `contradictory_definition`, `unfalsifiable_central_claim`. Fixable kinds: `missing_definition`, `notation_clash`, `underspecified_step`, `vague_secondary_claim`, `arithmetic_slip`; at most 8. `ok: true` requires both lists empty. `ok: false` requires at least one finding. |
| `reduction_critic` | `reduces_to` (the 12-value enum), `mapping`, `confidence` (0 to 1), `difference_if_any`, `radical_enough`, `why`, `textbook_transplant` (boolean), `textbook_why`. A claimed reduction requires a non-empty mapping. |
| `prior_art` | `label` (enum), `evidence` (items with `source_id`, `title`, `url`, `relation`), `notes`. Evidence is required unless the label is `novel`. |
| `feasibility_judge` | `verdict` (feasible, infeasible, undetermined), `obligations` (at least 1, each with `name`, `status`, `argument`), `fatal_issues`, `confidence`. A feasible verdict requires no fatal issues. |
| `disprover` | `attacks` (at least 1, each with `target_claim`, `counterexample`, `status`, optional `check_code`). |
| `analyst` | `claims` (at least 1, each with `claim`, `status` in supported, refuted, untested, and `evidence`). |

Tests check that the verdict enum is identical in `theory_card` and `verdict`.

## Inspect and report

- `python -m hybrid.research inspect <run-dir>` verifies the ledger first.
  - Exit 2 when there is no `ledger.jsonl`.
  - A broken ledger prints `BROKEN at seq N`, exits 1, and builds no report. The report is not built
    from a chain that failed verification.
  - File problems are printed, the report is still built, and the exit code is 1.
- The text summary covers run times, the ledger status, file status, cards per stage reached, verdict
  counts, calls and tokens per model with mean latency, and totals.
- Wall time is the span from the first to the last ledger event.
- The report is written to `<run-dir>/report/`:
  - `index.html`: overview.
  - `cards.html`: card table with the stage reached, verdicts, and the reasons for the stage reached.
  - `lineage.html`: nested lists built from `Store.lineage()`.
  - `models.html`: model usage.
  - `dossiers/<card-id>.html`: versions with a unified diff from the previous version, verdicts with
    links to each call transcript, literature receipts, artifacts, and bench results.
  - Every page footer shows the ledger status line.
- All links are relative, so the run folder can be zipped and opened from disk. The pages contain no
  scripts and no external resources.

## Contract assumptions for the other packages

These fields are read by the report. The report raises an error when a required key is missing, which
surfaces the mismatch at the first inspect run.

- `llm_call` event data must include `call_id`, `role`, `backend`, `model`, `usage` (with `input`,
  `output`, `cached`, each an integer or null), `latency_s`, and `transcript_path`. `error` is optional.
  Transcript links are resolved this way: an absolute path is used as is, and a relative path is
  taken relative to the run directory. A path outside the run directory, or a missing file, is listed
  as a warning and is not linked.
- `literature_receipt` events must match `literature_receipt.schema.json` exactly. The report
  validates each receipt.
- Bench summaries are written as `put_artifact("bench_result", name, bytes, card_id=...)`. The
  dossier parses the file with `bench_result.schema.json` only when its bytes match the recorded
  SHA-256. Otherwise it shows a warning.
- Executor receipts are artifacts with a `card_id`. A `by_calls` entry that starts with `job-` is
  shown as an executor job with no link.

## Limitations

- Tail truncation is not detected. Deleting the last events leaves a valid chain. A remedy needs the
  head hash recorded outside the run folder or in a later checkpoint event. Not built.
- A crash between writing a card or artifact file and appending its event leaves a file the ledger
  does not record. `inspect` reports it. Nothing reconciles it automatically.
- `append` does not re-verify the existing chain. `verify` is a separate full pass.
- The critic schemas check shape and internal consistency only. They cannot check whether a mapping
  or a proof is correct.
- The toy-example test runs the fixture's Python on the host with `sys.executable`, which is the venv
  interpreter under `.venv-hybrid`. Executor-based checks are not covered by these tests.
- The concurrency test confirms the lock behaves under two processes and 300 asyncio threads on one
  run. I did not run a mutation check that removes the lock to confirm the test fails without it.
- The example card at `hybrid/tests/research/fixtures/example_card.json` is a labeled toy fixture. It
  is not a research result, and its `references` list is empty on purpose.

## Test results

Command: `.venv-hybrid/bin/python -m pytest hybrid/tests/research -q -p no:cacheprovider`

```
71 passed in 0.66s
```

Test files: `test_ledger.py` (chain, tamper detection, partial writes, two processes, 300 asyncio
tasks), `test_schemas.py` (accept and reject cases, critic rules, toy-example execution),
`test_store.py` (versions, verdicts, artifacts, lineage, file audit), and `test_inspect.py` (report
build on a synthetic run, link resolution, no scripts, tamper and missing-file exit codes, CLI).
