# hybrid/llm status

## Works

- `LLMClient.complete` (async) and `complete_sync`, `LLMResult`, `EventSink` protocol, `JsonlSink`.
- `hybrid/config/models.toml`: spec routing table, per-backend concurrency and timeout, per-run budgets,
  parsed with tomllib and validated (unknown keys, unknown backends, bad numbers fail at load).
- Backends: llxprt (subprocess, stdin prompt, per-call scratch cwd, tools off or web-only, verified from
  `stats.tools`), openai_compat (live-tested against LM Studio), anthropic_messages (mock transport only),
  gateway (openai_compat preset).
- Fallback on transport errors with `llm_fallback` events; no fallback on 4xx, tool-policy, schema or budget
  errors.
- JSON mode with fence-tolerant extraction, jsonschema validation, one retry with the error, both attempts
  logged.
- Transcript per call under `runs/<run>/llm/calls/` and one `llm_call` event with prompt and response sha256.
- Budgets (calls and tokens, resumed from transcripts) and per-backend semaphores.
- Input overhead per llxprt call cut from about 19.8k tokens to a few hundred (see README).

## Tests

- `hybrid/tests/llm`: unit tests with fake runner, fake backends and mock HTTP transports.
- Four live tests (haiku JSON, sonnet JSON, haiku web, LM Studio), all passing on 2026-10-09.

## Left

- Run `anthropic_messages` against a real key once one exists.
- Point `gateway` at the native gateway once it exists.
- Add tokens from tool-policy-violating calls to the budget.
- Explain the one odd long-prompt case described in the README (random-word filler, question at the end).
- Re-measure `explorer` and `reduction_critic` latency with real prompt sizes; the figures so far use tiny prompts.
