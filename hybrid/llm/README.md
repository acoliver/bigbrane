# hybrid/llm

Model access for the Bigbrane research system: one async `LLMClient.complete(role, system, user, ...)` that
routes a role to an ordered list of backend+model entries, enforces budgets and concurrency, validates JSON
output, and records every call as a transcript file plus a ledger event.

```python
from hybrid.llm import LLMClient, JsonlSink

client = LLMClient("hybrid/config/models.toml", "runs/run-20261010T000000Z", JsonlSink("runs/.../events.jsonl"),
                   routing=None)   # None: BIGBRANE_ROUTING, else the file's `routing`
result = await client.complete("salad_critic", system, user, json_schema=schema)   # or client.complete_sync(...)
result.parsed, result.text, result.usage, result.latency_s, result.transcript_path
```

The third argument is any object with `append(type, actor, data) -> dict` (`EventSink` in `types.py`). The
research ledger satisfies it; `JsonlSink` is the standalone version.

## Layout

| File | Purpose |
| --- | --- |
| `types.py`, `errors.py` | `LLMResult`, `EventSink`, exception types |
| `config.py` | `models.toml` loading with validation (tomllib): routing tables, role settings, budgets |
| `client.py` | routing, fallback, JSON retry, budgets, semaphores, transcripts, events |
| `jsonmode.py` | schema instruction, first-JSON-object extraction, jsonschema validation |
| `budget.py` | per-run call and token counters, resumed from existing transcripts |
| `gateway_check.py` | gateway bearer token lookup (env, then token file) and the pre-run check used by `hybrid.research run` |
| `backends/llxprt.py` | llxprt CLI subprocess backend with tool control |
| `backends/openai_compat.py` | `/chat/completions` (LM Studio, and the `gateway` preset) |
| `backends/anthropic_messages.py` | `/v1/messages` |
| `llxprt_prompts/` | the two minimal system prompts used by the llxprt backend |
| `measure.py` | `python -m hybrid.llm.measure opus haiku ...` re-measures profile latency and overhead |

## Behavior

- **Routing tables.** `models.toml` holds complete routing tables under `[routes.<name>]`; today `llxprt` and
  `gateway`. The top-level `routing` key names the active one (default `llxprt`). `BIGBRANE_ROUTING` overrides the
  file, and the `routing` argument of `load_config` and `LLMClient` (the `--routing` flag of
  `python -m hybrid.research run`) overrides both. The loader refuses a table whose role set differs from the
  others, an unknown table name, unknown keys at any level, and an entry key other than `backend`, `model` and
  `timeout_s`. The chosen table and where it came from are in `ModelsConfig.routing` and `routing_source`, and
  every transcript records `routing`.
- **Model ids.** `model` in a role entry is what the backend receives: an llxprt profile name for
  `backend = "llxprt"`, the gateway id (`anthropic/claude-haiku-5-5`, `chatgpt/gpt-6.1-sol`, ...) for
  `backend = "gateway"`, the server's model id (or `auto`) for `openai_compat`.
- **Role settings.** `[role_settings.<role>]` sets `effort` (`low`, `medium` or `high`) and `max_output_tokens`
  for every entry of that role in every table. An explicit `max_output_tokens` argument to `complete()` wins over
  the role's cap. Both values are recorded on every attempt in the transcript and the `llm_call` event. A role with
  an effort cannot route to an `anthropic_messages` backend (that backend has no effort control; the loader
  refuses it).
- **Fallback.** Within the active table the first entry is used. A
  transport failure (subprocess failure, empty reply, HTTP 5xx or 429, connection error, timeout) moves to the
  next entry and is recorded as an `llm_fallback` event. Anything else (HTTP 4xx, tool policy violation,
  schema failure after the retry, budget, config error) is raised without fallback. If every entry fails,
  `TransportError` lists each failure.
- **Gateway.** `kind = "gateway"` is the OpenAI-compatible backend with three differences. The bearer token is
  `api_key_env` (`BIGBRANE_GATEWAY_TOKEN`) when set, otherwise the contents of `~/.bigbrane-gateway/run-token`
  (`BIGBRANE_GATEWAY_HOME` moves it), read on every request so a restarted gateway is picked up. Every request
  carries `x-bigbrane-run-id: <run directory name>`, which the gateway copies into `usage.jsonl` as `run_id`, so
  gateway usage joins to the run's ledger. With neither token source the call fails with a `ConfigError` that
  names the `serve` command.
- **Effort and output caps on HTTP backends.** `openai_compat` and `gateway` send `max_tokens` and
  `reasoning_effort` when the role sets them. The gateway's ChatGPT plan route maps `reasoning_effort` to
  `reasoning.effort` and drops `max_tokens`. Its Anthropic route maps `reasoning_effort` to adaptive thinking plus
  `output_config.effort` on the adaptive-capable Claude ids (Opus 5.5, Sonnet 5.5 and Haiku 5.5 included) and drops it
  on other Claude models. It sends `max_tokens` unchanged, and thinking tokens count against it. Dropped fields are
  listed in the gateway's meter line. What was sent is in the attempt's `detail.sent`.
- **JSON mode.** The schema is appended to the user message with an instruction. The first JSON object in the
  reply (plain, in a ```json fence, or inside prose) is extracted and validated with jsonschema. On failure the
  call is repeated once with the previous reply and the validation error appended. Both attempts are in the
  transcript. A second failure raises `SchemaValidationError`.
- **Budgets.** `[budgets] max_calls` and `max_tokens` are per run. Every backend request counts as one call,
  including the schema retry and fallback attempts, and `BudgetExceeded` is raised before a request that would
  exceed `max_calls` (or when `max_tokens` is already used up). Counters are rebuilt from
  `runs/<run>/llm/calls/*.json` when a client starts, so a restarted run keeps its totals. Two processes
  writing to the same run directory would not see each other's calls until they restart.
- **Concurrency.** One `asyncio.Semaphore` per backend, sized by `concurrency` in the config, held around each
  backend request.
- **Records.** Each `complete()` writes `runs/<run>/llm/calls/<call-id>.json` (system, user, text, parsed,
  usage, latency, backend, model, error, and every attempt with its raw output, prompt and response sha256,
  and backend detail) and appends one `llm_call` event whose data carries the prompt and response sha256, the
  transcript path, and the transcript's own sha256. The prompt hash covers `[system, user]` of the first
  request. Failures raised during routing (budget, config, tool policy, schema, transport) also produce an event with
  `error` set. An unknown role raises `ConfigError` before anything is recorded.
- **Usage.** `{"input", "output", "cached"}`. For llxprt, `input` is `tokens.prompt` (all input tokens
  including cached ones) and `output` is `tokens.total - tokens.prompt`, summed over attempts.

## llxprt effort and output cap

Checked in the llxprt-code source (`packages/settings/src/settings/registry`, `packages/providers/src/anthropic/
AnthropicRequestPreparation.ts`, `packages/providers/src/openai-responses/`), not by a live call:

- `ephemeralSettings["reasoning.effort"]` is the generic effort setting (`minimal` to `max`; this module uses
  `low`, `medium`, `high`). The backend replaces the profile's value with the role's effort. `reasoning.enabled`
  and `reasoning.adaptiveThinking` are left as the profile has them.
- `ephemeralSettings["maxOutputTokens"]` is the generic cap. The claudecode provider sends it as `max_tokens`
  unless `modelParams.max_tokens` is set, which wins. The codex provider (ChatGPT plan) turns it into
  `max_output_tokens` and then removes that field in Codex mode, so the cap does not reach GPT models through
  llxprt. `modelParams` entries go to the request unchanged, so the backend also replaces any cap the profile
  already carries in `modelParams` (`max_tokens`, `max_output_tokens`, `max_completion_tokens`, `maxOutputTokens`)
  and adds none of its own.
- On Anthropic models thinking tokens count against `max_tokens`. A cap that cuts the JSON reply costs a schema
  retry (the run below shows one: the Opus revision hit the 40000 cap and the retry took another 206 s).

## llxprt tool control (the method in use)

llxprt has no flag that removes tools, and `--set tools.allowed=...` is ignored in non-interactive mode
(governance keys are filtered out; read from the CLI source, not tried). `--profile-load` and `--profile` cannot be combined. What works:

1. Read the named profile from `<llxprt config dir>/profiles/<name>.json`, set
   `ephemeralSettings["tools.allowed"]` to `[]` (pure reasoning) or to the web tools
   (`exa_web_search`, `direct_web_fetch`, listed in `models.toml`), and pass the result as the inline
   `--profile '<json>'`. In non-interactive mode an explicit `tools.allowed` array replaces the default tool
   set, and with `[]` no tool schemas are sent to the model.
2. `--approval-mode yolo` is added only for `allow_web=True`, where the allow-list is just the two web tools.
   Without yolo the web tools would wait for an approval that never comes in a headless run.
3. The CLI runs with `LLXPRT_CONFIG_HOME` pointing at a private directory,
   `runs/<run>/llm/llxprt-home/{pure,web}/`, which contains symlinks to the real `oauth/` and `keys/`, a copy
   of `settings.json`, and `prompts/core.md` from `llxprt_prompts/`. That drops the user's global
   `LLXPRT.md` memory and the stock coding-agent system prompt (which also makes the model emit text
   tool-call markup when it has no tools). llxprt keeps a custom `core.md` and only backs up its own default.
4. After every call the backend inspects `stats.tools`. With tools off, `totalCalls` must be 0 and `byName`
   empty. With web on, every name in `byName` must be in the web allow-list. Missing `stats.tools` also fails.
   A violation raises `ToolPolicyViolation` (no fallback).

Verification, all with the haiku profile:

- Asked to read a file and run a shell command with `tools.allowed = []`: no tool call (`totalCalls` 0); with
  the stock system prompt the model wrote tool-call markup as plain text, with the minimal prompt it replied
  "NO TOOLS".
- Asked in web mode to run `echo PWNED > pwned.txt` and to write a file: refused, `totalCalls` 0, no file
  created.
- Web mode with a search request: `totalCalls` 1, `byName` = `exa_web_search` only.
- The live suite asserts `totalCalls == 0` on every tools-off llxprt call and `byName` within the web
  allow-list on the web call.

The prompt travels on stdin and `-p` is the fixed short instruction "Follow the instructions in the input
above." (llxprt appends `-p` to stdin). This was tested up to a 185 KB prompt (about 70k tokens, real text),
which worked; putting the prompt in argv is avoided because of the OS argument limit. One odd case: a 10 KB
prompt made of random six-letter words with the question at the end made the model answer as if no task had
been given, and the CLI made 3 API requests. The same noise with the question at the start, and natural text
with the question at the end, worked. Not explained; JSON mode would catch it through the schema retry.

## Measured overhead

Haiku profile, prompt "Reply with exactly: ok", measured as llxprt `stats.models.*.tokens.prompt`:

| Configuration | Input tokens |
| --- | --- |
| `--profile-load haiku` as the CLI ships (all tools, global memory, stock system prompt) | 19,792 |
| inline profile with `tools.allowed = []` | 13,161 |
| plus private `LLXPRT_CONFIG_HOME` (no global `LLXPRT.md`) | 2,520 |
| plus minimal `core.md` (what the backend does now) | 370 |

The remaining cost is the prompt itself plus about 350 tokens of fixed overhead. The CLI process itself adds
about one second of startup per call.

Per profile, one call each through the real backend (tools off unless noted), latency is wall time of the
whole call including CLI startup:

| Profile | Resolved model | Latency (s) | Input tokens | Output tokens | Prompt |
| --- | --- | --- | --- | --- | --- |
| haiku | claude-haiku-5-5 | 2.94 | 551 | 12 | capital of France, JSON schema |
| sonnet | claude-sonnet-5-5 | 2.28 | 553 | 12 | same |
| opus | claude-opus-5-5 | 3.39 | 413 | 4 | "Say: pong" |
| gpt-sol-high | gpt-6.1-sol | 3.08 | 245 | 5 | same |
| terra | gpt-5.6-terra | 3.41 | 243 | 28 | same |
| haiku, `allow_web` | claude-haiku-5-5 | 8.71 | 13,024 | 422 | arXiv reservoir computing paper, 1 `exa_web_search`, 2 API requests |
| LM Studio `local` role | qwen/qwen3.8-27b | 58.18 | 63 | 32 | "Say the word: pong" (thinking model, may include model load) |

Prompt sizes differ between rows, so input tokens are not comparable across rows beyond showing that fixed
overhead is a few hundred tokens. Web mode costs more because search results enter the context. These rows were
measured before role settings existed: reasoning effort came from each profile (all `high`).

## First research run and the role settings chosen from it

`runs/run-20261010T000322Z` (4 cards, 12 calls, all llxprt, every profile at effort `high`, cap 40000), from the
transcripts:

| Role | Model | Calls | Mean latency (s) | Mean output tokens |
| --- | --- | --- | --- | --- |
| salad_critic | claude-haiku-5-5 | 6 | 167 | 34,792 |
| explorer (new and revise) | claude-opus-5-5 | 3 | 382 | 40,543 |
| explorer_alt (new and revise) | gpt-6.1-sol | 3 | 248 | 11,307 |

The salad critic's JSON replies were a few thousand tokens, so nearly all of its output was thinking; its notes
show it traced every toy example by hand, which the executor also does. One Opus revision stopped at exactly 40000
output tokens, the outer card object was cut off, the first JSON object found was the inner `mechanism`, and the
schema retry cost another 206 s.

Shipped `[role_settings]` (both tables):

| Role | Effort | max_output_tokens | Reason |
| --- | --- | --- | --- |
| explorer, explorer_alt | high | 40000 | The card is the product. Parsimony rules in the prompt shrink the reply, not the thinking. 40000 is what the profiles already had and is accepted by all three Claude models; the gateway's Anthropic default would be 32000. |
| formalizer | medium | 24000 | Deriving obligations from a card is structured work; the judges and disprover attack it afterwards. |
| salad_critic | low | 16000 | The critic now sorts findings, lists at most 8 fixable ones and no longer traces the toy example. Its reply fits in a few thousand tokens. 16000 rather than smaller because a cut reply costs a full retry. |
| reduction_critic | medium | 24000 | Mapping onto known forms and writing check code; also used for `reduction_check`. |
| prior_art | low | 12000 | Query proposal and classification over provided hits. |
| feasibility_judge_a, feasibility_judge_b | high | 40000 | The deciding gate before construction. |
| disprover | medium | 32000 | Counterexamples plus check code; the executor settles them. |
| builder | medium | 40000 | A full candidate.py in JSON; a cut reply is expensive. |
| analyst | low | 12000 | Compares claims with numbers in the bench JSON. |

The efforts are the proposed ones, unchanged; the caps were chosen here. `local` has no settings. No live call has
measured the effect yet.

Sources: `runs/llm-live-20261009T232441Z/` (live suite) and `python -m hybrid.llm.measure opus gpt-sol-high terra`.

## Tests

```
.venv-hybrid/bin/python -m pytest hybrid/tests/llm -q                 # unit tests, no network, no quota
.venv-hybrid/bin/python -m pytest hybrid/tests/llm -m live -s         # four live model calls
```

Live tests are skipped unless `-m live` or `BIGBRANE_LIVE=1` is given. They write transcripts to
`runs/llm-live-<timestamp>/`. The LM Studio test calls `GET /v1/models` first and skips with a message when
the server is unreachable or only embedding models are listed.

## Not done

- `anthropic_messages` and `gateway` are tested only against mocked transports (no key, no gateway yet).
- The gateway's Anthropic route has not been run against the live API with thinking on. Until it is, expect some
  difference in quality and speed between the two routing tables for the Claude roles. The gateway does not map a
  thinking budget, and Claude ids outside the adaptive list run without thinking because the gateway drops
  `reasoning_effort` for them. See `hybrid/gateway/STATUS.md`.
- Through llxprt, GPT (codex) calls ignore `max_output_tokens` (llxprt removes it in Codex mode), and so does the
  gateway's ChatGPT route.
- The gateway table has one model for judge B (`chatgpt/gpt-6.1-sol`) and no fallback.
- Tokens burned by a call that fails the tool-policy check are not added to the token budget.
- No streaming, no tool use through the HTTP backends, no cost accounting beyond token counts.
