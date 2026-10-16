# Claude port from LLxprt Code

This file records how the gateway's Claude requests compare with LLxprt Code's Claude OAuth ("claudecode") requests,
what was changed to match, and what was left different on purpose.

Reference: `llxprt-code` at `~/projects/llxprt/branch-7/llxprt-code`, commit `c214b23f` (origin/main, 2026-10-09),
read only. Main files: `packages/auth/src/flows/anthropic-device-flow.ts`, `packages/providers/src/auth/anthropic-oauth-provider.ts`,
`packages/providers/src/auth/local-oauth-callback.ts`, `packages/providers/src/anthropic/*`,
`packages/providers/src/composition/aliases/claudecode.config`, `packages/core/src/utils/retry.ts`. The Rust port
(`~/projects/llxprt/agent/main/llxprt-code-rs`, vendored `serdes-ai-models/src/claude_code_oauth`) is an older, generic
implementation (tool prefix `cp_`, no model rules), so the TypeScript code is the reference.

Live checks ran on 2026-10-10 against the subscription endpoint with the existing login. Raw outputs are in
`results/gateway-20261011/` (`probe-*.log`).

## Difference table

"Same" means the gateway already matched before this work. "Ported" means this work changed the gateway. "Kept" means
the gateway differs from LLxprt and the difference stays, with the reason.

### Login and tokens

| Item | LLxprt Code | Gateway | Status |
| --- | --- | --- | --- |
| Authorize URL | `https://claude.ai/oauth/authorize`, params `code=true`, `client_id`, `response_type=code`, `redirect_uri`, `scope`, `code_challenge`, `code_challenge_method=S256`, `state` | identical | Same |
| Client id | `9d1c250a-e61b-44d9-88ed-5944d1962f5e` | identical | Same |
| Scopes | `org:create_api_key user:profile user:inference` | identical | Same |
| PKCE | S256, verifier = 32 random bytes base64url, verifier reused as `state` | identical | Same |
| Redirect, paste flow | `https://console.anthropic.com/oauth/code/callback`, user pastes `code#state` | identical | Same |
| Redirect, callback flow | tries `127.0.0.1` ports 8765 to 8795, serves `/callback`, advertises `http://localhost:<port>/callback`, checks `state`, 3 minute timeout, races the paste prompt | `anthropic_callback.py` and `login.py` do the same; `login anthropic` races the browser callback against a hidden paste prompt that can be called off (`tty_prompt.py`); `--paste-only` restores paste-only; no free port falls back to paste | Ported |
| Authorize URL shown to the user | shows the paste URL, opens the callback URL in the browser | same | Ported |
| Token endpoint | `https://console.anthropic.com/v1/oauth/token` | identical | Same |
| Code exchange body | JSON: `grant_type`, `code`, `state`, `client_id`, `redirect_uri`, `code_verifier` | identical; `redirect_uri` is now the one the authorize URL used (console or loopback) | Same, extended |
| Refresh body | form-encoded `grant_type=refresh_token`, `refresh_token`, `client_id` | identical | Same |
| Refresh timing | reactive buffer 30 s; a background timer renews at `max(300 s, 10 % of remaining life)` minus 0 to 30 s of jitter | on demand, when fewer than 300 s remain; no timer | Kept. The gateway has no idle timer; a token that expired while the gateway sat idle is refreshed by the next request. |
| Refresh rotation | new refresh token wins; the old one is kept when the response has none or an empty one; a changed refresh token on disk means another process refreshed, so skip | same, plus a cross-process flock around the re-read and the write | Same |
| Missing `expires_in` | defaults to 3600 s | rejected unless the access token carries a JWT `exp`; no lifetime is guessed | Kept (deliberate hardening) |
| Token storage | OS credential store or LLxprt's own files | private files under `~/.bigbrane-gateway/oauth/`; no OS credential store, no reading LLxprt's tokens | Kept (hard rule) |

### Request headers

| Header | LLxprt Code | Gateway | Status |
| --- | --- | --- | --- |
| `Authorization` | `Bearer <token>` (SDK `authToken`) | same | Same |
| `anthropic-version` | `2023-06-01` (SDK default) | same | Same |
| `anthropic-beta` | `oauth-2025-04-20, interleaved-thinking-2025-05-14` | same base set | Same |
| `anthropic-beta` + cache | adds `extended-cache-ttl-2025-04-11` when `prompt-caching` is `1h` (the default) | added when any block carries a 1 hour `cache_control` | Ported |
| `anthropic-beta` + binding | adds `thinking-binding-controls-2026-08-01` when `thinking` carries `block_binding` | same | Ported |
| `User-Agent` | `claude-cli/2.1.293 (external, cli)` | same | Same |
| `anthropic-dangerous-direct-browser-access` | `true` (the SDK adds it because LLxprt sets `dangerouslyAllowBrowser`) | same | Ported |
| `x-app` | not sent | not sent | Same |
| `context-1m` beta | not sent; the 1M window is only LLxprt's local `context-limit` | not sent | Same |
| `x-stainless-*`, SDK `Accept` | sent by the SDK | not sent | Kept. They are SDK telemetry with no effect on the answer. |

### Prompt and message shape

| Item | LLxprt Code | Gateway | Status |
| --- | --- | --- | --- |
| `system` field | only `You are Claude Code, Anthropic's official CLI for Claude.` | same | Same |
| Caller's system prompt | first user turn, `<system>\n...\n</system>\n\nUser provided conversation begins here:` | same | Same |
| Prompt caching | `cache_control: {type: ephemeral, ttl: 1h}` on the context-prefix block and on the last cacheable block of the last message (not on thinking blocks or empty text or results) | same two breakpoints | Ported |
| Anchor and media-purge cache breakpoints | extra breakpoints driven by LLxprt's history compaction state | none; the gateway keeps no history | Kept |
| Tool names | `llxprt_` prefix on declared tools; names unprefixed in replies and in replayed history | same | Same |
| Tool order | tools sorted by name, top-level schema properties sorted, for a stable cache prefix | same | Ported |
| Tool schema | LLxprt rewrites its Gemini-style schemas (lowercases types, keeps a fixed key list) | the caller's JSON Schema is passed through, with `type`, `properties` and `required` defaulted | Kept. OpenAI-style schemas are already JSON Schema, and the rewrite would drop `anyOf` and `additionalProperties`. |
| Tool ids | `toolu_` normalization | same | Same |
| Orphaned tool results | removed | removed | Ported |
| Missing tool results | synthesized as `[tool execution interrupted]` with `is_error` | same, including moving a later result next to its tool use | Ported |
| Sequence rules | starts with a user turn, never empty (`Hello`), placeholders `[Empty message]` and `[No content generated]` for empty turns, blank text blocks removed | same | Ported |
| Trailing assistant turn | a `Continue the conversation` user turn is added when thinking is on or the model refuses prefill (Fable 5 and the 5.5 models) | same | Ported |
| Oversized image replacement (`max-image-dimension`) | drops images over the limit before sending and retries once on the many-image dimension error | none; the API's 400 is returned | Not ported. It needs image header parsing; no current caller sends images. |
| Tool input coercion to schema, double-escape repair | applied to replies | none | Not ported. The gateway returns the arguments as the model sent them. |
| Thinking blocks in replies | kept in history with signatures | removed; the chat format has no place for the signature | Kept |
| `tool_choice` | not set by LLxprt | mapped from the OpenAI field (`auto`, `required`, function) | Kept (gateway feature) |
| `response_format: json_schema` | not an LLxprt feature | forced tool `llxprt_emit_json`; on Opus 5.5, Sonnet 5.5 and Fable 5.1, which refuse forced tools, native structured output (`output_config.format`) with a rewritten schema | Ported as a gateway fix (see below) |

### Models and parameters

| Item | LLxprt Code | Gateway | Status |
| --- | --- | --- | --- |
| Model list | `claudecode.config` `staticModels`: 19 ids (Opus 5.5, 5, 4.8, 4.7, 4.6, 4.5 dated and bare; Fable 5.1, 5; Sonnet 5.5, 5, 4.6, 4.5 dated and bare, 4 dated and bare; Haiku 5.5, 4.5 dated and bare) | `/v1/models` lists the same 19 as `anthropic/<id>`; routing passes any `anthropic/<id>` through, so `-latest` pointers and dated snapshots work | Ported (was 3 ids) |
| Adaptive-thinking models | Opus 4.6+, Sonnet 5 and 5.5, Haiku 5.5, Fable 5 and 5.1 (anchored patterns) | same patterns | Same |
| Adaptive request | `thinking: {type: adaptive, display: summarized}` + `output_config.effort` | same | Same |
| Effort names | `minimal` and `low` send `low`; `xhigh` and `max` send `max` on adaptive models, `high` elsewhere | same; `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` accepted (was `low`, `medium`, `high`) | Ported |
| Thinking off | Opus 5 and Haiku 5.5: `{type: disabled}`; Sonnet 5.5: `{type: between_tools}`; other models: field omitted | `reasoning_effort: "none"` sends the same; other models omit it and report `reasoning_effort` under `dropped` | Ported |
| Non-adaptive Claude models with an effort | `thinking: {type: enabled, budget_tokens: 10000}` + `output_config.effort` | same, but `output_config.effort` is left out on Sonnet 4, Sonnet 4.5 and Haiku 4.5 | Ported with a guard. The API answers 400 "does not support the effort parameter" there (probe-matrix-effort.log). |
| `block_binding` | `{prefix_mismatch_behavior: drop_block}` inside adaptive or enabled `thinking` on Fable 5.1, Opus 5.5, Sonnet 5.5, Haiku 5.5 | same | Ported |
| Default thinking | the alias profile turns thinking on with effort `high` for Claude ids | thinking is sent only when the caller sends `reasoning_effort` | Kept. That default belongs to LLxprt's profile, not to its request code; the API's default effort is already `high`. |
| Default `max_tokens` | the alias sets `maxOutputTokens` 128000 for every model | registry limit: 128000 for the 4.6+ and 5.x ids, 32000 Opus 4.5, 64000 Sonnet 4/4.5, 16000 Haiku 4.5; ids outside the registry follow `getMaxTokensForModel` (4096 for unknown) | Ported with a guard. The older ids would answer 400 to 128000. |
| Temperature, `top_p`, `top_k` | forbidden on Opus 5.5, Sonnet 5.5, Haiku 5.5 (`unallowedParameters`) | dropped and listed under `dropped` for those and for Opus 5, Sonnet 5, Fable 5 and 5.1, Opus 4.7 and 4.8, which the API also refuses ("`temperature` is deprecated for this model", probe-temp.log). Opus 4.6, Opus 4.5, Sonnet 4.6 and Haiku 4.5 still take them. `top_k` is accepted as an extension field. | Ported and widened |
| Stop sequences | `stop_sequences` | same | Same |
| Streaming | always streamed upstream | same, aggregated to one answer | Same |

### Errors, retries and usage

| Item | LLxprt Code | Gateway | Status |
| --- | --- | --- | --- |
| Retried | HTTP 429, any 5xx (529 included), in-stream `overloaded_error`, `rate_limit_error`, `api_error`, transient network errors, truncated streams | same | Ported |
| Not retried | 400, 403, and other 4xx | same | Ported |
| Attempts and delay | 6 attempts, 4 s first delay, doubling to a 30 s ceiling, +-30 % jitter | same (`RetryPolicy`) | Ported |
| `Retry-After` | honoured on 429 and in-stream overload or rate-limit errors (seconds or HTTP date), capped at 300 s; 5xx uses backoff | same | Ported |
| 401 | one forced credential refresh, then retry | same (this was already in the gateway) | Same |
| Pre-request throttle | uses `anthropic-ratelimit-*` headers of the last answer; below 5 % remaining it waits for the reset, at most 60 s | same | Ported |
| Usage | prompt tokens = input + cache read + cache creation; completion = output | same; cached tokens reported separately | Same |
| Metering | not applicable | one meter line per call, whatever the number of attempts | Kept |

## Fixes found by live probes that LLxprt does not have

1. **Forced tool choice is refused on Opus 5.5, Sonnet 5.5 and Fable 5.1** ("tool_choice: type "tool" and "any" are not
   supported for this model", probe-matrix-json.log). `response_format: json_schema` through those models failed
   before this work. They now use native structured output. That feature accepts a smaller subset of JSON Schema
   (objects must set `additionalProperties: false`; `minimum`, `minLength` and similar are refused), so
   `anthropic_schema.py` rewrites the caller's schema. A constraint the rewrite removes is appended to the property's
   description and is no longer enforced by the API. A caller's own `tool_choice` of `required` or a named function
   on those three models still returns the API's 400, because there is no equivalent to translate it to.
2. **Sampling parameters are refused on more models than LLxprt lists** (see the table). Before this work a request
   with `temperature` to these models failed with a 400.
3. **Opus 5.5 does not accept `thinking: {type: disabled}`** ("Use adaptive and output_config.effort"), which matches
   LLxprt's rule that only Opus 5 and Haiku 5.5 have a disabled mode. `reasoning_effort: "none"` on Opus 5.5 therefore
   omits `thinking`, as LLxprt does.

## Re-login

None is needed. The stored record format, the file locations and the refresh request are unchanged. The next
`login anthropic` uses the new callback flow (or `--paste-only` for the old one).

## Where each piece lives

| File | Contents |
| --- | --- |
| `anthropic_models.py` | model list, id predicates, thinking-off modes, prefill, sampling and effort rules, max tokens |
| `anthropic_translate.py` | request building: effort and thinking plan, caching, tool ordering, JSON schema mode, beta values |
| `anthropic_sequence.py` | message-sequence repair |
| `anthropic_schema.py` | schema rewrite for native structured output |
| `anthropic_retry.py` | retry policy, `Retry-After`, rate-limit headers and throttle |
| `anthropic_callback.py`, `tty_prompt.py`, `login.py` | loopback login and the cancellable hidden prompt |
| `backends.py` | headers, retry wiring, model list |

Tests: `test_claude_models.py`, `test_claude_request.py`, `test_claude_retry.py`, `test_claude_login.py`, plus the
updated `test_anthropic_thinking.py`, `test_translate_anthropic.py`, `test_server.py`.
