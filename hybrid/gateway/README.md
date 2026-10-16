# hybrid/gateway

A local OpenAI-compatible server that lets the Bigbrane research loop use two subscription logins: Claude Code
OAuth (Anthropic) and Sign in with ChatGPT (OpenAI). It is the only process that ever refreshes those tokens.
`hybrid/llm`'s `gateway` backend (an OpenAI-compatible preset) points at it. Login steps are in `LOGIN.md`.

```
python -m hybrid.gateway login anthropic|chatgpt   # interactive, run by a person
python -m hybrid.gateway status                    # which logins exist and when they expire, never a token
python -m hybrid.gateway serve --port 8765         # 127.0.0.1, per-run bearer token
python -m hybrid.gateway logout anthropic|chatgpt  # delete the local record (chatgpt: revoke at OpenAI first)
```

Dependencies: the standard library and `httpx`. The server is a small asyncio HTTP/1.1 layer (`http_min.py`), one
request per connection.

## HTTP surface

| Route | Behavior |
| --- | --- |
| `POST /v1/chat/completions` | Model `anthropic/<model>` or `chatgpt/<model>`. Non-streaming answer; with `stream: true` the finished answer is replayed as `chat.completion.chunk` events (buffered, not live). |
| `GET /v1/models` | Ids of every provider that has a login (`chatgpt/` ids come from the account's own catalog); providers without a login are listed under `not_logged_in`. Each list is fetched under a 10 s limit and cached for 10 minutes; a provider that times out serves its last good list (see `[models]` in LOGIN.md). |
| `GET /health` | `{"ok": true}`, no auth. |

Everything except `/health` needs `Authorization: Bearer <run token>`. Optional request header `x-bigbrane-run-id`
is copied to the meter line.

Supported request fields: `messages` (system, developer, user with text and images, assistant with tool calls, tool),
`tools`, `tool_choice`, `response_format` (`json_schema`; `json_object` on ChatGPT only), `max_tokens` /
`max_completion_tokens`, `temperature` / `top_p` / `top_k` / `stop` (Anthropic), `reasoning_effort`.
ChatGPT takes it as `reasoning.effort`. On Anthropic the values are `none`, `minimal`, `low`, `medium`, `high`, `xhigh`
and `max`, mapped per model the way LLxprt Code maps them (adaptive thinking plus `output_config.effort` on Opus 5.5,
Sonnet 5.5, Haiku 5.5 and the other adaptive-capable ids, budgeted thinking on older Claude ids, `none` turns thinking
off where the model has an off mode). Thinking text is never returned. Anything a backend cannot take is dropped and
listed under `dropped` in the meter line. The ChatGPT plan route rejects `max_output_tokens`, `temperature`, `top_p`,
`user` and `metadata`, so those are among the dropped fields there. The Anthropic 5.x models and Opus 4.7 and 4.8 refuse
`temperature`, `top_p` and `top_k`, so those fields are dropped for them. `CLAUDE-PORT.md` has the full comparison with
LLxprt Code and the per-model rules.

## Module map

| File | Responsibility |
| --- | --- |
| `__main__.py` | CLI |
| `server.py`, `http_min.py` | HTTP surface, bearer check, error bodies, buffered SSE |
| `gateway.py` | model routing (`provider/model`), metering around every call, `/v1/models` |
| `backends.py` | upstream calls for both providers (both streamed upstream, aggregated to one answer), one retry after a 401 on an unexpired token; the Anthropic backend also sends LLxprt's headers and retries 429, 5xx and in-stream overload errors |
| `anthropic_translate.py` | chat to Messages and back (system prompt placement, prompt caching, tool names and order, JSON schema, thinking from `reasoning_effort`, usage) |
| `anthropic_models.py` | the 19 Claude ids LLxprt serves over OAuth, id predicates, per-model thinking, sampling, effort and `max_tokens` rules |
| `anthropic_sequence.py` | message-sequence repair (orphaned and missing tool results, first and last turn, empty turns) |
| `anthropic_schema.py` | JSON Schema rewrite for native structured output on models that refuse a forced tool |
| `anthropic_retry.py` | retry policy, `Retry-After`, `anthropic-ratelimit-*` throttle |
| `anthropic_callback.py`, `tty_prompt.py` | loopback listener for the Claude login redirect and a hidden prompt that can be called off |
| `responses_translate.py` | chat to Responses and back (`instructions`, `function_call` items, `text.format`, usage) |
| `anthropic_auth.py`, `chatgpt_auth.py` | login URLs, PKCE, code exchange, refresh, ID token verification |
| `login.py` | interactive login drivers (Anthropic: loopback callback racing a paste; ChatGPT: loopback listener) and `status` |
| `refresher.py` | the single refresher: one `asyncio.Lock` and one cross-process flock per provider, re-read inside both, persist before use, dead-token memory |
| `store.py` | token records (`Credentials`), per-provider lock, snapshot and login publication |
| `fsstore.py` | private filesystem storage: owner, mode and type checks, no-follow bounded reads, flock, atomic publish |
| `singleton.py` | `flock` pidfile, one gateway per machine |
| `meter.py` | per-call usage JSONL |
| `redact.py` | scrubber applied to error messages and log lines |
| `paths.py` | state directory (`~/.bigbrane-gateway`, override `BIGBRANE_GATEWAY_HOME`), `oauth/` location, host id, `config.toml` |

## Token storage

Tokens live in `~/.bigbrane-gateway/oauth/` (`BIGBRANE_GATEWAY_HOME` moves the parent). The directory is mode 0700
and owned by the current user. Each provider has `anthropic.json` or `chatgpt.json` (mode 0600) and a stable
`<provider>.lock` file. The macOS keychain is never used or probed: its password prompts interrupt the owner, and
probing it was a likely cause of earlier prompts.

The rules follow the explicit filesystem OAuth in llxprt-code-rs (PR 309):

- Every read and write checks owner, mode, file type and link count. A wrong state fails with a message that names
  the path and the fix. Nothing is repaired, overwritten or moved aside automatically.
- Files open relative to the directory with `O_NOFOLLOW`, so a symlink is refused, and reads stop at 65536 bytes.
- A refresh holds an exclusive `flock` on the provider's lock file from the re-read until the new record is
  published. A gateway and a login command therefore cannot rotate the same refresh token twice. Waiting for the lock
  gives up after 120 seconds.
- A write goes to a private temp file in the same directory, is fsynced, renamed over the record, and the directory
  is fsynced. Readers never see a partial record and take no lock. A crash can leave an unreferenced temp file; the
  next write removes it.
- A login remembers the exact bytes of the record it started from and publishes only if the record is unchanged. If a
  refresh or another login got there first, the newer record is kept and the login fails with a conflict message.
- There is no fallback storage and no migration from anywhere else. An existing LLxprt login is never read.

## Refresh rules

- The access token is refreshed when fewer than 300 seconds remain.
- Inside the provider lock (in process and on disk) the record is read again, so callers that waited reuse the
  refresh that just happened, including a refresh or login done by another process.
- The replacement refresh token is published to the oauth directory (fsync, rename, directory fsync) before the new
  access token is returned to anyone. If the write fails the request fails and no call is made with the unsaved
  token.
- A token response without `expires_in` takes its expiry from the access token's JWT `exp` claim (read without
  verifying the signature, as lifetime metadata only). A response with neither is rejected; no lifetime is guessed.
- `invalid_grant`, `invalid_refresh_token`, `token_expired`, `refresh_token_expired`, `refresh_token_invalidated` and
  `refresh_token_reused` mark the stored refresh token dead. Later requests fail immediately with the login command and
  make no network call, until a login replaces the record. Timeouts, 5xx and other errors are not remembered.
- A 401 from the API on a token that looks valid forces one refresh and one retry. A second 401 is returned.

## What the metering file holds

One JSON object per call, appended to `usage.jsonl` (override with `serve --meter`): `ts`, `call_id`, `provider`,
`model` (as requested), `resolved_model`, `status`, `usage` (`input` includes cached tokens, `output`, `cached`),
`latency_s`, `run_id`, `dropped`, and `error` with `code` and `status` on failures. A response without usage fails the
call (`usage_missing`) and is recorded as an error.

## Tests

```
.venv-hybrid/bin/python -m pytest hybrid/tests/gateway -q
```

No network: upstream HTTP is `httpx.MockTransport`, and ID tokens are signed with a throwaway RSA key stored in the
fixtures. Token files are real files in the real filesystem store, under a pytest temp directory inside the
workspace (`tmp/pytest-<pid>`, set in the root `conftest.py`, never the system temp directory). A test scans every
Python module under `hybrid/` and fails if one references operating system credential store APIs, the `security` command or the
Security framework bindings.
