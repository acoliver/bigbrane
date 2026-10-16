# hybrid/gateway status

## /v1/models timeouts and cache (2026-10-11)

**Problem.** A research run failed its preflight `GET /v1/models` at 04:29 -03. The upstream `GET
https://api.openai.com/v1/models` took about 3.5 minutes, the gateway waited for it before answering, and the client had
already given up. The log showed `GET /v1/models -> 500 in 213092ms` and a traceback for the broken write.

**Changes.**
- Each provider's list is fetched concurrently, each under `[models] upstream_timeout_s` (default 10 s).
- A successful list is cached in memory for `[models] cache_ttl_s` (default 600 s; 0 fetches every time). A fresh cache
  makes no upstream call. A stale cache is refreshed by the request that finds it stale. There is no background task.
- A provider whose live fetch times out or fails serves its last good list, even if stale. With none, it serves its
  static ids (Anthropic's 19 claudecode ids; ChatGPT has none). With neither, it contributes nothing. Its entry under
  `upstream_errors` says why; a timeout is status 504, code `upstream_timeout`. The other providers still list.
- One WARNING line per failed live fetch, with the provider, the duration and what was served:
  `upstream chatgpt model list failed after 10.01 s (...); serving cached list of 19 models, fetched 412 s ago`.
- A login that is missing (`LoginRequired`) still goes to `not_logged_in` and is not papered over by the cache. Errors
  that are not upstream failures still return 500.
- ChatGPT list calls still pass through the concurrency slot and the pacer. A list call that times out while queued for
  the pacing gap never reaches OpenAI, so the timeout cannot skip pacing. Cache hits make no list call, so they use no
  pacing slot. Fewer list calls count against the 60 per hour cap.
- A client that disconnects while the response is written gets one INFO line,
  `GET /v1/models -> client disconnected after N ms; no response sent`, and no traceback. Other errors are unchanged.
  Checked on a real socket: a reset during a large `/v1/models` answer produced that one line.

**Limits.** While a provider is down and its cache is stale, each `/v1/models` call still waits the full timeout for
it; there is no failure backoff. A timed-out ChatGPT list that already reached OpenAI still counts as a pacing start.

**Restart.** The `serve` process that is running now started before this change and still has the old code. A restart
is needed for the new behavior and the `[models]` defaults to take effect. Until then the old blocking `/v1/models` and
the traceback on client disconnect remain.

**Tests.** `hybrid/tests/gateway/test_models_listing.py` (23 tests, fakes and a fake clock, no upstream calls). The
whole gateway suite passes: see "Tests" below.

## SIWC durability (2026-10-11)

**Verdict.** Our refresh request was already the same request every public SIWC client sends, and no public client
has shown a working SIWC refresh. Nothing on the request side is known to fix devkit#5. OpenAI's only staff
statement ties `refresh_token_invalidated` to the account's security state, not to the request. No browserless
recovery exists, documented or demonstrated. What the gateway can control is now done: it paces ChatGPT calls, it
refreshes as soon as the short refresh window opens even with no traffic, it keeps a valid access token after a
refused refresh, it never resends a consumed refresh token, and it records every refresh outcome with OpenAI's
request id. Live refresh result for tonight's token: see "Live" below.

### What working clients send (read 2026-10-11)

| | Endpoint | Body (form-encoded) | When |
| --- | --- | --- | --- |
| OpenAI devkit `packages/local` ([oauth.ts L330-L337](https://github.com/openai/sign-in-with-chatgpt-devkit/blob/0a36fefeb913055c8c7a1a29b63d82b96b2e841a/packages/local/src/oauth.ts#L330-L337), [index.ts L99-L127](https://github.com/openai/sign-in-with-chatgpt-devkit/blob/0a36fefeb913055c8c7a1a29b63d82b96b2e841a/packages/local/src/index.ts#L99-L127)) | discovered `token_endpoint` | `grant_type`, issued `client_id`, `refresh_token`, `resource=https://api.openai.com/v1`; headers `content-type: application/x-www-form-urlencoded`, `accept: application/json` | last 60 s before expiry, never before `earliest_refresh_at`; an expired token before that time fails with `refresh_not_ready` |
| T3 Code ([CodexChatGptAuth.ts L735-L752](https://github.com/pingdotgg/t3code/blob/43888b5ff14cce0a52b9fcb03e74c5c693141aa2/apps/server/src/provider/CodexChatGptAuth.ts#L735-L752)) | discovered `token_endpoint` | same four fields | last 60 s, after `earliest_refresh_at` |
| pi ([openai-chatgpt.ts L20, L29, L208-L221](https://github.com/earendil-works/pi/blob/ea448f454af474f3e2fe9733b331e17b0662f5d6/packages/ai/src/auth/oauth/openai-chatgpt.ts#L208-L221)) | `https://auth.openai.com/api/accounts/oauth/token` | same four fields | 3 min before expiry; ignores `earliest_refresh_at` |
| Codegraff ([oauth_chatgpt.zig L469, L505-L507](https://github.com/justrach/codegraff/blob/c71b9ad7ba51ef198f82b58eacf6ac15654b570a/src/oauth_chatgpt.zig#L469)) | `https://auth.openai.com/api/accounts/oauth/token` | same four fields | last 300 s, never before `earliest_refresh_at` ("OpenAI answers a refresh before earliest_refresh_at with invalid_grant") |
| Bigbrane gateway | `https://auth.openai.com/api/accounts/oauth/token` | same four fields, same order; `accept: application/json` added tonight | from `max(earliest_refresh_at, expires_at - 300 s)`; proactively since tonight |

- Endpoint: the issuer's discovery document names `https://auth.openai.com/api/accounts/oauth/token`
  (fetched 2026-10-11, saved in `tmp/siwc-durability-20261011/discovery.json`). `https://auth.openai.com/oauth/token`
  is the Codex CLI's endpoint ([openai/codex#39696](https://github.com/openai/codex/issues/39696)); no SIWC client uses it.
- Parameters: the [Refreshing tokens](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
  page lists exactly the four fields and says to omit `scope`. Nobody sends `audience`. The docs pages were unchanged
  from the 2026-10-10 snapshot when diffed tonight.
- Timing: devkit#5 tried 63 s after `earliest_refresh_at`, 60 s before expiry, 3 min 41 s before expiry and after
  expiry; every one was refused ([devkit#5](https://github.com/openai/sign-in-with-chatgpt-devkit/issues/5)). The docs
  give no meaning for `earliest_refresh_at` beyond listing it in the token response
  ([token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference)).
- Rotation: each successful refresh returns a replacement refresh token (token reference); refreshes must be
  serialized (profiles-and-sessions). The devkit persists a `pendingRefresh` checkpoint before it verifies the new ID
  token. Codex reports show reused refresh tokens invalidating sessions
  ([openai/codex#36525](https://github.com/openai/codex/issues/36525)).
- Silent re-authorization: no SIWC doc, the devkit, T3 Code, pi or Codegraff uses `prompt=none`. The devkit uses only
  `prompt=consent` and deliberately keeps `id_token_hint` out of browser URLs. The discovery document lists no
  `prompt_values_supported`. `prompt=none` would also need the browser's auth.openai.com session cookie, which a
  headless process does not have. Nothing was implemented for it.

### Why refreshes fail, as far as anyone has said

- devkit#5 is open with no OpenAI reply. Four clients reproduce `invalid_grant` on never-used refresh tokens inside
  the window, two with `error_reason: refresh_token_invalidated`
  (pi 1.0.2 capture in the 2026-10-04 comments on devkit#5). Another reporter has a Pro account "with advanced security enabled" (2026-10-06).
- OpenAI staff (vignesh-oai, a devkit committer) on [pingdotgg/t3code#14356](https://github.com/pingdotgg/t3code/issues/14356),
  2026-10-04: "This happens if your account had a change in security settings, this includes if you recently did a
  password reset, or enrolled in AAS / Daybreak Blue."
- [earendil-works/pi#10377](https://github.com/earendil-works/pi/issues/10377): the reporter had removed Google as a
  login method; "logging out of all sessions via the OpenAI website resolved this issue for me" (2026-10-04).
- [openai/codex#39696](https://github.com/openai/codex/issues/39696): with Advanced Account Security (or after moving
  to passkeys), fresh refresh tokens come back `refresh_token_invalidated` seconds after login, on the Codex client too.
- Grant withdrawal shortly after registration (our 13-minute admission 403): nothing public explains it. The errors page
  says OpenAI does not notify apps when a user disconnects them. Pacing is our own precaution, not a documented rule.

### Changes

- `chatgpt_pacing.py` (new): concurrency 1, at least 10 s between request starts, at most 60 requests in any sliding
  hour (extra requests wait in arrival order, they never fail), `reasoning.effort` lowered to `medium` and set to
  `medium` when a request names none. The catalog call is paced too. All four come from `[chatgpt]` in
  `config.toml` (`max_concurrency`, `min_start_interval_s`, `max_requests_per_hour`, `max_reasoning_effort`).
- `refresh_schedule.py` (new): the running gateway refreshes the ChatGPT token at `max(earliest_refresh_at,
  expires_at - 300 s)` without waiting for a request. Tonight's window was six minutes (03:39:14Z to 03:45:15Z), so an
  idle stretch in the loop would otherwise have missed it. After a refusal it sends nothing until a new login record
  appears; transient failures retry with backoff (15 s doubling to 120 s). `proactive_refresh = false` turns it off.
- `refresher.py`: after `invalid_grant` the current access token is still used until it expires (it stays valid, per
  the devkit#5 reports), the dead refresh token is never sent again, and every attempt is written to
  `~/.bigbrane-gateway/refresh.jsonl` (outcome, seconds before expiry, HTTP status, `error`, `error_reason`,
  `error_description`, `x-request-id`; never a token). `due_at` is the single refresh-time rule shared with the
  scheduler and `status`.
- `chatgpt_auth.py`: token requests send the devkit's `accept: application/json`; a 200 refresh without a replacement
  refresh token no longer keeps the consumed one.
- `oauth_common.py`: refresh failures carry `error_reason` and the request id into the error message, the log and the
  journal.
- `status` and the serve log print the pacing policy, ChatGPT calls in the last hour (from `usage.jsonl`), the refresh
  window and due time, and the last refresh outcome.
- Tests: `test_siwc_durability.py` (34 new, fakes only). The suite went from 468 to 502 passed.

### Live

- Gateway restarted on the new code at 2026-10-11T03:00:42Z, PID 59956, log `results/gateway-20261011/serve3.log`
  (an earlier restart at 02:58Z, PID 55500, is in the same log). The old PID 47199 was stopped first.
- Checks after the restart: `chatgpt/gpt-6.1-sol` effort low, 200, "OK", 3.3 s; `anthropic/claude-haiku-5-5`, 200,
  "OK", 0.8 s.
- Token timing (login at 02:45:15Z): `earliest_refresh_at` 2026-10-11T03:39:14Z, `expires_at` 2026-10-11T03:45:15Z,
  proactive refresh scheduled for 03:40:15Z.
- Refresh result: REFRESH-RESULT-PLACEHOLDER

### Still uncertain

- Whether tonight's refresh succeeds depends on OpenAI's side. If it fails with `refresh_token_invalidated`, the
  reported fix is account-side: check ChatGPT Settings > Security for Advanced Account Security, passkeys, a recent
  password reset or a removed login method, use "Log out of all devices", then one fresh `login chatgpt` (same client).
  That needs Andrew at the browser.
- Whether pacing prevents another admission 403 is untested; OpenAI publishes no rule to check it against.

## Works (verified by tests with mocked HTTP and the real filesystem store in a workspace temp directory)

- OpenAI-compatible server on 127.0.0.1: `POST /v1/chat/completions` (non-streaming; `stream: true` replays the
  finished answer as SSE chunks), `GET /v1/models`, `GET /health`. Per-run bearer token, compared in constant time.
- Routing by model prefix: `anthropic/<model>` and `chatgpt/<model>`.
- Anthropic backend, ported from LLxprt Code: client id `9d1c250a-...`, authorize URL `claude.ai/oauth/authorize`,
  token URL `console.anthropic.com/v1/oauth/token`, scopes `org:create_api_key user:profile user:inference`, PKCE
  with the verifier reused as state, a loopback callback (ports 8765 to 8795) racing a paste of `code#state`, JSON code
  exchange, form-encoded refresh. Requests go to
  `https://api.anthropic.com/v1/messages` with `Authorization: Bearer`, `anthropic-version: 2023-06-01`,
  `anthropic-beta: oauth-2025-04-20, interleaved-thinking-2025-05-14` (plus `extended-cache-ttl-2025-04-11` for the
  1 hour cache breakpoints and `thinking-binding-controls-2026-08-01` where `block_binding` is sent),
  `anthropic-dangerous-direct-browser-access: true`, `User-Agent: claude-cli/2.1.293 (external, cli)`.
  The `system` field is only "You are Claude Code, Anthropic's official CLI for Claude."; the caller's system prompt
  is placed at the top of the conversation as a `<system>...</system>` user turn followed by
  "User provided conversation begins here:". Declared tools get the `llxprt_` prefix and response tool names are
  unprefixed. Ids are normalized to `toolu_...`.
- Thinking from `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`), mapped per model as
  LLxprt maps it: adaptive-capable ids get `thinking: {type: adaptive, display: summarized}` and `output_config:
  {effort}` (plus `block_binding` on Fable 5.1 and the 5.5 models), older Claude ids get budgeted thinking, `none` uses
  the model's thinking-off mode. Thinking blocks are removed from the reply. `CLAUDE-PORT.md` is the item-by-item
  comparison with LLxprt Code (headers, caching, sequence repair, retries, model rules).
- Model list: the 19 ids of LLxprt's claudecode alias, served as `anthropic/<id>`.
- Retries for Anthropic: 429, 5xx (529 included), in-stream overload errors and transport failures, six attempts with
  LLxprt's backoff, `Retry-After` honoured, plus the `anthropic-ratelimit-*` throttle.
- ChatGPT backend, from OpenAI's Sign in with ChatGPT documentation (details below): dynamic client registration,
  PKCE, loopback callback, ID token verification (RS256 against the issuer JWKS, pure Python), documented refresh with
  rotating refresh tokens, requests to `https://api.openai.com/v1/responses` with `store: false` and `stream: true`.
  Request shaping from LLxprt's openai-responses provider where it applies (`instructions`, `function_call` and
  `function_call_output` items with `call_` ids, `tool_choice`, `parallel_tool_calls`, dropping unpaired calls and
  outputs). Function tools are sent as one `additional_tools` input item (see Decisions), not as flat top-level tools.
- ChatGPT hardening (from `results/gateway-20261010/gpt-support-review.md` section 5, SIWC for supervised runs only):
  1. Error code: `detail.error_code` (and `error.error_code`, `detail.error.code`) is read as the code, so the
     403 `{"detail":{"error_code":"access_denied"}}` now logs and meters `access_denied` and shows it in the message.
     On every non-200 from api.openai.com the gateway keeps `x-request-id`, `x-openai-ide-root-error-code` and
     `x-openai-authorization-error`: in the log line, in the `usage.jsonl` `error` object (`request_id`,
     `root_error_code`, `authorization_error`) and in `/v1/models` `upstream_errors`. A 401 that is retried also logs them.
  2. After an admission 403 (HTTP 403 on `POST /v1/responses` or `GET /v1/models`) the ChatGPT provider is blocked for the
     process (`chatgpt_plan.PlanAccessGuard`). Later calls fail locally, with no request to OpenAI and no refresh, as 403
     `plan_access_denied` with "ChatGPT plan access denied; check https://chatgpt.com/settings/usage, then run login
     chatgpt". The block lifts only when the stored record's bytes differ from the refused record's and a record exists
     (a new login); deleting the record keeps it. Calls queued behind the refused one are checked again after the wait.
  3. `earliest_refresh_at` is enforced (`Refresher._window_open`): a record that carries it is refreshed only from that
     time on, and only when the access token is within the 300 s skew of expiry (or the API rejected it with 401). Before
     the window the current token is used. Anthropic records have no such field and behave as before.
  4. A ChatGPT-only semaphore in `ChatGPTBackend`: default 1, 2 at most, from `serve --chatgpt-concurrency` or
     `[chatgpt] max_concurrency` in `config.toml` (the flag wins; other values are refused). It wraps inference calls;
     the cheap model-list call is not queued behind long streams. Claude calls are not limited.
  5. `logout chatgpt` revokes first: it reads `revocation_endpoint` from the issuer's openid-configuration, POSTs `token`,
     `token_type_hint=refresh_token` and `client_id`, retries network failures and 5xx twice with backoff, then deletes
     the record under the provider lock. When revocation is not confirmed the record is still deleted and the command
     says so. `logout anthropic` is unchanged.
  6. Tools: see Decisions.
  7. Attribution: `login chatgpt` ends with "You're using your ChatGPT plan. Manage usage: https://chatgpt.com/settings/usage";
     `status` and the `serve` startup lines add "Using ChatGPT plan (Manage usage: ...)" after the chatgpt line when a
     ChatGPT login exists; ChatGPT 403 and 429 messages (HTTP and in-stream) start with "Manage usage: <link>.".
  8. `login chatgpt --new-registration` registers a new client with `dynamic_agent_client` under the same host id. It is
     documented in `LOGIN.md` as the answer to `invalid_client` only.
- Translation: messages, tools, tool results (parallel results merged into one Anthropic user turn), `tool_choice`,
  `response_format` json_schema (Anthropic through a forced tool, ChatGPT through `text.format`), usage with cached
  tokens split out, finish reasons. A response without usage fails the call and is recorded as an error.
- Single refresher: per-provider `asyncio.Lock` plus a cross-process `flock`, re-read inside both, refresh 300
  seconds ahead of expiry, new refresh token published (fsync, rename, directory fsync) before the new access token is
  used, `invalid_grant` family fails closed with the login command and is remembered so there is no retry loop, one
  forced refresh and one retry after an API 401.
- Token storage: private files `~/.bigbrane-gateway/oauth/anthropic.json` and `chatgpt.json` (mode 0600) in a mode
  0700 directory, with a stable `<provider>.lock` per provider. Owner, mode, file type and link count are checked on
  every read and write and a wrong state fails with the path named; symlinks are refused (`O_NOFOLLOW`); reads are
  bounded to 65536 bytes. Writes use a same-directory temp file, fsync, rename and directory fsync. A login publishes
  only if the record still equals the bytes it started from. Token responses without `expires_in` take the expiry
  from the access token's JWT `exp`; a response with neither is rejected. The pattern is the explicit filesystem
  OAuth of llxprt-code-rs PR 309. There is no fallback storage and no migration. No real token has been read or
  written by these tests.
- One gateway per machine: `flock` on `~/.bigbrane-gateway/gateway.pid`; a second `serve` exits with status 2.
- Meter: `usage.jsonl`, one line per call, success or failure.
- Redaction: every token the process loads or receives is registered; error messages and log lines are scrubbed.
- CLI: `login`, `logout`, `status`, `serve`. Before the storage change, `serve` and `status` were run once against an
  empty scratch state dir: auth, `/v1/models` with `not_logged_in`, the login-required error, and the second-gateway
  refusal behaved as designed. They have not been re-run by hand since; the in-process and second-process tests cover
  the same paths on the filesystem store.

## Tests

`.venv-hybrid/bin/python -m pytest hybrid/tests/gateway -q -p no:cacheprovider` gives 525 passed (391 before the
ChatGPT hardening, plus 77 in `test_chatgpt_hardening.py`, plus 34 in `test_siwc_durability.py` on 2026-10-11, plus 23
in `test_models_listing.py` on 2026-10-11).
`.venv-hybrid/bin/python -m pytest hybrid/tests/gateway hybrid/tests/llm -q -p no:cacheprovider` gives 581 passed, 4 skipped
(the llm live tests).

Live check, 2026-10-10 (logs in `results/gateway-20261011/`): a gateway on port 8765 answered Haiku 5.5, Sonnet 5.5
and Opus 5.5 calls with effort low, high, max and none, JSON schema (forced tool on Haiku, native structured output on
Sonnet and Opus), a system prompt, a tool call and its result, dropped and kept sampling parameters, and a
16269 token prompt whose second call read 16265 tokens from the cache.

Requested coverage, by file in `hybrid/tests/gateway/`:

| Requirement | Tests |
| --- | --- |
| Golden request translation, both providers, tools and json schema | `test_translate_anthropic.py`, `test_translate_responses.py`, goldens in `fixtures/golden/` |
| Usage extraction, cached tokens, missing usage | same files, `test_server.py` |
| 10 concurrent requests on an expired token cause one refresh | `test_refresh.py::test_ten_concurrent_...` |
| Rotation persisted before use; crash between persist and use; failed write (file fsync, rename, directory fsync) | `test_refresh.py` |
| Two gateway processes on one store cause one refresh; a refresh waits for a login holding the lock | `test_refresh.py` |
| Wrong mode, wrong owner (check functions and a store built for another uid), symlinked record, lock or directory, FIFO, hard link, oversized record | `test_store_and_singleton.py` |
| Cross-process lock excludes a second process; a cancelled waiter leaks nothing | `test_store_and_singleton.py` |
| Login snapshot conflict keeps the newer record | `test_store_and_singleton.py` |
| Expiry from `expires_in`, from JWT `exp`, and rejection when neither exists | `test_auth_flows.py` |
| `invalid_grant` has no retry loop | `test_refresh.py::test_invalid_grant_fails_closed_...` |
| Second gateway fails at startup | `test_store_and_singleton.py` (in process and in a second OS process) |
| Writes only inside its own state directory | `test_store_and_singleton.py::test_a_gateway_run_writes_only_inside_its_own_state_directory` |
| No module under `hybrid/` uses operating system credential store APIs, the `security` command or Security framework bindings | `test_no_keychain.py` |
| No token value in any response, log or output | `test_no_token_leaks.py` (sentinels, hostile upstream that echoes credentials) |
| Login flows | `test_auth_flows.py` (URL parameters, exchange bodies, ID token rejections, a real loopback callback) |
| `hybrid/llm` compatibility | `test_server.py::test_hybrid_llm_openai_compat_backend_can_use_the_gateway` |
| `reasoning_effort` to thinking: golden request per effort level, each Claude 5.5 model, effort normalization, thinking off per model, legacy budget, `block_binding`, invalid effort, `max_tokens` unchanged, adaptive id boundaries, interleaved-thinking beta | `test_anthropic_thinking.py`, goldens `fixtures/golden/anthropic_request_reasoning_{low,medium,high}.json` |
| Claude model registry and per-model rules (list, max tokens, sampling, prefill, effort support, forced-tool refusal) | `test_claude_models.py` |
| Prompt caching, beta headers, tool order, sequence repair, sampling drops, native structured output and schema rewrite | `test_claude_request.py`, goldens `anthropic_request_json_schema*.json` |
| Retries (429, 5xx, 529, stream errors, transport errors, `Retry-After`, six attempts, no retry on 4xx), rate-limit throttle, ChatGPT not retried | `test_claude_retry.py` |
| Claude login: loopback listener, callback and paste race, bad callback, port fallback, timeout, paste-only | `test_claude_login.py` |
| ChatGPT hardening: `error_code` shapes, request id and root error headers in log, meter and `/v1/models`, Claude errors unchanged | `test_chatgpt_hardening.py` section 1 |
| Admission 403 stops ChatGPT locally (no request, no refresh), queued calls, lift on a changed record, kept when the record is deleted, per process, Claude unaffected | `test_chatgpt_hardening.py` section 2 |
| `earliest_refresh_at`: no refresh before the window, refresh at and after it, 401 before the window, field formats | `test_chatgpt_hardening.py` section 3 |
| ChatGPT concurrency cap (default 1, 2, config file, flag, invalid values, Claude unlimited) | `test_chatgpt_hardening.py` section 4 |
| Revoking logout (request form, order before delete, retries, 4xx, bad discovery, unreadable or missing record) | `test_chatgpt_hardening.py` section 5 |
| `additional_tools` placement, `tool_choice`, replayed calls | `test_chatgpt_hardening.py` section 6, golden `responses_request_with_tools.json` |
| Attribution text, Manage usage lead on 403 and 429 | `test_chatgpt_hardening.py` section 7, `test_upstream_errors.py` |
| `--new-registration` | `test_chatgpt_hardening.py` section 8 |
| Thinking blocks removed from the reply (text and redacted), tool calls kept, usage from `output_tokens` | `test_anthropic_thinking.py` |

I checked three of these by removing the safeguard (lock, dead-token memory, scrubbing): each made a test fail, then
the code was restored.

## Not verified

- The loopback Claude login (callback race, `tty_prompt.py`) is covered by fakes and a real loopback socket, but no
  real browser login was made with it, and `tty_prompt.read_hidden` has no automated test because it needs a terminal.
  The earlier paste login produced the stored Claude record that live calls use.
- `F_FULLFSYNC` on directories and files was exercised by the tests on this machine's disk only. A filesystem that
  rejects it makes writes fail with a clear error instead of falling back to plain `fsync`.
- ChatGPT ID token verification and the loopback flow follow the documents, but real claims, JWKS entries and the
  registration page have not been seen. If verification rejects a real token, the message says which check failed.
  The authorization code is single use, so a rejected login means running the login again.
- `earliest_refresh_at` is enforced from the stored number, but the real value format was only seen once (a Unix
  timestamp, 360 s before expiry); ISO strings are accepted because OpenAI's devkit accepts them.
- The `additional_tools` request shape, the revocation request and the admission-403 stop have been tested against
  fakes only. No call was made to OpenAI (the stored grant is dead; it needs a new login). Whether the Responses API
  accepts `tool_choice` `required` or a named function when the tools arrive only as an `additional_tools` item is not
  documented for this route and is unverified.
- Multi-hour behavior of the refresh path is unchanged and still hits OpenAI's open devkit issue #5.
- Whether Anthropic accepts a `state` equal to the PKCE verifier from this client today: LLxprt does the same.

## Decisions and deviations

- ChatGPT needs no dashboard step. The documentation says the direct flow needs neither a client secret nor a partner
  API key, and the issued client id comes back from the browser registration. `config.toml` can carry a
  pre-issued `client_id` if OpenAI ever requires one.
- Tools on the ChatGPT route: the preview-limitations page lists two supported ways ("group function/custom tools in
  namespaces or supply them through `additional_tools` input items"). Both shapes are documented in OpenAI's
  tool-search guide and SDK types (`{"type": "namespace", "name", "description", "tools": [...]}` and
  `{"type": "additional_tools", "role": "developer", "tools": [...]}`). The gateway uses `additional_tools`, placed first
  in `input`, because it leaves `function_call` items identical in both directions (a namespace adds a `namespace`
  field to calls and replayed calls) and needs no namespace name. The SIWC page itself does not say whether a flat
  top-level function tool is rejected; it is no longer sent. `hybrid/llm` sends no tools today, so this has not run live.
  `tool_choice` is now sent only when the caller sets it (it used to default to `auto`), and `parallel_tool_calls: true`
  is still sent with tools.
- Only an HTTP 403 at admission stops ChatGPT. An in-stream `response.failed` with a 403-class code
  (`subscription_sharing_user_not_eligible`) is reported but does not set the stop.
- The stop compares stored record bytes, so any rewrite of the record lifts it, including a refresh by another process.
  Only one gateway runs per machine and a blocked gateway does not refresh, so in practice that is a login.
- The Codex-specific parts of LLxprt's Responses provider (`backend-api/codex`, `ChatGPT-Account-ID`, `originator`,
  WebSocket transport, stateful chaining) do not apply to the documented route and were not ported. The documentation
  forbids `previous_response_id` over HTTP, `max_output_tokens`, `temperature`, `top_p`, `user` and `metadata`.
  Sampling parameters therefore cannot be matched across the two providers on the ChatGPT side. They are dropped and
  listed in the meter line.
- Both upstreams are streamed and aggregated, so long generations do not hit non-streaming limits. `stream: true`
  from the client is answered after the upstream finishes, not incrementally.
- Anthropic `response_format` json_schema is a forced tool call (native structured output on Opus 5.5, Sonnet 5.5 and
  Fable 5.1, which refuse forced tools) and cannot be combined with other tools; that combination returns 400. The root
  schema type must be `object`.
- Tool-use blocks replayed from history keep unprefixed names while declared tools are prefixed, which is what LLxprt
  does (`unprefixToolName` on history, prefix in `convertToolsToAnthropic`).
- Default `max_tokens` for Anthropic when the caller sets none is the model's limit from LLxprt's claudecode alias
  (128000 for the 4.6+ and 5.x ids, 32000 Opus 4.5, 64000 Sonnet 4 and 4.5, 16000 Haiku 4.5); ids outside that list
  follow `getMaxTokensForModel`. Callers that care should pass `max_tokens`.
- Extended thinking follows LLxprt's legacy auto path for `reasoning.enabled` true with an effort, on adaptive-capable
  ids: `buildThinkingConfig` (`AnthropicRequestBuilder.ts` from line 381, adaptive branch at 408-412) and `buildLegacyAutoConfig`
  (`anthropic-reasoning-config.ts` lines 275-303), which picks the adaptive branch through `selectEnabledThinkingType`
  (lines 426-472). The gateway has no budget input, so older Claude ids use LLxprt's 10000 token default budget.
  Non-Claude ids drop `reasoning_effort` and report it under `dropped`. An effort value outside LLxprt's vocabulary is a
  400.
- Thinking blocks are removed from the reply. Their signatures are not kept. The chat format has no field that could
  carry them back on a later tool-use turn, and the gateway keeps no conversation state. Thinking tokens are part of
  `output_tokens` and are metered there, as LLxprt reads them (`AnthropicResponseParser.ts` `extractUsageMetadata`,
  line 104 on).
- `max_tokens` has no thinking headroom rule in LLxprt. It sends the caller's positive override or the model default
  unchanged (`AnthropicRequestPreparation.ts` `resolveRequestMaxTokens`, lines 156-170). The gateway does the same.
  Thinking tokens count against this cap, so roles that think should set `max_output_tokens`.
- Temperature, `top_p` and `top_k` are passed to Anthropic with thinking on, except on the models that refuse them
  (dropped and reported). LLxprt's list is the 5.5 models; the API also refuses them on the other 5.x ids and Opus 4.7
  and 4.8 (`CLAUDE-PORT.md`).
- Anthropic login races LLxprt's loopback callback against the paste prompt; `--paste-only` keeps the old behavior.
- Plan limits are not read or enforced, as requested (OpenAI offers no quota API for SIWC tokens). ChatGPT limit failures arrive as HTTP 429
  `subscription_sharing_usage_limit_exceeded` and are returned as such.
- No ASGI server was needed, nothing was installed, and no dependency beyond httpx was added.

## Left for Andrew or the next step

- Run the two logins and the curl checks in `LOGIN.md`.
- `hybrid/config/models.toml` now has `base_url = "http://127.0.0.1:8765/v1"` and
  `api_key_env = "BIGBRANE_GATEWAY_TOKEN"` for the `gateway` backend, and lists the gateway model ids in comments. Every
  role still routes through the llxprt bridge. Switching a role to `gateway` entries with `anthropic/...` and
  `chatgpt/...` model names is an edit Andrew makes after the logins work.
- `hybrid/llm` does not send `x-bigbrane-run-id`; once it does, `usage.jsonl` lines can be joined to ledger events by
  run. Until then the join is by timestamp and model. Pass `serve --meter runs/<id>/gateway-usage.jsonl` to keep one
  file per run.
- Controller containers cannot reach a gateway bound to 127.0.0.1 from inside a Podman VM. If the loop runs in a
  container, start `serve` with `--host` set to an address the VM can reach and keep the bearer token on the
  host-to-container path only. Not tested.
- The ChatGPT catalog call in `/v1/models` is not covered by a live check; Anthropic's list is LLxprt's fixed
  claudecode list (19 ids), which includes ids this account may not be served (`claude-sonnet-4` and
  `claude-sonnet-4-20250514` answered 404 on 2026-10-10).
