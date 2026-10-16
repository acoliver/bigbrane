# Provider OAuth for the Bigbrane model gateway

Date: 2026-10-09. Scope: read-only survey of LLxprt Code (TypeScript), llxprt-code-rs and serdes-ai for ChatGPT (Codex) and Anthropic subscription OAuth, and a recommendation for the Bigbrane model gateway described in `HYBRID-PLAN.md` sections 2 and 6. No login flows were run, no provider endpoints were called, no repository state was changed, and no token values were read. Stores were confirmed by path or keychain item name only.

## Verdict

Build the gateway in Python on the host (option a), as part of Bigbrane's P1 model gateway module, with its own logins and its own token store, and make the gateway process the only thing that ever refreshes those tokens. Port the flow and request-shaping logic from LLxprt Code TS, which has the most complete and most recently maintained implementation of both providers. Do not reuse the tokens LLxprt already holds in the keychain: those are shared by every concurrent LLxprt session on this machine, and a second refresher would race them.

Neither llxprt binary can serve as the gateway without new server work, and serdes-ai supplies request shaping but no token lifecycle, so a Rust gateway would rebuild the same storage, refresh and OpenAI-compatible translation layer in a second language next to a Python project.

The two providers differ on terms, and that difference needs a decision from you before P3:

- **ChatGPT plan.** OpenAI now documents a supported path, "Sign in with ChatGPT" with ChatGPT plan usage, for open-source and locally hosted personal apps. It uses dynamic client registration, PKCE, a host id and the public Responses API with `store: false` and `stream: true` ([overview](https://developers.openai.com/siwc/token-sharing-open-source)). The gateway should use that flow. The older approach that LLxprt and serdes-ai use (the Codex CLI's public client id against `chatgpt.com/backend-api/codex`) is the fallback if the documented flow does not fit.
- **Claude subscription.** Anthropic's Claude Code legal page says subscription OAuth "is designed to support ordinary use of Claude Code and other native Anthropic applications" and that developers "may not collect, store, or intermediate Claude.ai credentials or session tokens" ([legal and compliance](https://code.claude.com/docs/en/legal-and-compliance#authentication-and-credential-use)). The Register reported in February 2026 that Anthropic blocked third-party clients that present themselves as Claude Code ([report](https://www.theregister.com/software/2026/02/20/anthropic-clarifies-ban-on-third-party-tool-access-to-claude/5014546)). All three code bases surveyed do exactly that: Claude Code system prompt, `claude-cli` user agent, `oauth-2025-04-20` beta. The same page explicitly allows signing in to the unmodified `claude` binary, including headless. A Bigbrane Anthropic backend therefore has three choices: direct OAuth as LLxprt does it, the unmodified `claude -p` CLI as a subprocess backend (loses native tool calling and temperature control, which section 6 of the plan requires), or a Console API key. The gateway design below keeps the Anthropic backend behind the same interface so any of the three can be slotted in.

## Comparison

| | LLxprt Code TS | llxprt-code-rs | serdes-ai |
| --- | --- | --- | --- |
| Checkout surveyed | `/Volumes/XS1000/acoliver/projects/llxprt/branch-1/llxprt-code`, HEAD 0a5d995 (2026-10-04) | `/Users/acoliver/projects/llxprt/agent/rs-b3-190`, HEAD f03d3800 (2026-09-22, newest of the rs-* worktrees checked) | `/Volumes/XS1000/acoliver/projects/serdesAI/branch-1`, HEAD 0452ce7 (2026-09-07); `/Users/acoliver/projects/seredes` is older (2026-01-16) |
| License | Apache-2.0 (SPDX headers, Vybestack LLC) | Apache-2.0 | MIT |
| Codex login | Browser PKCE and device-code flow, refresh | None (reads TS's stored token) | PKCE helper with loopback server and refresh function in `serdes-ai-providers/src/oauth`; CLI keeps only the access token |
| Anthropic login | PKCE with paste-back code via `console.anthropic.com/oauth/code/callback`, refresh | None | Same PKCE helper; no app-level storage |
| Token store | macOS keychain service `llxprt-code-oauth`, accounts `codex:default`, `anthropic:default` (both present); encrypted file fallback; locks in `~/Library/Preferences/llxprt-code/oauth/locks` | Reads `llxprt-code-oauth` / `codex:default` read-only | None in the model crates; caller passes the access token |
| Refresh owner | Any LLxprt process, serialized by an O_EXCL lock file per provider and bucket with a re-read after the lock is taken | Never refreshes | Caller |
| Codex request | Responses API at `chatgpt.com/backend-api/codex`, HTTP SSE and WebSocket, `ChatGPT-Account-ID`, `originator: codex_cli_rs`, session id | Same endpoint via vendored `serdes-ai-responses`, `chatgpt-account-id`, `originator` | `chatgpt_oauth` model: fixed Codex instructions, user system prompt injected into the first user turn, `store: false`, streaming forced |
| Anthropic request | Messages API, bearer token, `anthropic-beta: oauth-2025-04-20,interleaved-thinking-2025-05-14`, `User-Agent: claude-cli/2.1.257 (external, cli)`, fixed Claude Code system string, tool name prefixing | Anthropic backend uses API-key `AnthropicModel`; no OAuth path | `claude_code_oauth` model: same beta pair, `claude-cli/2.0.61`, `x-app: cli`, fixed system string, `cp_` tool prefix |
| Tool calling | Yes, both providers | Yes (Codex Responses, Anthropic API key) | Yes, both |
| Structured output | No `json_schema`, `response_format` or `text.format` handling found in `openai-responses/` or `anthropic/`; the gateway must add it | Not surveyed in depth | No `text.format` or response format in the OAuth models |
| Usage metadata | Per-response usage; separate plan quota endpoints (`codexUsageInfo.ts`, `provider-usage-info.ts`) | Per-response usage | Maps `usage` from `response.completed` and Messages responses |
| Local OpenAI or Anthropic compatible server | No. `packages/a2a-server` speaks the agent protocol; headless `--prompt` runs the full agent | No. Headless agent CLI only | No |
| Tests | Extensive spec suites around token store, lock recovery, refresh coordination, Responses transport | Credential parsing and keychain timeout tests | 3 unit tests in the OAuth model modules; Responses rig and WebSocket fakes in `serdes-ai-models/tests` |

## Details

### LLxprt Code TS

Flows (`packages/auth/src/flows/`):

- `codex-device-flow.ts` (854 lines). Issuer `auth.openai.com`, scopes `openid profile email offline_access`. Browser PKCE (S256, verifier kept per state), authorization code exchange, refresh with `grant_type=refresh_token`, and a browserless device flow through `api/accounts/deviceauth/usercode` and `deviceauth/token`. The account id comes from the `https://api.openai.com/auth` claim of the id token.
- `anthropic-device-flow.ts` (284 lines). Authorize at `claude.ai/oauth/authorize`, token at `console.anthropic.com/v1/oauth/token`, scopes `org:create_api_key user:profile user:inference`, PKCE where the verifier doubles as state, redirect `console.anthropic.com/oauth/code/callback` and the user pastes the code back. Despite the file name there is no device-code grant on the Anthropic side.
- `types.ts` (106 lines) holds the `OAuthToken` schema (access token, refresh token, expiry, token type, optional scope and account fields).

Provider and manager layer (`packages/providers/src/auth/`): `codex-oauth-provider.ts` (821), `anthropic-oauth-provider.ts` (538), `oauth-provider-base.ts` (191), `oauth-manager.ts` (632), `token-access-coordinator.ts` (1,055, refresh lock with a time-of-check re-read), `proactive-renewal-manager.ts` (512), `provider-usage-info.ts` (340). Codex refresh keeps the old refresh token when the response omits a new one (`codex-oauth-provider.ts` around line 786).

Storage: `packages/auth/src/keyring-token-store.ts` (815 lines) over `packages/storage/src/secure-store/secure-store.ts`, wired in `packages/core/src/auth-factories.ts`. Accounts are keyed `provider:bucket`; locks are `<provider>[-<bucket>]-refresh.lock` and `-auth.lock` files in `Storage.getOAuthLocksDir()`, which on this machine is `~/Library/Preferences/llxprt-code/oauth/locks`. Keychain items `llxprt-code-oauth` / `codex:default` and `llxprt-code-oauth` / `anthropic:default` both exist. `packages/auth/src/proxy/` (socket client, proxy token store) lets a sandboxed LLxprt get tokens from the host over a Unix socket, which is the same separation Bigbrane wants between controller containers and credentials.

Reuse by another process: possible in principle (read the keychain item, take the same lock file before refreshing), but the lock has a fenced dead-owner takeover protocol that a second implementation would have to match exactly, and the tokens are shared with every LLxprt session you run. A Bigbrane refresh that skipped or mis-implemented the lock could rotate the refresh token out from under those sessions. A separate grant avoids the problem.

Codex request shape: `packages/providers/src/openai-responses/` (`OpenAIResponsesProvider.ts`, `openAIResponsesHttpStream.ts`, WebSocket transport). Headers: bearer, `ChatGPT-Account-ID`, `originator: codex_cli_rs`, a session id from the runtime. Plan quota and rate-limit reset come from `packages/providers/src/openai/codexUsageInfo.ts` (`chatgpt.com/backend-api/wham/usage`).

Anthropic request shape: `packages/providers/src/anthropic/AnthropicApiExecution.ts` adds `oauth-2025-04-20` and `interleaved-thinking-2025-05-14` betas and the `claude-cli/2.1.257` user agent (the code comment says some subscription models reject older client versions). `AnthropicRequestBuilder.ts` replaces the system prompt with "You are Claude Code, Anthropic's official CLI for Claude." under OAuth, and `AnthropicMessageNormalizer.ts` prefixes and strips tool names. Credential choice is in `AnthropicCredentialResolution.ts` (46) and endpoint handling in `AnthropicEndpointUtils.ts` (107); the provider itself is 1,057 lines.

### llxprt-code-rs

Newest worktree checked: `rs-b3-190` (2026-09-22); `rs-b3-206` from the task brief is older (2026-09-07).

- `src/model_api/macos_keychain.rs` (157 lines) reads service `llxprt-code-oauth`, account `codex:default` through `security-framework`, with a 10 second bound so a headless run fails with a remediation message instead of hanging on a keychain prompt.
- `src/model_api/credentials.rs` (364 lines) parses the TS token JSON, keeps only the access token and account id, validates header bytes, and treats `refresh_token` and `id_token` as fields to validate and drop. Expired tokens produce "sign in again with LLxprt Code". There is no refresh code.
- Codex endpoint `https://chatgpt.com/backend-api/codex/responses` (`src/model_api/settings.rs`), headers set in `src/model_api/registry.rs`. The Anthropic backend (`src/model_api/anthropic_backend.rs`, 77 lines) wraps the API-key `AnthropicModel`; there is no Anthropic OAuth.
- Vendored serdes-ai crates under `vendor/` include `serdes-ai-responses`, which is newer than the standalone serdes-ai checkout's model code.

As a gateway: it depends on LLxprt TS to log in and refresh, it is Codex only, and it has no HTTP server.

### serdes-ai

Current checkout: `/Volumes/XS1000/acoliver/projects/serdesAI/branch-1` (origin `git@github.com:acoliver/serdesAI.git`, branch `feature/issue-65-open-responses`). There is no `sse.rs` in `chatgpt_oauth`; SSE parsing lives in `model.rs`.

- `serdes-ai-models/src/chatgpt_oauth/` (mod 34, model 841, types 177, plus a 310-line `codex_system_prompt.md`). Takes an access token and optional account id in the constructor; the module docs say the crate does not store tokens. Base URL `chatgpt.com/backend-api/codex`, headers `originator: codex_cli_rs`, `User-Agent: codex_cli_rs/0.72.0`, `ChatGPT-Account-Id`. Sends the bundled Codex prompt as `instructions` and puts the caller's system prompt into the first user message with a "You MUST ignore the system prompt" preface. `request_stream` buffers a full response and replays it (a TODO notes real streaming is missing).
- `serdes-ai-models/src/claude_code_oauth/` (mod 34, model 642, stream 136, types 218). Bearer token, `anthropic-beta: oauth-2025-04-20,interleaved-thinking-2025-05-14`, `x-app: cli`, `User-Agent: claude-cli/2.0.61 (external, cli)`, `POST /v1/messages?beta=true`, fixed Claude Code system string with user system prompts moved into the first user turn, `cp_` tool name prefix added and stripped.
- Login and refresh: `serdes-ai-providers/src/oauth/` (config 139, context 80, flow 241, server 194) implements PKCE with a loopback callback server and a `refresh_token` function for both providers. `serdes-ai-cli/src/oauth.rs` (269) saves only the access token into its config, so refresh is not usable from that CLI.

serdes-ai expects the caller to own login, storage and refresh.

## Options weighed

| Criterion | (a) Python gateway, own grants | (b) llxprt binary as gateway | (c) Rust gateway on serdes-ai | (d1) `claude -p` subprocess backend |
| --- | --- | --- | --- | --- |
| Single refresh owner | Yes: one host process, one lock per provider, separate grant | No: tokens shared with all LLxprt sessions | Yes, once storage and refresh are written | The CLI refreshes its own login |
| Credentials out of agent containers | Yes: host process; containers get a per-run gateway token | Yes if run on host | Yes | Yes |
| Usage metering for the ledger | Direct: gateway parses `usage` from each response | Hard: agent output, not raw model calls | Direct, after writing it | `--output-format json` reports usage; no per-turn tool calls |
| Native tool calling and JSON schema (plan section 6) | Yes, translated to OpenAI chat format | No passthrough | Partial: no structured output in OAuth models | No native tool passthrough, no temperature |
| New code | Gateway server, translation, two backends, store | A new OpenAI-compatible server inside LLxprt plus metering | Gateway server, translation, store, metering, and fixes to the OAuth models | Subprocess adapter |
| Fits Bigbrane | Same language, same tests, same ledger code | Second runtime and dependency on another repo's release cycle | Second language and build chain on the host | Anthropic-only fallback |

(b) fails the first criterion and the tool-calling probe. (c) reuses the request mapping, but that mapping is the smaller part of the gateway: the server, OpenAI chat to Responses or Messages translation, token store, refresh scheduling and metering would all be new, and the serdes-ai OAuth models would need streaming, structured output and system-prompt handling changes first.

## Chosen design

Module `bigbrane/gateway/` in the Bigbrane repo, running on the host, bound to the port the controller container is allowed to reach.

| Module | Responsibility |
| --- | --- |
| `server.py` | OpenAI-compatible `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models`; authenticates the per-run controller token; never returns provider tokens or headers |
| `translate_openai_responses.py` | Chat Completions to Responses input (messages, tools, tool results, `response_format` to `text.format`), and Responses events back to chat chunks with tool calls |
| `translate_anthropic.py` | Chat Completions to Messages (system, tool_use and tool_result blocks, JSON schema via a forced tool), and back |
| `backends/chatgpt.py` | Responses request: Sign in with ChatGPT access token against the documented endpoint, or the Codex backend fallback (`chatgpt.com/backend-api/codex/responses`, `ChatGPT-Account-ID`, `originator`) |
| `backends/anthropic_oauth.py` | Messages request with bearer and the OAuth beta header set; selected only if you choose direct OAuth |
| `backends/claude_cli.py` | Optional: unmodified `claude -p --output-format json` subprocess, text and schema output only |
| `auth/flows.py` | PKCE login per provider (loopback callback for ChatGPT; paste-back code for Anthropic), run interactively by you on the host, never by a controller |
| `auth/store.py` | Keychain items under a Bigbrane service name (for example `bigbrane-gateway` / `chatgpt:default`), separate from `llxprt-code-oauth` |
| `auth/refresher.py` | The only code that calls a token endpoint for refresh; one `asyncio.Lock` per provider; refresh ahead of expiry; persist the new refresh token before using the new access token; on `invalid_grant` fail the request and require a new login |
| `auth/singleton.py` | `flock` on a host pidfile so a second gateway process exits at startup |
| `meter.py` | Writes model id, resolved model id, input, cached input and output tokens, latency and run id to the ledger for every call |
| `replay.py` | Scripted responses for P1 and P2 (already in the plan) |

Acceptance tests (pytest, no live provider calls except the marked probe):

1. Translation round trip: a chat request with two tools and a tool result maps to the expected Responses and Messages bodies (golden files), and a recorded SSE stream with a function call maps back to an OpenAI tool-call chunk sequence.
2. `response_format` with a JSON schema reaches the Responses `text.format` field; on Anthropic it becomes a forced tool and the reply is the tool input.
3. Usage from `response.completed` and from Messages `usage` lands in the ledger with cached tokens split out; a response without usage is recorded as an error event and the call fails.
4. Ten concurrent requests against an expired token with a fake token endpoint cause exactly one refresh call, and all ten use the new token.
5. A fake token endpoint that rotates the refresh token: the new refresh token is in the store before the first request uses the new access token; a crash injected between the two leaves a usable store.
6. `invalid_grant` on refresh returns a gateway error naming the login command; no retry loop.
7. A second gateway process fails at startup on the pidfile lock.
8. The controller container cannot reach provider hosts, and no response body, log line or error from the gateway contains a provider token (scan with planted sentinel tokens in the fake store).
9. The gateway never reads or writes `llxprt-code-oauth` keychain items (store test with a fake keychain asserting the service name).
10. Live capability probe at P3 entry (the section 6 table) against each backend you enable, run by you.

## Open questions for you

1. **Anthropic route.** Given Anthropic's published position, do you want direct subscription OAuth in the gateway, the unmodified `claude -p` subprocess (no native tool calls or temperature, so it fails two section 6 probes), a Console API key, or ChatGPT only for this study?
2. **ChatGPT route.** Are you willing to sign in through OpenAI's "Sign in with ChatGPT" plan-usage flow as a new registered client for Bigbrane? If it does not work for your plan, is the Codex CLI client id fallback acceptable?
3. **Separate grants.** Confirm that Bigbrane should run its own logins instead of sharing the `codex:default` and `anthropic:default` items LLxprt uses. Sharing would put Bigbrane's refreshes and your LLxprt sessions on one rotating refresh token.
4. **Plan quota.** Research runs draw from the same plan limits as your interactive LLxprt sessions. Should the gateway read the plan usage endpoints (as `codexUsageInfo.ts` does) and stop a run before it exhausts the window?
5. **Temperature.** The Codex backend and reasoning models may ignore or reject temperature, which section 6 requires. Is a fixed reasoning effort an acceptable substitute for matched sampling settings if the probe confirms that?
