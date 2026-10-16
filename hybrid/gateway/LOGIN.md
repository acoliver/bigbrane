# Logging in to the gateway

Run everything below from `/Volumes/XS1000/acoliver/projects/bigbrane` on the Mac that will run the research
loop. Use the hybrid environment: `.venv-hybrid/bin/python`. Nothing here touches LLxprt's logins. The gateway has
its own grants, stored as private files in `~/.bigbrane-gateway/oauth/` (`anthropic.json` and `chatgpt.json`, mode
0600, in a directory with mode 0700). LLxprt's own login storage is never read or written. The gateway never triggers an operating system
password prompt, because it only reads and writes those files.

## 1. ChatGPT (Sign in with ChatGPT, ChatGPT plan usage)

```
.venv-hybrid/bin/python -m hybrid.gateway login chatgpt
```

1. The command starts a listener on `127.0.0.1:1455`, prints a URL and opens your browser. Port 1455 is also used
   by the Codex CLI login, so do not run a Codex or LLxprt ChatGPT login at the same moment. If the port is taken,
   put `callback_port = 1456` (any free port) under `[chatgpt]` in `~/.bigbrane-gateway/config.toml`.
2. Sign in to the ChatGPT account whose plan should pay for the runs (Plus or Pro).
3. The first time, ChatGPT asks you to name the app. Leave it as `Bigbrane` (you may edit it) and approve the
   request to use your ChatGPT plan. This is OpenAI's dynamic client registration: it needs no dashboard, client
   secret or API key. The page you land on says the login was received; close the tab.
4. The terminal prints `chatgpt: saved. Account <your email>, client oaiapp_...`, then the plan notice
   `You're using your ChatGPT plan. Manage usage: https://chatgpt.com/settings/usage`. The issued client id is stored
   with the tokens and reused for every later sign-in and refresh.

If the terminal says the grant lacks `chatgpt.tokens.use.direct`, you declined or the plan-usage permission is off.
Run the login again and approve it, or enable the app under ChatGPT Settings > Usage
(https://chatgpt.com/settings/usage). To undo the grant later, disconnect the app on that same page.

### Signing in again, and `--new-registration`

Sign in again with the same command. It reuses the saved issued client id and the saved host id
(`~/.bigbrane-gateway/host-id`), which is what OpenAI's docs ask for after a failed refresh, a sign-out or a revoked
session: all three leave the registered client in place.

```
.venv-hybrid/bin/python -m hybrid.gateway login chatgpt --new-registration
```

`--new-registration` ignores the saved and configured client id and registers a new client with
`dynamic_agent_client`. The host id stays the same. Use it only when OpenAI answers `invalid_client` (at the token
endpoint, or by refusing the saved client at the authorize step). Every registration adds another "Bigbrane" entry
under ChatGPT Settings, so do not use it for a 403, an `invalid_grant` or a dead session; none of those is fixed by a
new client. The old entry can be disconnected at https://chatgpt.com/settings/usage. The flag is accepted for
`login chatgpt` only.

Only if OpenAI later requires a pre-issued client id, put it in `~/.bigbrane-gateway/config.toml`:

```toml
[chatgpt]
client_id = "oaiapp_..."      # issued id; leave out to register a new client at login
agent_name_hint = "Bigbrane"
callback_port = 1455
max_concurrency = 1           # simultaneous ChatGPT calls through the gateway: 1 or 2
```

## 2. Anthropic (Claude Code subscription OAuth, LLxprt's flow)

```
.venv-hybrid/bin/python -m hybrid.gateway login anthropic
```

1. The command starts a listener on the first free `127.0.0.1` port from 8765 to 8795, prints a paste URL
   (`https://claude.ai/oauth/authorize?...`) and opens your browser on the same authorize URL with a
   `http://localhost:<port>/callback` redirect. A running gateway holds port 8765, so the login takes the next free
   port.
2. Sign in with the Claude account (Pro or Max) and approve. The browser lands on the listener's "Authorization
   finished" page and the terminal prints `anthropic: saved. Access token valid until ...`.
3. If the browser is on another machine, open the printed paste URL there instead. Anthropic shows a code of the form
   `code#state`; paste all of it at the terminal prompt (input is hidden) and press Enter. The browser callback and
   the paste prompt race; whichever answers first completes the login and the other is called off.
4. `login anthropic --paste-only` skips the listener and uses only the paste URL. The login also falls back to paste
   by itself when no port in the range is free. It gives up after 3 minutes without an answer.

This is the same flow LLxprt uses (same client id, scopes and request shape). PROVIDER-OAUTH.md records Anthropic's
published position on third-party use of subscription OAuth; you chose to proceed.

## 3. Check the logins

```
.venv-hybrid/bin/python -m hybrid.gateway status
```

Expected, with real dates:

```
anthropic: logged in, access token valid until 2026-10-10T08:12:44Z (479 min left); refresh token stored
chatgpt: logged in, access token valid until 2026-10-10T00:41:02Z (58 min left); account you@example.com, client oaiapp_..., refresh token stored
Using ChatGPT plan (Manage usage: https://chatgpt.com/settings/usage)
gateway: not running
```

The `Using ChatGPT plan` line appears after the chatgpt line whenever a ChatGPT login exists, and `serve` prints it at
startup with the other status lines. `status` never prints a token. An expired access token is normal; the gateway refreshes it on the next request.

## 4. Start the gateway and send one request

Terminal A:

```
.venv-hybrid/bin/python -m hybrid.gateway serve --port 8765
```

`--chatgpt-concurrency 1|2` caps simultaneous ChatGPT calls (default 1, or `max_concurrency` from `config.toml`;
anything else is refused). Claude calls are not limited by it. It prints the listening address, the path of the per-run bearer token file (`~/.bigbrane-gateway/run-token`, a new
token on every start unless `BIGBRANE_GATEWAY_TOKEN` is set) and the usage log path. A second `serve` on the same
machine exits at once with "another gateway already holds ...".

Terminal B, one line per provider:

```
curl -s http://127.0.0.1:8765/v1/chat/completions -H "Authorization: Bearer $(cat ~/.bigbrane-gateway/run-token)" -H "Content-Type: application/json" -d '{"model":"anthropic/claude-haiku-5-5","messages":[{"role":"user","content":"Reply with the single word: pong"}],"max_tokens":20}'
```

```
curl -s http://127.0.0.1:8765/v1/chat/completions -H "Authorization: Bearer $(cat ~/.bigbrane-gateway/run-token)" -H "Content-Type: application/json" -d '{"model":"chatgpt/gpt-6.1-sol","messages":[{"role":"user","content":"Reply with the single word: pong"}]}'
```

A good answer is a JSON `chat.completion` whose `choices[0].message.content` is `pong` and whose `usage` has
`prompt_tokens`, `completion_tokens` and `prompt_tokens_details.cached_tokens`. Model ids available to you:

```
curl -s http://127.0.0.1:8765/v1/models -H "Authorization: Bearer $(cat ~/.bigbrane-gateway/run-token)"
```

Each call adds a line to `~/.bigbrane-gateway/usage.jsonl` (provider, model, resolved model, input, output and cached
tokens, latency). To check it: `tail -n 2 ~/.bigbrane-gateway/usage.jsonl`.

## 5. Use it from hybrid/llm

`hybrid/config/models.toml` has the `gateway` backend, set to the port used above, and a complete `gateway`
routing table next to the `llxprt` one:

```toml
routing = "llxprt"          # the default until both logins are verified

[backends.gateway]
kind = "gateway"
base_url = "http://127.0.0.1:8765/v1"
api_key_env = "BIGBRANE_GATEWAY_TOKEN"
```

Run the research loop through the gateway with:

```
.venv-hybrid/bin/python -m hybrid.research run --routing gateway
```

`BIGBRANE_ROUTING=gateway` selects the same table wherever `models.toml` is loaded (the flag wins over the
variable). With either, `hybrid.research run` checks before anything is written that the gateway answers
`GET /v1/models` at `base_url` with the bearer token and serves every model id the gateway table routes to (`anthropic/claude-opus-5-5`, `anthropic/claude-sonnet-5-5`,
`anthropic/claude-haiku-5-5`, `chatgpt/gpt-6.1-sol`). If not, it exits with the exact `serve` or `login` command to
run. The token comes from `BIGBRANE_GATEWAY_TOKEN` when it is set, otherwise from `~/.bigbrane-gateway/run-token`,
which `serve` rewrites on every start. The file is read again on every request, so a restarted gateway needs no
export. If you do export the variable, it must match the running gateway: `export
BIGBRANE_GATEWAY_TOKEN=$(cat ~/.bigbrane-gateway/run-token)`. Every request carries `x-bigbrane-run-id` with the run
directory name, so `usage.jsonl` lines join to the run's ledger by `run_id`.

## When something fails

| Message | Meaning and fix |
| --- | --- |
| `<provider> login needed (...). Run: python -m hybrid.gateway login <provider>` | No stored login, or the refresh token was rejected (`invalid_grant`, revoked, expired, reused). The gateway does not retry; run the login command. |
| `the grant lacks chatgpt.tokens.use.direct` | ChatGPT plan usage was not granted. See the ChatGPT section above. |
| `cannot listen on 127.0.0.1:1455` | Another login or Codex process holds the port. Close it or set `callback_port`. |
| HTTP 429 `subscription_sharing_usage_limit_exceeded` | The ChatGPT plan or app limit is used up. The message starts with `Manage usage: https://chatgpt.com/settings/usage.` Check it there. |
| HTTP 403 from ChatGPT, for example `chatgpt returned HTTP 403 (access_denied)` | OpenAI refused the call at admission. The message starts with the Manage usage link, and the gateway log and `usage.jsonl` line carry the upstream `request_id`, `root_error_code` (`x-openai-ide-root-error-code`) and `authorization_error` for OpenAI support. Do not retry in a loop. |
| `ChatGPT plan access denied; check https://chatgpt.com/settings/usage, then run login chatgpt` (code `plan_access_denied`) | After one admission 403 the gateway stops calling OpenAI for ChatGPT until the end of the process. This message is produced locally, with no request to OpenAI. It clears when a new login record replaces the refused one (run `login chatgpt`; no restart needed). Deleting the record does not clear it. A restart also clears it, but a restart alone repeats the 403 if nothing changed in ChatGPT settings. |
| `another gateway already holds ...` | A gateway is already running (see `status`, last line). Stop it first. |
| `<path> has mode 0644, expected 0600` (or a wrong owner, a symlink, extra hard links) | The store refuses unsafe files and never repairs them. Fix the named path yourself (`chmod 600 <file>`, `chmod 700 ~/.bigbrane-gateway/oauth`) or remove the file and log in again. |
| `the <provider> record changed while this login was in progress` | A gateway refresh or another login published a newer record while you were in the browser. That record was kept. Run `login` again if you still need it. |
| `... is held by another process; waited 120s` | Another gateway or login holds the provider's lock file. Check `status`, stop the other process, retry. |
| `stored <provider> record is unreadable` | The record is not valid JSON. Run `logout <provider>` to discard it, then `login <provider>`. |

Remove a login from this machine with `.venv-hybrid/bin/python -m hybrid.gateway logout chatgpt` (or `anthropic`).

- `logout chatgpt` reads `revocation_endpoint` from `https://auth.openai.com/.well-known/openid-configuration`, POSTs the
  refresh token to it (`token`, `token_type_hint=refresh_token`, the issued `client_id`), and then deletes the local
  record. A network failure or 5xx is retried twice with backoff. If revocation cannot be confirmed the record is
  still deleted, as OpenAI's sign-out guidance says, and the command says so and points to
  https://chatgpt.com/settings/usage, where the app can be disconnected. Revoking does not delete the registered
  client, so the next `login chatgpt` reuses it.
- `logout anthropic` deletes the local record file only. Revoke the grant through the Claude account's connected apps.

## ChatGPT usage rules for this gateway

Sign in with ChatGPT is meant for interactive apps. Until OpenAI fixes refresh for SIWC clients
(openai/sign-in-with-chatgpt-devkit#5), use it for supervised runs only:

- One ChatGPT stream at a time, two at most. The gateway enforces the cap (`--chatgpt-concurrency`, `max_concurrency`).
- Set the Bigbrane app's weekly limit in ChatGPT settings and check https://chatgpt.com/settings/usage before a run.
- A person stays at the machine to sign in again when the session ends.
- No GPT roles through ChatGPT in overnight or unattended runs.
- The gateway refreshes a ChatGPT token only from the token's `earliest_refresh_at` on (and only when the access token
  is within 300 seconds of expiry). Before that time it keeps using the current token. Since 2026-10-11 the running
  gateway does this on its own at that moment, with or without traffic (see "SIWC durability" below).

## SIWC durability (2026-10-11)

Findings and sources are in `STATUS.md` under the same heading. In short: our refresh request matches OpenAI's
docs, the devkit, T3 Code, pi and Codegraff field for field; none of them has a working SIWC refresh in public
(openai/sign-in-with-chatgpt-devkit#5); OpenAI staff link `refresh_token_invalidated` to account security changes
(password reset, Advanced Account Security); no browserless recovery exists.

### Pacing settings

Defaults, all under `[chatgpt]` in `~/.bigbrane-gateway/config.toml`:

```toml
[chatgpt]
max_concurrency = 1              # 1 or 2 (also --chatgpt-concurrency)
min_start_interval_s = 10        # seconds between the starts of two ChatGPT requests
max_requests_per_hour = 60       # sliding hour; extra requests wait, they do not fail
max_reasoning_effort = "medium"  # requests above it (or with no effort) are sent at it; "" turns the cap off
proactive_refresh = true         # refresh as soon as the refresh window opens
```

A restart of `serve` picks up changes. `status` and the first lines of the serve log print the policy in force.

### Model list settings

`GET /v1/models` asks each provider for its list at the same time, and each list has its own time limit. Defaults,
under `[models]` in `~/.bigbrane-gateway/config.toml`:

```toml
[models]
upstream_timeout_s = 10    # a provider's list that has not arrived by then is replaced (see below)
cache_ttl_s = 600          # a successful list is served from memory this long; 0 asks upstream every time
```

A provider whose list times out or fails gets its last good list, else its built-in ids (Anthropic only), else nothing.
Its entry under `upstream_errors` says why, and the serve log gets one WARNING line with the duration.

### Watching a refresh

1. `.venv-hybrid/bin/python -m hybrid.gateway status` shows `earliest_refresh_at`, `expires_at`, when the refresh is
   due (`max(earliest_refresh_at, expires_at - 300 s)`) and the last refresh outcome.
2. The serve log announces `chatgpt: proactive refresh scheduled for ...`, then `refresh due ...; refreshing now`,
   then either `refreshed; new access token expires ...` or `the refresh token was refused`:
   `grep -E "chatgpt|oauth/token" results/gateway-20261011/serve3.log | tail`
3. `tail -1 ~/.bigbrane-gateway/refresh.jsonl` holds the outcome, seconds before expiry, HTTP status, `error`,
   `error_reason` and OpenAI's `request_id` (no tokens). Keep the request id for OpenAI support.

### When the refresh is refused

- The gateway keeps using the current access token until it expires, then answers `login_required`. It never resends
  the refused refresh token.
- `refresh_token_invalidated`: check ChatGPT Settings > Security (Advanced Account Security, passkeys, a recent password
  reset, a removed login method). One pi user fixed it with "Log out of all devices" before signing in again
  (earendil-works/pi#10377). Then run `login chatgpt` once, with the saved client (no `--new-registration`).
- Do not script repeated logins and do not call the token endpoint by hand: a refresh before `earliest_refresh_at` is
  refused, and resending a consumed refresh token counts as reuse.
