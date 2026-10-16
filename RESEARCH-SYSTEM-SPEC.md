# Bigbrane research system: prototype spec

Started 2026-10-09 20:08 -03. Prototype deadline 2026-10-10 20:08 -03.

## Goal

A system that is useful for finding a new mathematical theory of sequence learning and a small model built on it. The long-term target: no quadratic cost in context length (beyond partitioning or windowing tricks), and few-shot or conceptual learning that needs far less data than Chinchilla-style scaling. The system itself does not have to find that model by the deadline. It must run as a loop with parallel explorers that:

1. propose theories from many source fields (physics, biology, dynamical systems, information theory, algebra, and others), drawing on real papers;
2. separate prior art from new work, and keep under-tested prior art as worth exploring;
3. reject math salad and reject disguised known architectures (an RNN, linear attention, an SSM, a transformer variant, BitNet, MoE, partitioned attention, LCM-style concept models);
4. reach a mathematical feasibility verdict, or a proof of infeasibility, before any code is built;
5. only then build a minimal model and test it against baselines with trusted evaluation;
6. record everything so a person can inspect every decision.

Lesson from the October OpenAI math episode: one-shot generation is hit and miss, and the stated result can differ from what the proof or code actually does. The system must do both: generate boldly, then verify independently. Every claim is checked against what was actually computed or proved, by a different model family where judgment is involved.

## Layout

All new code lives under `hybrid/`. Do not modify `bigbrane/`, `tests/`, `EXPERIMENT.md`, `README.md` or old `results/` (Experiment 1's frozen-source checks compare against the live tree). The hybrid system uses its own virtual environment `.venv-hybrid` (Python 3.12), so the frozen Experiment 1 environment hash stays valid. Each subpackage has tests under `hybrid/tests/<subpackage>/`.

```
hybrid/
  executor/      existing P0 Podman executor (Job, Caps, Executor.run -> Receipt)
  llm/           model access: backends, role router, budgets, call log
  research/      store, ledger, theory cards, stages, orchestrator, inspect/report
  bench/         trusted tasks with exact oracles, baselines, evaluator, scaling probe
  literature/    paper search clients, cache, receipts, seed reading list
  gateway/       native OAuth gateway (built later; llm/ uses llxprt until then)
  config/        models.toml, lenses.toml, budgets.toml
runs/<run-id>/   all run state (gitignored-style; never /tmp)
```

## Shared conventions

- Python 3.12, standard library first. Allowed extra deps in `.venv-hybrid`: numpy, torch (CPU), sympy, httpx, jsonschema, jinja2, pytest. Ask before adding others.
- IDs: `card-<8 hex>`, `call-<12 hex>`, `job-<12 hex>`, `run-<UTC yyyymmddThhmmssZ>`.
- Times: UTC ISO 8601 in records.
- Fail fast on internal invariant violations. Handle errors only at real external boundaries (network, subprocess, model output parsing), and record those failures as events, never drop them.
- Every file written under a run directory is content-hashed (sha256) in the ledger.

## 1. `hybrid/llm` (model access)

```python
@dataclass(frozen=True)
class LLMResult:
    call_id: str
    role: str
    backend: str          # "llxprt" | "openai_compat" | "anthropic_messages" | "gateway"
    model: str            # resolved model id
    text: str             # final text
    parsed: dict | None   # set when json_schema given and parsing+validation succeeded
    usage: dict           # {"input": int, "output": int, "cached": int} best effort
    latency_s: float
    transcript_path: str  # file with full prompt+response

class LLMClient:
    def __init__(self, config_path: str, run_dir: str, ledger: "Ledger"): ...
    async def complete(self, role: str, system: str, user: str, *,
                       json_schema: dict | None = None,
                       allow_web: bool = False,
                       max_output_tokens: int | None = None) -> LLMResult: ...
```

- Roles map to an ordered list of backend+model entries in `hybrid/config/models.toml`; the first available entry is used, falling back on transport failure (recorded). Per-backend concurrency semaphores and per-run budgets (calls, tokens) live in the same file. Exceeding a budget raises `BudgetExceeded`.
- When `json_schema` is given: instruct JSON output, extract the first JSON object, validate with jsonschema, and retry once with the validation error appended. Both attempts are logged.
- Backends:
  - `llxprt`: subprocess `llxprt --profile-load <profile> --output-format json` with the prompt on stdin (system + user concatenated with clear headers), cwd = a fresh scratch dir under `runs/<run>/llm/scratch/<call-id>/`. Parse `response` and `stats.models.*.tokens`. For `allow_web=False`, disable all tools via `--profile` inline JSON or settings so the call is pure reasoning; for `allow_web=True`, allow only web search/fetch tools, never shell or file writes. Timeout per call from config.
  - `openai_compat`: POST `{base_url}/chat/completions`; `api_key_env` optional (no key for local servers). LM Studio is running at `http://127.0.0.1:1234/v1` for tests.
  - `anthropic_messages`: POST `{base_url}/v1/messages` with `x-api-key` from env.
  - `gateway`: placeholder entry pointing at the future native gateway (OpenAI-compatible, `http://127.0.0.1:<port>/v1`).
- Every call appends an `llm_call` event to the ledger (role, backend, model, prompt sha256, response sha256, usage, latency, transcript path, error if any).

Default role routing (prototype; adjust from measurements):

| Role | First choice | Fallback |
| --- | --- | --- |
| explorer | llxprt:opus | llxprt:gpt-sol-high |
| explorer_alt (diversity) | llxprt:gpt-sol-high | llxprt:opus |
| formalizer | llxprt:sonnet | llxprt:opus |
| salad_critic | llxprt:haiku | llxprt:sonnet |
| reduction_critic | llxprt:opus | llxprt:sonnet |
| prior_art | llxprt:haiku (allow_web) | llxprt:sonnet |
| feasibility_judge_a | llxprt:opus | llxprt:sonnet |
| feasibility_judge_b | llxprt:gpt-sol-high | llxprt:terra |
| disprover | llxprt:sonnet | llxprt:opus |
| builder | llxprt:sonnet | llxprt:haiku |
| analyst | llxprt:haiku | llxprt:sonnet |

## 2. `hybrid/research` store and ledger

- `Ledger(run_dir)`: append-only `ledger.jsonl`; each event `{"seq", "ts", "type", "actor", "data", "prev_hash", "hash"}` where `hash = sha256(prev_hash + canonical_json(event_without_hash))`. `verify()` recomputes the chain. Single writer per run (file lock).
- `Store(run_dir, ledger)`: writes cards and artifacts; every write emits an event with the content hash.
- Theory card JSON schema (`hybrid/research/schemas/theory_card.schema.json`), versioned `cards/<card-id>/v<N>.json`:

```
id, version, parent_ids[], created_by_call, lens, title, summary
mechanism:
  state: definitions of every object (name, type, shape, meaning)
  update: equations for processing one token / one step (LaTeX plus a plain-text form)
  readout: how predictions are produced
  learning: how parameters or structure are learned; what makes few-shot/conceptual learning possible
claims:
  complexity: {time_per_token: "O(...) in n and d", memory: "O(...)", total_sequence: "O(...)", constants_notes}
  sample_efficiency: {mechanism, prediction: numeric, measurable statement}
  capabilities: [{task, prediction}]   # e.g. associative recall of K pairs with state size d
known_relatives: [{name, relation, claimed_difference}]
falsifiers: [concrete observable results that would refute the theory]
toy_example: {description, python: "self-contained code computing a tiny instance", expected: {...}}
references: [{source_id, why}]          # from literature receipts
status: {stage, verdicts: {stage: {verdict, reasons, by_calls[]}}}
```

- Stage verdict values: `pass`, `fail`, `revise`, `infeasible_proved`, `reduces_to_known`, `undetermined`.
- `python -m hybrid.research inspect <run-dir>`: prints a summary and builds `report/index.html` (static, jinja2): run overview (time, calls, tokens per model, latency), card table with stage reached and why, lineage tree, per-card dossier (all versions, every critic transcript linked, literature receipts, feasibility arguments and disproof attempts, executor receipts, bench results vs baselines), ledger verification status.

## 3. Stages and gates

Each card moves through stages in order. A gate failure ends that card's line unless the gate allows one revision. Every gate writes its verdict and the calls behind it.

| Stage | What happens | Gate |
| --- | --- | --- |
| S0 explore | Explorer gets a lens, a sample of 3 to 6 papers from the literature cache, and the card summaries already in the run (to avoid repeats). Operators: new idea, mutate a passing card, cross two cards. | Card validates against schema. |
| S1 well-formed | Salad critic checks every symbol is defined, dimensions agree, claims are falsifiable and numeric. The toy example runs in the executor and must reproduce its `expected` values. | Both pass. One revision allowed. |
| S2 prior art and reduction | Literature search with recorded queries; classify as `novel`, `prior_art_tested`, `prior_art_undertested`, `known_disproven`. Reduction critic tries to write the card's update rule as an RNN, linear attention, SSM, Hopfield/kernel memory, transformer variant, MoE, BitNet, partitioned attention or LCM. A claimed reduction must come with an explicit mapping; when possible a numeric equivalence check runs in the executor. Also checks the "radical enough" rule. | `known_disproven` fails. A verified reduction gives `reduces_to_known` and fails unless the difference is in the learning rule and is stated as such. `prior_art_tested` fails unless the card adds a specific untested element. `prior_art_undertested` and `novel` pass. |
| S3 feasibility | Formalizer writes proof obligations: complexity derivation, an information or capacity bound for each capability claim (fixed-size state cannot recall arbitrarily many items; the card must address known lower bounds), and a learnability or sample-efficiency argument. Disprover tries to break each obligation with counterexamples, and small numeric simulations run in the executor. Two judges from different model families rule independently. | `pass` needs both judges to rule feasible with no surviving disproof. A surviving disproof gives `infeasible_proved`, which is recorded as a result. A split vote gets one more round, then `undetermined`. |
| S4 construct | Builder writes a minimal implementation of the Candidate API (section 4) from the card. Static checks: forbidden modules for the claimed class (for example no `nn.MultiheadAttention` or `nn.Transformer*` in a non-attention claim), causality test, parameter and state counts. | All checks pass and code runs in the executor. Two repair attempts allowed. |
| S5 evaluate | Trusted bench runs the candidate and baselines on oracle tasks in the executor. The scaling probe measures time and memory against context length. | Results recorded. No pass/fail here. |
| S6 verdict | Analyst compares each claim with measured results: supported, refuted or untested, using numbers from the bench only. Spawns mutations for promising cards. | Claims ledger written. |

Efficiency rule: nothing reaches S4 without passing S3. The scheduler prefers cards that are cheap to check and spends the most expensive models only on cards that survived the cheaper gates.

## 4. `hybrid/bench` (trusted)

Candidate API (written by the builder, run inside the executor):

```python
class Candidate:
    def __init__(self, vocab_size: int, seed: int, budget: dict): ...
    def fit(self, train: list[list[int]]) -> None: ...
    def predict_proba(self, prefixes: list[list[int]]) -> "np.ndarray":  # shape (B, vocab), causal
    def describe(self) -> dict:   # {"params": int, "state_floats_per_sequence": int, "notes": str}
```

- Tasks, each with an exact oracle or exact answer: HMM (reuse `bigbrane/core.py` math by copying, not importing), variable-order Markov, associative recall (K key-value pairs, then queries), hierarchical composition, few-shot rule induction (a new rule shown in context, then tested). Each task has train sizes for sample-efficiency curves.
- Baselines: n-gram, GRU, tiny transformer, a diagonal SSM. Same budgets.
- The candidate job only receives train data and test prefixes; it writes predictions. The evaluator, in a separate job, reads predictions plus oracle data and computes KL to oracle, accuracy, and tokens-to-target. The candidate never sees oracle files.
- Scaling probe: time and peak memory of `predict_proba` for context lengths 2^k, fit of the slope on log-log, compared to the card's complexity claim.
- Output: `bench_result.json` per candidate with per-task metrics, baseline metrics, scaling slopes, receipts.

## 5. `hybrid/literature`

- Clients: arXiv export API, OpenAlex, Semantic Scholar (no keys; respect rate limits; cache responses under `hybrid/literature/cache/`).
- `search(query) -> list[Paper]` with receipts (query, source, time, result ids) written to the ledger.
- `seed_reading_list.toml`: 60 to 120 papers across lenses (reservoir computing, vector symbolic and hyperdimensional computing, holographic reduced representations, modern Hopfield networks, predictive coding, active inference, dendritic and spiking computation, neural ODEs and Koopman operators, tensor networks and matrix product states for language, renormalization group and multiscale methods, kernel machines and Gaussian processes, Bayesian program induction, cellular automata, optimal transport, sheaves and category-theoretic models, compression and minimum description length, algorithmic information, sparse distributed memory, complementary learning systems, hyperbolic geometry, quantum-inspired models, chemical reaction networks, morphogenesis). Each entry: title, year, id, lens, one-line relevance. Built by searching, never invented.
- `classify_prior_art(card) -> {label, evidence[]}` combining search receipts with the prior_art role model.

## 6. Orchestrator

- `python -m hybrid.research run --config hybrid/config/run.toml --run-dir runs/<id>` starts the loop; `--resume` continues a run.
- asyncio. N explorers in parallel (default 4), each with a different lens; stage workers pull from per-stage queues; per-role concurrency follows `models.toml`.
- Diversity: before accepting a new card, compare its summary and mechanism with existing cards (TF-IDF cosine on text plus the critic's judgment); near-duplicates are recorded and dropped.
- Budgets: total calls, tokens and wall time per run; stage budgets; stop cleanly when exhausted.
- Everything recorded in the ledger. A crash loses at most the in-flight call.

## 7. Native gateway (later)

Claude Code OAuth following LLxprt Code patterns, Sign in with ChatGPT (not the Codex CLI client), OpenAI-compatible and Anthropic endpoints with API keys, local endpoints with no key. Own keychain items, single refresher. When it works, `models.toml` switches roles from `llxprt` to `gateway`.

## 8. Podman rules

Label every container, image, volume and pod `bigbrane.owner=bigbrane` and `bigbrane.run=<run-id>`. Run `scripts/podman-teardown.sh` at the end of each session. Never prune unlabeled resources. Stop the VM when no run is active.
