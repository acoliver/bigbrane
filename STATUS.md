# Bigbrane status, 2026-10-16

Bigbrane is an automated research lab for sequence learners outside the transformer and state-space families. The target is a learner that gets language-level skills from roughly a child's amount of input at well under quadratic compute. The system can run end to end, but it cannot yet tell good ideas from bad ones reliably, and no discovered model has beaten the baselines.

`README.md` describes Experiment 1, an earlier frozen study comparing a learned HMM with a small RNN and an attention model (result in `RESULT.md`). The current design is `DISCOVERY-PIPELINE-PLAN.md`; decisions taken along the way are rows in `DECISIONS.md`.

## How it works

1. **Ideas.** Explorer models write theory cards (`hybrid/research`, stages S0 to S2). A prior-art stage reduces each card against known families and kills rediscoveries such as linear attention, the delta rule and Hopfield memories.
2. **Math (stages M and C).** GPT writes the formal statements. Claude writes proofs and check programs. A local Qwen in LM Studio checks that each statement matches its claim, and Sonnet cross-checks. Lean 4 with Mathlib, Z3, SymPy and NumPy run in network-less Podman containers (`hybrid/executor`, `hybrid/math`). A challenger model then attacks the result at stage C.
3. **The credibility profile.** A card must show what a transformer can show about itself:
   - P1: exact cost;
   - P2: constructions for copy, recall, nesting and composition;
   - P3: a derived limit;
   - P4: learnability on toy tasks;
   - P5: information bounds.
4. **Bench.** Survivors are built and trained against tuned baselines (`hybrid/bench`).
5. **Infrastructure.**
   - Model access goes through a local gateway on port 8765 (`hybrid/gateway`) that holds the Claude and ChatGPT sign-ins. Tokens are stored outside the repo.
   - Every run writes a hash-chained ledger under `runs/`, which is not checked in.

## Where things stand

**The plumbing works.** Processes, the gateway, the Podman sandbox and the ledgers hold up under 8-hour runs. Ledger and artifact hashes re-verify.

**Calibration does not pass, and my scoring hid part of why.**
- Calibration runs the math stages on four disguised known models: K1 transformer, K2 LSTM, K3 HMM and K4 n-gram. Each must reproduce an answer key 3 times.
- I tracked agreement on the seven P1/P2/P3/P5 cells, and it climbed from 74/84 to 83/84 over fixes FIX-8 to FIX-21.
- The answer key also requires K1 and K2 to pass the whole profile, P4 included. They never did.
- Run 20261016T051933Z shows the gap:
  - K1 composition: 0.565 to 0.62 accuracy against a 0.99 threshold, in all three runs;
  - K2 recall: 0.82 to 0.99 across seeds;
  - K2 composition: 0.66 to 0.998 across seeds.
- Rescored with profile and target included (scripts/score_calibration.py uses the acceptance items): 20261015T200147Z, reported as 83/84, is 101/108, and every K1 and K2 run fails the profile. 20261016T051933Z is 99/108.
- The system's own report records "profile: expected pass, got fail" for these runs. Its acceptance rule is the right one; my tracked score was the misleading part.
- An outside review on 2026-10-16 pointed this out. I stopped the run in progress (20261016T141847Z), since another full run cannot fix it.

**Other open problems, in priority order:**
1. **P4 training recipes.** Known-good models fail the learnability checks. The recipes come from the statement fixer and differ between runs and seeds, so this needs the same treatment as earlier fixes: diff passing and failing inputs, and have the card or the host supply the training recipe instead of the model inventing it.
2. **Environment failures counted as science.** Fixed by FIX-22 (live as of 17:28Z): the local model's context is checked before every call, and card-runs killed by LM Studio or transport errors are marked invalid and rerun with `calibrate --rerun-invalid`.
3. **Cost accounting.** The cost checker can miss work hidden in built-ins (`sum`, `sorted`) or data movement, so a quadratic design could look cheap.
4. **No positive control for new ideas.** Every planted "valid new idea" card was reduced to a known family and removed (`DECISIONS.md` row T7c). Passing the planted test today shows only that known designs survive and bad ones die.
5. **Status labels.** The final stage can record "pass" when every claim inside it was refuted. Labels need to come from the claims.
6. **Bench reliability.** The same seed and settings gave both near-perfect and chance results on one configuration. The fixed bench has not been run.
7. **Speed.** GPT and local-model calls run one at a time; a full calibration takes about 8 hours.

## Where it is going

In order. Each step gates the next.

1. Diagnose P4 on K1 and K2 directly: train the recipes outside the pipeline, find which input varies between passing and failing seeds, and fix where that input comes from.
2. Make the tracked calibration score the system's own acceptance rule, profile and target included, and close items 3 and 5 above.
3. Run one clean calibration and accept it only at full agreement.
4. Build a positive control: at least one planted card that is new by construction and should survive. Then run the planted test.
5. Run the bench acceptance sweep.
6. Run the live batch of 30 ideas.

The real measure of progress is a surviving idea that beats tuned baselines on equal data and compute. Nothing has done that yet.

## Layout

| Path | Contents |
|---|---|
| `hybrid/research` | Orchestrator, stages, prompts, schemas, calibration and planted harnesses |
| `hybrid/math` | Check programs and their manifest |
| `hybrid/executor` | Podman sandbox |
| `hybrid/llm`, `hybrid/gateway` | Model routing and the sign-in gateway |
| `hybrid/bench` | Baselines, ladder and fingerprints |
| `hybrid/literature` | Literature clients and the seed reading list |
| `hybrid/fixtures` | Calibration cards (K1 to K4) and planted cards |
| `hybrid/config` | Run, model and family configuration |
| `hybrid/tests` | About 3,000 tests |
| `bigbrane/`, `tests/`, `EXPERIMENT.md`, `README.md` | Experiment 1, frozen |
| `reports/` | LaTeX and Markdown sources of written reports |

Run records (`runs/`, `results/`), caches, virtual environments and local state are kept out of the repo. Sealed answers for the planted test live outside the workspace.

Run the tests with `.venv-hybrid/bin/python -m pytest hybrid/tests -q -p no:cacheprovider --ignore=hybrid/tests/test_probes.py`. They need Python 3.12 with numpy, torch (CPU), sympy, httpx, jsonschema and jinja2, plus Podman for the executor and bench tests.
