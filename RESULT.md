# The HMM won the low-data comparison, but its advantage did not survive both stress comparisons

The completed experiment supports the prespecified low-data advantage of a learned four-state HMM against both tiny neural controls. Broadly fewer training tokens were not established: HMM reached the sustained error target earlier than both controls on only **1 of 8 generators**. Under second-order latent dynamics, RNN beat HMM at both tested sizes; attention beat it at the larger size.

This provides a usable measuring apparatus and a prior-versus-flexibility baseline for comparing mathematical theories. There is no new architecture, derived mechanism or natural-language evidence.

## Scope

Actual run: [`confirmatory-v1-followup-c482b1`](results/experiment-001/confirmatory-v1-followup-c482b1/). Manifest: `complete`, seed 20260930, protocol version 1, non-pilot; 384 matched and 192 stress candidates completed.

Eight HMM-friendly generators cover mixing/informative and persistent/ambiguous regimes, four each. The matched learner receives the correct state count. Families share data and have two search replicas. Validation adds 2,048 tokens: training budgets 512/2,048/8,192/16,384 therefore mean accessible totals 2,560/4,096/10,240/18,432. Each test contains 8,192 tokens. Neural widths and optimization budgets are tiny and fixed.

## Learning curves

Mean oracle conditional KL, **nats/target**, across generators and replicas; lower is better:

| Regime | Family | N=512 | 2,048 | 8,192 | 16,384 |
| --- | --- | ---: | ---: | ---: | ---: |
| Mixing/informative | HMM | .09817 | .04387 | .02698 | .04838 |
| Mixing/informative | RNN | .14456 | .03642 | .00938 | .00627 |
| Mixing/informative | Attention | .18983 | .08266 | .03917 | .02351 |
| Persistent/ambiguous | HMM | .08872 | .04227 | .03188 | .04157 |
| Persistent/ambiguous | RNN | .11221 | .07152 | .01617 | .01032 |
| Persistent/ambiguous | Attention | .15019 | .09976 | .04041 | .02602 |
| Both | HMM | .09345 | .04307 | .02943 | .04497 |
| Both | RNN | .12839 | .05397 | .01277 | .00830 |
| Both | Attention | .17001 | .09121 | .03979 | .02477 |

Primary delta is control KL minus HMM KL, averaging N=512 and 2,048. Positive favors HMM:

| Control | Mean delta | Descriptive 95% interval | Regime means, mixing/persistent | Positive generators |
| --- | ---: | --- | --- | ---: |
| RNN | .02292 | [.01200, .03208] | .01947 / .02637 | 7/8 |
| Attention | .06235 | [.05350, .07242] | .06522 / .05948 | 8/8 |

Both pass: mean ≥.02, positive regime means, ≥6 favorable generators, interval lower endpoint >0. The 5,000-draw stratified bootstrap is descriptive: four generators per stratum.

## Target crossings

The .05 target requires remaining below threshold at every larger sampled budget:

| Family | 2,048 | 8,192 | 16,384 | >16,384 |
| --- | ---: | ---: | ---: | ---: |
| HMM | 4 | 2 | 0 | 2 |
| RNN | 5 | 3 | 0 | 0 |
| Attention | 0 | 7 | 1 | 0 |

HMM crosses earlier/ties/later against RNN on 1/4/3 generators; against attention, 5/1/2. Only persistent generator 1 favors HMM against both. No crossing interpolation is justified. HMM's nonmonotonic curve leaves optimization adequacy unresolved.

## Triggered stress stage

Fresh second-order latent sequences have a sixteen-pair-state oracle; learned HMM remains four-state. Within-stage train and test distributions match.

| N | Control | Stress delta [95% interval] | Matched same-size delta | Outcome |
| --- | --- | --- | ---: | --- |
| 2,048 | RNN | -.01779 [-.02798, -.00711] | +.01090 | Reversed |
| 2,048 | Attention | +.05869 [.05153, .06594] | +.04814 | Retained |
| 16,384 | RNN | -.05108 [-.06035, -.03955] | -.03668 | Reversed |
| 16,384 | Attention | -.03215 [-.04208, -.02027] | -.02021 | Reversed |

Neither size retains advantage against both. Both controls already beat HMM on matched N=16,384 data, preventing attribution of the entire stress gap to mismatch.

## Parameters and costs

HMM has 43 free parameters, 52 stored float64 entries and 416 parameter bytes. RNN has 280 float32 parameters/1,120 bytes; attention has 712/2,848 bytes. Optimizer tensors use 0/2,264/5,756 bytes respectively. HMM filter and RNN hidden state each use 32 bytes. Attention's dense workspace grows with length.

Measured costs include losing candidates and both stages, 192 fits per family:

| Family | Training s | Validation s | Checkpoint I/O s | Token exposures |
| --- | ---: | ---: | ---: | ---: |
| HMM | 25.60 | 3.76 | 21.62 | 72,562,688 |
| RNN | 182.22 | 9.81 | 63.25 | 87,321,600 |
| Attention | 121.96 | 8.69 | 109.52 | 101,738,496 |

Training includes objective diagnostics. Retained-candidate training alone totals 12.66/70.91/51.70 seconds. Worker wall time totals 1,320.79 seconds including process overhead. Counts: 9,567 EM iterations, 43,746 RNN/51,048 attention optimizer steps, 34,711 validation evaluations. Peak RSS was 302,530,560 bytes, including runtime/library overhead.

Batch-32 scoring of 32 sequences, mean of ten-pass model medians across eight matched N=16,384 replica-0 models, **milliseconds**:

| Family | L=16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HMM | .100 | .178 | .336 | .647 | 1.316 |
| RNN | .210 | .401 | .776 | 1.543 | 3.098 |
| Attention | .188 | .279 | .536 | 1.355 | 4.687 |

For 128 length-64 sequences, median across all selected models was 30.78/73.36/11.75 ms at batch one and 1.334/3.120/2.063 ms at batch 32. Construction/I/O are excluded. Scaling KL from length 16 to 256: HMM .03564→.04227, RNN .00762→.01009, attention .02145→.21390. These CPU timings establish neither asymptotic laws nor cached attention performance.

## Integrity and meaning

Read-only checks passed for **41,222 distinct run files**: 41,050 result hashes, 152 data/scaling hashes, 18 frozen sources, environment and preflight log; 18 live sources match. Stage registries agree. All 576 candidate commits, 288 evaluations and 120 scaling records across 24 models are present, with zero invalid/missing candidates. Saved metrics reproduce both decisions. No tests/experiments were rerun; raw artifacts were unchanged.

Exact verification/aggregation command from the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python results/experiment-001-summary-c482b1/verify_and_summarize.py > results/experiment-001-summary-c482b1/actual-summary.json
```

Existing artifact functions verify integrity; [`actual-summary.json`](results/experiment-001-summary-c482b1/actual-summary.json) preserves outputs. `RESULT.md` is outside frozen inputs. Broader theory comparisons, unknown-state learning and language claims remain untested.
