# Bench baselines

Five baselines (n-gram, GRU, tiny transformer, diagonal SSM, linear attention) run through the Podman executor with identical budgets, then a mechanism fingerprint is computed for each of them. Every number below was computed by the evaluator job from prediction arrays the baseline job wrote (trusted: true, image `localhost/bigbrane/executor@sha256:7fa94b73...`, one CPU per job, five jobs in parallel on the 8-CPU VM). KL is in nats against the exact oracle (for answer tasks it is -log q(answer)); the second figure is tie-aware top-1 accuracy. Tokens-to-target is the log-linear interpolation described in [README.md](README.md); "not reached" means no train size met the target. The fingerprint columns are defined in the README section "Mechanism fingerprint".

Reading the numbers:

- Chance for `fewshot` is 1/13 = 0.077 and for `recall` it is 1/16 = 0.0625. Only the transformer (0.262 at 8 shots on default, 0.195 on quick) and linear attention (0.219 default, 0.188 quick) are clearly above chance on `fewshot`; no baseline learns rule induction well at these budgets. The disjoint train and test rules make memorization useless, so a candidate that scores well here is doing something the baselines do not.
- `recall` accuracy falls as K grows for every baseline. At K=32 (default) the best baseline gets 0.090 (linear attention). On quick, the transformer does not learn recall at all (0.062 at K=2, chance), so it has no capacity estimate there (`not_learned`).
- `compose`: all neural baselines pick the matching closing bracket 88% to 97% of the time at the largest size; the n-gram gets 79% to 81%.
- Scaling: GRU, SSM and linear attention have tail slopes near 1 (a claimed exponent of 1). The transformer's measured tail slope is 1.5 (default) and 1.45 (quick) against a claim of 2, because at lengths up to 4096 with this small model the linear terms and fixed overhead still dominate the quadratic attention term. The probe reports what the code does at the lengths it measures. The SSM read 1.37 on the default run below (verdict `mismatch` at tolerance 0.3) and 1.19 to 1.26 elsewhere, so single slopes carry timing noise from the shared VM.
- Recorded job times (`run_s`, five baseline jobs in parallel): default config ngram 0.7 s, gru 199.3 s, ssm 326.2 s, linear_attention 477.3 s, transformer 479.1 s (whole `baselines` command 491.6 s); quick config ngram 0.6 s, gru 95.7 s, ssm 124.1 s, transformer 137.1 s, linear_attention 141.1 s (whole command 147.2 s). These are slower than the earlier four-job runs (351.5 s and 114.7 s) because a fifth job shares the VM.
- Flags on the baselines are for reference, with the other baselines as the comparison set. Linear attention and the transformer are the one pair the fingerprint separates poorly: on the default config each is flagged `behaves_like` the other (3 of 5 tasks close, ratio 0.30), and on quick they are 2 of 5 at ratio 0.29. The time-slope flag is a threshold on a noisy number: the transformer read 1.53 on default (`quadratic_or_worse`) and 1.45 on quick (not flagged).

Sources (all produced by this pass; logs and rendered tables under `results/bench-final/`): default baselines `runs/bench-final-default-baselines` (`default-baselines.md`, 491.6 s), quick baselines `runs/bench-final-baselines` (`quick-baselines.md`, 147.2 s), both with a fresh cache `results/bench-final/cache`. The quick config's GRU-as-candidate evaluation is `runs/bench-final-gru`. Memory growth labels (`flat`, `growing`) rest on a peak-RSS delta with a 10 MB floor and moved between runs for the GRU and the transformer, so read them as weak evidence.

## Default config (`configs/default.toml`: train sizes 32, 128, 512; 800 steps)

#### Largest train size: KL to oracle (nats) / top-1 accuracy

| entrant | compose (n=512) | fewshot (n=512) | hmm (n=512) | recall (n=512) | vom (n=512) |
| --- | --- | --- | --- | --- | --- |
| gru | 0.114 / 0.764 | 2.626 / 0.094 | 0.014 / 0.945 | 2.284 / 0.258 | 0.012 / 0.992 |
| linear_attention | 0.109 / 0.760 | 2.232 / 0.129 | 0.055 / 0.830 | 2.326 / 0.314 | 0.078 / 0.871 |
| ngram | 0.309 / 0.525 | 2.689 / 0.044 | 0.076 / 0.852 | 2.897 / 0.204 | 0.017 / 0.984 |
| ssm | 0.143 / 0.703 | 2.640 / 0.108 | 0.019 / 0.930 | 2.411 / 0.228 | 0.040 / 0.953 |
| transformer | 0.090 / 0.848 | 2.281 / 0.150 | 0.056 / 0.854 | 2.418 / 0.245 | 0.072 / 0.883 |

#### fewshot accuracy by shots (n=512)

| entrant | shots=2 | shots=4 | shots=8 |
| --- | --- | --- | --- |
| gru | 0.062 | 0.094 | 0.125 |
| linear_attention | 0.047 | 0.121 | 0.219 |
| ngram | 0.047 | 0.027 | 0.059 |
| ssm | 0.086 | 0.094 | 0.145 |
| transformer | 0.059 | 0.129 | 0.262 |

#### recall accuracy by K (n=512)

| entrant | K=2 | K=8 | K=32 |
| --- | --- | --- | --- |
| gru | 0.496 | 0.203 | 0.074 |
| linear_attention | 0.688 | 0.164 | 0.090 |
| ngram | 0.430 | 0.148 | 0.035 |
| ssm | 0.445 | 0.164 | 0.074 |
| transformer | 0.508 | 0.164 | 0.062 |

#### compose close-bracket accuracy (n=512)

| entrant | close_acc |
| --- | --- |
| gru | 0.954 |
| linear_attention | 0.917 |
| ngram | 0.789 |
| ssm | 0.921 |
| transformer | 0.940 |

#### Sample efficiency: KL (acc) at each train size

| compose entrant | n=32 (3072 tok) | n=128 (12288 tok) | n=512 (49152 tok) | tokens to kl <= 0.15 |
| --- | --- | --- | --- | --- |
| gru | 0.274 (0.549) | 0.181 (0.652) | 0.114 (0.764) | 23392 |
| linear_attention | 0.340 (0.459) | 0.189 (0.693) | 0.109 (0.760) | 24020 |
| ngram | 0.625 (0.354) | 0.428 (0.420) | 0.309 (0.525) | not reached |
| ssm | 0.339 (0.502) | 0.236 (0.633) | 0.143 (0.703) | 44382 |
| transformer | 0.333 (0.463) | 0.222 (0.674) | 0.090 (0.848) | 26231 |

| fewshot entrant | n=32 (1136 tok) | n=128 (4648 tok) | n=512 (18080 tok) | tokens to acc >= 0.2 |
| --- | --- | --- | --- | --- |
| gru | 2.592 (0.052) | 2.595 (0.069) | 2.626 (0.094) | not reached |
| linear_attention | 2.570 (0.069) | 2.389 (0.128) | 2.232 (0.129) | 4294 |
| ngram | 2.905 (0.059) | 2.731 (0.057) | 2.689 (0.044) | not reached |
| ssm | 2.584 (0.104) | 2.599 (0.081) | 2.640 (0.108) | not reached |
| transformer | 2.619 (0.089) | 2.578 (0.129) | 2.281 (0.150) | 6816 |

| hmm entrant | n=32 (2048 tok) | n=128 (8192 tok) | n=512 (32768 tok) | tokens to kl <= 0.02 |
| --- | --- | --- | --- | --- |
| gru | 0.063 (0.844) | 0.031 (0.922) | 0.014 (0.945) | 20447 |
| linear_attention | 0.113 (0.740) | 0.074 (0.799) | 0.055 (0.830) | not reached |
| ngram | 0.368 (0.727) | 0.170 (0.824) | 0.076 (0.852) | not reached |
| ssm | 0.078 (0.812) | 0.028 (0.922) | 0.019 (0.930) | 28074 |
| transformer | 0.122 (0.740) | 0.087 (0.768) | 0.056 (0.854) | not reached |

| recall entrant | n=32 (2012 tok) | n=128 (8036 tok) | n=512 (31184 tok) | tokens to acc >= 0.15 |
| --- | --- | --- | --- | --- |
| gru | 3.124 (0.102) | 2.673 (0.151) | 2.284 (0.258) | 13987 |
| linear_attention | 3.011 (0.092) | 2.622 (0.163) | 2.326 (0.314) | 23777 |
| ngram | 3.927 (0.083) | 3.440 (0.104) | 2.897 (0.204) | not reached |
| ssm | 3.479 (0.085) | 2.652 (0.178) | 2.411 (0.228) | 20762 |
| transformer | 2.957 (0.090) | 2.853 (0.068) | 2.418 (0.245) | 26772 |

| vom entrant | n=32 (4096 tok) | n=128 (16384 tok) | n=512 (65536 tok) | tokens to kl <= 0.03 |
| --- | --- | --- | --- | --- |
| gru | 0.045 (0.906) | 0.023 (0.953) | 0.012 (0.992) | 10538 |
| linear_attention | 0.089 (0.859) | 0.088 (0.877) | 0.078 (0.871) | not reached |
| ngram | 0.068 (0.924) | 0.030 (0.957) | 0.017 (0.984) | 16416 |
| ssm | 0.083 (0.887) | 0.067 (0.895) | 0.040 (0.953) | not reached |
| transformer | 0.089 (0.848) | 0.082 (0.879) | 0.072 (0.883) | not reached |

#### Scaling of predict_proba with context length

| entrant | params | tail slope (time) | full slope | claimed | verdict | median s at max n | peak RSS delta MB at max n | stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 76812 | 1.03 | 1.00 | 1.0 | match | 0.2052 (n=4096) | 16.4 | no |
| linear_attention | 101644 | 0.94 | 1.10 | 1.0 | match | 0.2384 (n=4096) | 38.5 | no |
| ngram | 7146 | -0.08 | -0.01 | 0.0 | match | 0.0000 (n=4096) | 2.0 | no |
| ssm | 51724 | 1.37 | 1.15 | 1.0 | mismatch | 0.1427 (n=4096) | 48.5 | no |
| transformer | 101644 | 1.53 | 1.53 | 2.0 | mismatch | 1.2823 (n=4096) | 29.2 | no |

#### Mechanism fingerprint: capacity, scaling, few-shot slope, nearest baseline, flags

| entrant | recall status | K at half accuracy | state floats | floats per pair | time slope (tail) | memory growth | few-shot slope per doubling | nearest baseline (mean ratio / mean JS) | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru | knee | 6.5 | 128 | 19.8 | 1.03 | growing | 0.031 | ssm (0.36 / 0.0344) | none |
| linear_attention | knee | 5.0 | 8320 | 1673.8 | 0.94 | flat | 0.086 | transformer (0.30 / 0.0476) | behaves_like:transformer, no_capacity_advantage |
| ngram | knee | 5.8 | 3 | 0.5 | -0.08 | flat | 0.006 | gru (0.42 / 0.0533) | no_fewshot_gain |
| ssm | knee | 6.0 | 256 | 42.7 | 1.37 | flat | 0.029 | gru (0.36 / 0.0344) | no_capacity_advantage, no_fewshot_gain |
| transformer | knee | 5.6 | 65536 | 11769.2 | 1.53 | flat | 0.102 | linear_attention (0.30 / 0.0476) | behaves_like:linear_attention, no_capacity_advantage, quadratic_or_worse |

##### Length generalization: KL (change from the training-length test set) at 2x and 4x the training length

| entrant | compose x2 | compose x4 | hmm x2 | hmm x4 | vom x2 | vom x4 |
| --- | --- | --- | --- | --- | --- | --- |
| gru | 0.138 (+0.024) | 0.126 (+0.012) | 0.015 (+0.001) | 0.014 (-0.000) | 0.010 (-0.002) | 0.010 (-0.001) |
| linear_attention | 0.210 (+0.102) | 0.333 (+0.224) | 0.118 (+0.062) | 0.117 (+0.062) | 0.084 (+0.005) | 0.076 (-0.003) |
| ngram | 0.321 (+0.012) | 0.328 (+0.019) | 0.077 (+0.002) | 0.070 (-0.005) | 0.008 (-0.008) | 0.010 (-0.007) |
| ssm | 0.221 (+0.078) | 0.214 (+0.071) | 0.021 (+0.002) | 0.021 (+0.002) | 0.047 (+0.007) | 0.038 (-0.001) |
| transformer | 0.206 (+0.116) | 0.385 (+0.295) | 0.121 (+0.064) | 0.124 (+0.068) | 0.092 (+0.020) | 0.077 (+0.005) |

##### Behavioral similarity: mean ratio / mean JS (nats) / tasks close, to each baseline at the largest train size

| entrant | gru | linear_attention | ngram | ssm | transformer |
| --- | --- | --- | --- | --- | --- |
| gru | self | 0.47 / 0.0655 / 0 of 5 | 0.42 / 0.0533 / 0 of 5 | 0.36 / 0.0344 / 1 of 5 | 0.45 / 0.0559 / 0 of 5 |
| linear_attention | 0.47 / 0.0655 / 0 of 5 | self | 0.51 / 0.1031 / 0 of 5 | 0.44 / 0.0709 / 0 of 5 | 0.30 / 0.0476 / 3 of 5 |
| ngram | 0.42 / 0.0533 / 0 of 5 | 0.51 / 0.1031 / 0 of 5 | self | 0.48 / 0.0709 / 0 of 5 | 0.50 / 0.0941 / 0 of 5 |
| ssm | 0.36 / 0.0344 / 1 of 5 | 0.44 / 0.0709 / 0 of 5 | 0.48 / 0.0709 / 0 of 5 | self | 0.42 / 0.0614 / 1 of 5 |
| transformer | 0.45 / 0.0559 / 0 of 5 | 0.30 / 0.0476 / 3 of 5 | 0.50 / 0.0941 / 0 of 5 | 0.42 / 0.0614 / 1 of 5 | self |

## Quick config (`configs/quick.toml`: train sizes 32, 128, 256; 600 steps; shorter sequences)

#### Largest train size: KL to oracle (nats) / top-1 accuracy

| entrant | compose (n=256) | fewshot (n=256) | hmm (n=256) | recall (n=256) | vom (n=256) |
| --- | --- | --- | --- | --- | --- |
| gru | 0.145 / 0.695 | 2.614 / 0.052 | 0.030 / 0.926 | 2.649 / 0.190 | 0.022 / 0.965 |
| linear_attention | 0.158 / 0.789 | 2.324 / 0.120 | 0.057 / 0.844 | 2.599 / 0.177 | 0.078 / 0.891 |
| ngram | 0.421 / 0.418 | 2.712 / 0.044 | 0.171 / 0.824 | 3.254 / 0.130 | 0.026 / 0.980 |
| ssm | 0.190 / 0.641 | 2.569 / 0.076 | 0.034 / 0.922 | 2.689 / 0.182 | 0.060 / 0.883 |
| transformer | 0.160 / 0.750 | 2.475 / 0.125 | 0.074 / 0.832 | 2.779 / 0.062 | 0.080 / 0.875 |

#### fewshot accuracy by shots (n=256)

| entrant | shots=2 | shots=4 | shots=8 |
| --- | --- | --- | --- |
| gru | 0.070 | 0.031 | 0.055 |
| linear_attention | 0.070 | 0.102 | 0.188 |
| ngram | 0.039 | 0.031 | 0.062 |
| ssm | 0.062 | 0.078 | 0.086 |
| transformer | 0.094 | 0.086 | 0.195 |

#### recall accuracy by K (n=256)

| entrant | K=2 | K=8 | K=16 |
| --- | --- | --- | --- |
| gru | 0.320 | 0.133 | 0.117 |
| linear_attention | 0.320 | 0.125 | 0.086 |
| ngram | 0.188 | 0.109 | 0.094 |
| ssm | 0.289 | 0.141 | 0.117 |
| transformer | 0.062 | 0.070 | 0.055 |

#### compose close-bracket accuracy (n=256)

| entrant | close_acc |
| --- | --- |
| gru | 0.970 |
| linear_attention | 0.897 |
| ngram | 0.808 |
| ssm | 0.882 |
| transformer | 0.892 |

#### Sample efficiency: KL (acc) at each train size

| compose entrant | n=32 (1536 tok) | n=128 (6144 tok) | n=256 (12288 tok) | tokens to kl <= 0.2 |
| --- | --- | --- | --- | --- |
| gru | 0.310 (0.527) | 0.194 (0.699) | 0.145 (0.695) | 5700 |
| linear_attention | 0.324 (0.527) | 0.185 (0.680) | 0.158 (0.789) | 5283 |
| ngram | 0.671 (0.375) | 0.516 (0.379) | 0.421 (0.418) | not reached |
| ssm | 0.335 (0.430) | 0.247 (0.594) | 0.190 (0.641) | 10837 |
| transformer | 0.372 (0.492) | 0.229 (0.695) | 0.160 (0.750) | 8187 |

| fewshot entrant | n=32 (1136 tok) | n=128 (4648 tok) | n=256 (9056 tok) | tokens to acc >= 0.15 |
| --- | --- | --- | --- | --- |
| gru | 2.588 (0.083) | 2.594 (0.091) | 2.614 (0.052) | not reached |
| linear_attention | 2.569 (0.068) | 2.469 (0.102) | 2.324 (0.120) | 3133 |
| ngram | 2.884 (0.057) | 2.742 (0.065) | 2.712 (0.044) | not reached |
| ssm | 2.586 (0.107) | 2.628 (0.062) | 2.569 (0.076) | not reached |
| transformer | 2.672 (0.076) | 2.578 (0.130) | 2.475 (0.125) | 2330 |

| hmm entrant | n=32 (1024 tok) | n=128 (4096 tok) | n=256 (8192 tok) | tokens to kl <= 0.05 |
| --- | --- | --- | --- | --- |
| gru | 0.080 (0.805) | 0.039 (0.930) | 0.030 (0.926) | 2845 |
| linear_attention | 0.132 (0.797) | 0.074 (0.836) | 0.057 (0.844) | not reached |
| ngram | 0.428 (0.711) | 0.229 (0.816) | 0.171 (0.824) | not reached |
| ssm | 0.117 (0.801) | 0.054 (0.902) | 0.034 (0.922) | 4733 |
| transformer | 0.161 (0.738) | 0.088 (0.773) | 0.074 (0.832) | not reached |

| recall entrant | n=32 (1240 tok) | n=128 (4500 tok) | n=256 (9052 tok) | tokens to acc >= 0.12 |
| --- | --- | --- | --- | --- |
| gru | 3.378 (0.057) | 2.883 (0.133) | 2.649 (0.190) | 3773 |
| linear_attention | 3.311 (0.060) | 2.784 (0.115) | 2.599 (0.177) | 8094 |
| ngram | 4.039 (0.070) | 3.660 (0.104) | 3.254 (0.130) | 3661 |
| ssm | 3.577 (0.094) | 2.886 (0.154) | 2.689 (0.182) | 1393 |
| transformer | 3.214 (0.076) | 2.902 (0.128) | 2.779 (0.062) | not reached |

| vom entrant | n=32 (2048 tok) | n=128 (8192 tok) | n=256 (16384 tok) | tokens to kl <= 0.04 |
| --- | --- | --- | --- | --- |
| gru | 0.073 (0.887) | 0.029 (0.953) | 0.022 (0.965) | 5741 |
| linear_attention | 0.093 (0.855) | 0.091 (0.883) | 0.078 (0.891) | not reached |
| ngram | 0.071 (0.906) | 0.039 (0.957) | 0.026 (0.980) | 7822 |
| ssm | 0.106 (0.891) | 0.076 (0.898) | 0.060 (0.883) | not reached |
| transformer | 0.094 (0.867) | 0.086 (0.867) | 0.080 (0.875) | not reached |

#### Scaling of predict_proba with context length

| entrant | params | tail slope (time) | full slope | claimed | verdict | median s at max n | peak RSS delta MB at max n | stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 76812 | 1.14 | 1.04 | 1.0 | match | 0.1307 (n=2048) | 17.7 | no |
| linear_attention | 101644 | 0.98 | 0.90 | 1.0 | match | 0.0447 (n=2048) | 20.3 | no |
| ngram | 5003 | 0.01 | -0.04 | 0.0 | match | 0.0000 (n=2048) | 2.0 | no |
| ssm | 51724 | 1.26 | 1.09 | 1.0 | match | 0.0740 (n=2048) | 18.0 | no |
| transformer | 101644 | 1.45 | 1.26 | 2.0 | mismatch | 0.1784 (n=2048) | 20.4 | no |

#### Mechanism fingerprint: capacity, scaling, few-shot slope, nearest baseline, flags

| entrant | recall status | K at half accuracy | state floats | floats per pair | time slope (tail) | memory growth | few-shot slope per doubling | nearest baseline (mean ratio / mean JS) | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru | knee | 6.5 | 128 | 19.6 | 1.14 | flat | -0.008 | ssm (0.33 / 0.0197) | no_fewshot_gain |
| linear_attention | knee | 6.2 | 8320 | 1334.8 | 0.98 | flat | 0.059 | transformer (0.29 / 0.0330) | no_capacity_advantage |
| ngram | knee | 16.0 | 3 | 0.2 | 0.01 | flat | 0.012 | gru (0.40 / 0.0507) | no_fewshot_gain |
| ssm | knee | 7.7 | 256 | 33.2 | 1.26 | flat | 0.012 | gru (0.33 / 0.0197) | no_capacity_advantage, no_fewshot_gain |
| transformer | not_learned | n/a | 65536 | n/a | 1.45 | growing | 0.051 | linear_attention (0.29 / 0.0330) | no_capacity_advantage |

##### Length generalization: KL (change from the training-length test set) at 2x and 4x the training length

| entrant | compose x2 | compose x4 | hmm x2 | hmm x4 | vom x2 | vom x4 |
| --- | --- | --- | --- | --- | --- | --- |
| gru | 0.202 (+0.056) | 0.216 (+0.070) | 0.031 (+0.001) | 0.033 (+0.003) | 0.030 (+0.008) | 0.012 (-0.010) |
| linear_attention | 0.295 (+0.137) | 0.407 (+0.249) | 0.099 (+0.042) | 0.133 (+0.076) | 0.098 (+0.020) | 0.041 (-0.037) |
| ngram | 0.470 (+0.049) | 0.458 (+0.037) | 0.176 (+0.005) | 0.167 (-0.005) | 0.031 (+0.006) | 0.011 (-0.015) |
| ssm | 0.283 (+0.093) | 0.271 (+0.082) | 0.041 (+0.006) | 0.038 (+0.004) | 0.069 (+0.009) | 0.043 (-0.016) |
| transformer | 0.263 (+0.103) | 0.404 (+0.244) | 0.121 (+0.047) | 0.177 (+0.103) | 0.106 (+0.026) | 0.037 (-0.043) |

##### Behavioral similarity: mean ratio / mean JS (nats) / tasks close, to each baseline at the largest train size

| entrant | gru | linear_attention | ngram | ssm | transformer |
| --- | --- | --- | --- | --- | --- |
| gru | self | 0.41 / 0.0463 / 0 of 5 | 0.40 / 0.0507 / 1 of 5 | 0.33 / 0.0197 / 2 of 5 | 0.39 / 0.0314 / 0 of 5 |
| linear_attention | 0.41 / 0.0463 / 0 of 5 | self | 0.49 / 0.0869 / 0 of 5 | 0.35 / 0.0470 / 1 of 5 | 0.29 / 0.0330 / 2 of 5 |
| ngram | 0.40 / 0.0507 / 1 of 5 | 0.49 / 0.0869 / 0 of 5 | self | 0.44 / 0.0589 / 1 of 5 | 0.47 / 0.0718 / 0 of 5 |
| ssm | 0.33 / 0.0197 / 2 of 5 | 0.35 / 0.0470 / 1 of 5 | 0.44 / 0.0589 / 1 of 5 | self | 0.36 / 0.0342 / 1 of 5 |
| transformer | 0.39 / 0.0314 / 0 of 5 | 0.29 / 0.0330 / 2 of 5 | 0.47 / 0.0718 / 0 of 5 | 0.36 / 0.0342 / 1 of 5 | self |

### GRU submitted as the candidate (quick config, fingerprinted)

The GRU baseline's own code, staged as a candidate and evaluated against the five cached baselines with `python -m hybrid.bench eval` (`runs/bench-final-gru`, nothing else running on the VM). Measured wall time 78.7 s for the whole command (candidate job 74.3 s, evaluator job 0.4 s, baselines served from the cache). The candidate's row from `quick-gru-candidate.md`:

| entrant | recall status | K at half accuracy | state floats | time slope (tail) | memory growth | few-shot slope per doubling | nearest baseline (mean ratio / mean JS) | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate (GRU code) | knee | 6.5 | 128 | 1.03 | flat | -0.008 | gru (0.00 / 0.0000) | behaves_like:gru, no_capacity_advantage, no_fewshot_gain |

Its predictions equal the GRU baseline's, so the JS divergence is exactly 0 on all five tasks and the candidate is flagged `behaves_like:gru`. `no_capacity_advantage` is raised because its capacity per state float equals the best fixed-state baseline's, and `no_fewshot_gain` because its few-shot slope is under 0.03 per doubling. The loop would see three flags and the nearest baseline named `gru`.

### Calibration controls (quick config)

Five controls run through the executor as candidates against the same cache (`runs/bfp-final-<name>`, summary in `results/bench-final/calib.txt`). Each is a baseline's code with a different model seed (`seed1`), or the GRU unchanged (`same`). The columns are tasks close to the named baseline, out of 5, and the mean ratio.

| control | flags | nearest by ratio | gru | ssm | linear_attention | transformer | ngram |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru, same code | behaves_like:gru, no_capacity_advantage, no_fewshot_gain | gru | 5 (0.00) | 2 (0.33) | 0 (0.41) | 0 (0.39) | 1 (0.40) |
| gru, seed 1 | behaves_like:gru, no_capacity_advantage, no_fewshot_gain | gru | 4 (0.27) | 1 (0.35) | 0 (0.42) | 0 (0.41) | 1 (0.41) |
| ssm, seed 1 | behaves_like:ssm, no_capacity_advantage, no_fewshot_gain | ssm | 1 (0.37) | 3 (0.28) | 1 (0.37) | 1 (0.36) | 1 (0.45) |
| linear_attention, seed 1 | no_capacity_advantage | linear_attention | 0 (0.43) | 1 (0.37) | 2 (0.27) | 2 (0.31) | 0 (0.49) |
| transformer, seed 1 | behaves_like:transformer, no_capacity_advantage | linear_attention | 0 (0.41) | 1 (0.36) | 2 (0.27) | 3 (0.28) | 0 (0.47) |

Reading: a changed seed of the GRU, SSM or transformer is still flagged as that architecture (4, 3 and 3 of 5 tasks, flag needs more than 2.5). A different seed of linear attention is not flagged (2 of 5) and sits as close to the transformer (ratio 0.31) as to itself (0.27); the transformer with a new seed has linear attention as its nearest baseline by ratio (0.27 against 0.28 for itself), though the flag names the transformer. Different architectures never reached 3 of 5 tasks close in these controls, but on the default-config baseline run linear attention and the transformer did (above). The recorded limits are in the README section "Calibration and limits".


## Phase 10 rung 1, first slice (2026-10-12 to 2026-10-13): calibration failed

The new recall/copy ladder completed 90 jobs with zero execution failures, but it is not calibrated. B.5 criteria 1 and 7 pass on the configured one-thread data; criteria 2, 4, 6 and 9 fail. These results use different tasks and training recipes from the older tables above and are not directly comparable. Sources and full per-seed numbers: [summary](../../results/phase10-20261012/summary.md), [criteria JSON](../../results/phase10-20261012/criteria.json), [read-only analysis](../../results/phase10-20261012/criteria.py).

| Check | Result at the selected LR, seeds 0 / 1 / 2 |
| --- | --- |
| Full attention learns MQAR, d64 N256 | Minimum accuracy across K=4..64: 0.980 / 0.973 / 0.981, all above 0.95. |
| Small-state MQAR, d32 N512 | GDN K4: 0.921 / 0.766 / 0.020. Mamba K4: 0.002 / 0.002 / 0.002. LA K8-to-K64 drops: 0.036 / 0.131 / 0.329, only one meets 0.2. |
| Window limit, w32 d64 N256 | SWA gap 33-64 accuracy: 0.957 / 0.386 / 0.064; beyond 64: 0.0025 / 0.0033 / 0.0032. Two layers extend the receptive field; mixer state matches at 8192 floats, total state differs because of convolution buffers. |
| Copy at L64 | Both transformer recipes and all fixed-state models score zero whole strings out of 128 on every seed. Prefix lookup passes at training lengths, but degrades beyond them. No copy length exceeds FP32 state bits. |
| Cost at one thread | d64 attention tail slope 1.519 versus LA/GDN/Mamba 0.968 / 0.980 / 0.971. State and analytic FLOP ordering agree. Auxiliary two-thread d64 timing does not consistently reproduce that direction. |
| Rank stability | 10 reversals among 91 cross-family pair/grid-point comparisons with mean gap >0.1; 7 are on MQAR. |

The Mamba d32 LR sweep stays at chance at every rate, while GDN seeds differ in whether retrieval develops. These are learning failures or weak learning, not established capacity limits. Large-K MQAR separates LA from full attention, but the full `behaves_like` fingerprint and the state-size sweep are untested. Seed 0 also selected the LR, so the slice lacks three fresh post-selection seeds.

Elapsed slice time was 17,068 seconds (4 h 44 m 28 s), seven one-thread workers. The N256 d64 microbenchmark measured 4.29 / 5.52 / 4.98 / 3.81 / 3.48 steps/s for full attention / SWA / LA / GDN / Mamba. Opinion: rerun five weak/seed-sensitive recall cells plus both transformer copy cells at 18k steps with the LR grids in the summary, then three post-selection seeds. The proposed 42 runs extrapolate to 62.45 job-hours or about 9.94 elapsed hours at the observed scheduling ratio. No new runs were launched; no bench code or configuration was changed for this report.

## Phase 10 diag1 (2026-10-13): rung 1 still uncalibrated

The diagnostic slice completed 42 host-MPS jobs at 18,000 steps, with zero execution failures. Seed 0 selected each cell's LR; seeds 1,2,3 ran afterward. More training and revised LRs improve several cells, but retrieval is still seed-sensitive and neither transformer copies any complete L64 string. This comparison with the CPU first slice is suggestive because device, budget, schedule and LR grids changed. Full evidence: [summary](../../results/phase10-20261013/summary.md), [read-only analysis](../../results/phase10-20261013/diag1_analysis.py), [JSON](../../results/phase10-20261013/diag1_analysis.json).

| Cell | Selected LR | Post-selection result, seeds 1 / 2 / 3 |
| --- | ---: | --- |
| GDN d64, MQAR N256 | .0003 | K4 .979 / .247 / 1.000; K64 .990 / .013 / .979. Seed 2 mostly predicts one retained context value rather than matching query keys. |
| Mamba s32 d64, MQAR N256 | .001 | K4 .034 / .283 / .060. Partial retrieval, no reliable strong low-K solution. |
| GDN d32, MQAR N512 | .001 | K8-K64 drops .268 / .619 / .040; seed 3 fails the required .2 drop. |
| LA d32, MQAR N512 | .01 | K8-K64 drops .260 / .342 / .401, all pass. |
| Mamba s16 d32, MQAR N512 | .0003 | K4 .090 / .187 / .178, above global chance but close to the stored-context target-permutation null. All required drops fail. |
| RoPE transformer copy | .003 | L32 whole-string accuracy 1 / 1 / 1; L64 is 0 / 0 / 0. |
| Hard-ALiBi transformer copy | .001 | L32 whole-string accuracy .188 / .836 / .805; L64 is 0 / 0 / 0. |

Criteria 2 and 6 fail on the tested subsets; same-device attention, fixed-copy and prefix controls are missing. Criterion 9 fails with 8 reversals among 17 eligible cross-family comparisons on post-selection seeds. The current gate instead uses seeds 0,1,2 and token accuracy for copy; its diag1 report has seven incomplete criteria and criterion 9 failed. `ladder gate` only reads a stored report, and `rung1.json` remains absent. State sweeps and the distribution-based `behaves_like` fingerprint remain unestablished.

Measured elapsed time was 17,950 s (4 h 59 m 10 s), three MPS workers. Opinion: inspect the weak-learning recipes, then a six-cell, 36k-step diagnostic with the summary's LR grids and three post-selection seeds projects to about 9.59 elapsed hours from diag1's measured rates. Stage the wider state sweep after learnability is repeatable. The literal w32 cutoff conflicts with the two-layer receptive field, and L256 never exceeds the counted FP32 state-bit budgets; both need explicit experimental-design changes. No new training or benchmark code/configuration changes were made for this analysis.

## Gated DeltaNet recipe change and diag2 (2026-10-13)

`baselines/bb_recall.py` now builds DeltaNet and Gated DeltaNet the way the current FLA reference does. The q/k/v short convolution and the a/b gate projections have no bias, A is drawn from U[0, 16] instead of U[1, 16], and every Linear and Conv1d weight in those models starts at N(0, 0.02) with zero biases. These are patches 01 and 02 from the [recipe probe](../../results/phase10-20261013/recipe-probe/REPORT.md); the residual-scaling patch 03 was not applied because it is untested. Transformer, sliding-window, linear attention and Mamba-2 build bitwise-identical weights to the previous source for seeds 0 to 3 ([check](../../results/phase10-diag2-20261013T1334Z/recipe-check/family_equivalence.json)). Gated DeltaNet loses `layers * (3d + 2 heads)` bias parameters: 196 at d32 and 388 at d64.

Evidence: in the paired CPU [pilot](../../results/phase10-20261013/gdn-pilot/summary.md) on MQAR N256 at d64, LR 0.0003 and 18,000 steps, initialization seed 2 failed again under the old recipe with identical data (mixed validation 0.041) and learned under the new one (0.9998, test K4 to K64 all at least 0.9998). Seed 1 kept learning (0.9989). That is two initialization seeds on one cell, on CPU. The pilot's own summary held back from a wider launch because its fixed-rate LA control did not learn at 0.0003; diag2 addresses that with per-cell LR sweeps and same-device attention controls.

The code hash moves from `33f43fbe1715...` to `8e4c6caba4c7...`, matching the probe's patch manifest, so every run id changes and no diag1 result can count toward acceptance. Slice `diag2` reruns diag1's seven cells with diag1's grids, 18,000 steps and seeds 0 then 1, 2, 3 on host MPS, plus full-attention controls on MQAR N256 (tf-d64) and N512 (tf-d32) at LR 0.0003, 0.001 and 0.003: 54 runs. Status: [STATUS.md](../../results/phase10-diag2-20261013T1334Z/STATUS.md).


## Phase 10 diag2 (2026-10-13): GDN remains unreliable, attention controls pass

Diag2 completed 54 host-MPS jobs at 18,000 steps under `8e4c6caba4c7...`, including GDN recipe patches 01+02, with zero execution failures. Seed 0 selected each LR; seeds 1,2,3 supplied the diagnostic checks. Patched GDN does not learn reliably across those seeds at either width. Full evidence: [summary](../../results/phase10-diag2-20261013T1334Z/summary.md), [read-only analysis](../../results/phase10-diag2-20261013T1334Z/diag2_analysis.py), [JSON](../../results/phase10-diag2-20261013T1334Z/diag2_analysis.json).

| Cell | Selected LR | Post-selection result, seeds 1 / 2 / 3 |
| --- | ---: | --- |
| GDN d64, MQAR N256 | .0003 | K64 .615 / .999 / .0035. Seed 2 improves, seed 3 stays at chance, seed 1 fits incompletely. |
| Mamba s32 d64, MQAR N256 | .0003 | K4 .453 / .817 / .604, positive permutation-null excess on all three. LR changed from diag1's .001; initial weights were unchanged. |
| Attention d64, MQAR N256 | .001 | Every K passes .95 on all three; worst .998047. Criterion 1 passes on this MPS cell. |
| GDN d32, MQAR N512 | .001 | K64 .0018 / .778 / .494. Seed 1 stays at chance; only seed 2 meets the .2 recall drop. |
| LA d32, MQAR N512 | .01 | K8-K64 drops .268 / .346 / .413, all pass. |
| Mamba s16 d32, MQAR N512 | .0003 | K4 excess over permutation null .0012 / -.0020 / .0010. Content retention still does not establish keyed retrieval. |
| Attention d32, MQAR N512 | .003 | K64 1 / .999695 / 1; all attention-drop checks pass. Top-rate winner leaves optimum unbracketed. |
| RoPE transformer copy | .003 | L32 strings .992 / 1 / 1; L64 zero on every seed. |
| Hard-ALiBi transformer copy | .001 | L32 strings .219 / .719 / .891; L64 zero on every seed. |

Criterion 2 fails with four of nine fixed-state checks passing; all three attention checks pass. Criterion 6's observed strict copy inequality cannot hold with zero transformer strings, while fixed-copy/prefix and bit-bound evidence remain missing. Criterion 9 fails with 20 violations among 41 cross-family comparisons (18 strict reversals, two ties). On the seven cells shared with diag1, it has 13 violations among 16 comparisons, versus diag1's 8 among 17. The [in-memory gate observation](../../results/phase10-diag2-20261013T1334Z/gate_observation.json) instead uses seeds 0,1,2 and token copy accuracy: criterion 1 passes, 2 and 9 fail, five criteria are incomplete. No `rung1.json` was written.

The CPU pilot used different data streams and only two initialization seeds; it cannot establish device-only equivalence or MPS reliability. Missing state/window/copy/lookup/cost/fingerprint evidence and the two structural issues from diag1 remain: two layers extend the window receptive field to 2w, and L256 contains only 1024 payload bits against roughly 299k-303k counted FP32 state bits.

Measured elapsed time was 21789 s (6 h 3 m 9 s), with three MPS workers. Opinion: replay the two failed GDN configurations and run a bounded four-cell LR-stability tranche before widening states. The summary's 28-run MPS portion projects to 18972 s, about 5.27 hours, from diag2's measured per-cell rates; CPU device checks need separate rate measurements. This analysis launched no training and changed no benchmark code/configuration or existing run evidence.

## Phase 10 diag3 (2026-10-13 to 2026-10-14): GDN replay varies, lower rates do not restore reliability

Diag3 completed 28 host-MPS FP32 jobs at 18,000 steps under the same `8e4c6caba4c7...` hash as diag2, with zero execution failures. Four runs replay failed configurations; 24 form four seed-0 LR sweeps followed by seeds 1,2,3. Replays are excluded from calibration seed and rank checks. Evidence: [summary](../../results/phase10-diag3-20261013T2009Z/summary.md), [read-only analysis](../../results/phase10-diag3-20261013T2009Z/diag3_analysis.py), [JSON](../../results/phase10-diag3-20261013T2009Z/diag3_analysis.json).

The N256 GDN d64 seed-3 replay at LR .0003 learns near-perfect retrieval once (K64 .996887) and stays at chance once (.004272), against diag2's .003540. Identical recorded job specs, seeds and execution metadata therefore do not determine the outcome. Both N512 GDN d32 seed-1 replays at .001 stay at chance. Small early loss differences grow into a different d64 learning trajectory; the artifacts do not identify a particular MPS kernel as the cause.

| Cell | Selected LR | Post-selection result, seeds 1 / 2 / 3 |
| --- | ---: | --- |
| GDN d64, MQAR N256 | .0002 | K64 .250 / .999451 / .022888; seed 3 mostly retains content without clear keyed retrieval. Unreliable. |
| GDN d32, MQAR N512 | .0006 | K64 .002197 / .144653 / .001831; seeds 1,3 stay at chance. Lower selected LR does not fix learning. |
| Mamba s16 d32, MQAR N512 | .0006 | K4 .021484 / .226562 / .935547; null excess .014404 / .014893 / .689209. Seed 3 learns strong retrieval, so an absolute capacity-impossibility claim is unsupported. |
| LA d32, MQAR N512 | .01 | K8-K64 drops .295898 / .354248 / .353455; all pass. Learned knee persists. |

Criterion 2 still fails, with 5/9 fixed-state checks passing. Criterion 9 fails all three eligible diag3 comparisons through strict seed-order reversals. Explicitly retaining the five other same-hash diag2 cells gives 13 violations among 33 comparisons, including two tie-only violations; this is diagnostic carry-forward, not a promoted report. Rung 1 remains uncalibrated, and no `rung1.json` exists.

Opinion: resolve criterion 4's two-layer receptive field and criterion 6's unreachable state-bit premise before widening the full grid. Bound recipe triage to nine 18k-step runs: GDN d32 at .0003 on seeds 1,2,3, plus post-seed repeats of GDN d64 at .0002 and Mamba s16 d32 at .0006. These are diagnostics, not a fresh LR-selected calibration cell. They project to 7234 elapsed seconds, about 2.01 hours, from diag3's measured three-worker rates and .978617 occupied-slot ratio. Diag3 itself took 21604 seconds. The summary proposes explicit learnedness/capacity and practical-rank/repeatability distinctions for Andrew to decide; no thresholds or enforcement were changed. No training, benchmark code/configuration changes or calibration promotion were performed for this analysis.


## Phase 10 diag4 (2026-10-14): GDN still branches; instrument before more slices

Diag4 completed nine fixed-rate 18,000-step host-MPS FP32 runs under `8e4c6caba4c7...`, with zero execution failures. GDN d32 N512 at .0003 learns keyed retrieval only on seed 2: K64 .587952. Seed 1 stays at chance; seed 3's K4 .195312 is mostly explained by permutation null .193115. GDN d64 N256 at .0002 repeats the qualitative fits on seeds 1,2, but seed 3 changes from retention in diag3 to near-perfect retrieval, K64 .993347. Mamba s16 d32 at .0006 repeats strong learning on seed 3 with lower K64 .094727; seeds 1,2 lose their earlier small null excesses. Specs and execution metadata match the six diag3 repeat jobs. Sources: [summary](../../results/phase10-diag4-20261014T0250Z/summary.md), [analysis](../../results/phase10-diag4-20261014T0250Z/diag4_analysis.py), [full-precision JSON](../../results/phase10-diag4-20261014T0250Z/diag4_analysis.json).

Across diag2 to diag4, GDN has 34 attempts: 23 learned, nine chance-floor and two retention outcomes under the inherited descriptive screen; three learned attempts are only partial. Excluding seed-0 sweeps leaves 22 attempts: 12 learned, eight chance and two retention. Adaptive LR search and targeted same-seed repeats make these descriptive counts, not a pooled success-probability estimate. Rung 1 remains uncalibrated. No gate, criterion or threshold changed.

Opinion: stop broad sweeps, save initialization/stream digests and instrument gradients, clipping and gate/update statistics. Prefer a predeclared success rate with a bound for calibration; best-of-k should disclose failures and total compute. Prefer one-thread Podman CPU for acceptance after exact-repeat determinism checks and fresh CPU recipe/control runs. Earlier achieved CPU rates project to 9145/6670/5810 seconds per 18k job for GDN d64/GDN d32/Mamba d32, respectively, versus diag4 MPS 1917/3587/1075. The comparison uses different recipes and contention. Next: two instrumented d64 .0002 seed-3 MPS attempts and two matching one-thread Podman CPU attempts, serialized, project to 22123 process seconds (6.15 hours) before unmeasured instrumentation overhead. Diag4 measured 6598 elapsed seconds. No training or benchmark code/configuration edits were made for this analysis.


## Phase 10 instrumented failure (2026-10-14): active gates, near-uniform values, no retrieval

Two completed GDN d32 N512 seed-1 .001 MPS attempts finish at mean loss 6.238407, close to log(512). The full vocabulary is 1024; the model learns the value half. Saved-checkpoint CPU forwards show near-uniform but unequal value logits, nonzero gradients and updates, and active SiLU output gates. F1 clips on 140/18000 steps; f2 never clips. Reconstructed recurrent states stay finite. The final MLP residual grows to mean query norms 195/176, while normalized query features become nearly aligned. The successful d64 N256 .0002 traces escape a similar early common-feature plateau. A persistent value-support optimization basin, with residual growth suppressing relative memory features, is the leading explanation; an LR-induced explosion or dead gate is unsupported.

F1/f2 initial tensors and complete batch streams match bitwise. Their losses first differ at step 14 and weight/optimizer digests differ by step 900, so repeat failure is not bitwise determinism. F3 remains running in the snapshot and has no endpoint claim. Next diagnostic: two unchanged-model seed-1 attempts at peak LR .0001, retaining the existing 900-step warmup and 18000-step budget. Seed 1 already failed at .0003 in diag4. Measured two-worker MPS times of 3744/3749 s project to about 3749 elapsed seconds for the pair, or 7492 job-seconds. No new training or benchmark implementation changes were made. Sources: [summary](../../results/phase10-instfail-20261014T0718Z/summary.md), [inference-only analysis](../../results/phase10-instfail-20261014T0718Z/analysis.py), [JSON evidence](../../results/phase10-instfail-20261014T0718Z/analysis.json).


## Phase 10 LR .0001 (2026-10-14): feature collapse resolves, keyed retrieval remains weak

Both GDN d32 N512 seed-1 attempts at peak LR .0001 finish 18,000 training steps. L1's K4/K8/K16/K32/K64 accuracies are .135742/.059570/.026367/.013306/.006836, but K4 permutation-null excess is only .000977 and its K8-K64 drop .052734 fails B.5 criterion 2's .2 requirement. L2 writes K4 predictions, then fails loading an empty K8 input during final evaluation. Its partial K4 excess is -.001953. Endpoint normalized-feature variation rises to about .93 and final-MLP query norm falls to .15-.17, versus .019-.022 and 176-195 in the .001 failures. Persistent feature collapse resolves, while keyed retrieval remains weak.

Concurrent cache writers could truncate a published NPZ during another worker's staging copy. The reproduced race is fixed by closing unique same-directory archives before atomic replacement, with the oracle published last. Training math and code hash `b5452f050553...` are unchanged; host staging SHA-256 changes from `e4106cb50809...` to `7030a41339eb...`. None of 168 remaining Phase 10 cache files is empty; the two slices' archives match regeneration and recorded oracle hashes. Tests: **285 passed, 4 skipped**, including publication regressions and exact CPU instrumentation equivalence.

Opinion: stop expanding this recipe for calibration and record it as unreliable under the 18k budget. Keyed-excess slopes do not predict completion. A separate two-attempt 36k investigation would project to 6,930-7,360 elapsed seconds, or 14,713 job-seconds at whole-job rates, including instrumentation. Patch 03 remains a separate ablation. No new diagnostic training or criterion changes were made. Sources: [summary](../../results/phase10-instlr-20261014T0855Z/summary.md), [analysis](../../results/phase10-instlr-20261014T0855Z/analysis.py), [JSON](../../results/phase10-instlr-20261014T0855Z/analysis.json).


## Phase 10 criterion revisions (2026-10-16, decision B10)

The B.5 criteria 2, 4, 6 and 9 and the acceptance execution policy changed in `ladder_criteria.py` (see [B10-STATUS.md](B10-STATUS.md) and IMPROVEMENT-PLAN.md B.5). Acceptance numbers come only from the pinned one-thread Podman CPU container on seeds 1, 2 and 3; the MPS numbers above are diagnostics and the gate refuses them. Reading the stored diag3 results with the revised rules: criterion 2 has five learned passes (LA 3/3, GDN d32 seed 2, Mamba s16 d32 seed 3), attention d32 from diag2 passes 3/3, and the four seeds that never learn the K8 control (GDN d32 seeds 1 and 3, Mamba seeds 1 and 2) are now `unresolved` instead of failures. Criterion 9 still fails on the same three cross-family order reversals, and three cells with unstable onset still need same-seed repeats. Baseline numbers in this file are unchanged, and no calibration run was made under the new rules.

