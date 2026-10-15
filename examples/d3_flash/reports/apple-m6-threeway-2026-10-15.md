# D3-Flash three-way evaluation on Apple M6 / 32 GiB

On 2026-10-15, all three variants completed the same 1,356 frozen choice requests: **4,068 model requests, zero failures and full scoring coverage**. The final Mirai INT4 artifact exceeded simple MLX INT4 on three of four role metrics, with a smaller text payload and lower measured request latency. It did not preserve BF16 quality on every task; ANLI and sarcasm detection remain weaker.

Unrounded metrics, pairwise probability comparisons and artifact identities are in the [measurement JSON](apple-m6-threeway-2026-10-15.json). Scores below are metric values multiplied by 100, not a combined accuracy score.

## Primary quality results

| Benchmark | Requests | Metric | BF16 | MLX INT4 | Mirai INT4 |
| --- | ---: | --- | ---: | ---: | ---: |
| When2Call | 200 | accuracy | 86.50 | 89.50 | 88.50 |
| BANKING77 | 154 | macro-F1 | 88.83 | 87.58 | 88.23 |
| CLINC150+OOS | 302 | macro-F1 | 97.26 | 96.87 | 97.22 |
| iSarcasmEval-A-En | 400 | sarcastic positive F1 | 54.86 | 48.00 | 52.75 |
| ANLI | 100 | macro-F1 | 63.23 | 62.78 | 60.05 |
| WinoGrande | 100 | accuracy | 85.00 | 85.00 | 86.00 |
| HellaSwag | 100 | accuracy | 96.00 | 93.00 | 94.00 |

iSarcasm uses sarcastic-class positive F1 from the official scorer's subtask detail. Its native aggregate is null; treating that as zero or reporting majority-class accuracy would misrepresent this task. Every model answered every case; errors, unsupported, abstained and pending counts were zero.

Mirai exceeds MLX on BANKING77, CLINC150+OOS and iSarcasm, while When2Call is 1.00 accuracy point lower. Relative to BF16, Mirai loses 0.61 macro-F1 points on BANKING77, 0.04 on CLINC, 2.11 positive-F1 points on iSarcasm and 3.18 macro-F1 points on ANLI. Gains on other subsets do not establish general superiority or statistical significance.

## Probability quality and request latency

NLL and multiclass Brier sum are row-weighted within each suite; lower is better. These are descriptive metrics, not official Decision Index scores, and should not be compared between suites. Percentiles come from individual successful requests, not averages of benchmark medians.

| Suite | Model | NLL (nats) | Brier sum | Wall p50 (ms) | Wall p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| roles | bf16 | 0.3121 | 0.1770 | 1576.7 | 2564.4 |
| roles | mlx-int4 | 0.3494 | 0.2013 | 899.3 | 1503.1 |
| roles | mirai-int4 | 0.3190 | 0.1816 | 655.8 | 1139.9 |
| regression | bf16 | 0.4708 | 0.2657 | 592.8 | 756.6 |
| regression | mlx-int4 | 0.4938 | 0.2796 | 172.1 | 333.6 |
| regression | mirai-int4 | 0.4965 | 0.2773 | 130.4 | 225.6 |

Request wall time includes Python preparation, engine IPC/inference and postprocessing, with completion synchronization. Startup and one unscored warmup per process are excluded. All models ran serially, batch 1, on the same Metal GPU engine. This is one run per model/suite, not a throughput or repeated-run variance study. It does not measure an mlx-lm deployment.

Role predictions match BF16 on 1,029/1,056 requests (97.44%) for Mirai and 1,011/1,056 (95.74%) for MLX. Similar top labels do not imply identical probabilities: the largest BF16-to-Mirai single-option change is 45.40 percentage points in CLINC. The JSON retains all pairwise MAE, total variation, directed KL, JSD and gold-match changes by benchmark.

## Comparable text storage

| Model | Active text + one readout (GiB) | Whole native bundle (GiB) |
| --- | ---: | ---: |
| bf16 | 14.789 | 14.815 |
| mlx-int4 | 5.527 | 5.552 |
| mirai-int4 | 4.289 | 4.315 |

Mirai's active payload is **22.40% smaller than MLX INT4** and 71.00% smaller than BF16. This is substantially an embedding-policy difference: quantizing the embedding saves 1,446,199,296 bytes relative to MLX, while Mirai's projection payload is actually 116,955,136 bytes larger. It is not an isolated test of RHT versus affine quantization.

All three evaluated native bundles are text-only. Active payload includes embeddings, projections and quantization parameters, norms, stored DeltaNet parameters and one FP32 decision readout. Whole-bundle sizes also count the duplicate readout sidecar and other files, excluding Python bytecode. These are logical file bytes, not measured GPU memory, RSS or allocated disk blocks.

## Model and engine provenance

- Source: `vllm-sr/d3-flash` at `521525effa79469d899a570726547dd7f00da3af`. The pinned original BF16 checkpoint was acquired through the Hub cache and source hashes were checked.
- BF16 reference: value-preserving native format export. Norm/state tensors are widened from BF16 to exactly representable FP32 values as required by the native format; projection and embedding values remain BF16.
- Simple INT4: freshly quantized with MLX 0.32.3 affine INT4/group64, retaining the BF16 embedding and FP32 decision head. Its native export checked all 248 source Linear modules and 6,918,504,448 packed INT4 codes without re-quantization.
- Mirai: the preserved final INT4/group32/RHT32 artifact, including an INT4 embedding and FP32 readout, created using the modified Lalamo checkout based on `831accb8`. No Fisher calibration or QAD was applied. **This is fresh evaluation of that artifact, not a fresh full conversion at the current PR head**, whose upstream base is `ad76c7d0`.
- All full evaluations use the same Uzu Metal D3 engine, based on `432da1b7bf42d5f5ffd30d7acf9ca201c9c70455` plus a separate local adapter. The adapter and model weights are not included in the Lalamo PR. Binary SHA-256: `42bd6ce9740721f605ebb3212ee2ff5fd7d7b02669c5a97f3bcb7673c4ebf640`.
- Original Transformers/MPS-to-native paired checks cover eight selected role requests per BF16/MLX variant: 8/8 matching top labels each. Maximum probability differences are 0.00876944 (BF16) and 0.00424312 (MLX). These nonzero differences and limited coverage do not establish full-suite engine equivalence or original Transformers/MPS accuracy.

## Sampling and limits

The frozen scorer is pinned to Decision Index kit commit `9eb2dbe2a358004c8782c66e40a83ac07b953fec`. Roles comprise When2Call 200, BANKING77 154 (77 classes x 2), CLINC150+OOS 302 (151 classes x 2) and iSarcasm 400. Regression comprises ANLI 100 (rounds 1/2/3: 33/33/34), WinoGrande 100 and HellaSwag 100. The same selected rows, prompts, option order, answer-code assignments, temperature and readout contract were checked for every model.

- These are fixed held-out subsets, not the full Decision Index or an official leaderboard run. BANKING77 and CLINC gold labels are used only for class stratification; no gold labels are sent in prompt payloads. No prompt/quantization tuning was performed on evaluation results.
- When2Call has gold B/C/D counts 76/65/59 and no A (direct-answer) examples. It cannot validate positive direct-answer routing. iSarcasm has 51 sarcastic and 349 non-sarcastic examples.
- All 1,356 cases have one choice question. This run does not establish noul/score/multi-question quality, Korean production-bot accuracy or broad D3-series compatibility.
- Different group sizes, embedding policies and quantizers are evaluated together. Their individual causal effects are not separated.
- Full raw responses, frozen suite/scorer files, conversion/oracle reports, source/environment archives and hashes are retained locally. This public summary omits benchmark payloads and application details. Published hashes identify the local evidence; they do not make the omitted engine/data archive independently reproducible from this PR alone.
- Final hash checks confirmed the operating Mirai weights, configuration, readout and inference binary were unchanged. Official scorecards were recomputed from the complete raw responses; all six model-suite comparisons and frozen-input/reproduction checks passed.
