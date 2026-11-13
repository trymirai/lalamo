# On-policy distillation pilot: Qwen3.8-27B → Qwen3.5-9B

This documents the pilot run of `scripts/on_policy_distillation_pilot.py`: what was trained, on what, with which
loss, how it was evaluated, and what came out. Section 8 adds three follow-up runs with other training signals.
Every run is a single run with one seed and small evaluation samples. They are meant to validate the pipeline and
to show where the recipe needs work, not to support claims about a method.

- [1. Summary](#1-summary)
- [2. Notation](#2-notation)
- [3. Loss](#3-loss)
- [4. Setup](#4-setup)
- [5. Evaluation protocol](#5-evaluation-protocol)
- [6. Logged training metrics](#6-logged-training-metrics)
- [7. Results](#7-results)
- [8. Follow-up runs](#8-follow-up-runs)
- [9. Action items](#9-action-items)
- [10. Reproduction](#10-reproduction)

## 1. Summary

| | |
|---|---|
| Teacher | `Qwen/Qwen3.8-27B`, frozen |
| Student | `Qwen/Qwen3.5-9B`, frozen base weights plus trained rank-64 low-rank adapters (141,656,064 parameters) |
| Method | On-policy distillation: the student samples responses, the teacher scores the sampled tokens, and the adapters take one optimizer step per batch on a sampled-token estimate of the reverse KL |
| Training signal | Log-probabilities only. No reward, no reference answers, no teacher-written text |
| Training data | 3,200 math problems from DeepScaleR-Preview, each used once (50 steps × 64 rollouts, up to 4,096 response tokens each; 10,031,161 response tokens in total) |
| Evaluation | MATH-500 (200 problems, the distilled skill) and MMLU-Pro (300 questions, never trained on) |
| Result | MATH-500 49.0% → 80.0% (teacher 86.5%), MMLU-Pro 31.7% → 70.3%. Most of the gain is the student finishing inside the token budget. Evaluation accuracy is flat from step 10; the reverse-KL estimate falls until step 14 and then doubles between steps 28 and 34 |
| Follow-up runs (section 8) | Same budget, other signals; MATH-500 / MMLU-Pro at step 50: supervised fine-tuning on teacher-written responses 78.5% / 67.3%; reward for correct answers only 67.5% / 76.0%, after 83.5% / 76.7% at step 10; distillation combined with that reward 82.0% / 65.0% |

## 2. Notation

All logarithms are natural, so every information quantity is in nats.

| Symbol | Meaning |
|---|---|
| $`𝒱`$ | Token vocabulary, shared by both models: $`\lvert𝒱\rvert = 248{,}320`$ |
| $`x`$ | A training problem (text) |
| $`p^{\mathrm{S}}(x)`$, $`p^{\mathrm{T}}(x)`$ | Token sequences obtained by rendering $`x`$ with the student's and with the teacher's chat template. They differ (section 4.2) |
| $`y = (y_1, \dots, y_L)`$ | A response: the tokens the student sampled after $`p^{\mathrm{S}}(x)`$, with $`L \le L_{\max} = 4096`$. It contains the reasoning, the final reply and, if it was reached, the stop token |
| $`s_t = (p^{\mathrm{S}}(x),\, y_{<t})`$ | The student's context before response token $`t`$ |
| $`\tilde{s}_t = (p^{\mathrm{T}}(x),\, y_{<t})`$ | The same context as the teacher sees it: the teacher's prompt followed by the same response prefix |
| $`\theta`$ | The trained parameters: all low-rank adapter weights. $`\theta_{k-1}`$ is their value at the start of step $`k`$, and $`\theta_0`$ makes the student identical to the base model |
| $`\pi_\theta(v \mid s)`$ | Student's next-token distribution over $`v \in 𝒱`$: the softmax of its logits, with no temperature or truncation |
| $`\pi_{\mathrm{T}}(v \mid \tilde{s})`$ | Teacher's next-token distribution, defined the same way |
| $`B`$, $`i`$ | Number of rollouts per step ($`B = 64`$) and the rollout index |
| $`L_i`$, $`N`$ | Length of response $`i`$, and the number of response tokens in the step, $`N = \sum_{i=1}^{B} L_i`$ |
| $`g_{i,t}`$ | Log-probability gap of a sampled token, teacher minus student (equation 2) |
| $`b`$, $`A_{i,t}`$ | Baseline and advantage (equations 3 and 4) |
| $`\mathrm{sg}[\cdot]`$ | Stop-gradient: the argument is treated as a constant when differentiating |
| $`K`$, $`k`$ | Number of optimizer steps ($`K = 50`$) and the step index, $`k = 1, \dots, K`$ |

## 3. Loss

### 3.1 Target quantity

For one context, the quantity being reduced is the reverse KL divergence from the student's next-token
distribution to the teacher's:

```math
\ell(\theta;\, s, \tilde{s})
= D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot \mid s) \,\big\Vert\, \pi_{\mathrm{T}}(\cdot \mid \tilde{s})\bigr)
= \sum_{v \in 𝒱} \pi_\theta(v \mid s)\,\bigl[\log\,\pi_\theta(v \mid s) - \log\,\pi_{\mathrm{T}}(v \mid \tilde{s})\bigr].
\qquad (1)
```

### 3.2 Sampled-token estimator used in the run

In step $`k`$ the student with parameters $`\theta_{k-1}`$ samples $`B`$ responses, one per problem. Every
response token $`y_{i,t}`$ is scored by both models, giving the log-probability gap

```math
g_{i,t} = \log\,\pi_{\mathrm{T}}\bigl(y_{i,t} \mid \tilde{s}_{i,t}\bigr) - \log\,\pi_{\theta_{k-1}}\bigl(y_{i,t} \mid s_{i,t}\bigr).
\qquad (2)
```

The baseline is the mean gap over all response tokens of the step, and the advantage is the centred gap:

```math
b = \frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i} g_{i,t},
\qquad (3)
```

```math
A_{i,t} = g_{i,t} - b.
\qquad (4)
```

The loss of step $`k`$ is the policy-gradient surrogate

```math
ℒ_k(\theta) = -\frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i}
\mathrm{sg}\bigl[A_{i,t}\bigr]\, \log\,\pi_\theta\bigl(y_{i,t} \mid s_{i,t}\bigr).
\qquad (5)
```

The step's gradient $`\hat{G}_k`$ is the gradient of (5) with respect to $`\theta`$ at $`\theta_{k-1}`$. It is a
sampled-token estimate of the gradient of (1), averaged over the response tokens of the step, at the contexts
the student visited. Only the sampled token of each context is used; the sum over the vocabulary in (1) is
never formed. `tests/unit/test_on_policy_distillation.py` checks on a tiny model that the expectation of the
estimator equals the exact gradient of (1), and that the gradient is zero when teacher and student are the same
model.

### 3.3 What the estimator leaves out

- **Credit assignment across tokens.** A token is pushed up or down according to how the teacher rates that
  token alone. The gaps of the tokens that follow it do not enter its advantage, so where a token leads the
  rest of the response has no effect on its update. The contexts the student visits are treated as fixed, and
  nothing in the loss stops the student from moving to contexts where (1) is larger.
- **An independent baseline.** $`b`$ is computed from the same batch. It contains the token's own gap and the
  gaps of later tokens of the same rollout, so it is not strictly independent of the sampled token. The
  affected share of the baseline is at most $`L_i / N \approx 1/B \approx 1.6\%`$ per token. This bias was not
  corrected.
- **Per-rollout weighting.** The loss is divided by $`N`$, so every token weighs the same and long rollouts
  contribute more. It is not a mean over rollouts.
- **Truncated responses.** Responses that reach $`L_{\max}`$ are cut off; no term accounts for what would have
  followed.

### 3.4 Not in the loss

There is no outcome reward, no cross-entropy on teacher-generated text, no forward-KL term, no penalty tying
the student to its initial weights, no entropy bonus other than the one implicit in (1), and no weight decay.

### 3.5 Accumulation over micro-batches

One 4,096-token rollout needs about 92 GB of working memory for the backward pass, so the gradient of (5) is
accumulated over micro-batches of one rollout. Micro-batch $`m`$ holds $`N_m`$ response tokens and contributes
the gradient of its own token-mean loss, weighted by its share of the step's tokens:

```math
\hat{G}_k = \sum_{m} \nabla_\theta \Bigl[\frac{N_m}{N}\, ℒ_{k,m}(\theta)\Bigr]_{\theta = \theta_{k-1}},
\qquad
ℒ_{k,m}(\theta) = -\frac{1}{N_m} \sum_{(i,t) \in m} \mathrm{sg}\bigl[A_{i,t}\bigr]\, \log\,\pi_\theta\bigl(y_{i,t} \mid s_{i,t}\bigr).
\qquad (6)
```

The advantages are computed once for the whole step, before any gradient, so $`b`$ is the mean over all 64
rollouts and not over a micro-batch. There is exactly one optimizer update per step.

## 4. Setup

### 4.1 Models

| | Student | Teacher |
|---|---|---|
| Hugging Face repository | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.8-27B` |
| Revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` |
| Decoder parameters | 8,953,803,264 | 26,895,998,464 |
| Layers | 32 | 64 |
| Hidden size | 4,096 | 5,120 |
| Output vocabulary | 248,320 | 248,320 |
| Weight precision | bfloat16 | bfloat16 |
| Role | Samples responses; adapters are trained | Scores the student's tokens; never updated, never samples during training |

Both tokenizers have the same BPE vocabulary and merges, and both models produce 248,320 logits, so one token
sequence can be scored by both without re-tokenization. The teacher's tokenizer names seven special tokens
(ids 248070–248076) that the student's does not.

### 4.2 Prompts

A problem is rendered as a single user message with each model's own chat template and default settings. With
`{problem}` standing for the problem text, the user message is

```text
{problem}

Put your final answer within \boxed{}.
```

The student's prompt $`p^{\mathrm{S}}(x)`$ is

```text
<|im_start|>user
{user message}<|im_end|>
<|im_start|>assistant
<think>

```

and the teacher's prompt $`p^{\mathrm{T}}(x)`$ is the same text preceded by a system message that its template
adds by default:

```text
<|im_start|>system
Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>
```

Both prompts end with `<think>` and a newline, so reasoning is on for both models and every response starts
inside the reasoning block. The teacher's prompt is 42 tokens longer for every problem. The teacher is thus
conditioned on an instruction that the student never sees.

### 4.3 Training data

- Source: `agentica-org/DeepScaleR-Preview-Dataset`, file `deepscaler.json`, revision
  `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`: 40,315 problems. Only the `problem` field is used. The reference
  answers and solutions are never read by the training code.
- Overlap removal against all 500 MATH-500 problems (not only the 200 evaluated): a training problem is dropped
  if its lower-cased, whitespace-normalised text equals that of a MATH-500 problem, or if it shares any sequence
  of 8 consecutive words with one (words are maximal `\w+` matches of the lower-cased text). 38,252 problems
  remain.
- Selection: the remaining problems are shuffled with `random.Random(0)`. Going through them in order, a
  problem is kept if both of its prompts have at most 384 tokens, until 3,200 are kept. 3,228 were examined and
  28 skipped. Kept student prompts have 26–342 tokens (median 84), teacher prompts 68–384 (median 126).
- Step $`k`$ uses kept problems $`64(k-1)+1, \dots, 64k`$. No problem is used twice.

### 4.4 Rollout sampling

- The student samples one response per problem by ancestral sampling from $`\pi_{\theta_{k-1}}`$: temperature 1,
  no top-k, top-p or min-p truncation, no repetition, presence or frequency penalty.
- A response ends with the first stop token (`<|im_end|>`, id 248046, or `<|endoftext|>`, id 248044), which is
  part of the response and of the loss, or after $`L_{\max} = 4096`$ tokens without one.
- The sampling key of step $`k`$ is derived from the integer `seed + k` with seed 0.
- The 64 prompts are sampled as one batch. Student sequences are right-padded to 342 + 4,096 = 4,438 tokens and
  teacher sequences to 384 + 4,096 = 4,480; padding is masked out of attention and of the loss.

### 4.5 Adapters

Every two-dimensional full-precision weight matrix of the student's transformer stack gets an adapter. With
$`W \in ℝ^{d_{\mathrm{out}} \times d_{\mathrm{in}}}`$ the frozen matrix and $`r = 64`$,

```math
W' = W + U D, \qquad U \in ℝ^{d_{\mathrm{out}} \times r}, \quad D \in ℝ^{r \times d_{\mathrm{in}}},
\qquad (7)
```

with no scaling factor. Initialization: $`U = 0`$ and $`D_{ab} \sim 𝒩(0,\, 1/d_{\mathrm{in}})`$
independently, from `jax.random.key(0)` split once per matrix. Both $`U`$ and $`D`$ are trained, in float32.
Because $`U = 0`$ at the start, the student with $`\theta_0`$ computes exactly what the base model computes.

| Adapted matrix (per layer) | Shape $`d_{\mathrm{out}} \times d_{\mathrm{in}}`$ | Layers | Adapter parameters |
|---|---|---:|---:|
| Mixer input projection | 12,352 × 4,096 | 24 | 25,264,128 |
| Mixer input projection | 10,240 × 4,096 | 8 | 7,340,032 |
| Mixer output projection | 4,096 × 4,096 | 32 | 16,777,216 |
| MLP up-projection | 24,576 × 4,096 | 32 | 58,720,256 |
| MLP down-projection | 4,096 × 12,288 | 32 | 33,554,432 |
| Total | 128 matrices | | 141,656,064 |

Not adapted: the input and output embeddings (248,320 × 4,096 each) and all normalization parameters.

### 4.6 Optimizer

One update per step on $`\hat{G}_k`$:

1. Global-norm clipping at 1.0: $`\hat{G}_k \leftarrow \hat{G}_k \cdot \min\bigl(1,\, 1 / \lVert \hat{G}_k \rVert_2\bigr)`$.
   The largest norm in the run was 0.314, so clipping never changed a gradient.
2. Adam (`optax.adamw` with `weight_decay=0.0`): learning rate $`10^{-4}`$, constant, no warm-up or decay;
   $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$, $`\varepsilon = 10^{-8}`$, with bias correction.

### 4.7 Numerics

- Base weights are bfloat16 and stay frozen; adapters, optimizer state, losses and gradients are float32.
- Logits are cast to float32 before the log-softmax over the full vocabulary.
- The log-probabilities in (2), for both models, come from lalamo's inference forward pass
  (`DecoderForwardPassConfig.for_inference()`), as does sampling. The gradient of (5) is taken through the
  training forward pass (`DecoderForwardPassConfig.for_training()`), which uses a different attention kernel.
  The two passes compute the same function up to floating-point differences.
- The gradient is compiled under JAX's standard dtype promotion instead of lalamo's strict setting, because the
  backward pass of the attention kernel multiplies float32 cotangents with bfloat16 activations.

### 4.8 One step

1. Sample 64 responses from the student (section 4.4).
2. Place the responses behind the teacher's prompts and compute
   $`\log\,\pi_{\mathrm{T}}(y_{i,t} \mid \tilde{s}_{i,t})`$ for every response token, two rollouts at a time.
   Shift the values by the difference in prompt length so that they line up with the student's positions.
3. Compute $`\log\,\pi_{\theta_{k-1}}(y_{i,t} \mid s_{i,t})`$ with the student, two rollouts at a time.
4. Form gaps, baseline and advantages (equations 2–4) over the whole step.
5. Accumulate the gradient (equation 6), one rollout at a time.
6. Clip and apply one Adam update, giving $`\theta_k`$.
7. Append the step's metrics to `metrics.jsonl`. After steps 5, 10, …, 50, save a checkpoint (adapters,
   optimizer state, step count).

### 4.9 Configuration

All values are the defaults of the script and were written to the run's `config.json`.

| Setting | Value |
|---|---|
| Steps $`K`$ | 50 |
| Rollouts per step $`B`$ | 64 |
| Maximum response length $`L_{\max}`$ | 4,096 tokens |
| Maximum prompt length | 384 tokens |
| Adapter rank $`r`$ | 64 |
| Learning rate | $`10^{-4}`$ |
| Gradient clip norm | 1.0 |
| Advantage baseline | batch mean (equation 3) |
| Seed | 0 |
| Gradient micro-batch | 1 rollout |
| Student scoring micro-batch | 2 rollouts |
| Teacher scoring micro-batch | 2 rollouts |
| Checkpoint interval | 5 steps |
| Evaluation interval | 10 steps |

### 4.10 Hardware, software and cost

- Two NVIDIA B200 GPUs (183,359 MiB each): the student on the first visible device, the teacher on the last.
- Python 3.12.9, JAX 0.11.0, Optax 0.2.8, Equinox 0.13.7, lalamo at commit `f3d9f59c` plus the code of this
  pull request.
- The bfloat16 weights take about 16.7 GiB (student) and 50.1 GiB (teacher).
- Time per step, mean over steps 11–50: sampling 54.8 s, teacher scoring 28.7 s, student scoring 10.5 s,
  gradient and update 62.8 s; about 2.6 minutes in total.
- Peak GPU memory of the process during a step (steps 11–50, read from `nvidia-smi` every 2 seconds over one
  step): about 122 GB on the student's GPU, during the backward pass, and about 94 GB on the teacher's, during
  scoring.
- Evaluations: the teacher's MATH-500 evaluation took 31 minutes; one round of student evaluations (both
  benchmarks) took 11–13 minutes.
- Wall-clock time from launch to the last evaluation: 4 hours 3 minutes, including the two restarts below.

### 4.11 Restarts during the run

The run was stopped and resumed from a checkpoint twice, to reduce memory use. Resuming restores the adapters
and the optimizer state, and prompts and sampling keys are functions of the step index, so a resumed step is
the step an uninterrupted run would have taken. `tests/unit/test_on_policy_distillation_runner.py` checks this
equivalence on a tiny model. Finished evaluations are stored in `results.json` and were not repeated.

| Segment | Steps | Evaluations run | Difference from the previous segment |
|---|---|---|---|
| 1 | 1–5 | Teacher; student at step 0 | — |
| 2 | 6–10 | Student at step 10 | Micro-batch gradients are added to a running sum instead of being held all at once. Evaluation results are logged to wandb on their own axis |
| 3 | 11–50 | Student at steps 20, 30, 40, 50 | Only the adapters pass through the compiled optimizer update, so the frozen weights are no longer copied every step. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` |

None of these changes alters the arithmetic of a step. The distillation package, the script and the tests in
this pull request are identical to the files used for segment 3. Gradient time rose from about 50 s to about 63 s per step with the `platform`
allocator.

## 5. Evaluation protocol

| | MATH-500 | MMLU-Pro |
|---|---|---|
| Purpose | The distilled skill | Check against forgetting: never trained on |
| Source | `HuggingFaceH4/MATH-500`, `test.jsonl`, revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | `TIGER-Lab/MMLU-Pro`, `data/test-00000-of-00001.parquet`, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| Pool | 500 problems | 12,032 questions |
| Evaluated | 200, drawn with `random.Random(0).sample` | 300, drawn with `random.Random(0).sample`; 4 to 10 options each |
| Student evaluated | Before training and after steps 10, 20, 30, 40, 50 | Same |
| Teacher evaluated | Once | Not evaluated |
| Response limit | 8,192 tokens | 4,096 tokens |
| Batch size | 100 (student), 50 (teacher) | 100 (student) |

**Prompts.** MATH-500 uses the training prompt format of section 4.2. An MMLU-Pro question is rendered as

```text
{question}

A. {first option}
B. {second option}
...

Finish your reply with a line of the form "Answer: X", where X is the letter of the correct option.
```

with the same chat templates as in section 4.2, so reasoning is on.

**Sampling.** Each model uses its own default sampling settings, which differ from training and from each
other:

| | Temperature | Top-k | Top-p |
|---|---:|---:|---:|
| Student (`Qwen/Qwen3.5-9B`; lalamo takes these from `Qwen/Qwen3.5-27B`'s `generation_config.json`) | 0.6 | 20 | 0.95 |
| Teacher (`Qwen/Qwen3.8-27B`) | 1.0 | 20 | 0.95 |
| Training rollouts, for comparison | 1.0 | off | off |

One response is sampled per question. Batch $`j`$ (counting from 0) of an evaluation uses the sampling key
derived from the integer $`j`$, so every checkpoint is evaluated with the same keys.

**Grading.**

1. The *final reply* is the text after the last `</think>`. A response with no `</think>` (cut off while
   reasoning) has no final reply.
2. MATH-500: the answer is the content of the last `\boxed{...}` in the final reply, with braces matched.
   Answer and reference are normalised as strings: `\text{}`, `\textbf{}`, `\mbox{}` and `\mathrm{}` wrappers,
   `\left`, `\right`, spacing commands, dollar signs, degree and percent signs are removed; `\dfrac` and
   `\tfrac` become `\frac`; `\frac12` becomes `\frac{1}{2}`; whitespace and a trailing period are removed; a
   leading `x=` (left side of at most two characters) is dropped. The answer is correct if the two strings are
   equal. Mathematically equal forms that differ as strings, such as `1/2` and `0.5`, count as wrong.
3. MMLU-Pro: the answer is the letter of the last match of `Answer:\s*\(?([A-J])\)?` in the final reply, and is
   correct if it equals the reference letter.

**Reported numbers.**

- *Accuracy*: correct answers divided by all questions. A question without an answer counts as wrong.
- *Answered*: questions with an extracted answer (a final reply containing a `\boxed{}` or an `Answer:` line)
  divided by all questions.
- *Mean response length*: mean number of generated tokens per question, including the stop token. Responses
  that hit the limit count with the limit.

Only these totals were saved; per-question outcomes and the generated texts were not.

## 6. Logged training metrics

All except the adapter norm are computed from the rollouts of the step, before the update, so they describe
the student with parameters $`\theta_{k-1}`$.

| Metric | Definition |
|---|---|
| Reverse KL | $`-\frac{1}{N} \sum_{i,t} g_{i,t}`$: sampled-token estimate of the token-averaged $`\ell`$ at the visited contexts |
| Student entropy | $`-\frac{1}{N} \sum_{i,t} \log\,\pi_{\theta_{k-1}}(y_{i,t} \mid s_{i,t})`$: sampled-token estimate of the student's mean next-token entropy |
| Teacher cross-entropy | $`-\frac{1}{N} \sum_{i,t} \log\,\pi_{\mathrm{T}}(y_{i,t} \mid \tilde{s}_{i,t})`$. Reverse KL = teacher cross-entropy − student entropy |
| Gradient norm | $`\lVert \hat{G}_k \rVert_2`$ over all adapter parameters, before clipping |
| Adapter norm | $`\bigl(\sum \lVert U \rVert_F^2\bigr)^{1/2}`$ over all adapters, after the update. Zero at initialization |
| Mean length | $`N / B`$ |
| Finished | Fraction of the 64 responses that end with a stop token |
| Gap quantiles | 1st, 10th, 50th, 90th and 99th percentile of $`g_{i,t}`$ over all response tokens of the step |
| Gap < −1 | Fraction of response tokens with $`g_{i,t} < -1`$: the teacher gives the sampled token less than $`e^{-1} \approx 37\%`$ of the probability the student gives it |
| Reverse KL by position | Reverse KL restricted to response positions 0–1,023, 1,024–2,047, 2,048–3,071 and 3,072–4,095 |
| Rollout reverse KL | Minimum, median and maximum over the 64 rollouts of $`-\frac{1}{L_i} \sum_t g_{i,t}`$ |

## 7. Results

### 7.1 Evaluations

MATH-500, 200 problems, response limit 8,192 tokens:

| Model | Correct | Accuracy | Answered | Accuracy among answered | Mean response length |
|---|---:|---:|---:|---:|---:|
| Teacher | 173 | 86.5% | 92.5% | 93.5% | 1,622 |
| Student, before training | 98 | 49.0% | 51.0% | 96.1% | 5,862 |
| Student, step 10 | 155 | 77.5% | 84.5% | 91.7% | 3,254 |
| Student, step 20 | 149 | 74.5% | 80.0% | 93.1% | 2,787 |
| Student, step 30 | 163 | 81.5% | 87.0% | 93.7% | 2,298 |
| Student, step 40 | 150 | 75.0% | 82.0% | 91.5% | 2,259 |
| Student, step 50 | 160 | 80.0% | 87.0% | 92.0% | 1,862 |

MMLU-Pro, 300 questions, response limit 4,096 tokens:

| Model | Correct | Accuracy | Answered | Accuracy among answered | Mean response length |
|---|---:|---:|---:|---:|---:|
| Student, before training | 95 | 31.7% | 34.0% | 93.1% | 3,390 |
| Student, step 10 | 192 | 64.0% | 71.7% | 89.3% | 2,433 |
| Student, step 20 | 206 | 68.7% | 78.3% | 87.7% | 2,015 |
| Student, step 30 | 215 | 71.7% | 83.7% | 85.7% | 1,493 |
| Student, step 40 | 206 | 68.7% | 75.0% | 91.6% | 1,663 |
| Student, step 50 | 211 | 70.3% | 77.0% | 91.3% | 1,532 |

The binomial standard error of one accuracy is about 2.8 points at 80% with 200 questions and 2.6 points at
70% with 300 questions.

### 7.2 Training metrics

<details>
<summary>Scalar metrics for all 50 steps</summary>

| Step | Reverse KL | Student entropy | Teacher cross-entropy | Gradient norm | Adapter norm | Response tokens | Mean length | Finished | Gap < −1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0997 | 0.3541 | 0.4538 | 0.167 | 0.94 | 255,284 | 3989 | 0.08 | 4.5% |
| 2 | 0.0785 | 0.2622 | 0.3407 | 0.109 | 1.47 | 229,713 | 3589 | 0.25 | 3.3% |
| 3 | 0.0720 | 0.2285 | 0.3004 | 0.103 | 1.88 | 225,111 | 3517 | 0.33 | 3.0% |
| 4 | 0.0739 | 0.2299 | 0.3038 | 0.094 | 2.20 | 231,746 | 3621 | 0.27 | 3.0% |
| 5 | 0.0702 | 0.2259 | 0.2961 | 0.089 | 2.48 | 246,881 | 3858 | 0.16 | 2.9% |
| 6 | 0.0749 | 0.2558 | 0.3307 | 0.081 | 2.73 | 240,789 | 3762 | 0.20 | 3.1% |
| 7 | 0.0799 | 0.2844 | 0.3643 | 0.081 | 2.96 | 235,264 | 3676 | 0.27 | 3.3% |
| 8 | 0.0742 | 0.2669 | 0.3411 | 0.092 | 3.18 | 240,897 | 3764 | 0.17 | 3.1% |
| 9 | 0.0810 | 0.3029 | 0.3839 | 0.071 | 3.38 | 236,424 | 3694 | 0.22 | 3.3% |
| 10 | 0.0789 | 0.2971 | 0.3760 | 0.070 | 3.57 | 233,150 | 3643 | 0.27 | 3.2% |
| 11 | 0.0787 | 0.3206 | 0.3993 | 0.062 | 3.75 | 240,685 | 3761 | 0.20 | 3.2% |
| 12 | 0.0733 | 0.2918 | 0.3651 | 0.066 | 3.93 | 219,829 | 3435 | 0.28 | 2.9% |
| 13 | 0.0725 | 0.3127 | 0.3853 | 0.063 | 4.10 | 238,726 | 3730 | 0.20 | 2.8% |
| 14 | 0.0649 | 0.2840 | 0.3489 | 0.058 | 4.26 | 228,323 | 3568 | 0.25 | 2.5% |
| 15 | 0.0678 | 0.3160 | 0.3838 | 0.072 | 4.40 | 221,220 | 3457 | 0.25 | 2.5% |
| 16 | 0.0668 | 0.3266 | 0.3934 | 0.058 | 4.55 | 214,049 | 3345 | 0.33 | 2.5% |
| 17 | 0.0694 | 0.3667 | 0.4361 | 0.051 | 4.68 | 222,039 | 3469 | 0.30 | 2.6% |
| 18 | 0.0706 | 0.3860 | 0.4566 | 0.055 | 4.82 | 209,682 | 3276 | 0.31 | 2.6% |
| 19 | 0.0741 | 0.4171 | 0.4912 | 0.058 | 4.95 | 209,331 | 3271 | 0.33 | 2.9% |
| 20 | 0.0706 | 0.4252 | 0.4957 | 0.060 | 5.07 | 198,935 | 3108 | 0.39 | 2.8% |
| 21 | 0.0776 | 0.4862 | 0.5638 | 0.055 | 5.19 | 203,288 | 3176 | 0.42 | 3.0% |
| 22 | 0.0715 | 0.4120 | 0.4835 | 0.055 | 5.31 | 200,572 | 3134 | 0.38 | 2.8% |
| 23 | 0.0785 | 0.4553 | 0.5338 | 0.055 | 5.43 | 213,410 | 3335 | 0.36 | 3.0% |
| 24 | 0.0772 | 0.4264 | 0.5035 | 0.057 | 5.54 | 222,358 | 3474 | 0.30 | 2.9% |
| 25 | 0.0821 | 0.4304 | 0.5125 | 0.070 | 5.65 | 189,549 | 2962 | 0.48 | 3.1% |
| 26 | 0.0838 | 0.4475 | 0.5313 | 0.063 | 5.75 | 196,811 | 3075 | 0.44 | 3.2% |
| 27 | 0.0840 | 0.4266 | 0.5107 | 0.064 | 5.85 | 203,807 | 3184 | 0.42 | 3.3% |
| 28 | 0.0956 | 0.5254 | 0.6210 | 0.074 | 5.95 | 214,651 | 3354 | 0.34 | 3.9% |
| 29 | 0.1083 | 0.5854 | 0.6937 | 0.097 | 6.04 | 169,319 | 2646 | 0.58 | 4.3% |
| 30 | 0.1072 | 0.6579 | 0.7651 | 0.083 | 6.13 | 198,007 | 3094 | 0.41 | 4.5% |
| 31 | 0.1157 | 0.6870 | 0.8027 | 0.086 | 6.23 | 186,689 | 2917 | 0.48 | 4.9% |
| 32 | 0.1256 | 0.7329 | 0.8585 | 0.094 | 6.34 | 170,180 | 2659 | 0.50 | 5.4% |
| 33 | 0.1432 | 0.8234 | 0.9666 | 0.119 | 6.45 | 183,251 | 2863 | 0.53 | 6.3% |
| 34 | 0.1605 | 0.9016 | 1.0622 | 0.132 | 6.57 | 165,953 | 2593 | 0.53 | 7.2% |
| 35 | 0.1648 | 0.8531 | 1.0180 | 0.137 | 6.68 | 176,008 | 2750 | 0.53 | 7.3% |
| 36 | 0.1649 | 0.8186 | 0.9836 | 0.110 | 6.80 | 149,815 | 2341 | 0.59 | 7.2% |
| 37 | 0.1560 | 0.8811 | 1.0371 | 0.117 | 6.91 | 195,466 | 3054 | 0.38 | 6.9% |
| 38 | 0.1423 | 0.8160 | 0.9583 | 0.107 | 7.03 | 189,343 | 2958 | 0.42 | 6.4% |
| 39 | 0.1373 | 0.8000 | 0.9373 | 0.095 | 7.15 | 196,100 | 3064 | 0.36 | 6.0% |
| 40 | 0.1463 | 0.7594 | 0.9057 | 0.116 | 7.28 | 193,106 | 3017 | 0.44 | 6.4% |
| 41 | 0.1502 | 0.7752 | 0.9253 | 0.107 | 7.40 | 164,145 | 2565 | 0.50 | 6.5% |
| 42 | 0.1555 | 0.7764 | 0.9319 | 0.101 | 7.53 | 166,105 | 2595 | 0.58 | 6.6% |
| 43 | 0.1617 | 0.7665 | 0.9282 | 0.107 | 7.65 | 179,312 | 2802 | 0.47 | 6.9% |
| 44 | 0.1547 | 0.7159 | 0.8706 | 0.117 | 7.77 | 145,378 | 2272 | 0.59 | 6.6% |
| 45 | 0.1644 | 0.8432 | 1.0076 | 0.111 | 7.88 | 160,323 | 2505 | 0.56 | 7.2% |
| 46 | 0.1650 | 0.7555 | 0.9206 | 0.314 | 7.99 | 160,914 | 2514 | 0.56 | 7.3% |
| 47 | 0.1642 | 0.7905 | 0.9547 | 0.133 | 8.11 | 162,367 | 2537 | 0.52 | 7.2% |
| 48 | 0.1625 | 0.8029 | 0.9654 | 0.097 | 8.23 | 191,638 | 2994 | 0.42 | 7.1% |
| 49 | 0.1661 | 0.7602 | 0.9263 | 0.098 | 8.35 | 163,838 | 2560 | 0.53 | 7.2% |
| 50 | 0.1565 | 0.6899 | 0.8464 | 0.105 | 8.46 | 141,380 | 2209 | 0.66 | 6.5% |

</details>

Distribution of the log-probability gap, reverse KL by response position, and reverse KL per rollout, at
selected steps:

| Step | Gap p01 | p10 | p50 | p90 | p99 | KL at 0–1,023 | 1,024–2,047 | 2,048–3,071 | 3,072–4,095 | Rollout KL min | median | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | −2.42 | −0.39 | −0.0000 | 0.09 | 0.91 | 0.114 | 0.097 | 0.095 | 0.093 | 0.053 | 0.098 | 0.171 |
| 5 | −1.86 | −0.22 | −0.0000 | 0.02 | 0.80 | 0.087 | 0.064 | 0.063 | 0.065 | 0.034 | 0.065 | 0.142 |
| 10 | −1.94 | −0.30 | −0.0000 | 0.05 | 0.84 | 0.098 | 0.072 | 0.070 | 0.072 | 0.031 | 0.074 | 0.150 |
| 14 | −1.67 | −0.25 | −0.0000 | 0.03 | 0.80 | 0.084 | 0.057 | 0.055 | 0.060 | 0.037 | 0.064 | 0.106 |
| 20 | −1.75 | −0.31 | −0.0000 | 0.08 | 0.87 | 0.094 | 0.062 | 0.060 | 0.057 | 0.026 | 0.070 | 0.171 |
| 25 | −1.85 | −0.36 | −0.0001 | 0.08 | 0.93 | 0.108 | 0.072 | 0.070 | 0.065 | 0.043 | 0.083 | 0.178 |
| 27 | −1.82 | −0.38 | −0.0001 | 0.07 | 0.95 | 0.112 | 0.070 | 0.072 | 0.073 | 0.017 | 0.082 | 0.220 |
| 30 | −2.00 | −0.54 | −0.0004 | 0.16 | 1.10 | 0.136 | 0.094 | 0.097 | 0.091 | 0.039 | 0.115 | 0.297 |
| 34 | −2.67 | −0.77 | −0.0005 | 0.24 | 1.22 | 0.203 | 0.146 | 0.137 | 0.135 | 0.096 | 0.174 | 0.435 |
| 40 | −2.43 | −0.70 | −0.0005 | 0.19 | 1.18 | 0.181 | 0.140 | 0.127 | 0.124 | 0.087 | 0.151 | 0.354 |
| 45 | −2.62 | −0.77 | −0.0007 | 0.21 | 1.20 | 0.199 | 0.156 | 0.149 | 0.128 | 0.112 | 0.171 | 0.337 |
| 50 | −2.55 | −0.71 | −0.0005 | 0.16 | 1.16 | 0.185 | 0.144 | 0.131 | 0.141 | 0.085 | 0.159 | 0.318 |

### 7.3 Observations

These are readings of the tables above. None was tested with a further experiment.

1. **Most of the accuracy gain is the student finishing.** Before training the student left 49% of MATH-500
   and 66% of MMLU-Pro without an answer. Its mean response lengths are consistent with those responses still
   reasoning at the limit, but this was not checked per question. After training it answers 87% and 77%, with a mean response length close to the teacher's (1,862 against
   1,622 tokens on MATH-500). Accuracy among answered questions did not rise (96.1% → 92.0% on MATH-500),
   though the answered set now includes questions the base student could not finish.
2. **Accuracy is flat from step 10.** From step 10 to step 50 accuracy stays within 74.5–81.5% on MATH-500
   and 64.0–71.7% on MMLU-Pro, a spread of about two standard errors of a difference between two checkpoints
   (about 4 points). Mean response length is the only evaluation number that keeps moving: it falls at every
   evaluation on MATH-500.
3. **The gain carries over to MMLU-Pro**, which was never trained on, and nothing collapsed there. The
   evaluation cannot detect small losses of knowledge, because the set of answered questions changes.
4. **The reverse-KL estimate has three phases.** It falls from 0.100 to 0.065 by step 14 while the student's
   entropy drops and then slowly rises; it stays below 0.085 until step 27; it doubles to 0.16 between steps
   28 and 34 and stays at 0.14–0.17. From step 27 to step 34 the student's entropy rises from 0.43 to 0.90.
   Reverse KL as a fraction of student entropy is 28–32% over the first five steps and 16–23% from step 16
   on.
5. **The rise is spread over everything.** All four position ranges and the minimum, median and maximum
   rollout rise together. The median gap stays at zero while both tails widen (10th percentile from −0.25 to
   about −0.75, 90th percentile from 0.03 to about 0.2). The first 1,024 response positions have the highest
   reverse KL throughout.
6. **The student's reasoning style changes late in training.** Eight rollouts were saved at steps 1, 10, 20,
   30, 40 and 50. Sentences that begin with a bare "Need" (as in "Need solve. Let's parse.") appear in none of
   them through step 30, in five of eight at step 40 and in seven of eight at step 50. Whether this is the teacher's style was not checked: no teacher
   samples were saved.
7. **Training did not converge.** The adapter norm grows by about 0.115 per step from step 20 to step 50
   without flattening, and the gradient norm at the end (about 0.10) is above its minimum (0.05 at step 17).

## 8. Follow-up runs

Three further runs use the same student, adapters, optimizer, step count, training prompts and evaluation
protocol as the pilot (sections 4 and 5). Each changes only where the training signal comes from. Like the
pilot, each is a single run with one seed.

### 8.1 What each run changes

| Run | Responses trained on | Signal | Teacher during training | Per step |
|---|---|---|---|---|
| Pilot (sections 3–7) | Sampled by the student | Log-probability gap to the teacher | Scores the student's tokens | 64 prompts × 1 rollout |
| Supervised baseline | Written by the teacher beforehand | Cross-entropy on the teacher's tokens | Not used | 64 prompts × 1 teacher response |
| Reward only | Sampled by the student | Correctness of the final answer | Not used | 8 prompts × 8 rollouts |
| Distillation + reward | Sampled by the student | Both, with weights 1.0 and 0.2 | Scores the student's tokens | 8 prompts × 8 rollouts |

### 8.2 Losses

**Supervised baseline.** Let $`y^{\mathrm{T}}_i`$ be the teacher's response to problem $`x_i`$ and $`s'_{i,t} = (p^{\mathrm{S}}(x_i),\, y^{\mathrm{T}}_{i,<t})`$ the student's prompt followed by the teacher's response prefix. The loss of step $`k`$ is the cross-entropy on the teacher's tokens, averaged over the $`N`$ response tokens of the step:

```math
ℒ^{\mathrm{sup}}_k(\theta) = -\frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i} \log\,\pi_\theta\bigl(y^{\mathrm{T}}_{i,t} \mid s'_{i,t}\bigr).
\qquad (8)
```

It is computed as the loss (5) with every advantage set to 1.

**Reward.** A step samples $`G = 8`$ rollouts for each of 8 prompts. Rollout $`j`$ of prompt $`i`$ earns $`r_{i,j} = 1`$ if its final reply contains a `\boxed{}` answer equal to the reference answer of the problem under the string normalisation of section 5, and $`r_{i,j} = 0`$ otherwise. A rollout cut off while reasoning earns 0. Rewards are standardized among the rollouts of the same prompt, with $`m_i`$ and $`\sigma_i`$ the mean and the population standard deviation of $`r_{i,1}, \dots, r_{i,G}`$:

```math
\hat{r}_{i,j} = \frac{r_{i,j} - m_i}{\sigma_i} \quad \text{if } \sigma_i > 0, \qquad \hat{r}_{i,j} = 0 \quad \text{otherwise}.
\qquad (9)
```

A prompt whose rollouts all earned the same reward therefore contributes nothing to the step. Every response token of a rollout gets the standardized reward of the rollout, and the advantage of token $`t`$ of rollout $`(i,j)`$ is

```math
A_{i,j,t} = \alpha\,\bigl(g_{i,j,t} - b\bigr) + \beta\,\hat{r}_{i,j},
\qquad (10)
```

used in the loss (5). The reward-only run has $`\alpha = 0`$ and $`\beta = 1`$, and does not load the teacher. The distillation + reward run has $`\alpha = 1.0`$ and $`\beta = 0.2`$, with $`g`$ and $`b`$ as in equations (2) and (3) over the 64 rollouts of the step.

### 8.3 Setup differences from the pilot

- **Code.** The runs used this branch at commit `587b8b7f`, which is based on lalamo `main` at `1edce60e`. The
  pilot used `f3d9f59c`.
- **Evaluations before training.** The evaluations of the untrained student and of the teacher were not
  repeated. Their numbers are the pilot's.
- **Teacher responses** (supervised baseline). `scripts/generate_teacher_responses.py` sampled one response
  per training prompt from the teacher behind its own prompt (section 4.2), with the teacher's default
  sampling settings (temperature 1.0, top-k 20, top-p 0.95), in batches of 48 with the sampling key of batch
  $`j`$ derived from the integer $`j`$, cut off at 4,096 tokens. Of the
  3,200 responses, 1,927 (60.2%) end with a stop token and 1,371 (42.8%) are correct by the reward rule above;
  their mean length is 2,392 tokens. All of them were used for training, including the cut-off and the
  incorrect ones. Step $`k`$ trains on the responses to the prompts that the pilot used in step $`k`$:
  7,655,782 response tokens in total.
- **Prompts of the reward runs.** Step $`k`$ uses prompts $`8(k-1)+1, \dots, 8k`$ of the same selection, so the
  two runs see the first 400 of the pilot's 3,200 prompts. The 64 rollouts of a step are sampled in one batch
  as in section 4.4.
- **Gradient clipping.** The gradient norm exceeded 1.0 once, in step 1 of the supervised run (1.256), where
  it was clipped.
- **Hardware.** The student's GPU was shared with another job during all three runs, so step times are not
  comparable with the pilot's. The reward-only run ran out of memory twice (during steps 25 and 30) and
  resumed from the checkpoints of steps 20 and 25.

### 8.4 Evaluations

MATH-500, 200 problems: accuracy, with the answered fraction in parentheses. The teacher scores 86.5% (92.5%).

| Step | Pilot | Supervised | Reward only | Distillation + reward |
|---:|---:|---:|---:|---:|
| 0 | 49.0% (51.0%) | 49.0% (51.0%) | 49.0% (51.0%) | 49.0% (51.0%) |
| 10 | 77.5% (84.5%) | 78.5% (87.0%) | 83.5% (93.5%) | 78.0% (84.0%) |
| 20 | 74.5% (80.0%) | 75.5% (81.5%) | 75.0% (88.5%) | 61.5% (67.0%) |
| 30 | 81.5% (87.0%) | 75.5% (84.0%) | 65.5% (95.5%) | 78.5% (85.0%) |
| 40 | 75.0% (82.0%) | 79.0% (86.0%) | 65.5% (97.5%) | 70.0% (76.5%) |
| 50 | 80.0% (87.0%) | 78.5% (86.5%) | 67.5% (98.5%) | 82.0% (87.5%) |

MMLU-Pro, 300 questions: accuracy, with the answered fraction in parentheses.

| Step | Pilot | Supervised | Reward only | Distillation + reward |
|---:|---:|---:|---:|---:|
| 0 | 31.7% (34.0%) | 31.7% (34.0%) | 31.7% (34.0%) | 31.7% (34.0%) |
| 10 | 64.0% (71.7%) | 56.7% (61.3%) | 76.7% (88.3%) | 67.3% (76.0%) |
| 20 | 68.7% (78.3%) | 61.0% (66.3%) | 74.7% (91.0%) | 57.0% (64.0%) |
| 30 | 71.7% (83.7%) | 61.3% (64.3%) | 71.7% (95.7%) | 72.7% (83.7%) |
| 40 | 68.7% (75.0%) | 66.0% (71.0%) | 70.3% (100.0%) | 58.7% (63.0%) |
| 50 | 70.3% (77.0%) | 67.3% (72.0%) | 76.0% (97.3%) | 65.0% (68.7%) |

Mean response length in tokens (the teacher's is 1,622 on MATH-500):

| Step | MATH-500: Pilot | Supervised | Reward only | Distillation + reward | MMLU-Pro: Pilot | Supervised | Reward only | Distillation + reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5,862 | 5,862 | 5,862 | 5,862 | 3,390 | 3,390 | 3,390 | 3,390 |
| 10 | 3,254 | 1,986 | 2,227 | 3,373 | 2,433 | 2,463 | 1,790 | 2,299 |
| 20 | 2,787 | 2,177 | 1,729 | 3,505 | 2,015 | 2,061 | 1,238 | 2,243 |
| 30 | 2,298 | 2,141 | 1,163 | 2,459 | 1,493 | 2,011 | 962 | 1,571 |
| 40 | 2,259 | 1,970 | 1,177 | 2,825 | 1,663 | 1,867 | 960 | 2,143 |
| 50 | 1,862 | 1,885 | 1,393 | 1,880 | 1,532 | 1,746 | 1,234 | 1,935 |

Accuracy among answered questions at step 50:

| Run | MATH-500 | MMLU-Pro |
|---|---:|---:|
| Student before training | 96.1% | 93.1% |
| Pilot | 92.0% | 91.3% |
| Supervised | 90.8% | 93.5% |
| Reward only | 68.5% | 78.1% |
| Distillation + reward | 93.7% | 94.7% |

One accuracy has a standard error of about 2.8 points on MATH-500 and 2.6 on MMLU-Pro, so the difference
between two runs has one of about 4 points.

### 8.5 Training metrics

Supervised baseline. "Finished" is the fraction of the step's teacher responses that end with a stop token.

| Step | Cross-entropy | Gradient norm | Adapter norm | Response tokens | Mean length | Finished |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1544 | 1.256 | 0.95 | 141,051 | 2204 | 0.62 |
| 5 | 0.8508 | 0.350 | 2.88 | 159,557 | 2493 | 0.59 |
| 10 | 0.8220 | 0.177 | 4.33 | 164,123 | 2564 | 0.56 |
| 15 | 0.7900 | 0.136 | 5.43 | 169,411 | 2647 | 0.53 |
| 20 | 0.7428 | 0.117 | 6.25 | 141,391 | 2209 | 0.62 |
| 25 | 0.7328 | 0.114 | 6.91 | 146,937 | 2296 | 0.64 |
| 30 | 0.7378 | 0.105 | 7.45 | 151,621 | 2369 | 0.62 |
| 35 | 0.7374 | 0.106 | 7.90 | 180,302 | 2817 | 0.48 |
| 40 | 0.6797 | 0.106 | 8.27 | 149,561 | 2337 | 0.61 |
| 45 | 0.7364 | 0.095 | 8.61 | 155,895 | 2436 | 0.58 |
| 50 | 0.6690 | 0.100 | 8.92 | 140,756 | 2199 | 0.66 |

Reward only. "Mean reward" is the fraction of the step's 64 rollouts with a correct answer, and "informative
prompts" counts the prompts whose rollouts did not all earn the same reward.

| Step | Student entropy | Mean reward | Informative prompts | Gradient norm | Adapter norm | Mean length | Finished |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3502 | 0.031 | 1/8 | 0.014 | 0.90 | 4079 | 0.03 |
| 5 | 0.3130 | 0.344 | 3/8 | 0.028 | 2.35 | 3621 | 0.36 |
| 10 | 0.2679 | 0.344 | 3/8 | 0.030 | 3.42 | 3321 | 0.47 |
| 15 | 0.3147 | 0.375 | 2/8 | 0.038 | 4.22 | 2365 | 0.64 |
| 20 | 0.3159 | 0.484 | 5/8 | 0.054 | 4.88 | 2327 | 0.72 |
| 25 | 0.4163 | 0.094 | 3/8 | 0.041 | 5.44 | 2070 | 0.88 |
| 30 | 0.3689 | 0.266 | 4/8 | 0.073 | 5.98 | 929 | 1.00 |
| 35 | 0.3280 | 0.328 | 2/8 | 0.048 | 6.49 | 1178 | 1.00 |
| 40 | 0.3447 | 0.219 | 3/8 | 0.050 | 7.01 | 1677 | 1.00 |
| 45 | 0.3699 | 0.219 | 4/8 | 0.065 | 7.46 | 1566 | 0.98 |
| 50 | 0.2519 | 0.500 | 5/8 | 0.063 | 7.87 | 1489 | 1.00 |

Distillation + reward.

| Step | Reverse KL | Student entropy | Mean reward | Informative prompts | Gradient norm | Adapter norm | Mean length | Finished |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0968 | 0.3502 | 0.031 | 1/8 | 0.171 | 0.94 | 4079 | 0.03 |
| 5 | 0.0768 | 0.2561 | 0.219 | 3/8 | 0.105 | 2.48 | 3866 | 0.22 |
| 10 | 0.0668 | 0.2553 | 0.141 | 1/8 | 0.073 | 3.57 | 3592 | 0.23 |
| 15 | 0.0626 | 0.2799 | 0.203 | 2/8 | 0.078 | 4.39 | 2781 | 0.45 |
| 20 | 0.0789 | 0.4588 | 0.094 | 2/8 | 0.065 | 5.02 | 3592 | 0.27 |
| 25 | 0.0795 | 0.4857 | 0.141 | 3/8 | 0.062 | 5.54 | 3906 | 0.16 |
| 30 | 0.1196 | 0.6436 | 0.297 | 4/8 | 0.103 | 5.99 | 2548 | 0.64 |
| 35 | 0.1486 | 0.7925 | 0.438 | 1/8 | 0.136 | 6.47 | 2419 | 0.56 |
| 40 | 0.1537 | 0.8002 | 0.203 | 3/8 | 0.111 | 7.03 | 3451 | 0.31 |
| 45 | 0.1709 | 0.6853 | 0.172 | 4/8 | 0.169 | 7.60 | 2817 | 0.44 |
| 50 | 0.1941 | 0.8308 | 0.531 | 2/8 | 0.212 | 8.19 | 2342 | 0.69 |

### 8.6 Observations

These are readings of the tables above. None was tested with a further experiment. A difference of less than
about 8 points between two runs is within two standard errors.

1. **The supervised baseline matches the pilot on MATH-500.** The two differ by at most 6 points at any
   evaluation and by 1.5 points at step 50. On MMLU-Pro the pilot is ahead at every evaluation, by 2.7 to 10.4
   points (3.0 at step 50). The supervised student reaches the teacher's response length sooner (1,986
   tokens on MATH-500 at step 10, against 3,254).
2. **The supervised baseline did this on imperfect data:** 40% of the teacher responses it trained on were cut
   off and 57% were not graded correct.
3. **Reward only is best early and then trades accuracy for answering.** At step 10 it has the highest
   accuracy of any run at any evaluation on both benchmarks (83.5% and 76.7%). Afterwards it answers almost
   every question (98.5% and 97.3% at step 50) with shorter responses, while accuracy among answered MATH-500
   questions falls from 89.3% at step 10 to 68.5% at step 50. In training, at least 97% of the rollouts of
   every step finish from step 28 on, at a mean length of 929–1,811 tokens against a limit of 4,096.
4. **The training reward does not rise after the first steps.** Averaged over ten steps, the mean reward of
   the reward-only run is 0.31, 0.42, 0.36, 0.29 and 0.32. A step sees only 8 prompts, on average 3.3 of them
   informative, so a single step's reward mostly reflects which prompts it drew.
5. **Distillation + reward is the least stable run.** Its MATH-500 accuracy moves between 61.5% and 82.0%
   from one evaluation to the next, and its MMLU-Pro accuracy between 57.0% and 72.7%. Its reverse-KL estimate
   falls to 0.058 at step 13 and rises to 0.194 at step 50, above the pilot's 0.157. Each of its steps scores
   8 distinct prompts where the pilot's scores 64.

## 9. Action items

Last updated: 2026-11-13. Items are ticked off, and their results added to sections 7 and 8, as the work is
done.

**Open: evaluation**

- [ ] Evaluate on all of MATH-500 and a larger MMLU-Pro sample, and repeat the four runs with more seeds. One
  accuracy now has a standard error of 2.6–2.8 points, which is too coarse to rank the runs.
- [ ] Save per-question outcomes and generated texts, then compare runs and checkpoints on the same questions
  and measure forgetting directly. Only totals were saved so far.
- [ ] Evaluate the base student with a larger response limit, to separate "finishes sooner" from "answers
  better". Its score in these runs is set by the limit, and its mean response length is censored by it.
- [ ] Evaluate teacher and student with the same sampling settings, and add an evaluation at the training
  temperature (section 5).
- [ ] Evaluate the teacher on MMLU-Pro.
- [ ] Grade MATH-500, and the training reward, by mathematical equivalence instead of string match.

**Open: training**

- [ ] Supervised baseline on the finished and correct teacher responses only. The first baseline trained on
  all of them, of which 40% were cut off and 57% not graded correct (section 8.3).
- [ ] Reward runs with more prompts per step. With 8 prompts, about 3 carry a signal in a step (section 8.6).
- [ ] A reward that does not pay for brevity alone. The reward-only run kept shortening its responses after
  accuracy had peaked at step 10 (section 8.6); options are a higher response limit in training or stopping on
  an evaluation.
- [ ] Find out why distillation + reward is unstable, and try other weights than 1.0 and 0.2 (section 8.6).
- [ ] Add a learning-rate schedule and try a lower learning rate. The adapters were still moving at step 50 in
  every run, and the reverse-KL estimate rose late in both runs that have one.
- [ ] Log two reference values: the teacher's entropy on its own samples, and the reverse KL computed over the
  full vocabulary. The sampled-token estimate alone is not a progress measure, because the contexts it
  averages over change during training (section 3.3).
- [ ] Check whether the terse reasoning style of the pilot's late checkpoints is the teacher's, using the
  teacher responses generated for the supervised baseline.
- [ ] Credit assignment across tokens (section 3.3).
- [ ] A baseline that does not depend on the sampled token (section 3.3).
- [ ] Give teacher and student the same prompt, or measure what the teacher's extra system message changes
  (section 4.2).
- [ ] Reduce the memory of the backward pass, so that more than one rollout fits in a gradient micro-batch
  (section 3.5).
- [ ] Re-run the pilot on the current base commit. It used lalamo at `f3d9f59c`; the follow-up runs used a
  later `main` (section 8.3).

**Done**

- [x] Pilot run of on-policy distillation (sections 4–7).
- [x] Supervised and reward-based training in `lalamo.distillation`, with unit tests.
- [x] Supervised baseline on teacher-written responses at the pilot's budget (section 8): level with the pilot
  on MATH-500, behind it on MMLU-Pro.
- [x] Reward-only run (section 8): best of all runs at step 10, then lower accuracy as it answered faster.
- [x] On-policy distillation combined with the reward (section 8): unstable from one evaluation to the next.

## 10. Reproduction

From the repository root, in an environment with lalamo's CUDA dependencies, on a machine where the first and
the last visible GPU can each hold one of the models (about 122 GB and 94 GB at peak during training with the
settings below; the teacher's evaluation at batch size 50 needs more):

```sh
XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python scripts/on_policy_distillation_pilot.py --output-directory <directory>
```

Every other option was left at its default. Passing `--wandb-project <project>` additionally logs to Weights &
Biases, which needs the `distillation` extra. The output directory receives `config.json`, `metrics.jsonl` (one
line per step), `results.json` (evaluations) and `checkpoints/step_XXXXX.eqx`. Running the same command again
with the same output directory resumes from the latest checkpoint.

Seeds are fixed, but bit-identical results across hardware, library versions or allocator settings were not
tested.

The follow-up runs of section 8 first need the teacher's responses, which take one GPU that can hold the
teacher:

```sh
python -m scripts.generate_teacher_responses --output-directory <responses>
```

Each follow-up run is then the pilot command with a method. The supervised and the reward-only run need only
the student's GPU:

```sh
python scripts/on_policy_distillation_pilot.py --output-directory <directory> \
    --method supervised --teacher-responses <responses>
python scripts/on_policy_distillation_pilot.py --output-directory <directory> \
    --method reward --teacher-responses <responses> --prompts-per-step 8 --rollouts-per-prompt 8
python scripts/on_policy_distillation_pilot.py --output-directory <directory> \
    --method distillation-and-reward --teacher-responses <responses> --prompts-per-step 8 --rollouts-per-prompt 8 \
    --distillation-weight 1.0 --reward-weight 0.2
```

With `--teacher-responses`, a run takes its training prompts from that directory. The follow-up runs also
started from a `results.json` holding the pilot's evaluations of the untrained student and of the teacher, which
the script then does not repeat.
