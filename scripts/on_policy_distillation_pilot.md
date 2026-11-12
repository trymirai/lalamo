# On-policy distillation pilot: Qwen3.8-27B → Qwen3.5-9B

This documents one run of `scripts/on_policy_distillation_pilot.py`: what was trained, on what, with which loss,
how it was evaluated, and what came out. It is a pilot (one run, one seed, small evaluation samples), meant to
validate the pipeline and to show where the recipe needs work, not to support claims about the method.

- [1. Summary](#1-summary)
- [2. Notation](#2-notation)
- [3. Loss](#3-loss)
- [4. Setup](#4-setup)
- [5. Evaluation protocol](#5-evaluation-protocol)
- [6. Logged training metrics](#6-logged-training-metrics)
- [7. Results](#7-results)
- [8. Limitations](#8-limitations)
- [9. Reproduction](#9-reproduction)

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

## 2. Notation

All logarithms are natural, so every information quantity is in nats.

| Symbol | Meaning |
|---|---|
| $`\mathcal{V}`$ | Token vocabulary, shared by both models: $`\lvert\mathcal{V}\rvert = 248{,}320`$ |
| $`x`$ | A training problem (text) |
| $`p^{\mathrm{S}}(x)`$, $`p^{\mathrm{T}}(x)`$ | Token sequences obtained by rendering $`x`$ with the student's and with the teacher's chat template. They differ (section 4.2) |
| $`y = (y_1, \dots, y_L)`$ | A response: the tokens the student sampled after $`p^{\mathrm{S}}(x)`$, with $`L \le L_{\max} = 4096`$. It contains the reasoning, the final reply and, if it was reached, the stop token |
| $`s_t = (p^{\mathrm{S}}(x),\, y_{<t})`$ | The student's context before response token $`t`$ |
| $`\tilde{s}_t = (p^{\mathrm{T}}(x),\, y_{<t})`$ | The same context as the teacher sees it: the teacher's prompt followed by the same response prefix |
| $`\theta`$ | The trained parameters: all low-rank adapter weights. $`\theta_{k-1}`$ is their value at the start of step $`k`$, and $`\theta_0`$ makes the student identical to the base model |
| $`\pi_\theta(v \mid s)`$ | Student's next-token distribution over $`v \in \mathcal{V}`$: the softmax of its logits, with no temperature or truncation |
| $`\pi_{\mathrm{T}}(v \mid \tilde{s})`$ | Teacher's next-token distribution, defined the same way |
| $`B`$, $`i`$ | Number of rollouts per step ($`B = 64`$) and the rollout index |
| $`L_i`$, $`N`$ | Length of response $`i`$, and the number of response tokens in the step, $`N = \sum_{i=1}^{B} L_i`$ |
| $`g_{i,t}`$ | Log-probability gap of a sampled token, teacher minus student (equation 3) |
| $`b`$, $`A_{i,t}`$ | Baseline and advantage (equations 4 and 5) |
| $`\mathrm{sg}[\cdot]`$ | Stop-gradient: the argument is treated as a constant when differentiating |
| $`K`$, $`k`$ | Number of optimizer steps ($`K = 50`$) and the step index, $`k = 1, \dots, K`$ |

## 3. Loss

### 3.1 Target quantity

For one context, the quantity being reduced is the reverse KL divergence from the student's next-token
distribution to the teacher's:

```math
\ell(\theta;\, s, \tilde{s})
= D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot \mid s) \,\big\Vert\, \pi_{\mathrm{T}}(\cdot \mid \tilde{s})\bigr)
= \sum_{v \in \mathcal{V}} \pi_\theta(v \mid s)\,\bigl[\log \pi_\theta(v \mid s) - \log \pi_{\mathrm{T}}(v \mid \tilde{s})\bigr].
\qquad (1)
```

Its gradient can be written as an expectation over a token sampled from the student, which is what makes a
sampled-token estimator possible. For any $`c`$ that does not depend on $`v`$,

```math
\nabla_\theta\, \ell(\theta;\, s, \tilde{s})
= \mathbb{E}_{v \sim \pi_\theta(\cdot \mid s)}
\Bigl[\bigl(\log \pi_\theta(v \mid s) - \log \pi_{\mathrm{T}}(v \mid \tilde{s}) - c\bigr)\,
\nabla_\theta \log \pi_\theta(v \mid s)\Bigr].
\qquad (2)
```

Derivation: differentiating (1) with the product rule gives

```math
\sum_{v} \nabla_\theta \pi_\theta(v \mid s)\,\bigl[\log \pi_\theta(v \mid s) - \log \pi_{\mathrm{T}}(v \mid \tilde{s})\bigr]
\;+\; \sum_{v} \pi_\theta(v \mid s)\, \nabla_\theta \log \pi_\theta(v \mid s).
```

The second sum is $`\nabla_\theta \sum_v \pi_\theta(v \mid s) = 0`$. In the first,
$`\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta`$ turns the sum into the expectation. The
constant drops out because $`\mathbb{E}_v[c\, \nabla_\theta \log \pi_\theta(v \mid s)] = c\, \nabla_\theta 1 = 0`$.

### 3.2 Sampled-token estimator used in the run

In step $`k`$ the student with parameters $`\theta_{k-1}`$ samples $`B`$ responses, one per problem. Every
response token $`y_{i,t}`$ is scored by both models, giving the log-probability gap

```math
g_{i,t} = \log \pi_{\mathrm{T}}\bigl(y_{i,t} \mid \tilde{s}_{i,t}\bigr) - \log \pi_{\theta_{k-1}}\bigl(y_{i,t} \mid s_{i,t}\bigr).
\qquad (3)
```

The baseline is the mean gap over all response tokens of the step, and the advantage is the centred gap:

```math
b = \frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i} g_{i,t},
\qquad (4)
```

```math
A_{i,t} = g_{i,t} - b.
\qquad (5)
```

The loss of step $`k`$ is the policy-gradient surrogate

```math
\mathcal{L}_k(\theta) = -\frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i}
\mathrm{sg}\bigl[A_{i,t}\bigr]\, \log \pi_\theta\bigl(y_{i,t} \mid s_{i,t}\bigr),
\qquad (6)
```

and the step's gradient is

```math
\hat{G}_k = \nabla_\theta \mathcal{L}_k(\theta)\Big|_{\theta = \theta_{k-1}}
= \frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i}
\bigl(\log \pi_{\theta_{k-1}}(y_{i,t} \mid s_{i,t}) - \log \pi_{\mathrm{T}}(y_{i,t} \mid \tilde{s}_{i,t}) + b\bigr)\,
\nabla_\theta \log \pi_\theta(y_{i,t} \mid s_{i,t})\Big|_{\theta = \theta_{k-1}}.
\qquad (7)
```

Each summand of (7) is the integrand of (2) with $`c = -b`$, evaluated at the sampled token. So, taking the
contexts as given, $`\hat{G}_k`$ estimates the token-averaged gradient of the per-context reverse KL at the
contexts the student visited:

```math
\hat{G}_k \;\approx\; \frac{1}{N} \sum_{i=1}^{B} \sum_{t=1}^{L_i}
\nabla_\theta\, \ell\bigl(\theta;\, s_{i,t}, \tilde{s}_{i,t}\bigr)\Big|_{\theta = \theta_{k-1}}.
\qquad (8)
```

Only one token per context is used (the sampled one); the sum over the vocabulary in (1) is never formed.
`tests/unit/test_on_policy_distillation.py` checks on a tiny model that the expectation of the estimator equals
the exact gradient of (1), and that the gradient is zero when teacher and student are the same model.

### 3.3 What the estimator is not

The approximation in (8) hides three things, listed from most to least important.

**(a) No credit for downstream effects.** The sequence-level reverse KL between the two models' response
distributions decomposes, by the chain rule, into the expected sum of per-context terms,
$`D_{\mathrm{KL}}\bigl(P_\theta(\cdot \mid x) \,\Vert\, P_{\mathrm{T}}(\cdot \mid x)\bigr) = \mathbb{E}_{y \sim P_\theta}\bigl[\sum_t \ell(\theta;\, s_t, \tilde{s}_t)\bigr]`$,
and its gradient is

```math
\nabla_\theta D_{\mathrm{KL}}\bigl(P_\theta(\cdot \mid x) \,\big\Vert\, P_{\mathrm{T}}(\cdot \mid x)\bigr)
= \mathbb{E}_{y \sim P_\theta}\Bigl[\sum_{t=1}^{L} \nabla_\theta \log \pi_\theta(y_t \mid s_t)
\sum_{t'=t}^{L} \bigl(-g_{t'}\bigr)\Bigr].
\qquad (9)
```

Here $`g_{t'}`$ is the gap of equation (3) for the token at position $`t'`$, evaluated at the current $`\theta`$.
The estimator (7) keeps only the $`t' = t`$ term of the inner sum. A token is therefore pushed up or down
according to how the teacher rates that token alone, not according to where it leads the rest of the response.
Equivalently, the distribution of visited contexts is treated as fixed. Nothing in the loss stops the student
from moving to contexts where the per-context divergence is larger.

**(b) The baseline is computed from the same batch.** $`b`$ contains the token's own gap and the gaps of later
tokens of the same rollout, which depend on that token. It is therefore not exactly independent of the sampled
token, as (2) requires. The dependent tokens carry weight $`1/N`$ each, at most
$`L_i / N \approx 1/B \approx 1.6\%`$ in total. This bias was not corrected.

**(c) Token-mean normalisation.** The loss is divided by $`N`$, which depends on the sampled lengths, so long
rollouts contribute more tokens and each token weighs the same. The loss is not a mean over rollouts.

Responses that reach $`L_{\max}`$ are cut off; no term accounts for what would have followed.

### 3.4 Not in the loss

There is no outcome reward, no cross-entropy on teacher-generated text, no forward-KL term, no penalty tying
the student to its initial weights, no entropy bonus other than the one implicit in (1), and no weight decay.

### 3.5 Accumulation over micro-batches

One 4,096-token rollout needs about 92 GB of working memory for the backward pass, so the gradient of (6) is
accumulated over micro-batches of one rollout. Micro-batch $`m`$ holds $`N_m`$ response tokens and contributes

```math
\hat{G}_k = \sum_{m} \nabla_\theta \Bigl[\frac{N_m}{N}\, \mathcal{L}_{k,m}(\theta)\Bigr]_{\theta = \theta_{k-1}},
\qquad
\mathcal{L}_{k,m}(\theta) = -\frac{1}{N_m} \sum_{(i,t) \in m} \mathrm{sg}\bigl[A_{i,t}\bigr]\, \log \pi_\theta\bigl(y_{i,t} \mid s_{i,t}\bigr),
\qquad (10)
```

which equals (7) exactly. The advantages are computed once for the whole step, before any gradient, so $`b`$ is
the mean over all 64 rollouts and not over a micro-batch. There is exactly one optimizer update per step.

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
$`W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}`$ the frozen matrix and $`r = 64`$,

```math
W' = W + U D, \qquad U \in \mathbb{R}^{d_{\mathrm{out}} \times r}, \quad D \in \mathbb{R}^{r \times d_{\mathrm{in}}},
\qquad (11)
```

with no scaling factor. Initialization: $`U = 0`$ and $`D_{ab} \sim \mathcal{N}(0,\, 1/d_{\mathrm{in}})`$
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
- The log-probabilities in (3), for both models, come from lalamo's inference forward pass
  (`DecoderForwardPassConfig.for_inference()`), as does sampling. The gradient in (7) is taken through the
  training forward pass (`DecoderForwardPassConfig.for_training()`), which uses a different attention kernel.
  The two passes compute the same function up to floating-point differences.
- The gradient is compiled under JAX's standard dtype promotion instead of lalamo's strict setting, because the
  backward pass of the attention kernel multiplies float32 cotangents with bfloat16 activations.

### 4.8 One step

1. Sample 64 responses from the student (section 4.4).
2. Place the responses behind the teacher's prompts and compute
   $`\log \pi_{\mathrm{T}}(y_{i,t} \mid \tilde{s}_{i,t})`$ for every response token, two rollouts at a time.
   Shift the values by the difference in prompt length so that they line up with the student's positions.
3. Compute $`\log \pi_{\theta_{k-1}}(y_{i,t} \mid s_{i,t})`$ with the student, two rollouts at a time.
4. Form gaps, baseline and advantages (equations 3–5) over the whole step.
5. Accumulate the gradient (equation 10), one rollout at a time.
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
| Advantage baseline | batch mean (equation 4) |
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
| Student entropy | $`-\frac{1}{N} \sum_{i,t} \log \pi_{\theta_{k-1}}(y_{i,t} \mid s_{i,t})`$: sampled-token estimate of the student's mean next-token entropy |
| Teacher cross-entropy | $`-\frac{1}{N} \sum_{i,t} \log \pi_{\mathrm{T}}(y_{i,t} \mid \tilde{s}_{i,t})`$. Reverse KL = teacher cross-entropy − student entropy |
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

## 8. Limitations

- **One run, one seed.** No repeat, no ablation, no baseline method (in particular no supervised fine-tuning
  on teacher-written responses with the same budget).
- **Small evaluation samples.** 200 and 300 questions give standard errors of 2.6–2.8 points. Because all
  checkpoints share sampling keys, differences between checkpoints are not independent draws either.
- **Token-budget confound.** The base student's score is limited by the response limit, not measured at its
  capability. Its mean response length (5,862 on MATH-500) is censored by the 8,192-token limit. The run does
  not show how the base student scores with a larger limit, so it does not separate "shorter" from "better".
- **Different sampling settings.** Training samples at temperature 1 without truncation; the student is
  evaluated at temperature 0.6 with top-k 20 and top-p 0.95; the teacher at temperature 1.0 with the same
  truncation. The teacher's 86.5% is therefore not measured under the student's settings, and no evaluation
  was run at the training temperature.
- **String-match grading** on MATH-500 undercounts answers given in an equivalent form.
- **Totals only.** Per-question outcomes and evaluation texts were not saved, so there is no paired comparison
  between checkpoints and no direct measurement of forgetting.
- **The teacher was not evaluated on MMLU-Pro.**
- **The reverse-KL estimate is not a progress measure.** It is an average over the contexts the student
  visits, and those change during training (section 3.3, a). Two reference values were not logged: the
  teacher's entropy on its own samples, and the per-context reverse KL computed over the full vocabulary.
- **Truncated rollouts.** Between 8% and 66% of training rollouts finish within 4,096 tokens, so early in
  training nearly all of the loss falls on reasoning text and little on final replies.
- **Different prompts for teacher and student.** The teacher is conditioned on a system message about
  reasoning effort that the student does not see (section 4.2).
- **Constant learning rate, no schedule**, and training was stopped while the adapters were still moving.
- **Restarts.** The run consists of three process segments with small code differences (section 4.11).
- **Base commit.** The run used lalamo at `f3d9f59c`. This pull request is rebased onto a later `main`; the
  unit tests were re-run there, the experiment was not.

## 9. Reproduction

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
