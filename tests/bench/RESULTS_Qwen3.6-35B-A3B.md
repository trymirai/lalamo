# Qwen3.6-35B-A3B — Typical-Prompt Bench Results

- Run date: 2026-04-16
- Model: `Qwen/Qwen3.6-35B-A3B` (converted to lalamo, bf16, 35B total / 3B active MoE)
- Hardware: Mac, 512 GB RAM
- Max tokens per reply: 256
- Total: 20 prompts, 4916 tokens in 35.2 min (2.33 tok/s avg)
- Raw log: [models/Qwen3.6-35B-A3B/bench_1776361086.jsonl](../models/Qwen3.6-35B-A3B/bench_1776361086.jsonl)

## Summary

Qwen3.6 defaults to thinking mode: every reply begins with an extended "Thinking Process: 1. Analyze User Input..." trace before a final answer. With `--max-tokens 256`, **only 3 of 20 prompts reached `</think>`** and produced a final answer. The other 17 ran out of tokens mid-trace. In most of those, the thinking itself contained the correct answer — the model simply didn't finish.

| | Count |
|---|---|
| Prompts that produced a final answer | 3 |
| Final answers that were correct | 3 |
| Prompts truncated mid-`<think>` | 17 |
| Truncated prompts whose trace contained the right answer | ~14 (spot-checked) |

## Per-prompt results

| ID | Category | Tokens | Time (s) | Tok/s | Finished? | Verdict |
|---|---|---:|---:|---:|---|---|
| `fact_capital` | factual | 140 | 78.1 | 1.79 | ✓ | Correct: Canberra |
| `fact_element` | factual | 178 | 83.5 | 2.13 | ✓ | Correct: Gold |
| `arith_small` | arithmetic | 256 | 109.1 | 2.35 | ✗ (trunc) | Trace computed 2491 correctly; no final answer emitted |
| `arith_words` | arithmetic | 256 | 109.1 | 2.35 | ✗ (trunc) | Trace reached 7 apples; truncated before final |
| `reason_syllogism` | reasoning | 256 | 110.0 | 2.33 | ✗ (trunc) | Trace clearly reasons yes; truncated |
| `reason_order` | reasoning | 256 | 110.3 | 2.32 | ✗ (trunc) | Trace identifies Sue; truncated |
| `format_words` | instruction_following | 256 | 110.0 | 2.33 | ✗ (trunc) | Thinking about 3-word beach descriptions; did not emit a 3-word final answer |
| `format_json` | instruction_following | 246 | 106.7 | 2.31 | ✓ | Correct: `{"name": "Alex", "age": 30}` — slight whitespace diff from strict expect |
| `code_fizzbuzz` | code_generation | 256 | 108.8 | 2.35 | ✗ (trunc) | Correct fizzbuzz code drafted inside trace; truncated before clean emit |
| `code_reverse` | code_generation | 256 | 108.6 | 2.36 | ✗ (trunc) | Trace lists `xs[::-1]` as the answer; truncated |
| `code_debug` | code_debugging | 256 | 105.4 | 2.43 | ✗ (trunc) | Correctly identifies `+ 1` bug; truncated mid-fix |
| `common_sense` | common_sense | 256 | 108.9 | 2.35 | ✗ (trunc) | Trace in progress; unclear if final sentence would land |
| `trap_presidents` | trap | 256 | 107.6 | 2.38 | ✗ (trunc) | Trace recognizes Antarctica has no government; truncated |
| `trap_math` | trap | 256 | 110.7 | 2.31 | ✗ (trunc) | Trace acknowledges no real solution (2i); truncated |
| `creative_haiku` | creative | 256 | 109.9 | 2.33 | ✗ (trunc) | Brainstorming imagery; did not emit a finished haiku |
| `summarize` | summarization | 256 | 103.8 | 2.47 | ✗ (trunc) | Extracting key facts; truncated before 1-sentence summary |
| `translate_fr` | translation | 256 | 108.1 | 2.37 | ✗ (trunc) | Thinking through French phrasing; truncated before translation |
| `edge_empty_reason` | reasoning | 256 | 107.9 | 2.37 | ✗ (trunc) | Reasoning about closed container; truncated |
| `multi_step` | reasoning | 256 | 107.7 | 2.38 | ✗ (trunc) | Trace computes 12:00 PM correctly; truncated before final sentence |
| `self_knowledge` | meta | 256 | 107.8 | 2.38 | ✗ (trunc) | Brainstorming hallucination as main limitation; truncated |

## Full answers for completed prompts

### `fact_capital`

**Prompt:** What is the capital of Australia? Reply in one sentence.

**Answer:** `The capital of Australia is Canberra.`

### `fact_element`

**Prompt:** What element has the atomic number 79? Just the name.

**Answer:** `Gold`

### `format_json`

**Prompt:** Return a JSON object with keys "name" and "age" for a fictional 30-year-old named Alex. Only the JSON.

**Answer:** `{"name": "Alex", "age": 30}`

## Notes

- ~2.3 tok/s throughput is JAX-on-CPU inference at 35B total params, 3B active. No Metal acceleration was used.
- The model card documents `chat_template_kwargs: {enable_thinking: False}` to disable thinking mode; lalamo's chat path does not surface this option, so the bench was run with thinking on.
- No automated grading. The "Verdict" column reflects eyeballing the response text.