"""Evaluate speculative decoding on MT-Bench.

Reports MAE (mean accepted length), acceptance rate, tokens/step.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from lalamo.message_processor import UserMessage
from lalamo.speculator.speculate import (
    SpeculationContext,
    SpeculationRun,
    SpeculativeDecodingResult,
)

if TYPE_CHECKING:
    from lalamo.message_processor import MessageProcessor
    from lalamo.modules.decoder import Decoder
    from lalamo.speculator.drafter import Drafter, SamplerConfig

MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
MTBENCH_CACHE = Path(".hikettei/mtbench_questions.jsonl")


def load_mtbench() -> list[dict]:
    """Load MT-Bench first-turn questions (80 total)."""
    if not MTBENCH_CACHE.exists():
        MTBENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading MT-Bench questions...", file=sys.stderr)
        urllib.request.urlretrieve(MTBENCH_URL, MTBENCH_CACHE)
    questions = []
    with open(MTBENCH_CACHE) as f:
        for line in f:
            d = json.loads(line)
            questions.append(
                {
                    "id": d["question_id"],
                    "category": d["category"],
                    "prompt": d["turns"][0],
                }
            )
    return questions


def evaluate_prompt(
    decoder: Decoder,
    mp: MessageProcessor,
    drafter: Drafter,
    config: SamplerConfig,
    prompt: str,
    max_tokens: int,
    eos_set: set[int],
    use_gumbel: bool = False,
    seed: int = 42,
) -> SpeculativeDecodingResult:
    """Run one prompt through the full speculative decoding pipeline."""
    prompt_ids = mp.tokenize_request([UserMessage(content=prompt)])

    ctx = SpeculationContext.create(decoder, drafter, config, eos_set, use_gumbel)
    session = SpeculationRun(ctx, prompt_ids, max_tokens, seed=seed)
    for _ in session:
        pass
    return session.result


def run_mtbench(
    decoder: Decoder,
    mp: MessageProcessor,
    drafter: Drafter,
    config: SamplerConfig,
    eos_set: set[int],
    questions: list[dict],
    max_tokens: int = 2048,
    use_gumbel: bool = False,
) -> dict:
    """Run MT-Bench evaluation, return per-category and overall statistics."""
    results_by_cat: dict[str, dict] = {}
    total_tokens, total_steps, total_accepted, total_proposed = 0, 0, 0, 0

    pbar = tqdm(questions, desc="MT-Bench", file=sys.stderr)
    for i, q in enumerate(pbar):
        cat = q["category"]

        result = evaluate_prompt(
            decoder,
            mp,
            drafter,
            config,
            q["prompt"],
            max_tokens,
            eos_set,
            use_gumbel=use_gumbel,
            seed=42 + i,
        )

        n_tok = len(result.generated)
        n_step = result.num_steps
        mae = result.mean_accepted_length

        if cat not in results_by_cat:
            results_by_cat[cat] = {"tokens": 0, "steps": 0, "accepted": 0, "proposed": 0, "count": 0}
        r = results_by_cat[cat]
        r["tokens"] += n_tok
        r["steps"] += n_step
        r["accepted"] += result.total_accepted
        r["proposed"] += result.total_proposed
        r["count"] += 1

        total_tokens += n_tok
        total_steps += n_step
        total_accepted += result.total_accepted
        total_proposed += result.total_proposed

        running_mae = total_accepted / max(total_steps, 1)
        pbar.set_description(f"MT-Bench (mae={running_mae:.2f})")
        pbar.set_postfix_str(f"{cat}: mae={mae:.2f}")
        print(
            f"  [{i + 1:2d}] {cat:12s} | {n_tok:4d} tok, {n_step:3d} steps, "
            f"mae={mae:.2f}, tok/step={result.tokens_per_step:.2f} | {q['prompt'][:50]}",
            file=sys.stderr,
        )

    return {
        "by_category": results_by_cat,
        "total_tokens": total_tokens,
        "total_steps": total_steps,
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
    }


def print_results(results: dict, label: str = "") -> None:
    prefix = f" [{label}]" if label else ""
    print(f"\n{'=' * 70}{prefix}")
    print(f"{'Category':>15s}  {'tok/step':>10s}  {'mae':>10s}  {'acc_rate':>10s}  {'questions':>10s}")
    print(f"{'-' * 70}")
    for cat in sorted(results["by_category"]):
        r = results["by_category"][cat]
        ts = r["tokens"] / max(r["steps"], 1)
        mae_cat = r["accepted"] / max(r["steps"], 1) if r["steps"] else 0
        acc = r["accepted"] / max(r["proposed"], 1) if r["proposed"] else 0
        print(f"{cat:>15s}  {ts:>10.2f}  {mae_cat:>10.2f}  {acc:>10.2%}  {r['count']:>10d}")

    ts = results["total_tokens"] / max(results["total_steps"], 1)
    mae = results["total_accepted"] / max(results["total_steps"], 1)
    acc = results["total_accepted"] / max(results["total_proposed"], 1)
    print(f"{'-' * 70}")
    total_count = sum(r["count"] for r in results["by_category"].values())
    print(f"{'OVERALL':>15s}  {ts:>10.2f}  {mae:>10.2f}  {acc:>10.2%}  {total_count:>10d}")
    print(f"{'=' * 70}")
