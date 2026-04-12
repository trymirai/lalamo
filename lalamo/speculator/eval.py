import json
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

from lalamo.message_processor import MessageProcessor, UserMessage
from lalamo.modules.decoder import Decoder
from lalamo.speculator.drafter import Drafter
from lalamo.speculator.speculate import (
    SamplerConfig,
    SpeculationContext,
    SpeculationRun,
    SpeculativeDecodingResult,
)

MTBENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


def load_mtbench(cache_path: Path | str) -> list[dict]:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading MT-Bench questions...", file=sys.stderr)
        urllib.request.urlretrieve(MTBENCH_URL, cache_path)
    questions = []
    with open(cache_path) as f:
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
    eos_set: set[int],
    seed: int = 42,
) -> SpeculativeDecodingResult:
    prompt_ids = mp.tokenize_request([UserMessage(content=prompt)])
    ctx = SpeculationContext.create(decoder, drafter, config, eos_set)
    session = SpeculationRun(ctx, prompt_ids, seed=seed)
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
) -> dict:
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
            eos_set,
            seed=42 + i,
        )

        n_tok = len(result.generated)
        n_step = result.num_steps
        draft_acc = result.mean_draft_accepted

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

        running_acc = total_accepted / max(total_steps, 1)
        pbar.set_description(f"MT-Bench (draft_acc={running_acc:.2f})")
        pbar.set_postfix_str(f"{cat}: draft_acc={draft_acc:.2f}")
        print(
            f"  [{i + 1:2d}] {cat:12s} | {n_tok:4d} tok, {n_step:3d} steps, "
            f"draft_acc={draft_acc:.2f}, tok/step={result.tokens_per_step:.2f} | {q['prompt'][:50]}",
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
    print(f"\n{'=' * 78}{prefix}")
    print(f"{'Category':>15s}  {'tok/step':>10s}  {'draft_acc':>10s}  {'acc_rate':>10s}  {'questions':>10s}")
    print(f"{'-' * 78}")
    for cat in sorted(results["by_category"]):
        r = results["by_category"][cat]
        ts = r["tokens"] / max(r["steps"], 1)
        da = r["accepted"] / max(r["steps"], 1) if r["steps"] else 0
        acc = r["accepted"] / max(r["proposed"], 1) if r["proposed"] else 0
        print(f"{cat:>15s}  {ts:>10.2f}  {da:>10.2f}  {acc:>10.2%}  {r['count']:>10d}")

    ts = results["total_tokens"] / max(results["total_steps"], 1)
    da = results["total_accepted"] / max(results["total_steps"], 1)
    acc = results["total_accepted"] / max(results["total_proposed"], 1)
    print(f"{'-' * 78}")
    total_count = sum(r["count"] for r in results["by_category"].values())
    print(f"{'OVERALL':>15s}  {ts:>10.2f}  {da:>10.2f}  {acc:>10.2%}  {total_count:>10d}")
    print(f"{'=' * 78}")
