from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from lalamo.qwen_moe_eval_common import (
    LossAccumulator,
    LossStatistics,
    PreparedSamples,
    SequenceEWMARouter,
    patched_ewma_routing,
    prepare_samples,
    teacher_forced_token_nll,
)
from lalamo.qwen_moe_payloads import write_payload
from lalamo.qwen_moe_routing import (
    AUTO_DEVICE_MAP_HEADROOM_GIB,
    ModelRoutingConfig,
    RuntimeInfo,
    dataset_fingerprint,
    indexed_conversation_samples,
    load_model,
    load_rows,
    model_routing_config,
    parse_ewma_alphas,
    runtime_info,
)

if TYPE_CHECKING:
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    from lalamo.qwen_moe_routing import ProcessedSample


NLL_AUTO_DEVICE_MAP_HEADROOM_GIB = AUTO_DEVICE_MAP_HEADROOM_GIB + 12


@dataclass(frozen=True)
class EvalConfig:
    model_repo: str
    dataset: str
    dataset_split: str
    output_path: Path
    seed: int
    max_rows: int | None
    max_prompts: int | None
    max_prompt_tokens: int
    max_continuation_tokens: int
    ewma_alphas: tuple[float, ...]
    device_map_mode: str


@dataclass(frozen=True)
class VariantResult:
    name: str
    alpha: float
    statistics: LossStatistics
    delta_token_weighted_mean_continuation_nll_vs_baseline: float
    delta_sequence_weighted_mean_continuation_nll_vs_baseline: float


@dataclass(frozen=True)
class EwmaEvalResult:
    config: EvalConfig
    model: ModelRoutingConfig
    runtime: RuntimeInfo
    dataset_fingerprint: str
    dataset_rows_total: int
    assistant_turns_total: int
    baseline: VariantResult
    ewma_variants: tuple[VariantResult, ...]
    dataset_rows_processed: int
    prompts_processed: int
    skipped_prompt_too_long: int
    skipped_continuation_too_long: int
    skipped_empty_continuation: int
    processed_samples: tuple[ProcessedSample, ...]


def evaluate_variant(
    model: Qwen3_5MoeForCausalLM,
    prepared_samples: PreparedSamples,
    routing_state: SequenceEWMARouter | None = None,
) -> LossAccumulator:
    accumulator = LossAccumulator()
    for sample in prepared_samples.samples:
        if routing_state is not None:
            routing_state.reset_sequence()
        accumulator.update(
            teacher_forced_token_nll(
                model=model,
                prompt_ids=sample.prompt_ids,
                continuation_ids=sample.continuation_ids,
            )
        )
    return accumulator


def variant_result(name: str, alpha: float, statistics: LossStatistics, baseline: LossStatistics) -> VariantResult:
    return VariantResult(
        name=name,
        alpha=alpha,
        statistics=statistics,
        delta_token_weighted_mean_continuation_nll_vs_baseline=(
            statistics.token_weighted_mean_continuation_nll - baseline.token_weighted_mean_continuation_nll
        ),
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=(
            statistics.sequence_weighted_mean_continuation_nll - baseline.sequence_weighted_mean_continuation_nll
        ),
    )


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Local dataset path or Hugging Face dataset repo id.")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--model-repo", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-continuation-tokens", type=int, default=512)
    parser.add_argument("--ewma-alphas", default="0.8")
    parser.add_argument("--device-map-mode", choices=("single-gpu", "auto", "balanced-low-0"), default=None)
    args = parser.parse_args()
    device_map_mode = args.device_map_mode
    if device_map_mode is None:
        device_map_mode = "single-gpu" if torch.cuda.is_available() else "auto"
    return EvalConfig(
        model_repo=args.model_repo,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output_path=args.output_path,
        seed=args.seed,
        max_rows=args.max_rows,
        max_prompts=args.max_prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        max_continuation_tokens=args.max_continuation_tokens,
        ewma_alphas=parse_ewma_alphas(args.ewma_alphas),
        device_map_mode=device_map_mode,
    )


def main() -> None:
    config = parse_args()
    routing_config = model_routing_config(config.model_repo)
    rows, dataset_rows_total = load_rows(config.dataset, config.dataset_split, config.max_rows, config.seed)
    samples = indexed_conversation_samples(rows, config.seed)
    assistant_turns_total = len(samples)
    if config.max_prompts is not None:
        samples = samples[: config.max_prompts]
    tokenizer = AutoTokenizer.from_pretrained(config.model_repo)
    prepared_samples = prepare_samples(
        samples=samples,
        tokenizer=tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        max_continuation_tokens=config.max_continuation_tokens,
    )
    if not prepared_samples.samples:
        raise ValueError("Evaluation produced no valid prompts.")
    model = load_model(
        config.model_repo,
        config.device_map_mode,
        auto_headroom_gib=NLL_AUTO_DEVICE_MAP_HEADROOM_GIB,
    )
    model.eval()

    baseline_statistics = evaluate_variant(model, prepared_samples).finalize()
    baseline = VariantResult(
        name="baseline",
        alpha=0.0,
        statistics=baseline_statistics,
        delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
    )
    ewma_variants = []
    for alpha in config.ewma_alphas:
        with patched_ewma_routing(model, alpha) as routing_state:
            statistics = evaluate_variant(model, prepared_samples, routing_state=routing_state).finalize()
        ewma_variants.append(variant_result(f"ewma_{alpha:.3f}", alpha, statistics, baseline_statistics))

    write_payload(
        config.output_path,
        EwmaEvalResult(
            config=config,
            model=routing_config,
            runtime=runtime_info(model, config.device_map_mode),
            dataset_fingerprint=dataset_fingerprint(rows),
            dataset_rows_total=dataset_rows_total,
            assistant_turns_total=assistant_turns_total,
            baseline=baseline,
            ewma_variants=tuple(ewma_variants),
            dataset_rows_processed=len(rows),
            prompts_processed=prepared_samples.prompts_processed,
            skipped_prompt_too_long=prepared_samples.skipped_prompt_too_long,
            skipped_continuation_too_long=prepared_samples.skipped_continuation_too_long,
            skipped_empty_continuation=prepared_samples.skipped_empty_continuation,
            processed_samples=prepared_samples.processed_samples,
        ),
    )


if __name__ == "__main__":
    main()
