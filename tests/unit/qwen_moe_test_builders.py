from __future__ import annotations

from pathlib import Path

from lalamo.qwen_moe_eval_common import LossStatistics
from lalamo.qwen_moe_ewma_eval import EvalConfig, EwmaEvalResult, VariantResult
from lalamo.qwen_moe_ewma_study_summary import (
    AlphaSummary,
    DatasetAlphaSummary,
    StudySummary,
    WindowDelta,
)
from lalamo.qwen_moe_offload_eval import (
    CacheBudgetResult,
    OffloadEvalConfig,
    OffloadEvalResult,
    TransferStatistics,
)
from lalamo.qwen_moe_routing import (
    CacheHitRateStatistics,
    EwmaStatistics,
    ExperimentConfig,
    ModelRoutingConfig,
    PhaseStatistics,
    ProcessedSample,
    ResidentBudgetStatistics,
    RoutingAnalysisResult,
    RuntimeInfo,
    TopKAgreementStatistics,
    WindowStatistics,
)
from lalamo.qwen_moe_transfer_budget_summary import (
    CacheBudgetTradeoff,
    DatasetTradeoff,
    TransferBudgetSummary,
)


def runtime_info() -> RuntimeInfo:
    return RuntimeInfo(
        transformers_version="4.test",
        torch_version="2.test",
        cuda_available=False,
        device_count=0,
        devices=(),
        model_dtype="bfloat16",
        model_class="Qwen3_5MoeForCausalLM",
        device_map_mode="auto",
        hf_device_map=None,
    )


def model_config() -> ModelRoutingConfig:
    return ModelRoutingConfig(
        repo="Qwen/Qwen3.6-35B-A3B",
        revision="main",
        num_layers=1,
        num_experts=16,
        num_active_experts=8,
        expert_parameters=1,
        expert_bytes=1,
    )


def experiment_config(dataset: str) -> ExperimentConfig:
    return ExperimentConfig(
        model_repo="Qwen/Qwen3.6-35B-A3B",
        dataset=dataset,
        dataset_split="train",
        output_path=Path("/tmp/out.json"),
        seed=0,
        max_rows=4,
        max_prompts=4,
        max_prompt_tokens=2048,
        max_continuation_tokens=512,
        window_sizes=(8, 16, 32),
        ewma_alphas=(0.8,),
        device_map_mode="balanced-low-0",
    )


def eval_config(dataset: str) -> EvalConfig:
    return EvalConfig(
        model_repo="Qwen/Qwen3.6-35B-A3B",
        dataset=dataset,
        dataset_split="train",
        output_path=Path("/tmp/out.json"),
        seed=0,
        max_rows=4,
        max_prompts=4,
        max_prompt_tokens=2048,
        max_continuation_tokens=512,
        ewma_alphas=(0.8,),
        device_map_mode="balanced-low-0",
    )


def offload_config(dataset: str) -> OffloadEvalConfig:
    return OffloadEvalConfig(
        model_repo="Qwen/Qwen3.6-35B-A3B",
        dataset=dataset,
        dataset_split="train",
        output_path=Path("/tmp/out.json"),
        seed=0,
        max_rows=4,
        max_prompts=4,
        max_prompt_tokens=2048,
        max_continuation_tokens=512,
        ewma_alphas=(0.2, 0.8),
        cache_sizes=(16,),
        device_map_mode="balanced-low-0",
    )


def cache_hit(cache_size: int, hit_rate: float, ci95: float = 0.0) -> CacheHitRateStatistics:
    return CacheHitRateStatistics(
        cache_size=cache_size,
        cache_fraction=0.5,
        window_weighted_hit_rate=hit_rate,
        sequence_weighted_hit_rate=hit_rate,
        sequence_weighted_hit_rate_std=0.0,
        sequence_weighted_hit_rate_sem=0.0,
        sequence_weighted_hit_rate_ci95=ci95,
    )


def resident_budget(
    cache_size: int,
    resident_gib_total: float,
    hit_rate: float,
    loads_per_token: float,
    transfer_mib_per_token: float,
    *,
    hit_ci95: float = 0.0,
    loads_ci95: float = 0.0,
    transfer_ci95: float = 0.0,
) -> ResidentBudgetStatistics:
    transfer_bytes = transfer_mib_per_token * 1024**2
    return ResidentBudgetStatistics(
        cache_size=cache_size,
        cache_fraction=0.5,
        resident_experts_total=cache_size,
        resident_bytes_total=int(resident_gib_total * 1024**3),
        resident_gib_total=resident_gib_total,
        token_weighted_hit_rate=hit_rate,
        sequence_weighted_hit_rate=hit_rate,
        sequence_weighted_hit_rate_ci95=hit_ci95,
        token_weighted_expert_loads_per_token=loads_per_token,
        sequence_weighted_expert_loads_per_token=loads_per_token,
        sequence_weighted_expert_loads_per_token_ci95=loads_ci95,
        token_weighted_transfer_bytes_per_token=transfer_bytes,
        sequence_weighted_transfer_bytes_per_token=transfer_bytes,
        sequence_weighted_transfer_bytes_per_token_ci95=transfer_ci95 * 1024**2,
    )


def window(
    window_size: int,
    distinct: float,
    cache_entries: tuple[CacheHitRateStatistics, ...],
    *,
    distinct_ci95: float = 0.0,
    random_baseline: float = 1.0,
    num_windows: int = 1,
    sequence_count_with_windows: int = 1,
) -> WindowStatistics:
    ratio = distinct / max(random_baseline, 1.0)
    return WindowStatistics(
        window_size=window_size,
        num_windows=num_windows,
        sequence_count_with_windows=sequence_count_with_windows,
        window_weighted_mean_distinct_experts_per_layer=[distinct],
        window_weighted_mean_distinct_experts_overall=distinct,
        window_weighted_mean_distinct_experts_fraction_overall=ratio,
        sequence_weighted_mean_distinct_experts_per_layer=[distinct],
        sequence_weighted_mean_distinct_experts_overall=distinct,
        sequence_weighted_mean_distinct_experts_fraction_overall=ratio,
        sequence_weighted_mean_distinct_experts_overall_std=0.0,
        sequence_weighted_mean_distinct_experts_overall_sem=0.0,
        sequence_weighted_mean_distinct_experts_overall_ci95=distinct_ci95,
        random_baseline_distinct_experts=random_baseline,
        random_baseline_fraction=1.0,
        window_weighted_observed_to_random_ratio=ratio,
        sequence_weighted_observed_to_random_ratio=ratio,
        window_weighted_mean_distinct_layer_expert_pairs=distinct,
        window_weighted_mean_distinct_layer_expert_pair_fraction=0.0,
        sequence_weighted_mean_distinct_layer_expert_pairs=distinct,
        sequence_weighted_mean_distinct_layer_expert_pair_fraction=0.0,
        oracle_cache_hit_rates=cache_entries,
    )


def phase_stats(
    windows: tuple[WindowStatistics, ...],
    resident_budgets: tuple[ResidentBudgetStatistics, ...],
) -> PhaseStatistics:
    return PhaseStatistics(sequence_count=1, token_count=1, windows=windows, resident_budgets=resident_budgets)


def agreement(
    retained: float,
    exact: float,
    *,
    retained_ci95: float = 0.0,
    exact_ci95: float = 0.0,
) -> TopKAgreementStatistics:
    return TopKAgreementStatistics(
        token_weighted_mean_retained_fraction_per_layer=[retained],
        token_weighted_mean_retained_fraction_overall=retained,
        sequence_weighted_mean_retained_fraction_per_layer=[retained],
        sequence_weighted_mean_retained_fraction_overall=retained,
        sequence_weighted_mean_retained_fraction_overall_std=0.0,
        sequence_weighted_mean_retained_fraction_overall_sem=0.0,
        sequence_weighted_mean_retained_fraction_overall_ci95=retained_ci95,
        token_weighted_exact_match_rate=exact,
        sequence_weighted_exact_match_rate=exact,
        sequence_weighted_exact_match_rate_std=0.0,
        sequence_weighted_exact_match_rate_sem=0.0,
        sequence_weighted_exact_match_rate_ci95=exact_ci95,
    )


def routing_result(
    dataset: str,
    *,
    prompt_statistics: PhaseStatistics,
    continuation_statistics: PhaseStatistics,
    ewma_statistics: tuple[EwmaStatistics, ...],
    prompts_processed: int = 1,
) -> RoutingAnalysisResult:
    return RoutingAnalysisResult(
        config=experiment_config(dataset),
        model=model_config(),
        runtime=runtime_info(),
        dataset_fingerprint="fingerprint",
        dataset_rows_total=4,
        assistant_turns_total=4,
        prompt_statistics=prompt_statistics,
        continuation_statistics=continuation_statistics,
        ewma_statistics=ewma_statistics,
        dataset_rows_processed=4,
        prompts_processed=prompts_processed,
        skipped_prompt_too_long=0,
        skipped_continuation_too_long=0,
        skipped_empty_continuation=0,
        processed_samples=(ProcessedSample(0, 0, 0, 2, 2),),
    )


def ewma_statistics(
    alpha: float,
    *,
    prompt_statistics: PhaseStatistics,
    continuation_statistics: PhaseStatistics,
    prompt_agreement: TopKAgreementStatistics,
    continuation_agreement: TopKAgreementStatistics,
) -> EwmaStatistics:
    return EwmaStatistics(
        alpha=alpha,
        prompt_statistics=prompt_statistics,
        continuation_statistics=continuation_statistics,
        prompt_agreement=prompt_agreement,
        continuation_agreement=continuation_agreement,
    )


def loss_statistics(token_nll: float, sequence_nll: float | None = None) -> LossStatistics:
    value = token_nll if sequence_nll is None else sequence_nll
    return LossStatistics(
        token_weighted_mean_continuation_nll=token_nll,
        token_weighted_continuation_perplexity=token_nll + 1.0,
        sequence_weighted_mean_continuation_nll=value,
        sequence_weighted_mean_continuation_nll_std=0.0,
        sequence_weighted_mean_continuation_nll_sem=0.0,
        sequence_weighted_mean_continuation_nll_ci95=0.0,
        sequence_weighted_continuation_perplexity=value + 1.0,
    )


def ewma_eval_result(
    dataset: str,
    baseline: VariantResult,
    variants: tuple[VariantResult, ...],
) -> EwmaEvalResult:
    return EwmaEvalResult(
        config=eval_config(dataset),
        model=model_config(),
        runtime=runtime_info(),
        dataset_fingerprint="fingerprint",
        dataset_rows_total=4,
        assistant_turns_total=4,
        baseline=baseline,
        ewma_variants=variants,
        dataset_rows_processed=4,
        prompts_processed=4,
        skipped_prompt_too_long=0,
        skipped_continuation_too_long=0,
        skipped_empty_continuation=0,
        processed_samples=(ProcessedSample(0, 0, 0, 2, 2),),
    )


def offload_transfer(
    cache_size: int,
    resident_gib_total: float,
    transfer_mib_per_token: float,
    loads_per_token: float,
    hit_rate: float,
    tokens_per_second: float,
) -> TransferStatistics:
    return TransferStatistics(
        cache_size=cache_size,
        resident_gib_total=resident_gib_total,
        prompt_transfer_bytes=0,
        continuation_transfer_bytes=int(transfer_mib_per_token * 1024**2),
        continuation_transfer_mib_per_token=transfer_mib_per_token,
        continuation_expert_loads_per_token=loads_per_token,
        continuation_hit_rate=hit_rate,
        prompt_elapsed_seconds=0.0,
        continuation_elapsed_seconds=1.0,
        continuation_tokens_per_second=tokens_per_second,
    )


def offload_eval_result(
    dataset: str,
    cache_budgets: tuple[CacheBudgetResult, ...],
) -> OffloadEvalResult:
    return OffloadEvalResult(
        config=offload_config(dataset),
        model=model_config(),
        runtime=runtime_info(),
        dataset_fingerprint="fingerprint",
        dataset_rows_total=4,
        assistant_turns_total=4,
        cache_budgets=cache_budgets,
    )


def study_summary(
    alpha: float,
    window_delta: WindowDelta,
    *,
    dataset: str = "hermes",
    retained: float = 0.88,
    exact: float = 0.34,
) -> StudySummary:
    return StudySummary(
        phase="continuation",
        cache_size=16,
        requested_windows=(window_delta.window_size,),
        effective_windows=(window_delta.window_size,),
        alphas=(
            AlphaSummary(
                alpha=alpha,
                datasets=(
                    DatasetAlphaSummary(
                        dataset=dataset,
                        retained_fraction=retained,
                        exact_match_rate=exact,
                        windows=(window_delta,),
                        passes_minimal_intervention_rule=True,
                    ),
                ),
                passes_all_datasets=True,
            ),
        ),
        recommended_alpha=alpha,
    )


def transfer_budget_summary(dataset: str, budget: CacheBudgetTradeoff) -> TransferBudgetSummary:
    return TransferBudgetSummary(datasets=(DatasetTradeoff(dataset=dataset, budgets=(budget,)),))
