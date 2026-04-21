from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lalamo.qwen_moe_payloads import alpha_label, dataset_stem, variant_label

if TYPE_CHECKING:
    from lalamo.qwen_moe_ewma_eval import EwmaEvalResult, VariantResult
    from lalamo.qwen_moe_ewma_study_summary import StudySummary
    from lalamo.qwen_moe_offload_eval import OffloadEvalResult, OffloadVariantResult
    from lalamo.qwen_moe_routing import PhaseStatistics, RoutingAnalysisResult
    from lalamo.qwen_moe_transfer_budget_summary import TransferBudgetSummary


@dataclass(frozen=True)
class RoutingLocalityRow:
    dataset: str
    phase: str
    alpha: str
    window_size: int
    distinct_experts: float
    distinct_ci95: float
    random_baseline: float
    observed_to_random_ratio: float


@dataclass(frozen=True)
class RoutingCacheRow:
    dataset: str
    phase: str
    alpha: str
    window_size: int
    cache_size: int
    cache_hit_rate: float
    cache_hit_ci95: float


@dataclass(frozen=True)
class RoutingResidentBudgetRow:
    dataset: str
    phase: str
    alpha: str
    cache_size: int
    resident_gib_total: float
    hit_rate: float
    hit_rate_ci95: float
    expert_loads_per_token: float
    expert_loads_per_token_ci95: float
    transfer_mib_per_token: float
    transfer_mib_per_token_ci95: float


@dataclass(frozen=True)
class AgreementRow:
    dataset: str
    phase: str
    alpha: str
    retained_fraction: float
    retained_ci95: float
    exact_match_rate: float
    exact_match_ci95: float


@dataclass(frozen=True)
class NllRow:
    dataset: str
    alpha: str
    token_nll: float
    token_ppl: float
    sequence_nll: float
    sequence_ppl: float
    delta_token_nll_vs_baseline: float
    delta_sequence_nll_vs_baseline: float


@dataclass(frozen=True)
class QualityTransferRow:
    dataset: str
    alpha: str
    cache_size: int
    resident_gib_total: float
    transfer_mib_per_token: float
    token_nll: float


@dataclass(frozen=True)
class StudyRow:
    alpha: float
    dataset: str
    window_size: int
    baseline_distinct: float
    ewma_distinct: float
    delta_distinct: float
    baseline_cache_hit_rate: float
    ewma_cache_hit_rate: float
    delta_cache_hit_rate: float
    retained_fraction: float
    exact_match_rate: float
    passes_rule: bool


@dataclass(frozen=True)
class OffloadVariantRow:
    dataset: str
    cache_size: int
    resident_gib_total: float
    variant: str
    alpha: float
    token_nll: float
    token_ppl: float
    sequence_nll: float
    sequence_ppl: float
    delta_token_nll_vs_baseline: float
    delta_sequence_nll_vs_baseline: float
    transfer_mib_per_token: float
    loads_per_token: float
    hit_rate: float
    continuation_tokens_per_second: float
    prompts_processed: int


@dataclass(frozen=True)
class OffloadBudgetSummaryRow:
    dataset: str
    cache_size: int
    resident_gib_total: float
    best_transfer_variant: str
    best_transfer_mib_per_token: float
    best_nll_variant: str
    best_token_nll: float
    best_tps_variant: str
    best_continuation_tokens_per_second: float


@dataclass(frozen=True)
class TransferBudgetRow:
    dataset: str
    cache_size: int
    resident_gib_total: float
    recommended_alpha_without_nll_regression: str
    alpha: str
    token_nll: float
    delta_token_nll_vs_baseline: float
    transfer_mib_per_token: float
    delta_transfer_mib_per_token_vs_baseline: float
    hit_rate: float


def _phase_statistics_rows(
    dataset: str,
    phase: str,
    alpha: float,
    statistics: PhaseStatistics,
) -> tuple[tuple[RoutingLocalityRow, ...], tuple[RoutingCacheRow, ...], tuple[RoutingResidentBudgetRow, ...]]:
    alpha_name = variant_label(alpha)
    locality_rows = tuple(
        RoutingLocalityRow(
            dataset=dataset,
            phase=phase,
            alpha=alpha_name,
            window_size=window.window_size,
            distinct_experts=window.sequence_weighted_mean_distinct_experts_overall,
            distinct_ci95=window.sequence_weighted_mean_distinct_experts_overall_ci95,
            random_baseline=window.random_baseline_distinct_experts,
            observed_to_random_ratio=window.sequence_weighted_observed_to_random_ratio,
        )
        for window in statistics.windows
    )
    cache_rows = tuple(
        RoutingCacheRow(
            dataset=dataset,
            phase=phase,
            alpha=alpha_name,
            window_size=window.window_size,
            cache_size=cache.cache_size,
            cache_hit_rate=cache.sequence_weighted_hit_rate,
            cache_hit_ci95=cache.sequence_weighted_hit_rate_ci95,
        )
        for window in statistics.windows
        for cache in window.oracle_cache_hit_rates
    )
    resident_rows = tuple(
        RoutingResidentBudgetRow(
            dataset=dataset,
            phase=phase,
            alpha=alpha_name,
            cache_size=budget.cache_size,
            resident_gib_total=budget.resident_gib_total,
            hit_rate=budget.sequence_weighted_hit_rate,
            hit_rate_ci95=budget.sequence_weighted_hit_rate_ci95,
            expert_loads_per_token=budget.sequence_weighted_expert_loads_per_token,
            expert_loads_per_token_ci95=budget.sequence_weighted_expert_loads_per_token_ci95,
            transfer_mib_per_token=budget.sequence_weighted_transfer_bytes_per_token / (1024**2),
            transfer_mib_per_token_ci95=budget.sequence_weighted_transfer_bytes_per_token_ci95 / (1024**2),
        )
        for budget in statistics.resident_budgets
    )
    return locality_rows, cache_rows, resident_rows


def routing_rows(
    result: RoutingAnalysisResult,
) -> tuple[
    tuple[RoutingLocalityRow, ...],
    tuple[RoutingCacheRow, ...],
    tuple[RoutingResidentBudgetRow, ...],
    tuple[AgreementRow, ...],
]:
    dataset = dataset_stem(result.config.dataset)
    locality_rows: list[RoutingLocalityRow] = []
    cache_rows: list[RoutingCacheRow] = []
    resident_rows: list[RoutingResidentBudgetRow] = []
    for phase, statistics in (("prompt", result.prompt_statistics), ("continuation", result.continuation_statistics)):
        phase_locality, phase_cache, phase_resident = _phase_statistics_rows(dataset, phase, 0.0, statistics)
        locality_rows.extend(phase_locality)
        cache_rows.extend(phase_cache)
        resident_rows.extend(phase_resident)
    agreement_rows: list[AgreementRow] = []
    for ewma in result.ewma_statistics:
        for phase, statistics, agreement in (
            ("prompt", ewma.prompt_statistics, ewma.prompt_agreement),
            ("continuation", ewma.continuation_statistics, ewma.continuation_agreement),
        ):
            phase_locality, phase_cache, phase_resident = _phase_statistics_rows(
                dataset,
                phase,
                ewma.alpha,
                statistics,
            )
            locality_rows.extend(phase_locality)
            cache_rows.extend(phase_cache)
            resident_rows.extend(phase_resident)
            agreement_rows.append(
                AgreementRow(
                    dataset=dataset,
                    phase=phase,
                    alpha=alpha_label(ewma.alpha),
                    retained_fraction=agreement.sequence_weighted_mean_retained_fraction_overall,
                    retained_ci95=agreement.sequence_weighted_mean_retained_fraction_overall_ci95,
                    exact_match_rate=agreement.sequence_weighted_exact_match_rate,
                    exact_match_ci95=agreement.sequence_weighted_exact_match_rate_ci95,
                )
            )
    return tuple(locality_rows), tuple(cache_rows), tuple(resident_rows), tuple(agreement_rows)


def continuation_locality_chart_rows(result: RoutingAnalysisResult) -> tuple[RoutingLocalityRow, ...]:
    locality_rows, _, _, _ = routing_rows(result)
    return tuple(row for row in locality_rows if row.phase == "continuation")


def continuation_cache_chart_rows(result: RoutingAnalysisResult, cache_size: int) -> tuple[RoutingCacheRow, ...]:
    _, cache_rows, _, _ = routing_rows(result)
    return tuple(row for row in cache_rows if row.phase == "continuation" and row.cache_size == cache_size)


def continuation_transfer_chart_rows(result: RoutingAnalysisResult) -> tuple[RoutingResidentBudgetRow, ...]:
    _, _, resident_rows, _ = routing_rows(result)
    return tuple(row for row in resident_rows if row.phase == "continuation")


def continuation_agreement_chart_rows(result: RoutingAnalysisResult) -> tuple[AgreementRow, ...]:
    _, _, _, agreement_rows = routing_rows(result)
    return tuple(row for row in agreement_rows if row.phase == "continuation")


def _nll_row(dataset: str, alpha: float, variant: VariantResult, *, baseline: VariantResult) -> NllRow:
    return NllRow(
        dataset=dataset,
        alpha=variant_label(alpha),
        token_nll=variant.statistics.token_weighted_mean_continuation_nll,
        token_ppl=variant.statistics.token_weighted_continuation_perplexity,
        sequence_nll=variant.statistics.sequence_weighted_mean_continuation_nll,
        sequence_ppl=variant.statistics.sequence_weighted_continuation_perplexity,
        delta_token_nll_vs_baseline=(
            variant.statistics.token_weighted_mean_continuation_nll
            - baseline.statistics.token_weighted_mean_continuation_nll
        ),
        delta_sequence_nll_vs_baseline=(
            variant.statistics.sequence_weighted_mean_continuation_nll
            - baseline.statistics.sequence_weighted_mean_continuation_nll
        ),
    )


def nll_rows(result: EwmaEvalResult) -> tuple[NllRow, ...]:
    dataset = dataset_stem(result.config.dataset)
    baseline = result.baseline
    return (
        _nll_row(dataset, 0.0, baseline, baseline=baseline),
        *tuple(_nll_row(dataset, variant.alpha, variant, baseline=baseline) for variant in result.ewma_variants),
    )


def quality_transfer_rows(routing: RoutingAnalysisResult, nll: EwmaEvalResult) -> tuple[QualityTransferRow, ...]:
    routing_dataset = dataset_stem(routing.config.dataset)
    nll_dataset = dataset_stem(nll.config.dataset)
    if routing_dataset != nll_dataset:
        raise ValueError(f"Routing and NLL datasets must match, got {routing_dataset} and {nll_dataset}.")
    nll_by_alpha = {row.alpha: row.token_nll for row in nll_rows(nll)}
    return tuple(
        QualityTransferRow(
            dataset=row.dataset,
            alpha=row.alpha,
            cache_size=row.cache_size,
            resident_gib_total=row.resident_gib_total,
            transfer_mib_per_token=row.transfer_mib_per_token,
            token_nll=nll_by_alpha[row.alpha],
        )
        for row in continuation_transfer_chart_rows(routing)
    )


def study_rows(summary: StudySummary) -> tuple[StudyRow, ...]:
    return tuple(
        StudyRow(
            alpha=alpha_summary.alpha,
            dataset=dataset_summary.dataset,
            window_size=window.window_size,
            baseline_distinct=window.baseline_distinct,
            ewma_distinct=window.ewma_distinct,
            delta_distinct=window.delta_distinct,
            baseline_cache_hit_rate=window.baseline_cache_hit_rate,
            ewma_cache_hit_rate=window.ewma_cache_hit_rate,
            delta_cache_hit_rate=window.delta_cache_hit_rate,
            retained_fraction=dataset_summary.retained_fraction,
            exact_match_rate=dataset_summary.exact_match_rate,
            passes_rule=dataset_summary.passes_minimal_intervention_rule,
        )
        for alpha_summary in summary.alphas
        for dataset_summary in alpha_summary.datasets
        for window in dataset_summary.windows
    )


def _offload_variant_row(
    dataset: str,
    cache_size: int,
    resident_gib_total: float,
    prompts_processed: int,
    variant: OffloadVariantResult,
) -> OffloadVariantRow:
    return OffloadVariantRow(
        dataset=dataset,
        cache_size=cache_size,
        resident_gib_total=resident_gib_total,
        variant=variant_label(variant.alpha),
        alpha=variant.alpha,
        token_nll=variant.statistics.token_weighted_mean_continuation_nll,
        token_ppl=variant.statistics.token_weighted_continuation_perplexity,
        sequence_nll=variant.statistics.sequence_weighted_mean_continuation_nll,
        sequence_ppl=variant.statistics.sequence_weighted_continuation_perplexity,
        delta_token_nll_vs_baseline=variant.delta_token_weighted_mean_continuation_nll_vs_baseline,
        delta_sequence_nll_vs_baseline=variant.delta_sequence_weighted_mean_continuation_nll_vs_baseline,
        transfer_mib_per_token=variant.transfer.continuation_transfer_mib_per_token,
        loads_per_token=variant.transfer.continuation_expert_loads_per_token,
        hit_rate=variant.transfer.continuation_hit_rate,
        continuation_tokens_per_second=variant.transfer.continuation_tokens_per_second,
        prompts_processed=prompts_processed,
    )


def offload_variant_rows(result: OffloadEvalResult) -> tuple[OffloadVariantRow, ...]:
    dataset = dataset_stem(result.config.dataset)
    rows: list[OffloadVariantRow] = []
    for budget in result.cache_budgets:
        rows.append(
            _offload_variant_row(
                dataset,
                budget.cache_size,
                budget.resident_gib_total,
                budget.prompts_processed,
                budget.baseline,
            )
        )
        rows.extend(
            _offload_variant_row(
                dataset,
                budget.cache_size,
                budget.resident_gib_total,
                budget.prompts_processed,
                variant,
            )
            for variant in budget.ewma_variants
        )
    return tuple(rows)


def offload_budget_summary_rows(result: OffloadEvalResult) -> tuple[OffloadBudgetSummaryRow, ...]:
    dataset = dataset_stem(result.config.dataset)
    rows: list[OffloadBudgetSummaryRow] = []
    for budget in result.cache_budgets:
        variants = (budget.baseline, *budget.ewma_variants)
        best_transfer = min(variants, key=lambda variant: variant.transfer.continuation_transfer_mib_per_token)
        best_nll = min(variants, key=lambda variant: variant.statistics.token_weighted_mean_continuation_nll)
        best_tps = max(variants, key=lambda variant: variant.transfer.continuation_tokens_per_second)
        rows.append(
            OffloadBudgetSummaryRow(
                dataset=dataset,
                cache_size=budget.cache_size,
                resident_gib_total=budget.resident_gib_total,
                best_transfer_variant=variant_label(best_transfer.alpha),
                best_transfer_mib_per_token=best_transfer.transfer.continuation_transfer_mib_per_token,
                best_nll_variant=variant_label(best_nll.alpha),
                best_token_nll=best_nll.statistics.token_weighted_mean_continuation_nll,
                best_tps_variant=variant_label(best_tps.alpha),
                best_continuation_tokens_per_second=best_tps.transfer.continuation_tokens_per_second,
            )
        )
    return tuple(rows)


def transfer_budget_rows(summary: TransferBudgetSummary) -> tuple[TransferBudgetRow, ...]:
    return tuple(
        TransferBudgetRow(
            dataset=dataset.dataset,
            cache_size=budget.cache_size,
            resident_gib_total=budget.resident_gib_total,
            recommended_alpha_without_nll_regression=budget.recommended_alpha_without_nll_regression,
            alpha=variant.alpha,
            token_nll=variant.token_nll,
            delta_token_nll_vs_baseline=variant.delta_token_nll_vs_baseline,
            transfer_mib_per_token=variant.transfer_mib_per_token,
            delta_transfer_mib_per_token_vs_baseline=variant.delta_transfer_mib_per_token_vs_baseline,
            hit_rate=variant.hit_rate,
        )
        for dataset in summary.datasets
        for budget in dataset.budgets
        for variant in budget.variants
    )
