import pytest

from lalamo.qwen_moe_ewma_eval import EwmaEvalResult, VariantResult
from lalamo.qwen_moe_ewma_study_summary import StudySummary, WindowDelta
from lalamo.qwen_moe_routing import RoutingAnalysisResult
from lalamo.qwen_moe_rows import (
    continuation_agreement_chart_rows,
    continuation_cache_chart_rows,
    continuation_locality_chart_rows,
    continuation_transfer_chart_rows,
    nll_rows,
    quality_transfer_rows,
    routing_rows,
    study_rows,
)
from tests.unit.qwen_moe_test_builders import (
    agreement,
    cache_hit,
    ewma_eval_result,
    ewma_statistics,
    loss_statistics,
    phase_stats,
    resident_budget,
    routing_result,
    study_summary,
    window,
)

pytestmark = pytest.mark.fast


def sample_routing_result() -> RoutingAnalysisResult:
    prompt = phase_stats(
        windows=(window(4, 10.0, (cache_hit(8, 0.4, 0.03),), distinct_ci95=0.5, random_baseline=12.0),),
        resident_budgets=(resident_budget(8, 1.5, 0.4, 10.0, 10.0, hit_ci95=0.03, loads_ci95=0.5, transfer_ci95=0.5),),
    )
    continuation = phase_stats(
        windows=(window(8, 20.0, (cache_hit(16, 0.6, 0.02),), distinct_ci95=1.0, random_baseline=24.0),),
        resident_budgets=(resident_budget(16, 3.0, 0.6, 8.0, 8.0, hit_ci95=0.02, loads_ci95=0.4, transfer_ci95=0.4),),
    )
    ewma = ewma_statistics(
        0.8,
        prompt_statistics=phase_stats(
            windows=(window(4, 9.5, (cache_hit(8, 0.42, 0.02),), distinct_ci95=0.4, random_baseline=12.0),),
            resident_budgets=(
                resident_budget(8, 1.5, 0.42, 9.0, 9.0, hit_ci95=0.02, loads_ci95=0.4, transfer_ci95=0.4),
            ),
        ),
        continuation_statistics=phase_stats(
            windows=(window(8, 19.0, (cache_hit(16, 0.64, 0.02),), distinct_ci95=0.8, random_baseline=24.0),),
            resident_budgets=(
                resident_budget(16, 3.0, 0.64, 7.0, 7.0, hit_ci95=0.02, loads_ci95=0.3, transfer_ci95=0.3),
            ),
        ),
        prompt_agreement=agreement(0.9, 0.3, retained_ci95=0.01, exact_ci95=0.02),
        continuation_agreement=agreement(0.88, 0.34, retained_ci95=0.01, exact_ci95=0.02),
    )
    return routing_result(
        "/tmp/hermes.parquet",
        prompt_statistics=prompt,
        continuation_statistics=continuation,
        ewma_statistics=(ewma,),
    )


def sample_nll_result() -> EwmaEvalResult:
    baseline = VariantResult(
        name="baseline",
        alpha=0.0,
        statistics=loss_statistics(1.0, 1.1),
        delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
    )
    variant = VariantResult(
        name="ewma_0.800",
        alpha=0.8,
        statistics=loss_statistics(1.05, 1.15),
        delta_token_weighted_mean_continuation_nll_vs_baseline=0.05,
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.05,
    )
    return ewma_eval_result("/tmp/hermes.parquet", baseline, (variant,))


def test_routing_row_extractors_keep_window_and_alpha() -> None:
    locality_rows, cache_rows, resident_rows, agreement_rows = routing_rows(sample_routing_result())

    continuation_locality = next(row for row in locality_rows if row.phase == "continuation" and row.alpha == "0.8")
    continuation_cache = next(row for row in cache_rows if row.phase == "continuation" and row.alpha == "0.8")
    continuation_resident = next(row for row in resident_rows if row.phase == "continuation" and row.alpha == "0.8")
    continuation_agreement = next(row for row in agreement_rows if row.phase == "continuation")

    assert continuation_locality.dataset == "hermes"
    assert (continuation_locality.window_size, continuation_locality.distinct_experts) == (8, 19.0)
    assert (continuation_cache.cache_size, continuation_cache.cache_hit_rate) == (16, 0.64)
    assert (continuation_resident.cache_size, continuation_resident.transfer_mib_per_token) == (16, 7.0)
    assert (continuation_agreement.retained_fraction, continuation_agreement.exact_match_rate) == (0.88, 0.34)


def test_continuation_chart_rows_are_dataset_scoped() -> None:
    payload = sample_routing_result()

    assert [
        (row.dataset, row.alpha, row.window_size, row.distinct_experts)
        for row in continuation_locality_chart_rows(payload)
    ] == [("hermes", "baseline", 8, 20.0), ("hermes", "0.8", 8, 19.0)]
    assert [
        (row.dataset, row.alpha, row.window_size, row.cache_hit_rate)
        for row in continuation_cache_chart_rows(payload, cache_size=16)
    ] == [("hermes", "baseline", 8, 0.6), ("hermes", "0.8", 8, 0.64)]
    assert [
        (row.dataset, row.alpha, row.cache_size, row.transfer_mib_per_token)
        for row in continuation_transfer_chart_rows(payload)
    ] == [("hermes", "baseline", 16, 8.0), ("hermes", "0.8", 16, 7.0)]
    assert [
        (row.dataset, row.alpha, row.retained_fraction, row.exact_match_rate)
        for row in continuation_agreement_chart_rows(payload)
    ] == [("hermes", "0.8", 0.88, 0.34)]


def test_nll_rows_include_baseline_and_variant() -> None:
    payload = sample_nll_result()
    rows = nll_rows(payload)

    assert [(row.dataset, row.alpha, row.token_nll, row.token_ppl) for row in rows] == [
        ("hermes", "baseline", 1.0, 2.0),
        ("hermes", "0.8", 1.05, 2.05),
    ]
    assert [
        (row.dataset, row.alpha, row.transfer_mib_per_token, row.token_nll)
        for row in quality_transfer_rows(sample_routing_result(), payload)
    ] == [("hermes", "baseline", 8.0, 1.0), ("hermes", "0.8", 7.0, 1.05)]


def test_study_rows_expand_each_window() -> None:
    payload: StudySummary = study_summary(
        0.8,
        WindowDelta(
            window_size=32,
            baseline_distinct=77.0,
            ewma_distinct=70.0,
            delta_distinct=-7.0,
            baseline_cache_hit_rate=0.55,
            ewma_cache_hit_rate=0.58,
            delta_cache_hit_rate=0.03,
        ),
    )
    rows = study_rows(payload)

    assert len(rows) == 1
    row = rows[0]
    assert (row.alpha, row.dataset, row.window_size) == (0.8, "hermes", 32)
    assert (row.delta_distinct, row.delta_cache_hit_rate, row.passes_rule) == (-7.0, 0.03, True)
