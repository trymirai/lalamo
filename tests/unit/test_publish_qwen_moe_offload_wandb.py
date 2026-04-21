import pytest

from lalamo.qwen_moe_offload_eval import CacheBudgetResult, OffloadEvalResult, OffloadVariantResult
from lalamo.qwen_moe_rows import offload_budget_summary_rows, offload_variant_rows
from tests.unit.qwen_moe_test_builders import loss_statistics, offload_eval_result, offload_transfer

pytestmark = pytest.mark.fast


def sample_offload_result() -> OffloadEvalResult:
    baseline = OffloadVariantResult(
        name="baseline",
        alpha=0.0,
        statistics=loss_statistics(1.5, 1.6),
        transfer=offload_transfer(16, 3.75, 900.0, 150.0, 0.4, 2.5),
        delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
        delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
    )
    variants = (
        OffloadVariantResult(
            name="ewma_0.200",
            alpha=0.2,
            statistics=loss_statistics(1.3, 1.4),
            transfer=offload_transfer(16, 3.75, 250.0, 40.0, 0.78, 5.5),
            delta_token_weighted_mean_continuation_nll_vs_baseline=-0.2,
            delta_sequence_weighted_mean_continuation_nll_vs_baseline=-0.2,
        ),
        OffloadVariantResult(
            name="ewma_0.800",
            alpha=0.8,
            statistics=loss_statistics(1.45, 1.5),
            transfer=offload_transfer(16, 3.75, 600.0, 100.0, 0.55, 4.0),
            delta_token_weighted_mean_continuation_nll_vs_baseline=-0.05,
            delta_sequence_weighted_mean_continuation_nll_vs_baseline=-0.1,
        ),
    )
    return offload_eval_result(
        "/tmp/openhermes.parquet",
        (
            CacheBudgetResult(
                cache_size=16,
                resident_gib_total=3.75,
                baseline=baseline,
                ewma_variants=variants,
                dataset_rows_processed=4,
                prompts_processed=4,
                skipped_prompt_too_long=0,
                skipped_continuation_too_long=0,
                skipped_empty_continuation=0,
                processed_samples=(),
            ),
        ),
    )


def test_offload_rows_expand_baseline_and_variants() -> None:
    rows = offload_variant_rows(sample_offload_result())

    assert [
        (row.dataset, row.cache_size, row.variant, row.alpha, row.token_nll, row.transfer_mib_per_token)
        for row in rows
    ] == [
        ("openhermes", 16, "baseline", 0.0, 1.5, 900.0),
        ("openhermes", 16, "0.2", 0.2, 1.3, 250.0),
        ("openhermes", 16, "0.8", 0.8, 1.45, 600.0),
    ]


def test_budget_summary_rows_pick_best_variant_per_metric() -> None:
    rows = offload_budget_summary_rows(sample_offload_result())

    assert [
        (row.dataset, row.cache_size, row.best_transfer_variant, row.best_nll_variant, row.best_tps_variant)
        for row in rows
    ] == [("openhermes", 16, "0.2", "0.2", "0.2")]
