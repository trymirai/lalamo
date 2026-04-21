import pytest

from lalamo.qwen_moe_ewma_eval import VariantResult
from lalamo.qwen_moe_transfer_budget_summary import summarize_transfer_budget
from tests.unit.qwen_moe_test_builders import (
    agreement,
    cache_hit,
    ewma_eval_result,
    ewma_statistics,
    loss_statistics,
    phase_stats,
    resident_budget,
    routing_result,
    window,
)

pytestmark = pytest.mark.fast


def test_summarize_transfer_budget_picks_lowest_transfer_without_nll_regression() -> None:
    routing_payload = routing_result(
        "/tmp/hermes.parquet",
        prompt_statistics=phase_stats(windows=(), resident_budgets=()),
        continuation_statistics=phase_stats(
            windows=(window(8, 10.0, (cache_hit(16, 0.60),)),),
            resident_budgets=(resident_budget(16, 3.0, 0.60, 8.0, 8.0),),
        ),
        ewma_statistics=(
            ewma_statistics(
                0.5,
                prompt_statistics=phase_stats(windows=(), resident_budgets=()),
                continuation_statistics=phase_stats(
                    windows=(window(8, 9.0, (cache_hit(16, 0.70),)),),
                    resident_budgets=(resident_budget(16, 3.0, 0.70, 6.0, 6.0),),
                ),
                prompt_agreement=agreement(0.0, 0.0),
                continuation_agreement=agreement(0.7, 0.05),
            ),
            ewma_statistics(
                0.8,
                prompt_statistics=phase_stats(windows=(), resident_budgets=()),
                continuation_statistics=phase_stats(
                    windows=(window(8, 8.5, (cache_hit(16, 0.65),)),),
                    resident_budgets=(resident_budget(16, 3.0, 0.65, 7.0, 7.0),),
                ),
                prompt_agreement=agreement(0.0, 0.0),
                continuation_agreement=agreement(0.9, 0.35),
            ),
        ),
    )
    nll_payload = ewma_eval_result(
        "/tmp/hermes.parquet",
        VariantResult(
            name="baseline",
            alpha=0.0,
            statistics=loss_statistics(1.0),
            delta_token_weighted_mean_continuation_nll_vs_baseline=0.0,
            delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.0,
        ),
        (
            VariantResult(
                name="ewma_0.500",
                alpha=0.5,
                statistics=loss_statistics(1.1),
                delta_token_weighted_mean_continuation_nll_vs_baseline=0.1,
                delta_sequence_weighted_mean_continuation_nll_vs_baseline=0.1,
            ),
            VariantResult(
                name="ewma_0.800",
                alpha=0.8,
                statistics=loss_statistics(0.99),
                delta_token_weighted_mean_continuation_nll_vs_baseline=-0.01,
                delta_sequence_weighted_mean_continuation_nll_vs_baseline=-0.01,
            ),
        ),
    )

    summary = summarize_transfer_budget([routing_payload], [nll_payload])

    budget = summary.datasets[0].budgets[0]
    assert budget.recommended_alpha_without_nll_regression == "0.8"
    assert budget.variants[0].alpha == "baseline"
    assert budget.variants[1].delta_transfer_mib_per_token_vs_baseline == pytest.approx(-2.0)
    assert budget.variants[2].delta_token_nll_vs_baseline == pytest.approx(-0.01)
