import pytest

from lalamo.qwen_moe_transfer_budget_summary import summarize_transfer_budget

pytestmark = pytest.mark.fast


def test_summarize_transfer_budget_picks_lowest_transfer_without_nll_regression() -> None:
    routing_payload = {
        "config": {"dataset": "/tmp/hermes.parquet"},
        "continuation_statistics": {
            "resident_budgets": [
                {
                    "cache_size": 16,
                    "resident_gib_total": 3.0,
                    "sequence_weighted_hit_rate": 0.60,
                    "sequence_weighted_transfer_bytes_per_token": 8.0 * 1024**2,
                }
            ]
        },
        "ewma_statistics": [
            {
                "alpha": 0.5,
                "continuation_statistics": {
                    "resident_budgets": [
                        {
                            "cache_size": 16,
                            "resident_gib_total": 3.0,
                            "sequence_weighted_hit_rate": 0.70,
                            "sequence_weighted_transfer_bytes_per_token": 6.0 * 1024**2,
                        }
                    ]
                },
            },
            {
                "alpha": 0.8,
                "continuation_statistics": {
                    "resident_budgets": [
                        {
                            "cache_size": 16,
                            "resident_gib_total": 3.0,
                            "sequence_weighted_hit_rate": 0.65,
                            "sequence_weighted_transfer_bytes_per_token": 7.0 * 1024**2,
                        }
                    ]
                },
            },
        ],
    }
    nll_payload = {
        "config": {"dataset": "/tmp/hermes.parquet"},
        "baseline": {"statistics": {"token_weighted_mean_continuation_nll": 1.0}},
        "ewma_variants": [
            {
                "alpha": 0.5,
                "statistics": {"token_weighted_mean_continuation_nll": 1.1},
            },
            {
                "alpha": 0.8,
                "statistics": {"token_weighted_mean_continuation_nll": 0.99},
            },
        ],
    }

    summary = summarize_transfer_budget([routing_payload], [nll_payload])

    budget = summary.datasets[0].budgets[0]
    assert budget.recommended_alpha_without_nll_regression == "0.8"
    assert budget.variants[0].alpha == "baseline"
    assert budget.variants[1].delta_transfer_mib_per_token_vs_baseline == pytest.approx(-2.0)
    assert budget.variants[2].delta_token_nll_vs_baseline == pytest.approx(-0.01)
