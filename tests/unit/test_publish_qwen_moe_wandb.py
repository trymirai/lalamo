import pytest

from lalamo.publish_qwen_moe_wandb import (
    agreement_rows,
    continuation_agreement_chart_rows,
    continuation_cache_chart_rows,
    continuation_locality_chart_rows,
    continuation_transfer_chart_rows,
    dataset_name,
    nll_chart_rows,
    nll_rows,
    phase_cache_rows,
    phase_locality_rows,
    phase_resident_budget_rows,
    quality_transfer_chart_rows,
    study_rows,
)

pytestmark = pytest.mark.fast


def sample_routing_payload() -> dict:
    return {
        "config": {"dataset": "/tmp/hermes.parquet"},
        "prompt_statistics": {
            "windows": [
                {
                    "window_size": 4,
                    "sequence_weighted_mean_distinct_experts_overall": 10.0,
                    "sequence_weighted_mean_distinct_experts_overall_ci95": 0.5,
                    "random_baseline_distinct_experts": 12.0,
                    "sequence_weighted_observed_to_random_ratio": 0.8,
                    "oracle_cache_hit_rates": [
                        {"cache_size": 8, "sequence_weighted_hit_rate": 0.4, "sequence_weighted_hit_rate_ci95": 0.03},
                    ],
                }
            ],
            "resident_budgets": [
                {
                    "cache_size": 8,
                    "resident_gib_total": 1.5,
                    "sequence_weighted_hit_rate": 0.4,
                    "sequence_weighted_hit_rate_ci95": 0.03,
                    "sequence_weighted_expert_loads_per_token": 10.0,
                    "sequence_weighted_expert_loads_per_token_ci95": 0.5,
                    "sequence_weighted_transfer_bytes_per_token": 10.0 * 1024**2,
                    "sequence_weighted_transfer_bytes_per_token_ci95": 0.5 * 1024**2,
                }
            ],
        },
        "continuation_statistics": {
            "windows": [
                {
                    "window_size": 8,
                    "sequence_weighted_mean_distinct_experts_overall": 20.0,
                    "sequence_weighted_mean_distinct_experts_overall_ci95": 1.0,
                    "random_baseline_distinct_experts": 24.0,
                    "sequence_weighted_observed_to_random_ratio": 0.9,
                    "oracle_cache_hit_rates": [
                        {"cache_size": 16, "sequence_weighted_hit_rate": 0.6, "sequence_weighted_hit_rate_ci95": 0.02},
                    ],
                }
            ],
            "resident_budgets": [
                {
                    "cache_size": 16,
                    "resident_gib_total": 3.0,
                    "sequence_weighted_hit_rate": 0.6,
                    "sequence_weighted_hit_rate_ci95": 0.02,
                    "sequence_weighted_expert_loads_per_token": 8.0,
                    "sequence_weighted_expert_loads_per_token_ci95": 0.4,
                    "sequence_weighted_transfer_bytes_per_token": 8.0 * 1024**2,
                    "sequence_weighted_transfer_bytes_per_token_ci95": 0.4 * 1024**2,
                }
            ],
        },
        "ewma_statistics": [
            {
                "alpha": 0.8,
                "prompt_statistics": {
                    "windows": [
                        {
                            "window_size": 4,
                            "sequence_weighted_mean_distinct_experts_overall": 9.5,
                            "sequence_weighted_mean_distinct_experts_overall_ci95": 0.4,
                            "random_baseline_distinct_experts": 12.0,
                            "sequence_weighted_observed_to_random_ratio": 0.79,
                            "oracle_cache_hit_rates": [
                                {
                                    "cache_size": 8,
                                    "sequence_weighted_hit_rate": 0.42,
                                    "sequence_weighted_hit_rate_ci95": 0.02,
                                },
                            ],
                        }
                    ],
                    "resident_budgets": [
                        {
                            "cache_size": 8,
                            "resident_gib_total": 1.5,
                            "sequence_weighted_hit_rate": 0.42,
                            "sequence_weighted_hit_rate_ci95": 0.02,
                            "sequence_weighted_expert_loads_per_token": 9.0,
                            "sequence_weighted_expert_loads_per_token_ci95": 0.4,
                            "sequence_weighted_transfer_bytes_per_token": 9.0 * 1024**2,
                            "sequence_weighted_transfer_bytes_per_token_ci95": 0.4 * 1024**2,
                        }
                    ],
                },
                "continuation_statistics": {
                    "windows": [
                        {
                            "window_size": 8,
                            "sequence_weighted_mean_distinct_experts_overall": 19.0,
                            "sequence_weighted_mean_distinct_experts_overall_ci95": 0.8,
                            "random_baseline_distinct_experts": 24.0,
                            "sequence_weighted_observed_to_random_ratio": 0.85,
                            "oracle_cache_hit_rates": [
                                {
                                    "cache_size": 16,
                                    "sequence_weighted_hit_rate": 0.64,
                                    "sequence_weighted_hit_rate_ci95": 0.02,
                                },
                            ],
                        }
                    ],
                    "resident_budgets": [
                        {
                            "cache_size": 16,
                            "resident_gib_total": 3.0,
                            "sequence_weighted_hit_rate": 0.64,
                            "sequence_weighted_hit_rate_ci95": 0.02,
                            "sequence_weighted_expert_loads_per_token": 7.0,
                            "sequence_weighted_expert_loads_per_token_ci95": 0.3,
                            "sequence_weighted_transfer_bytes_per_token": 7.0 * 1024**2,
                            "sequence_weighted_transfer_bytes_per_token_ci95": 0.3 * 1024**2,
                        }
                    ],
                },
                "prompt_agreement": {
                    "sequence_weighted_mean_retained_fraction_overall": 0.9,
                    "sequence_weighted_mean_retained_fraction_overall_ci95": 0.01,
                    "sequence_weighted_exact_match_rate": 0.3,
                    "sequence_weighted_exact_match_rate_ci95": 0.02,
                },
                "continuation_agreement": {
                    "sequence_weighted_mean_retained_fraction_overall": 0.88,
                    "sequence_weighted_mean_retained_fraction_overall_ci95": 0.01,
                    "sequence_weighted_exact_match_rate": 0.34,
                    "sequence_weighted_exact_match_rate_ci95": 0.02,
                },
            }
        ],
    }


def test_dataset_name_uses_path_stem() -> None:
    assert dataset_name(sample_routing_payload()) == "hermes"


def test_phase_row_extractors_keep_window_and_alpha() -> None:
    payload = sample_routing_payload()
    locality = phase_locality_rows(payload["continuation_statistics"], "hermes", "continuation", 0.8)
    cache = phase_cache_rows(payload["continuation_statistics"], "hermes", "continuation", 0.8)
    resident = phase_resident_budget_rows(payload["continuation_statistics"], "hermes", "continuation", 0.8)

    assert locality == [["hermes", "continuation", "0.8", 8, 20.0, 1.0, 24.0, 0.9]]
    assert cache == [["hermes", "continuation", "0.8", 8, 16, 0.6, 0.02]]
    assert resident == [["hermes", "continuation", "0.8", 16, 3.0, 0.6, 0.02, 8.0, 0.4, 8.0, 0.4]]


def test_agreement_rows_emit_one_row_per_phase() -> None:
    rows = agreement_rows(sample_routing_payload(), "hermes")

    assert rows == [
        ["hermes", "prompt", "0.8", 0.9, 0.01, 0.3, 0.02],
        ["hermes", "continuation", "0.8", 0.88, 0.01, 0.34, 0.02],
    ]


def test_continuation_chart_rows_are_dataset_scoped() -> None:
    payload = sample_routing_payload()

    assert continuation_locality_chart_rows(payload, "hermes") == [
        ["hermes", "baseline", 8, 20.0],
        ["hermes", "0.8", 8, 19.0],
    ]
    assert continuation_cache_chart_rows(payload, "hermes", cache_size=16) == [
        ["hermes", "baseline", 8, 0.6],
        ["hermes", "0.8", 8, 0.64],
    ]
    assert continuation_transfer_chart_rows(payload, "hermes") == [
        ["hermes", "baseline", 16, 3.0, 8.0],
        ["hermes", "0.8", 16, 3.0, 7.0],
    ]
    assert continuation_agreement_chart_rows(payload, "hermes") == [[
        "hermes",
        0.8,
        0.88,
        0.34,
    ]]


def test_nll_rows_include_baseline_and_variant() -> None:
    payload = {
        "baseline": {
            "statistics": {
                "token_weighted_mean_continuation_nll": 1.0,
                "token_weighted_continuation_perplexity": 2.0,
                "sequence_weighted_mean_continuation_nll": 1.1,
                "sequence_weighted_continuation_perplexity": 2.2,
            }
        },
        "ewma_variants": [
            {
                "alpha": 0.8,
                "statistics": {
                    "token_weighted_mean_continuation_nll": 1.05,
                    "token_weighted_continuation_perplexity": 2.1,
                    "sequence_weighted_mean_continuation_nll": 1.15,
                    "sequence_weighted_continuation_perplexity": 2.3,
                },
                "delta_token_weighted_mean_continuation_nll_vs_baseline": 0.05,
                "delta_sequence_weighted_mean_continuation_nll_vs_baseline": 0.05,
            }
        ],
    }

    rows = nll_rows(payload, "hermes")

    assert rows == [
        ["hermes", "baseline", 1.0, 2.0, 1.1, 2.2, 0.0, 0.0],
        ["hermes", "0.8", 1.05, 2.1, 1.15, 2.3, 0.05, 0.05],
    ]
    assert nll_chart_rows(payload, "hermes") == [
        ["hermes", "baseline", 1.0, 2.0],
        ["hermes", "0.8", 1.05, 2.1],
    ]
    assert quality_transfer_chart_rows(sample_routing_payload(), payload, "hermes") == [
        ["hermes", "baseline", 16, 3.0, 8.0, 1.0],
        ["hermes", "0.8", 16, 3.0, 7.0, 1.05],
    ]


def test_study_rows_expand_each_window() -> None:
    payload = {
        "alphas": [
            {
                "alpha": 0.8,
                "datasets": [
                    {
                        "dataset": "hermes",
                        "retained_fraction": 0.88,
                        "exact_match_rate": 0.34,
                        "passes_minimal_intervention_rule": True,
                        "windows": [
                            {
                                "window_size": 32,
                                "baseline_distinct": 77.0,
                                "ewma_distinct": 70.0,
                                "delta_distinct": -7.0,
                                "baseline_cache_hit_rate": 0.55,
                                "ewma_cache_hit_rate": 0.58,
                                "delta_cache_hit_rate": 0.03,
                            }
                        ],
                    }
                ],
            }
        ]
    }

    assert study_rows(payload) == [[0.8, "hermes", 32, 77.0, 70.0, -7.0, 0.55, 0.58, 0.03, 0.88, 0.34, True]]
