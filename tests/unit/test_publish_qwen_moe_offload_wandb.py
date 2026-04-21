import pytest

from lalamo.publish_qwen_moe_offload_wandb import budget_summary_rows, cache_budget_rows, dataset_name

pytestmark = pytest.mark.fast


def sample_offload_payload() -> dict:
    return {
        "config": {
            "dataset": "/tmp/openhermes.parquet",
            "model_repo": "Qwen/Qwen3.6-35B-A3B",
            "seed": 0,
            "cache_sizes": [16],
            "ewma_alphas": [0.2, 0.8],
            "max_rows": 4,
            "max_prompts": 4,
            "max_prompt_tokens": 2048,
            "max_continuation_tokens": 512,
            "device_map_mode": "balanced-low-0",
        },
        "dataset_rows_total": 4,
        "assistant_turns_total": 4,
        "cache_budgets": [
            {
                "cache_size": 16,
                "resident_gib_total": 3.75,
                "prompts_processed": 4,
                "baseline": {
                    "alpha": 0.0,
                    "statistics": {
                        "token_weighted_mean_continuation_nll": 1.5,
                        "token_weighted_continuation_perplexity": 4.5,
                        "sequence_weighted_mean_continuation_nll": 1.6,
                        "sequence_weighted_continuation_perplexity": 4.8,
                    },
                    "transfer": {
                        "continuation_transfer_mib_per_token": 900.0,
                        "continuation_expert_loads_per_token": 150.0,
                        "continuation_hit_rate": 0.4,
                        "continuation_tokens_per_second": 2.5,
                    },
                    "delta_token_weighted_mean_continuation_nll_vs_baseline": 0.0,
                    "delta_sequence_weighted_mean_continuation_nll_vs_baseline": 0.0,
                },
                "ewma_variants": [
                    {
                        "alpha": 0.2,
                        "statistics": {
                            "token_weighted_mean_continuation_nll": 1.3,
                            "token_weighted_continuation_perplexity": 3.7,
                            "sequence_weighted_mean_continuation_nll": 1.4,
                            "sequence_weighted_continuation_perplexity": 4.1,
                        },
                        "transfer": {
                            "continuation_transfer_mib_per_token": 250.0,
                            "continuation_expert_loads_per_token": 40.0,
                            "continuation_hit_rate": 0.78,
                            "continuation_tokens_per_second": 5.5,
                        },
                        "delta_token_weighted_mean_continuation_nll_vs_baseline": -0.2,
                        "delta_sequence_weighted_mean_continuation_nll_vs_baseline": -0.2,
                    },
                    {
                        "alpha": 0.8,
                        "statistics": {
                            "token_weighted_mean_continuation_nll": 1.45,
                            "token_weighted_continuation_perplexity": 4.3,
                            "sequence_weighted_mean_continuation_nll": 1.5,
                            "sequence_weighted_continuation_perplexity": 4.5,
                        },
                        "transfer": {
                            "continuation_transfer_mib_per_token": 600.0,
                            "continuation_expert_loads_per_token": 100.0,
                            "continuation_hit_rate": 0.55,
                            "continuation_tokens_per_second": 4.0,
                        },
                        "delta_token_weighted_mean_continuation_nll_vs_baseline": -0.05,
                        "delta_sequence_weighted_mean_continuation_nll_vs_baseline": -0.1,
                    },
                ],
            }
        ],
    }


def test_dataset_name_uses_path_stem() -> None:
    assert dataset_name(sample_offload_payload()) == "openhermes"


def test_cache_budget_rows_expand_baseline_and_variants() -> None:
    rows = cache_budget_rows(sample_offload_payload(), "openhermes")

    assert rows == [
        ["openhermes", 16, 3.75, "baseline", 0.0, 1.5, 4.5, 1.6, 4.8, 0.0, 0.0, 900.0, 150.0, 0.4, 2.5, 4],
        ["openhermes", 16, 3.75, "0.2", 0.2, 1.3, 3.7, 1.4, 4.1, -0.2, -0.2, 250.0, 40.0, 0.78, 5.5, 4],
        ["openhermes", 16, 3.75, "0.8", 0.8, 1.45, 4.3, 1.5, 4.5, -0.05, -0.1, 600.0, 100.0, 0.55, 4.0, 4],
    ]


def test_budget_summary_rows_pick_best_variant_per_metric() -> None:
    rows = budget_summary_rows(sample_offload_payload(), "openhermes")

    assert rows == [["openhermes", 16, 3.75, "0.2", 250.0, "0.2", 1.3, "0.2", 5.5]]
