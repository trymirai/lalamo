import pytest

from lalamo.qwen_moe_ewma_study_summary import summarize_study

pytestmark = pytest.mark.fast


def payload(
    dataset: str,
    window_values: dict[int, tuple[float, float]],
    alpha_values: dict[float, dict[int, tuple[float, float]]],
    *,
    unsupported_windows: tuple[int, ...] = (),
) -> dict:
    return {
        "config": {"dataset": dataset},
        "continuation_statistics": {
            "windows": [
                {
                    "window_size": window_size,
                    "sequence_weighted_mean_distinct_experts_overall": distinct,
                    "oracle_cache_hit_rates": [{"cache_size": 16, "sequence_weighted_hit_rate": cache_hit}],
                    "num_windows": 0 if window_size in unsupported_windows else 1,
                    "sequence_count_with_windows": 0 if window_size in unsupported_windows else 1,
                }
                for window_size, (distinct, cache_hit) in sorted(window_values.items())
            ]
        },
        "ewma_statistics": [
            {
                "alpha": alpha,
                "continuation_statistics": {
                    "windows": [
                        {
                            "window_size": window_size,
                            "sequence_weighted_mean_distinct_experts_overall": distinct,
                            "oracle_cache_hit_rates": [{"cache_size": 16, "sequence_weighted_hit_rate": cache_hit}],
                            "num_windows": 0 if window_size in unsupported_windows else 1,
                            "sequence_count_with_windows": 0 if window_size in unsupported_windows else 1,
                        }
                        for window_size, (distinct, cache_hit) in sorted(alpha_windows.items())
                    ]
                },
                "continuation_agreement": {
                    "sequence_weighted_mean_retained_fraction_overall": 0.9 if alpha == 0.8 else 0.7,
                    "sequence_weighted_exact_match_rate": 0.35 if alpha == 0.8 else 0.05,
                },
            }
            for alpha, alpha_windows in sorted(alpha_values.items())
        ],
    }


def test_summarize_study_selects_largest_alpha_that_improves_all_datasets() -> None:
    baseline = {
        "a": payload("a", {8: (10.0, 0.50), 16: (20.0, 0.40)}, {}),
        "b": payload("b", {8: (12.0, 0.55), 16: (22.0, 0.45)}, {}),
    }
    ewma = {
        "a": payload(
            "a",
            {},
            {
                0.5: {8: (8.0, 0.60), 16: (18.0, 0.50)},
                0.8: {8: (9.0, 0.56), 16: (19.0, 0.46)},
            },
        ),
        "b": payload(
            "b",
            {},
            {
                0.5: {8: (10.0, 0.61), 16: (20.0, 0.48)},
                0.8: {8: (11.0, 0.57), 16: (21.0, 0.46)},
            },
        ),
    }

    summary = summarize_study(
        baseline_payloads=baseline,
        ewma_payloads=ewma,
        phase="continuation",
        selected_windows=(8, 16),
        cache_size=16,
    )

    assert summary.recommended_alpha == 0.8
    assert all(alpha_summary.passes_all_datasets for alpha_summary in summary.alphas)
    assert summary.requested_windows == (8, 16)
    assert summary.effective_windows == (8, 16)


def test_summarize_study_rejects_alpha_that_fails_any_dataset() -> None:
    baseline = {
        "a": payload("a", {8: (10.0, 0.50)}, {}),
        "b": payload("b", {8: (12.0, 0.55)}, {}),
    }
    ewma = {
        "a": payload("a", {}, {0.8: {8: (9.0, 0.60)}}),
        "b": payload("b", {}, {0.8: {8: (13.0, 0.60)}}),
    }

    summary = summarize_study(
        baseline_payloads=baseline,
        ewma_payloads=ewma,
        phase="continuation",
        selected_windows=(8,),
        cache_size=16,
    )

    assert summary.recommended_alpha is None
    assert summary.alphas[0].passes_all_datasets is False


def test_summarize_study_drops_windows_without_shared_support() -> None:
    baseline = {
        "a": payload("a", {32: (10.0, 0.50), 128: (20.0, 0.40)}, {}),
        "b": payload("b", {32: (12.0, 0.55), 128: (0.0, 0.0)}, {}, unsupported_windows=(128,)),
    }
    ewma = {
        "a": payload("a", {}, {0.8: {32: (9.0, 0.60), 128: (19.0, 0.50)}}),
        "b": payload("b", {}, {0.8: {32: (11.0, 0.57), 128: (0.0, 0.0)}}, unsupported_windows=(128,)),
    }

    summary = summarize_study(
        baseline_payloads=baseline,
        ewma_payloads=ewma,
        phase="continuation",
        selected_windows=(32, 128),
        cache_size=16,
    )

    assert summary.recommended_alpha == 0.8
    assert summary.requested_windows == (32, 128)
    assert summary.effective_windows == (32,)
