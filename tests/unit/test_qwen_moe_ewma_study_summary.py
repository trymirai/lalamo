import pytest

from lalamo.qwen_moe_ewma_study_summary import summarize_study
from lalamo.qwen_moe_routing import RoutingAnalysisResult, WindowStatistics
from tests.unit.qwen_moe_test_builders import (
    agreement,
    cache_hit,
    ewma_statistics,
    phase_stats,
    routing_result,
    window,
)

pytestmark = pytest.mark.fast


def make_result(
    dataset: str,
    window_values: dict[int, tuple[float, float]],
    alpha_values: dict[float, dict[int, tuple[float, float]]],
    *,
    unsupported_windows: tuple[int, ...] = (),
) -> RoutingAnalysisResult:
    def sample_window(window_size: int, distinct: float, cache_rate: float) -> WindowStatistics:
        supported = window_size not in unsupported_windows
        return window(
            window_size,
            distinct,
            (cache_hit(16, cache_rate),),
            num_windows=1 if supported else 0,
            sequence_count_with_windows=1 if supported else 0,
        )

    baseline = phase_stats(
        windows=tuple(
            sample_window(window_size, distinct, cache_rate)
            for window_size, (distinct, cache_rate) in sorted(window_values.items())
        ),
        resident_budgets=(),
    )
    ewma_payloads = tuple(
        ewma_statistics(
            alpha,
            prompt_statistics=phase_stats(windows=(), resident_budgets=()),
            continuation_statistics=phase_stats(
                windows=tuple(
                    sample_window(window_size, distinct, cache_rate)
                    for window_size, (distinct, cache_rate) in sorted(alpha_windows.items())
                ),
                resident_budgets=(),
            ),
            prompt_agreement=agreement(0.0, 0.0),
            continuation_agreement=agreement(0.9 if alpha == 0.8 else 0.7, 0.35 if alpha == 0.8 else 0.05),
        )
        for alpha, alpha_windows in sorted(alpha_values.items())
    )
    return routing_result(
        dataset,
        prompt_statistics=phase_stats(windows=(), resident_budgets=()),
        continuation_statistics=baseline,
        ewma_statistics=ewma_payloads,
    )


def test_summarize_study_selects_largest_alpha_that_improves_all_datasets() -> None:
    baseline = {
        "a": make_result("a", {8: (10.0, 0.50), 16: (20.0, 0.40)}, {}),
        "b": make_result("b", {8: (12.0, 0.55), 16: (22.0, 0.45)}, {}),
    }
    ewma = {
        "a": make_result("a", {}, {0.5: {8: (8.0, 0.60), 16: (18.0, 0.50)}, 0.8: {8: (9.0, 0.56), 16: (19.0, 0.46)}}),
        "b": make_result(
            "b",
            {},
            {0.5: {8: (10.0, 0.61), 16: (20.0, 0.48)}, 0.8: {8: (11.0, 0.57), 16: (21.0, 0.46)}},
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
        "a": make_result("a", {8: (10.0, 0.50)}, {}),
        "b": make_result("b", {8: (12.0, 0.55)}, {}),
    }
    ewma = {
        "a": make_result("a", {}, {0.8: {8: (9.0, 0.60)}}),
        "b": make_result("b", {}, {0.8: {8: (13.0, 0.60)}}),
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
        "a": make_result("a", {32: (10.0, 0.50), 128: (20.0, 0.40)}, {}),
        "b": make_result("b", {32: (12.0, 0.55), 128: (0.0, 0.0)}, {}, unsupported_windows=(128,)),
    }
    ewma = {
        "a": make_result("a", {}, {0.8: {32: (9.0, 0.60), 128: (19.0, 0.50)}}),
        "b": make_result("b", {}, {0.8: {32: (11.0, 0.57), 128: (0.0, 0.0)}}, unsupported_windows=(128,)),
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
