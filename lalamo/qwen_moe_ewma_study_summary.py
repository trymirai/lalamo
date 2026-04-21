from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lalamo.qwen_moe_payloads import read_payload, write_payload
from lalamo.qwen_moe_routing import (
    EwmaStatistics,
    PhaseStatistics,
    RoutingAnalysisResult,
    WindowStatistics,
    parse_window_sizes,
)


@dataclass(frozen=True)
class WindowDelta:
    window_size: int
    baseline_distinct: float
    ewma_distinct: float
    delta_distinct: float
    baseline_cache_hit_rate: float
    ewma_cache_hit_rate: float
    delta_cache_hit_rate: float


@dataclass(frozen=True)
class DatasetAlphaSummary:
    dataset: str
    retained_fraction: float
    exact_match_rate: float
    windows: tuple[WindowDelta, ...]
    passes_minimal_intervention_rule: bool


@dataclass(frozen=True)
class AlphaSummary:
    alpha: float
    datasets: tuple[DatasetAlphaSummary, ...]
    passes_all_datasets: bool


@dataclass(frozen=True)
class StudySummary:
    phase: str
    cache_size: int
    requested_windows: tuple[int, ...]
    effective_windows: tuple[int, ...]
    alphas: tuple[AlphaSummary, ...]
    recommended_alpha: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_results", nargs="+", type=Path)
    parser.add_argument("--ewma-results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("prompt", "continuation"), default="continuation")
    parser.add_argument("--cache-size", type=int, default=16)
    parser.add_argument("--windows", default="8,16,32,64,128,256")
    return parser.parse_args()


def load_payloads(paths: list[Path]) -> dict[str, RoutingAnalysisResult]:
    payloads: dict[str, RoutingAnalysisResult] = {}
    for path in paths:
        payload = read_payload(path, RoutingAnalysisResult)
        dataset = payload.config.dataset
        if dataset in payloads:
            raise ValueError(f"Duplicate dataset payload for {dataset}.")
        payloads[dataset] = payload
    return payloads


def phase_statistics(payload: RoutingAnalysisResult, phase: str) -> PhaseStatistics:
    return payload.prompt_statistics if phase == "prompt" else payload.continuation_statistics


def phase_windows(payload: RoutingAnalysisResult, phase: str) -> dict[int, WindowStatistics]:
    return {window.window_size: window for window in phase_statistics(payload, phase).windows}


def supported_window(window: WindowStatistics) -> bool:
    return window.num_windows > 0 and window.sequence_count_with_windows > 0


def cache_hit_rate(window: WindowStatistics, cache_size: int) -> float:
    matches = [cache for cache in window.oracle_cache_hit_rates if cache.cache_size == cache_size]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cache entry for size {cache_size}, got {len(matches)}.")
    return matches[0].sequence_weighted_hit_rate


def alpha_lookup(payload: RoutingAnalysisResult) -> dict[float, EwmaStatistics]:
    lookup: dict[float, EwmaStatistics] = {}
    for ewma in payload.ewma_statistics:
        if ewma.alpha in lookup:
            raise ValueError(f"Duplicate alpha {ewma.alpha} in {payload.config.dataset}.")
        lookup[ewma.alpha] = ewma
    return lookup


def phase_ewma_statistics(ewma: EwmaStatistics, phase: str) -> PhaseStatistics:
    return ewma.prompt_statistics if phase == "prompt" else ewma.continuation_statistics


def common_supported_windows(
    baseline_payloads: dict[str, RoutingAnalysisResult],
    ewma_payloads: dict[str, RoutingAnalysisResult],
    phase: str,
    requested_windows: tuple[int, ...],
) -> tuple[int, ...]:
    supported = set(requested_windows)
    for dataset, baseline_payload in baseline_payloads.items():
        baseline_windows = phase_windows(baseline_payload, phase)
        supported &= {
            window_size
            for window_size in requested_windows
            if window_size in baseline_windows and supported_window(baseline_windows[window_size])
        }
        for ewma in alpha_lookup(ewma_payloads[dataset]).values():
            ewma_windows = {window.window_size: window for window in phase_ewma_statistics(ewma, phase).windows}
            supported &= {
                window_size
                for window_size in requested_windows
                if window_size in ewma_windows and supported_window(ewma_windows[window_size])
            }
    effective_windows = tuple(window_size for window_size in requested_windows if window_size in supported)
    if not effective_windows:
        raise ValueError(f"No requested windows have support across all datasets and alphas: {requested_windows}.")
    return effective_windows


def common_alphas(ewma_payloads: dict[str, RoutingAnalysisResult]) -> tuple[float, ...]:
    alpha_sets = {dataset: tuple(sorted(alpha_lookup(payload))) for dataset, payload in ewma_payloads.items()}
    first_dataset, first_alphas = next(iter(alpha_sets.items()))
    for dataset, alphas in alpha_sets.items():
        if alphas != first_alphas:
            raise ValueError(f"EWMA alpha set mismatch between {first_dataset} {first_alphas} and {dataset} {alphas}.")
    return first_alphas


def dataset_alpha_summary(
    dataset: str,
    baseline_payload: RoutingAnalysisResult,
    ewma_payload: RoutingAnalysisResult,
    phase: str,
    alpha: float,
    selected_windows: tuple[int, ...],
    cache_size: int,
) -> DatasetAlphaSummary:
    baseline_windows = phase_windows(baseline_payload, phase)
    ewma = alpha_lookup(ewma_payload)[alpha]
    ewma_windows = {window.window_size: window for window in phase_ewma_statistics(ewma, phase).windows}
    deltas = tuple(
        WindowDelta(
            window_size=window_size,
            baseline_distinct=baseline_windows[window_size].sequence_weighted_mean_distinct_experts_overall,
            ewma_distinct=ewma_windows[window_size].sequence_weighted_mean_distinct_experts_overall,
            delta_distinct=(
                ewma_windows[window_size].sequence_weighted_mean_distinct_experts_overall
                - baseline_windows[window_size].sequence_weighted_mean_distinct_experts_overall
            ),
            baseline_cache_hit_rate=cache_hit_rate(baseline_windows[window_size], cache_size),
            ewma_cache_hit_rate=cache_hit_rate(ewma_windows[window_size], cache_size),
            delta_cache_hit_rate=(
                cache_hit_rate(ewma_windows[window_size], cache_size)
                - cache_hit_rate(baseline_windows[window_size], cache_size)
            ),
        )
        for window_size in selected_windows
    )
    agreement = ewma.prompt_agreement if phase == "prompt" else ewma.continuation_agreement
    return DatasetAlphaSummary(
        dataset=dataset,
        retained_fraction=agreement.sequence_weighted_mean_retained_fraction_overall,
        exact_match_rate=agreement.sequence_weighted_exact_match_rate,
        windows=deltas,
        passes_minimal_intervention_rule=(
            all(delta.delta_distinct < 0.0 for delta in deltas)
            and all(delta.delta_cache_hit_rate > 0.0 for delta in deltas)
        ),
    )


def summarize_study(
    baseline_payloads: dict[str, RoutingAnalysisResult],
    ewma_payloads: dict[str, RoutingAnalysisResult],
    phase: str,
    selected_windows: tuple[int, ...],
    cache_size: int,
) -> StudySummary:
    if set(baseline_payloads) != set(ewma_payloads):
        raise ValueError(
            "Baseline and EWMA datasets must match exactly, "
            f"got {sorted(baseline_payloads)} and {sorted(ewma_payloads)}."
        )
    datasets = tuple(sorted(baseline_payloads))
    alphas = common_alphas(ewma_payloads)
    effective_windows = common_supported_windows(baseline_payloads, ewma_payloads, phase, selected_windows)
    summaries = tuple(
        AlphaSummary(
            alpha=alpha,
            datasets=tuple(
                dataset_alpha_summary(
                    dataset=dataset,
                    baseline_payload=baseline_payloads[dataset],
                    ewma_payload=ewma_payloads[dataset],
                    phase=phase,
                    alpha=alpha,
                    selected_windows=effective_windows,
                    cache_size=cache_size,
                )
                for dataset in datasets
            ),
            passes_all_datasets=False,
        )
        for alpha in alphas
    )
    finalized_summaries = tuple(
        AlphaSummary(
            alpha=summary.alpha,
            datasets=summary.datasets,
            passes_all_datasets=all(dataset.passes_minimal_intervention_rule for dataset in summary.datasets),
        )
        for summary in summaries
    )
    passing_alphas = [summary.alpha for summary in finalized_summaries if summary.passes_all_datasets]
    return StudySummary(
        phase=phase,
        cache_size=cache_size,
        requested_windows=selected_windows,
        effective_windows=effective_windows,
        alphas=finalized_summaries,
        recommended_alpha=max(passing_alphas) if passing_alphas else None,
    )


def main() -> None:
    args = parse_args()
    write_payload(
        args.output,
        summarize_study(
            baseline_payloads=load_payloads(args.baseline_results),
            ewma_payloads=load_payloads(args.ewma_results),
            phase=args.phase,
            selected_windows=parse_window_sizes(args.windows),
            cache_size=args.cache_size,
        ),
    )


if __name__ == "__main__":
    main()
