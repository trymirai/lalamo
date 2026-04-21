from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    alpha: float
    retained_fraction: float
    exact_match_rate: float
    windows: tuple[WindowDelta, ...]
    improves_distinct_everywhere: bool
    improves_cache_everywhere: bool
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


def parse_windows(text: str) -> tuple[int, ...]:
    windows = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not windows:
        raise ValueError("windows must not be empty.")
    if tuple(sorted(windows)) != windows:
        raise ValueError(f"windows must be sorted ascending, got {windows}.")
    return windows


def load_payloads(paths: list[Path]) -> dict[str, dict[str, Any]]:
    payloads = {}
    for path in paths:
        payload = json.loads(path.read_text())
        dataset = str(payload["config"]["dataset"])
        if dataset in payloads:
            raise ValueError(f"Duplicate dataset payload for {dataset}.")
        payloads[dataset] = payload
    return payloads


def phase_windows(payload: dict[str, Any], phase: str) -> dict[int, dict[str, Any]]:
    return {int(window["window_size"]): window for window in payload[f"{phase}_statistics"]["windows"]}


def supported_window(window: dict[str, Any]) -> bool:
    return int(window.get("num_windows", 1)) > 0 and int(window.get("sequence_count_with_windows", 1)) > 0


def cache_hit_rate(window: dict[str, Any], cache_size: int) -> float:
    matches = [cache for cache in window["oracle_cache_hit_rates"] if int(cache["cache_size"]) == cache_size]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one cache entry for size {cache_size}, got {len(matches)}.")
    return float(matches[0]["sequence_weighted_hit_rate"])


def alpha_lookup(payload: dict[str, Any], phase: str) -> dict[float, dict[str, Any]]:
    result = {}
    for ewma_payload in payload["ewma_statistics"]:
        ewma = dict(ewma_payload)
        ewma["phase_statistics"] = {
            int(window["window_size"]): window for window in ewma[f"{phase}_statistics"]["windows"]
        }
        ewma["phase_agreement"] = ewma[f"{phase}_agreement"]
        result[float(ewma["alpha"])] = ewma
    return result


def common_supported_windows(
    baseline_payloads: dict[str, dict[str, Any]],
    ewma_payloads: dict[str, dict[str, Any]],
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
        for ewma_payload in alpha_lookup(ewma_payloads[dataset], phase).values():
            supported &= {
                window_size
                for window_size in requested_windows
                if window_size in ewma_payload["phase_statistics"]
                and supported_window(ewma_payload["phase_statistics"][window_size])
            }
    effective_windows = tuple(window_size for window_size in requested_windows if window_size in supported)
    if not effective_windows:
        raise ValueError(f"No requested windows have support across all datasets and alphas: {requested_windows}.")
    return effective_windows


def dataset_alpha_summary(
    dataset: str,
    baseline_payload: dict[str, Any],
    ewma_payload: dict[str, Any],
    phase: str,
    alpha: float,
    selected_windows: tuple[int, ...],
    cache_size: int,
) -> DatasetAlphaSummary:
    baseline = phase_windows(baseline_payload, phase)
    ewma = alpha_lookup(ewma_payload, phase)[alpha]
    deltas = []
    for window_size in selected_windows:
        baseline_window = baseline[window_size]
        ewma_window = ewma["phase_statistics"][window_size]
        baseline_distinct = float(baseline_window["sequence_weighted_mean_distinct_experts_overall"])
        ewma_distinct = float(ewma_window["sequence_weighted_mean_distinct_experts_overall"])
        baseline_cache = cache_hit_rate(baseline_window, cache_size)
        ewma_cache = cache_hit_rate(ewma_window, cache_size)
        deltas.append(
            WindowDelta(
                window_size=window_size,
                baseline_distinct=baseline_distinct,
                ewma_distinct=ewma_distinct,
                delta_distinct=ewma_distinct - baseline_distinct,
                baseline_cache_hit_rate=baseline_cache,
                ewma_cache_hit_rate=ewma_cache,
                delta_cache_hit_rate=ewma_cache - baseline_cache,
            )
        )
    improves_distinct_everywhere = all(delta.delta_distinct < 0.0 for delta in deltas)
    improves_cache_everywhere = all(delta.delta_cache_hit_rate > 0.0 for delta in deltas)
    agreement = ewma["phase_agreement"]
    return DatasetAlphaSummary(
        dataset=dataset,
        alpha=alpha,
        retained_fraction=float(agreement["sequence_weighted_mean_retained_fraction_overall"]),
        exact_match_rate=float(agreement["sequence_weighted_exact_match_rate"]),
        windows=tuple(deltas),
        improves_distinct_everywhere=improves_distinct_everywhere,
        improves_cache_everywhere=improves_cache_everywhere,
        passes_minimal_intervention_rule=improves_distinct_everywhere and improves_cache_everywhere,
    )


def summarize_study(
    baseline_payloads: dict[str, dict[str, Any]],
    ewma_payloads: dict[str, dict[str, Any]],
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
    alphas = tuple(sorted(alpha_lookup(next(iter(ewma_payloads.values())), phase)))
    effective_windows = common_supported_windows(baseline_payloads, ewma_payloads, phase, selected_windows)
    summaries = []
    for alpha in alphas:
        dataset_summaries = tuple(
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
        )
        summaries.append(
            AlphaSummary(
                alpha=alpha,
                datasets=dataset_summaries,
                passes_all_datasets=all(summary.passes_minimal_intervention_rule for summary in dataset_summaries),
            )
        )
    passing_alphas = [summary.alpha for summary in summaries if summary.passes_all_datasets]
    return StudySummary(
        phase=phase,
        cache_size=cache_size,
        requested_windows=selected_windows,
        effective_windows=effective_windows,
        alphas=tuple(summaries),
        recommended_alpha=max(passing_alphas) if passing_alphas else None,
    )


def main() -> None:
    args = parse_args()
    summary = summarize_study(
        baseline_payloads=load_payloads(args.baseline_results),
        ewma_payloads=load_payloads(args.ewma_results),
        phase=args.phase,
        selected_windows=parse_windows(args.windows),
        cache_size=args.cache_size,
    )
    args.output.write_text(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
