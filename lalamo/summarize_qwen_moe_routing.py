from __future__ import annotations

import argparse
from pathlib import Path

from lalamo.qwen_moe_payloads import read_payload
from lalamo.qwen_moe_routing import PhaseStatistics, RoutingAnalysisResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--phase", choices=("prompt", "continuation", "both"), default="both")
    parser.add_argument("--metric", choices=("sequence", "window"), default="sequence")
    parser.add_argument("--section", choices=("distinct", "cache", "agreement", "both", "all"), default="both")
    return parser.parse_args()


def float_cell(value: float) -> str:
    return f"{value:.4f}"


def phase_statistics(payload: RoutingAnalysisResult, phase: str) -> PhaseStatistics:
    return payload.prompt_statistics if phase == "prompt" else payload.continuation_statistics


def print_phase_table(result_path: Path, payload: RoutingAnalysisResult, phase: str, metric: str) -> None:
    statistics = phase_statistics(payload, phase)
    prefix = f"{metric}_weighted"
    print(f"\n[{result_path.name}] phase={phase} metric={metric}")
    print("dataset\tprompts\ttokens\twindow\tmean\tfraction\tci95\trandom\tratio")
    for window in statistics.windows:
        print(
            "\t".join(
                [
                    payload.config.dataset,
                    str(payload.prompts_processed),
                    str(statistics.token_count),
                    str(window.window_size),
                    float_cell(getattr(window, f"{prefix}_mean_distinct_experts_overall")),
                    float_cell(getattr(window, f"{prefix}_mean_distinct_experts_fraction_overall")),
                    float_cell(getattr(window, f"{prefix}_mean_distinct_experts_overall_ci95")),
                    float_cell(window.random_baseline_distinct_experts),
                    float_cell(getattr(window, f"{prefix}_observed_to_random_ratio")),
                ]
            )
        )


def print_cache_table(result_path: Path, payload: RoutingAnalysisResult, phase: str, metric: str) -> None:
    statistics = phase_statistics(payload, phase)
    hit_key = f"{metric}_weighted_hit_rate"
    print(f"\n[{result_path.name}] phase={phase} cache metric={metric}")
    print("window\tcache_size\tcache_fraction\thit_rate\tci95")
    for window in statistics.windows:
        for cache_statistics in window.oracle_cache_hit_rates:
            print(
                "\t".join(
                    [
                        str(window.window_size),
                        str(cache_statistics.cache_size),
                        float_cell(cache_statistics.cache_fraction),
                        float_cell(getattr(cache_statistics, hit_key)),
                        float_cell(cache_statistics.sequence_weighted_hit_rate_ci95),
                    ]
                )
            )


def print_agreement_table(result_path: Path, payload: RoutingAnalysisResult, phase: str) -> None:
    print(f"\n[{result_path.name}] phase={phase} agreement")
    print("alpha\tretained_token\tretained_sequence\tretained_ci95\texact_token\texact_sequence\texact_ci95")
    for ewma in payload.ewma_statistics:
        agreement = ewma.prompt_agreement if phase == "prompt" else ewma.continuation_agreement
        print(
            "\t".join(
                [
                    float_cell(ewma.alpha),
                    float_cell(agreement.token_weighted_mean_retained_fraction_overall),
                    float_cell(agreement.sequence_weighted_mean_retained_fraction_overall),
                    float_cell(agreement.sequence_weighted_mean_retained_fraction_overall_ci95),
                    float_cell(agreement.token_weighted_exact_match_rate),
                    float_cell(agreement.sequence_weighted_exact_match_rate),
                    float_cell(agreement.sequence_weighted_exact_match_rate_ci95),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    phases = ("prompt", "continuation") if args.phase == "both" else (args.phase,)
    for result_path in args.results:
        payload = read_payload(result_path, RoutingAnalysisResult)
        for phase in phases:
            if args.section in ("distinct", "both", "all"):
                print_phase_table(result_path, payload, phase, args.metric)
            if args.section in ("cache", "both", "all"):
                print_cache_table(result_path, payload, phase, args.metric)
            if args.section in ("agreement", "all"):
                print_agreement_table(result_path, payload, phase)


if __name__ == "__main__":
    main()
