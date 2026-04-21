from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--phase", choices=("prompt", "continuation", "both"), default="both")
    parser.add_argument("--metric", choices=("sequence", "window"), default="sequence")
    parser.add_argument("--section", choices=("distinct", "cache", "agreement", "both", "all"), default="both")
    return parser.parse_args()


def float_cell(value: float) -> str:
    return f"{value:.4f}"


def phase_payload(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    key = f"{phase}_statistics"
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Missing {key} in {payload}.")
    return value


def print_phase_table(result_path: Path, payload: dict[str, Any], phase: str, metric: str) -> None:
    config = payload["config"]
    statistics = phase_payload(payload, phase)
    prefix = f"{metric}_weighted"
    print(f"\n[{result_path.name}] phase={phase} metric={metric}")
    print("dataset\tprompts\ttokens\twindow\tmean\tfraction\tci95\trandom\tratio")
    for window in statistics["windows"]:
        ci95 = window.get(f"{prefix}_mean_distinct_experts_overall_ci95", 0.0)
        print(
            "\t".join(
                [
                    str(config["dataset"]),
                    str(payload["prompts_processed"]),
                    str(statistics["token_count"]),
                    str(window["window_size"]),
                    float_cell(window[f"{prefix}_mean_distinct_experts_overall"]),
                    float_cell(window[f"{prefix}_mean_distinct_experts_fraction_overall"]),
                    float_cell(ci95),
                    float_cell(window["random_baseline_distinct_experts"]),
                    float_cell(window[f"{prefix}_observed_to_random_ratio"]),
                ]
            )
        )


def print_cache_table(result_path: Path, payload: dict[str, Any], phase: str, metric: str) -> None:
    statistics = phase_payload(payload, phase)
    hit_key = f"{metric}_weighted_hit_rate"
    print(f"\n[{result_path.name}] phase={phase} cache metric={metric}")
    print("window\tcache_size\tcache_fraction\thit_rate\tci95")
    for window in statistics["windows"]:
        for cache_statistics in window["oracle_cache_hit_rates"]:
            print(
                "\t".join(
                    [
                        str(window["window_size"]),
                        str(cache_statistics["cache_size"]),
                        float_cell(cache_statistics["cache_fraction"]),
                        float_cell(cache_statistics[hit_key]),
                        float_cell(cache_statistics["sequence_weighted_hit_rate_ci95"]),
                    ]
                )
            )


def print_agreement_table(result_path: Path, payload: dict[str, Any], phase: str) -> None:
    ewma_statistics = payload.get("ewma_statistics", [])
    if not ewma_statistics:
        return
    print(f"\n[{result_path.name}] phase={phase} agreement")
    print("alpha\tretained_token\tretained_sequence\tretained_ci95\texact_token\texact_sequence\texact_ci95")
    for ewma in ewma_statistics:
        agreement = ewma[f"{phase}_agreement"]
        print(
            "\t".join(
                [
                    float_cell(ewma["alpha"]),
                    float_cell(agreement["token_weighted_mean_retained_fraction_overall"]),
                    float_cell(agreement["sequence_weighted_mean_retained_fraction_overall"]),
                    float_cell(agreement["sequence_weighted_mean_retained_fraction_overall_ci95"]),
                    float_cell(agreement["token_weighted_exact_match_rate"]),
                    float_cell(agreement["sequence_weighted_exact_match_rate"]),
                    float_cell(agreement["sequence_weighted_exact_match_rate_ci95"]),
                ]
            )
        )


def main() -> None:
    args = parse_args()
    phases = ("prompt", "continuation") if args.phase == "both" else (args.phase,)
    for result_path in args.results:
        payload = json.loads(result_path.read_text())
        for phase in phases:
            if args.section in ("distinct", "both", "all"):
                print_phase_table(result_path, payload, phase, args.metric)
            if args.section in ("cache", "both", "all"):
                print_cache_table(result_path, payload, phase, args.metric)
            if args.section in ("agreement", "all"):
                print_agreement_table(result_path, payload, phase)


if __name__ == "__main__":
    main()
