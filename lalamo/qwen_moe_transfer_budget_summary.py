from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from lalamo.qwen_moe_ewma_eval import EwmaEvalResult
from lalamo.qwen_moe_payloads import read_payload, variant_label, write_payload
from lalamo.qwen_moe_routing import RoutingAnalysisResult


@dataclass(frozen=True)
class VariantTradeoff:
    alpha: str
    token_nll: float
    delta_token_nll_vs_baseline: float
    transfer_mib_per_token: float
    delta_transfer_mib_per_token_vs_baseline: float
    hit_rate: float


@dataclass(frozen=True)
class CacheBudgetTradeoff:
    cache_size: int
    resident_gib_total: float
    recommended_alpha_without_nll_regression: str
    variants: tuple[VariantTradeoff, ...]


@dataclass(frozen=True)
class DatasetTradeoff:
    dataset: str
    budgets: tuple[CacheBudgetTradeoff, ...]


@dataclass(frozen=True)
class TransferBudgetSummary:
    datasets: tuple[DatasetTradeoff, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-results", nargs="+", type=Path, required=True)
    parser.add_argument("--nll-results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def continuation_transfer_budget_lookup(payload: RoutingAnalysisResult) -> dict[str, dict[int, float]]:
    lookup = {
        "baseline": {
            budget.cache_size: budget.sequence_weighted_transfer_bytes_per_token / (1024**2)
            for budget in payload.continuation_statistics.resident_budgets
        }
    }
    for ewma in payload.ewma_statistics:
        lookup[variant_label(ewma.alpha)] = {
            budget.cache_size: budget.sequence_weighted_transfer_bytes_per_token / (1024**2)
            for budget in ewma.continuation_statistics.resident_budgets
        }
    return lookup


def continuation_hit_rate_lookup(payload: RoutingAnalysisResult) -> dict[str, dict[int, float]]:
    lookup = {
        "baseline": {
            budget.cache_size: budget.sequence_weighted_hit_rate
            for budget in payload.continuation_statistics.resident_budgets
        }
    }
    for ewma in payload.ewma_statistics:
        lookup[variant_label(ewma.alpha)] = {
            budget.cache_size: budget.sequence_weighted_hit_rate
            for budget in ewma.continuation_statistics.resident_budgets
        }
    return lookup


def continuation_resident_gib_lookup(payload: RoutingAnalysisResult) -> dict[int, float]:
    return {
        budget.cache_size: budget.resident_gib_total for budget in payload.continuation_statistics.resident_budgets
    }


def continuation_nll_lookup(payload: EwmaEvalResult) -> dict[str, float]:
    return {
        "baseline": payload.baseline.statistics.token_weighted_mean_continuation_nll,
        **{
            variant_label(variant.alpha): variant.statistics.token_weighted_mean_continuation_nll
            for variant in payload.ewma_variants
        },
    }


def summarize_transfer_budget(
    routing_payloads: list[RoutingAnalysisResult],
    nll_payloads: list[EwmaEvalResult],
) -> TransferBudgetSummary:
    routing_by_dataset: dict[str, RoutingAnalysisResult] = {}
    nll_by_dataset: dict[str, EwmaEvalResult] = {}
    for payload in routing_payloads:
        dataset = payload.config.dataset
        if dataset in routing_by_dataset:
            raise ValueError(f"Duplicate routing payload for {dataset}.")
        routing_by_dataset[dataset] = payload
    for payload in nll_payloads:
        dataset = payload.config.dataset
        if dataset in nll_by_dataset:
            raise ValueError(f"Duplicate NLL payload for {dataset}.")
        nll_by_dataset[dataset] = payload
    if set(routing_by_dataset) != set(nll_by_dataset):
        raise ValueError(
            "Expected routing and NLL payloads for the same datasets, got "
            f"{sorted(routing_by_dataset)} and {sorted(nll_by_dataset)}."
        )
    datasets = []
    for dataset in sorted(routing_by_dataset):
        routing_payload = routing_by_dataset[dataset]
        nll_payload = nll_by_dataset[dataset]
        transfer_lookup = continuation_transfer_budget_lookup(routing_payload)
        hit_rate_lookup = continuation_hit_rate_lookup(routing_payload)
        resident_gib_lookup = continuation_resident_gib_lookup(routing_payload)
        nll_lookup = continuation_nll_lookup(nll_payload)
        baseline_nll = nll_lookup["baseline"]
        budgets = []
        for cache_size in sorted(resident_gib_lookup):
            baseline_transfer = transfer_lookup["baseline"][cache_size]
            variants = tuple(
                VariantTradeoff(
                    alpha=alpha,
                    token_nll=nll_lookup[alpha],
                    delta_token_nll_vs_baseline=nll_lookup[alpha] - baseline_nll,
                    transfer_mib_per_token=transfer_lookup[alpha][cache_size],
                    delta_transfer_mib_per_token_vs_baseline=transfer_lookup[alpha][cache_size] - baseline_transfer,
                    hit_rate=hit_rate_lookup[alpha][cache_size],
                )
                for alpha in ("baseline", *sorted(alpha for alpha in transfer_lookup if alpha != "baseline"))
            )
            non_regressing = [variant for variant in variants if variant.token_nll <= baseline_nll]
            budgets.append(
                CacheBudgetTradeoff(
                    cache_size=cache_size,
                    resident_gib_total=resident_gib_lookup[cache_size],
                    recommended_alpha_without_nll_regression=min(
                        non_regressing,
                        key=lambda variant: variant.transfer_mib_per_token,
                    ).alpha,
                    variants=variants,
                )
            )
        datasets.append(DatasetTradeoff(dataset=dataset, budgets=tuple(budgets)))
    return TransferBudgetSummary(datasets=tuple(datasets))


def main() -> None:
    args = parse_args()
    write_payload(
        args.output,
        summarize_transfer_budget(
            routing_payloads=[read_payload(path, RoutingAnalysisResult) for path in args.routing_results],
            nll_payloads=[read_payload(path, EwmaEvalResult) for path in args.nll_results],
        ),
    )


if __name__ == "__main__":
    main()
