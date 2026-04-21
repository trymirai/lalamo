from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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
    recommended_alpha_without_nll_regression: str | None
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dataset_name(payload: dict[str, Any]) -> str:
    dataset = str(payload["config"]["dataset"])
    return Path(dataset).stem or dataset


def alpha_label(alpha: float | str) -> str:
    return str(alpha)


def continuation_transfer_budget_lookup(payload: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    lookup = {
        "baseline": {
            int(budget["cache_size"]): budget for budget in payload["continuation_statistics"]["resident_budgets"]
        }
    }
    lookup.update(
        {
            alpha_label(ewma["alpha"]): {
                int(budget["cache_size"]): budget for budget in ewma["continuation_statistics"]["resident_budgets"]
            }
            for ewma in payload["ewma_statistics"]
        }
    )
    return lookup


def continuation_nll_lookup(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "baseline": float(payload["baseline"]["statistics"]["token_weighted_mean_continuation_nll"]),
        **{
            alpha_label(variant["alpha"]): float(variant["statistics"]["token_weighted_mean_continuation_nll"])
            for variant in payload["ewma_variants"]
        },
    }


def summarize_transfer_budget(
    routing_payloads: list[dict[str, Any]],
    nll_payloads: list[dict[str, Any]],
) -> TransferBudgetSummary:
    routing_by_dataset = {dataset_name(payload): payload for payload in routing_payloads}
    nll_by_dataset = {dataset_name(payload): payload for payload in nll_payloads}
    if set(routing_by_dataset) != set(nll_by_dataset):
        raise ValueError(
            f"Expected routing and NLL payloads for the same datasets, got {sorted(routing_by_dataset)} and "
            f"{sorted(nll_by_dataset)}."
        )
    datasets: list[DatasetTradeoff] = []
    for dataset in sorted(routing_by_dataset):
        routing_payload = routing_by_dataset[dataset]
        nll_payload = nll_by_dataset[dataset]
        budget_lookup = continuation_transfer_budget_lookup(routing_payload)
        nll_lookup = continuation_nll_lookup(nll_payload)
        baseline_nll = nll_lookup["baseline"]
        baseline_budgets = budget_lookup["baseline"]
        budgets: list[CacheBudgetTradeoff] = []
        for cache_size in sorted(baseline_budgets):
            baseline_budget = baseline_budgets[cache_size]
            baseline_transfer = float(baseline_budget["sequence_weighted_transfer_bytes_per_token"]) / (1024**2)
            variants = [
                VariantTradeoff(
                    alpha=alpha,
                    token_nll=nll_lookup[alpha],
                    delta_token_nll_vs_baseline=nll_lookup[alpha] - baseline_nll,
                    transfer_mib_per_token=float(budget_lookup[alpha][cache_size]["sequence_weighted_transfer_bytes_per_token"])
                    / (1024**2),
                    delta_transfer_mib_per_token_vs_baseline=(
                        float(budget_lookup[alpha][cache_size]["sequence_weighted_transfer_bytes_per_token"])
                        / (1024**2)
                        - baseline_transfer
                    ),
                    hit_rate=float(budget_lookup[alpha][cache_size]["sequence_weighted_hit_rate"]),
                )
                for alpha in ("baseline", *sorted(alpha for alpha in budget_lookup if alpha != "baseline"))
            ]
            non_regressing = [variant for variant in variants if variant.token_nll <= baseline_nll]
            recommended_alpha = min(non_regressing, key=lambda variant: variant.transfer_mib_per_token).alpha
            budgets.append(
                CacheBudgetTradeoff(
                    cache_size=cache_size,
                    resident_gib_total=float(baseline_budget["resident_gib_total"]),
                    recommended_alpha_without_nll_regression=recommended_alpha,
                    variants=tuple(variants),
                )
            )
        datasets.append(DatasetTradeoff(dataset=dataset, budgets=tuple(budgets)))
    return TransferBudgetSummary(datasets=tuple(datasets))


def main() -> None:
    args = parse_args()
    summary = summarize_transfer_budget(
        routing_payloads=[load_json(path) for path in args.routing_results],
        nll_payloads=[load_json(path) for path in args.nll_results],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
