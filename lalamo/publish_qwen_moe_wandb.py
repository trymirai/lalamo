from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-results", nargs="+", type=Path, required=True)
    parser.add_argument("--nll-results", nargs="*", type=Path, default=[])
    parser.add_argument("--study-summary", type=Path, default=None)
    parser.add_argument("--project", default="qwen-moe-locality")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--job-type", default="analysis")
    parser.add_argument("--tags", default="qwen3.6,moe,locality,ewma")
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dataset_name(payload: dict[str, Any]) -> str:
    dataset = str(payload["config"]["dataset"])
    return Path(dataset).stem or dataset


def alpha_label(alpha: float | str) -> str:
    return str(alpha)


def continuation_locality_chart_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows = [
        [
            dataset,
            "baseline",
            window["window_size"],
            window["sequence_weighted_mean_distinct_experts_overall"],
        ]
        for window in payload["continuation_statistics"]["windows"]
    ]
    rows.extend(
        [
            dataset,
            alpha_label(ewma["alpha"]),
            window["window_size"],
            window["sequence_weighted_mean_distinct_experts_overall"],
        ]
        for ewma in payload["ewma_statistics"]
        for window in ewma["continuation_statistics"]["windows"]
    )
    return rows


def continuation_cache_chart_rows(payload: dict[str, Any], dataset: str, cache_size: int) -> list[list[object]]:
    rows = []

    def append_rows(alpha: float | str, windows: list[dict[str, Any]]) -> None:
        for window in windows:
            for cache in window["oracle_cache_hit_rates"]:
                if int(cache["cache_size"]) != cache_size:
                    continue
                rows.append(
                    [
                        dataset,
                        alpha_label(alpha),
                        window["window_size"],
                        cache["sequence_weighted_hit_rate"],
                    ]
                )

    append_rows("baseline", payload["continuation_statistics"]["windows"])
    for ewma in payload["ewma_statistics"]:
        append_rows(ewma["alpha"], ewma["continuation_statistics"]["windows"])
    return rows


def continuation_transfer_chart_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows = [
        [
            dataset,
            "baseline",
            budget["cache_size"],
            budget["resident_gib_total"],
            budget["sequence_weighted_transfer_bytes_per_token"] / (1024**2),
        ]
        for budget in payload["continuation_statistics"]["resident_budgets"]
    ]
    rows.extend(
        [
            dataset,
            alpha_label(ewma["alpha"]),
            budget["cache_size"],
            budget["resident_gib_total"],
            budget["sequence_weighted_transfer_bytes_per_token"] / (1024**2),
        ]
        for ewma in payload["ewma_statistics"]
        for budget in ewma["continuation_statistics"]["resident_budgets"]
    )
    return rows


def continuation_agreement_chart_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    return [
        [
            dataset,
            float(ewma["alpha"]),
            ewma["continuation_agreement"]["sequence_weighted_mean_retained_fraction_overall"],
            ewma["continuation_agreement"]["sequence_weighted_exact_match_rate"],
        ]
        for ewma in payload["ewma_statistics"]
    ]


def nll_chart_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows = [
        [
            dataset,
            "baseline",
            payload["baseline"]["statistics"]["token_weighted_mean_continuation_nll"],
            payload["baseline"]["statistics"]["token_weighted_continuation_perplexity"],
        ]
    ]
    rows.extend(
        [
            dataset,
            alpha_label(variant["alpha"]),
            variant["statistics"]["token_weighted_mean_continuation_nll"],
            variant["statistics"]["token_weighted_continuation_perplexity"],
        ]
        for variant in payload["ewma_variants"]
    )
    return rows


def quality_transfer_chart_rows(
    routing_payload: dict[str, Any],
    nll_payload: dict[str, Any],
    dataset: str,
) -> list[list[object]]:
    nll_by_alpha = {
        "baseline": nll_payload["baseline"]["statistics"]["token_weighted_mean_continuation_nll"],
        **{
            alpha_label(variant["alpha"]): variant["statistics"]["token_weighted_mean_continuation_nll"]
            for variant in nll_payload["ewma_variants"]
        },
    }
    rows = [
        [
            dataset,
            "baseline",
            budget["cache_size"],
            budget["resident_gib_total"],
            budget["sequence_weighted_transfer_bytes_per_token"] / (1024**2),
            nll_by_alpha["baseline"],
        ]
        for budget in routing_payload["continuation_statistics"]["resident_budgets"]
    ]
    rows.extend(
        [
            dataset,
            alpha_label(ewma["alpha"]),
            budget["cache_size"],
            budget["resident_gib_total"],
            budget["sequence_weighted_transfer_bytes_per_token"] / (1024**2),
            nll_by_alpha[alpha_label(ewma["alpha"])],
        ]
        for ewma in routing_payload["ewma_statistics"]
        for budget in ewma["continuation_statistics"]["resident_budgets"]
    )
    return rows


def phase_locality_rows(
    phase_statistics: dict[str, Any],
    dataset: str,
    phase: str,
    alpha: float | str,
) -> list[list[object]]:
    return [
        [
            dataset,
            phase,
            alpha_label(alpha),
            window["window_size"],
            window["sequence_weighted_mean_distinct_experts_overall"],
            window["sequence_weighted_mean_distinct_experts_overall_ci95"],
            window["random_baseline_distinct_experts"],
            window["sequence_weighted_observed_to_random_ratio"],
        ]
        for window in phase_statistics["windows"]
    ]


def phase_cache_rows(
    phase_statistics: dict[str, Any],
    dataset: str,
    phase: str,
    alpha: float | str,
) -> list[list[object]]:
    return [
        [
            dataset,
            phase,
            alpha_label(alpha),
            window["window_size"],
            cache["cache_size"],
            cache["sequence_weighted_hit_rate"],
            cache["sequence_weighted_hit_rate_ci95"],
        ]
        for window in phase_statistics["windows"]
        for cache in window["oracle_cache_hit_rates"]
    ]


def phase_resident_budget_rows(
    phase_statistics: dict[str, Any],
    dataset: str,
    phase: str,
    alpha: float | str,
) -> list[list[object]]:
    return [
        [
            dataset,
            phase,
            alpha_label(alpha),
            budget["cache_size"],
            budget["resident_gib_total"],
            budget["sequence_weighted_hit_rate"],
            budget["sequence_weighted_hit_rate_ci95"],
            budget["sequence_weighted_expert_loads_per_token"],
            budget["sequence_weighted_expert_loads_per_token_ci95"],
            budget["sequence_weighted_transfer_bytes_per_token"] / (1024**2),
            budget["sequence_weighted_transfer_bytes_per_token_ci95"] / (1024**2),
        ]
        for budget in phase_statistics["resident_budgets"]
    ]


def agreement_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows: list[list[object]] = []
    for ewma in payload["ewma_statistics"]:
        for phase in ("prompt", "continuation"):
            agreement = ewma[f"{phase}_agreement"]
            rows.append(
                [
                    dataset,
                    phase,
                    alpha_label(ewma["alpha"]),
                    agreement["sequence_weighted_mean_retained_fraction_overall"],
                    agreement["sequence_weighted_mean_retained_fraction_overall_ci95"],
                    agreement["sequence_weighted_exact_match_rate"],
                    agreement["sequence_weighted_exact_match_rate_ci95"],
                ]
            )
    return rows


def nll_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    baseline = payload["baseline"]
    rows: list[list[object]] = [
        [
            dataset,
            alpha_label("baseline"),
            baseline["statistics"]["token_weighted_mean_continuation_nll"],
            baseline["statistics"]["token_weighted_continuation_perplexity"],
            baseline["statistics"]["sequence_weighted_mean_continuation_nll"],
            baseline["statistics"]["sequence_weighted_continuation_perplexity"],
            0.0,
            0.0,
        ]
    ]
    rows.extend(
        [
            dataset,
            alpha_label(variant["alpha"]),
            variant["statistics"]["token_weighted_mean_continuation_nll"],
            variant["statistics"]["token_weighted_continuation_perplexity"],
            variant["statistics"]["sequence_weighted_mean_continuation_nll"],
            variant["statistics"]["sequence_weighted_continuation_perplexity"],
            variant["delta_token_weighted_mean_continuation_nll_vs_baseline"],
            variant["delta_sequence_weighted_mean_continuation_nll_vs_baseline"],
        ]
        for variant in payload["ewma_variants"]
    )
    return rows


def study_rows(payload: dict[str, Any]) -> list[list[object]]:
    return [
        [
            alpha_summary["alpha"],
            dataset_summary["dataset"],
            window["window_size"],
            window["baseline_distinct"],
            window["ewma_distinct"],
            window["delta_distinct"],
            window["baseline_cache_hit_rate"],
            window["ewma_cache_hit_rate"],
            window["delta_cache_hit_rate"],
            dataset_summary["retained_fraction"],
            dataset_summary["exact_match_rate"],
            dataset_summary["passes_minimal_intervention_rule"],
        ]
        for alpha_summary in payload["alphas"]
        for dataset_summary in alpha_summary["datasets"]
        for window in dataset_summary["windows"]
    ]


def routing_tables(payloads: list[dict[str, Any]]) -> dict[str, wandb.Table]:
    locality_columns = [
        "dataset",
        "phase",
        "alpha",
        "window_size",
        "distinct_experts",
        "distinct_ci95",
        "random_baseline",
        "observed_to_random_ratio",
    ]
    cache_columns = [
        "dataset",
        "phase",
        "alpha",
        "window_size",
        "cache_size",
        "cache_hit_rate",
        "cache_hit_ci95",
    ]
    agreement_columns = [
        "dataset",
        "phase",
        "alpha",
        "retained_fraction",
        "retained_ci95",
        "exact_match_rate",
        "exact_match_ci95",
    ]
    resident_budget_columns = [
        "dataset",
        "phase",
        "alpha",
        "cache_size",
        "resident_gib_total",
        "hit_rate",
        "hit_rate_ci95",
        "expert_loads_per_token",
        "expert_loads_per_token_ci95",
        "transfer_mib_per_token",
        "transfer_mib_per_token_ci95",
    ]
    locality_data: list[list[object]] = []
    cache_data: list[list[object]] = []
    agreement_data: list[list[object]] = []
    resident_budget_data: list[list[object]] = []
    for payload in payloads:
        dataset = dataset_name(payload)
        for phase in ("prompt", "continuation"):
            phase_statistics = payload[f"{phase}_statistics"]
            locality_data.extend(phase_locality_rows(phase_statistics, dataset, phase, "baseline"))
            cache_data.extend(phase_cache_rows(phase_statistics, dataset, phase, "baseline"))
            resident_budget_data.extend(phase_resident_budget_rows(phase_statistics, dataset, phase, "baseline"))
        agreement_data.extend(agreement_rows(payload, dataset))
        for ewma in payload["ewma_statistics"]:
            for phase in ("prompt", "continuation"):
                phase_statistics = ewma[f"{phase}_statistics"]
                locality_data.extend(phase_locality_rows(phase_statistics, dataset, phase, ewma["alpha"]))
                cache_data.extend(phase_cache_rows(phase_statistics, dataset, phase, ewma["alpha"]))
                resident_budget_data.extend(
                    phase_resident_budget_rows(phase_statistics, dataset, phase, ewma["alpha"])
                )
    return {
        "routing/locality_table": wandb.Table(columns=locality_columns, data=locality_data),
        "routing/cache_table": wandb.Table(columns=cache_columns, data=cache_data),
        "routing/agreement_table": wandb.Table(columns=agreement_columns, data=agreement_data),
        "routing/resident_budget_table": wandb.Table(columns=resident_budget_columns, data=resident_budget_data),
    }


def routing_charts(payloads: list[dict[str, Any]]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        dataset = dataset_name(payload)
        locality_table = wandb.Table(
            columns=["dataset", "alpha", "window_size", "distinct_experts"],
            data=continuation_locality_chart_rows(payload, dataset),
        )
        cache_table = wandb.Table(
            columns=["dataset", "alpha", "window_size", "cache_hit_rate"],
            data=continuation_cache_chart_rows(payload, dataset, cache_size=16),
        )
        transfer_table = wandb.Table(
            columns=["dataset", "alpha", "cache_size", "resident_gib_total", "transfer_mib_per_token"],
            data=continuation_transfer_chart_rows(payload, dataset),
        )
        agreement_table = wandb.Table(
            columns=["dataset", "alpha", "retained_fraction", "exact_match_rate"],
            data=continuation_agreement_chart_rows(payload, dataset),
        )
        charts[f"charts/{dataset}/continuation_distinct_experts"] = wandb.plot.line(
            locality_table,
            x="window_size",
            y="distinct_experts",
            stroke="alpha",
            title=f"{dataset}: continuation distinct experts",
        )
        charts[f"charts/{dataset}/continuation_cache16_hit_rate"] = wandb.plot.line(
            cache_table,
            x="window_size",
            y="cache_hit_rate",
            stroke="alpha",
            title=f"{dataset}: continuation cache-16 hit rate",
        )
        charts[f"charts/{dataset}/continuation_transfer_vs_resident"] = wandb.plot.line(
            transfer_table,
            x="resident_gib_total",
            y="transfer_mib_per_token",
            stroke="alpha",
            title=f"{dataset}: continuation transfer vs resident memory",
        )
        charts[f"charts/{dataset}/continuation_routing_agreement"] = wandb.plot.line_series(
            xs=[agreement_table.get_column("alpha"), agreement_table.get_column("alpha")],
            ys=[
                agreement_table.get_column("retained_fraction"),
                agreement_table.get_column("exact_match_rate"),
            ],
            keys=["retained_fraction", "exact_match_rate"],
            title=f"{dataset}: continuation agreement vs alpha",
            xname="alpha",
        )
    return charts


def nll_table(payloads: list[dict[str, Any]]) -> wandb.Table | None:
    if not payloads:
        return None
    columns = [
        "dataset",
        "alpha",
        "token_nll",
        "token_ppl",
        "sequence_nll",
        "sequence_ppl",
        "delta_token_nll_vs_baseline",
        "delta_sequence_nll_vs_baseline",
    ]
    data: list[list[object]] = []
    for payload in payloads:
        data.extend(nll_rows(payload, dataset_name(payload)))
    return wandb.Table(columns=columns, data=data)


def nll_charts(payloads: list[dict[str, Any]]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        dataset = dataset_name(payload)
        table = wandb.Table(
            columns=["dataset", "variant", "token_nll", "token_ppl"],
            data=nll_chart_rows(payload, dataset),
        )
        charts[f"charts/{dataset}/continuation_token_nll"] = wandb.plot.bar(
            table,
            label="variant",
            value="token_nll",
            title=f"{dataset}: continuation token NLL",
        )
        charts[f"charts/{dataset}/continuation_token_perplexity"] = wandb.plot.bar(
            table,
            label="variant",
            value="token_ppl",
            title=f"{dataset}: continuation token perplexity",
        )
    return charts


def quality_transfer_charts(
    routing_payloads: list[dict[str, Any]],
    nll_payloads: list[dict[str, Any]],
) -> dict[str, object]:
    charts: dict[str, object] = {}
    nll_by_dataset = {dataset_name(payload): payload for payload in nll_payloads}
    for routing_payload in routing_payloads:
        dataset = dataset_name(routing_payload)
        nll_payload = nll_by_dataset.get(dataset)
        if nll_payload is None:
            continue
        tradeoff_rows = quality_transfer_chart_rows(routing_payload, nll_payload, dataset)
        for cache_size in sorted({int(row[2]) for row in tradeoff_rows}):
            tradeoff_table = wandb.Table(
                columns=[
                    "dataset",
                    "alpha",
                    "cache_size",
                    "resident_gib_total",
                    "transfer_mib_per_token",
                    "token_nll",
                ],
                data=[row for row in tradeoff_rows if int(row[2]) == cache_size],
            )
            charts[f"charts/{dataset}/quality_vs_transfer_cache{cache_size}"] = wandb.plot.scatter(
                tradeoff_table,
                x="transfer_mib_per_token",
                y="token_nll",
                title=f"{dataset}: token NLL vs transfer (cache={cache_size})",
            )
    return charts


def study_table(payload: dict[str, Any] | None) -> wandb.Table | None:
    if payload is None:
        return None
    columns = [
        "alpha",
        "dataset",
        "window_size",
        "baseline_distinct",
        "ewma_distinct",
        "delta_distinct",
        "baseline_cache_hit_rate",
        "ewma_cache_hit_rate",
        "delta_cache_hit_rate",
        "retained_fraction",
        "exact_match_rate",
        "passes_rule",
    ]
    return wandb.Table(columns=columns, data=study_rows(payload))


def log_artifact(
    run: object,
    routing_paths: list[Path],
    nll_paths: list[Path],
    study_path: Path | None,
) -> None:
    artifact = wandb.Artifact(name=f"{run.id}-outputs", type="qwen-moe-study")
    for path in routing_paths + nll_paths:
        artifact.add_file(str(path), name=path.name)
    if study_path is not None:
        artifact.add_file(str(study_path), name=study_path.name)
    run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    routing_payloads = [load_json(path) for path in args.routing_results]
    nll_payloads = [load_json(path) for path in args.nll_results]
    study_payload = None if args.study_summary is None else load_json(args.study_summary)
    first = routing_payloads[0]
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=args.run_name,
        job_type=args.job_type,
        tags=[tag for tag in args.tags.split(",") if tag],
        notes=args.notes,
        config={
            "model_repo": first["config"]["model_repo"],
            "seed": first["config"]["seed"],
            "window_sizes": first["config"]["window_sizes"],
            "ewma_alphas": first["config"]["ewma_alphas"],
            "max_rows": first["config"]["max_rows"],
            "max_prompts": first["config"]["max_prompts"],
            "max_prompt_tokens": first["config"]["max_prompt_tokens"],
            "max_continuation_tokens": first["config"]["max_continuation_tokens"],
        },
    )
    assert run is not None
    try:
        for payload in routing_payloads:
            dataset = dataset_name(payload)
            run.summary[f"{dataset}/prompts_processed"] = payload["prompts_processed"]
            run.summary[f"{dataset}/dataset_rows_processed"] = payload["dataset_rows_processed"]
        if study_payload is not None:
            run.summary["study/recommended_alpha"] = study_payload["recommended_alpha"]
        run.log(routing_tables(routing_payloads))
        run.log(routing_charts(routing_payloads))
        quality_table = nll_table(nll_payloads)
        if quality_table is not None:
            run.log({"quality/nll_table": quality_table})
            run.log(nll_charts(nll_payloads))
            run.log(quality_transfer_charts(routing_payloads, nll_payloads))
        selection_table = study_table(study_payload)
        if selection_table is not None:
            run.log({"study/selection_table": selection_table})
        log_artifact(run, args.routing_results, args.nll_results, args.study_summary)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
