from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-results", nargs="+", type=Path, required=True)
    parser.add_argument("--project", default="qwen-moe-locality")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--job-type", default="runtime-offload")
    parser.add_argument("--tags", default="qwen3.6,moe,offload,ewma,runtime")
    parser.add_argument("--notes", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dataset_name(payload: dict[str, Any]) -> str:
    dataset = str(payload["config"]["dataset"])
    return Path(dataset).stem or dataset


def alpha_label(alpha: float | str) -> str:
    return str(alpha)


def cache_budget_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows: list[list[object]] = []
    for budget in payload["cache_budgets"]:
        baseline = budget["baseline"]
        rows.append(_variant_row(dataset, budget, baseline, "baseline"))
        rows.extend(
            _variant_row(dataset, budget, variant, alpha_label(variant["alpha"]))
            for variant in budget["ewma_variants"]
        )
    return rows


def _variant_row(dataset: str, budget: dict[str, Any], variant: dict[str, Any], variant_label: str) -> list[object]:
    transfer = variant["transfer"]
    stats = variant["statistics"]
    return [
        dataset,
        budget["cache_size"],
        budget["resident_gib_total"],
        variant_label,
        variant["alpha"],
        stats["token_weighted_mean_continuation_nll"],
        stats["token_weighted_continuation_perplexity"],
        stats["sequence_weighted_mean_continuation_nll"],
        stats["sequence_weighted_continuation_perplexity"],
        variant["delta_token_weighted_mean_continuation_nll_vs_baseline"],
        variant["delta_sequence_weighted_mean_continuation_nll_vs_baseline"],
        transfer["continuation_transfer_mib_per_token"],
        transfer["continuation_expert_loads_per_token"],
        transfer["continuation_hit_rate"],
        transfer["continuation_tokens_per_second"],
        budget["prompts_processed"],
    ]


def offload_table(payloads: list[dict[str, Any]]) -> wandb.Table:
    columns = [
        "dataset",
        "cache_size",
        "resident_gib_total",
        "variant",
        "alpha",
        "token_nll",
        "token_ppl",
        "sequence_nll",
        "sequence_ppl",
        "delta_token_nll_vs_baseline",
        "delta_sequence_nll_vs_baseline",
        "transfer_mib_per_token",
        "loads_per_token",
        "hit_rate",
        "continuation_tokens_per_second",
        "prompts_processed",
    ]
    data: list[list[object]] = []
    for payload in payloads:
        data.extend(cache_budget_rows(payload, dataset_name(payload)))
    return wandb.Table(columns=columns, data=data)


def budget_summary_rows(payload: dict[str, Any], dataset: str) -> list[list[object]]:
    rows: list[list[object]] = []
    for budget in payload["cache_budgets"]:
        variants = [budget["baseline"], *budget["ewma_variants"]]
        best_transfer = min(variants, key=lambda variant: variant["transfer"]["continuation_transfer_mib_per_token"])
        best_nll = min(variants, key=lambda variant: variant["statistics"]["token_weighted_mean_continuation_nll"])
        best_tps = max(variants, key=lambda variant: variant["transfer"]["continuation_tokens_per_second"])
        rows.append(
            [
                dataset,
                budget["cache_size"],
                budget["resident_gib_total"],
                alpha_label(best_transfer["alpha"] if best_transfer["alpha"] != 0.0 else "baseline"),
                best_transfer["transfer"]["continuation_transfer_mib_per_token"],
                alpha_label(best_nll["alpha"] if best_nll["alpha"] != 0.0 else "baseline"),
                best_nll["statistics"]["token_weighted_mean_continuation_nll"],
                alpha_label(best_tps["alpha"] if best_tps["alpha"] != 0.0 else "baseline"),
                best_tps["transfer"]["continuation_tokens_per_second"],
            ]
        )
    return rows


def budget_summary_table(payloads: list[dict[str, Any]]) -> wandb.Table:
    columns = [
        "dataset",
        "cache_size",
        "resident_gib_total",
        "best_transfer_variant",
        "best_transfer_mib_per_token",
        "best_nll_variant",
        "best_token_nll",
        "best_tps_variant",
        "best_continuation_tokens_per_second",
    ]
    data: list[list[object]] = []
    for payload in payloads:
        data.extend(budget_summary_rows(payload, dataset_name(payload)))
    return wandb.Table(columns=columns, data=data)


def charts(payloads: list[dict[str, Any]]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        dataset = dataset_name(payload)
        rows = cache_budget_rows(payload, dataset)
        for cache_size in sorted({int(row[1]) for row in rows}):
            cache_rows = [row for row in rows if int(row[1]) == cache_size]
            transfer_nll_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "token_nll"],
                data=[[row[0], row[3], row[11], row[5]] for row in cache_rows],
            )
            transfer_tps_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "continuation_tokens_per_second"],
                data=[[row[0], row[3], row[11], row[14]] for row in cache_rows],
            )
            variant_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "hit_rate", "loads_per_token"],
                data=[[row[0], row[3], row[11], row[13], row[12]] for row in cache_rows],
            )
            charts[f"charts/{dataset}/cache{cache_size}/nll_vs_transfer"] = wandb.plot.scatter(
                transfer_nll_table,
                x="transfer_mib_per_token",
                y="token_nll",
                title=f"{dataset}: token NLL vs transfer (cache={cache_size})",
            )
            charts[f"charts/{dataset}/cache{cache_size}/throughput_vs_transfer"] = wandb.plot.scatter(
                transfer_tps_table,
                x="transfer_mib_per_token",
                y="continuation_tokens_per_second",
                title=f"{dataset}: throughput vs transfer (cache={cache_size})",
            )
            charts[f"charts/{dataset}/cache{cache_size}/hit_rate_vs_transfer"] = wandb.plot.scatter(
                variant_table,
                x="transfer_mib_per_token",
                y="hit_rate",
                title=f"{dataset}: hit rate vs transfer (cache={cache_size})",
            )
        resident_transfer_table = wandb.Table(
            columns=["dataset", "variant", "resident_gib_total", "transfer_mib_per_token"],
            data=[[row[0], row[3], row[2], row[11]] for row in rows],
        )
        resident_tps_table = wandb.Table(
            columns=["dataset", "variant", "resident_gib_total", "continuation_tokens_per_second"],
            data=[[row[0], row[3], row[2], row[14]] for row in rows],
        )
        charts[f"charts/{dataset}/transfer_vs_resident"] = wandb.plot.line(
            resident_transfer_table,
            x="resident_gib_total",
            y="transfer_mib_per_token",
            stroke="variant",
            title=f"{dataset}: transfer vs resident memory",
        )
        charts[f"charts/{dataset}/throughput_vs_resident"] = wandb.plot.line(
            resident_tps_table,
            x="resident_gib_total",
            y="continuation_tokens_per_second",
            stroke="variant",
            title=f"{dataset}: throughput vs resident memory",
        )
    return charts


def log_artifact(run: object, result_paths: list[Path]) -> None:
    artifact = wandb.Artifact(name=f"{run.id}-outputs", type="qwen-moe-offload-runtime")
    for path in result_paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    payloads = [load_json(path) for path in args.offload_results]
    first = payloads[0]
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
            "cache_sizes": first["config"]["cache_sizes"],
            "ewma_alphas": first["config"]["ewma_alphas"],
            "max_rows": first["config"]["max_rows"],
            "max_prompts": first["config"]["max_prompts"],
            "max_prompt_tokens": first["config"]["max_prompt_tokens"],
            "max_continuation_tokens": first["config"]["max_continuation_tokens"],
            "device_map_mode": first["config"]["device_map_mode"],
        },
    )
    assert run is not None
    try:
        for payload in payloads:
            dataset = dataset_name(payload)
            run.summary[f"{dataset}/dataset_rows_total"] = payload["dataset_rows_total"]
            run.summary[f"{dataset}/assistant_turns_total"] = payload["assistant_turns_total"]
            for budget in payload["cache_budgets"]:
                baseline = budget["baseline"]
                run.summary[f"{dataset}/cache{budget['cache_size']}/baseline_transfer_mib_per_token"] = (
                    baseline["transfer"]["continuation_transfer_mib_per_token"]
                )
                run.summary[f"{dataset}/cache{budget['cache_size']}/baseline_token_nll"] = (
                    baseline["statistics"]["token_weighted_mean_continuation_nll"]
                )
        run.log({"offload/variant_table": offload_table(payloads)})
        run.log({"offload/budget_summary_table": budget_summary_table(payloads)})
        run.log(charts(payloads))
        log_artifact(run, args.offload_results)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
