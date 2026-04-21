from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from lalamo.qwen_moe_offload_eval import OffloadEvalResult
from lalamo.qwen_moe_payloads import read_payload
from lalamo.qwen_moe_rows import offload_budget_summary_rows, offload_variant_rows


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


def offload_table(payloads: list[OffloadEvalResult]) -> wandb.Table:
    rows = [row for payload in payloads for row in offload_variant_rows(payload)]
    return wandb.Table(
        columns=[
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
        ],
        data=[
            [
                row.dataset,
                row.cache_size,
                row.resident_gib_total,
                row.variant,
                row.alpha,
                row.token_nll,
                row.token_ppl,
                row.sequence_nll,
                row.sequence_ppl,
                row.delta_token_nll_vs_baseline,
                row.delta_sequence_nll_vs_baseline,
                row.transfer_mib_per_token,
                row.loads_per_token,
                row.hit_rate,
                row.continuation_tokens_per_second,
                row.prompts_processed,
            ]
            for row in rows
        ],
    )


def budget_summary_table(payloads: list[OffloadEvalResult]) -> wandb.Table:
    rows = [row for payload in payloads for row in offload_budget_summary_rows(payload)]
    return wandb.Table(
        columns=[
            "dataset",
            "cache_size",
            "resident_gib_total",
            "best_transfer_variant",
            "best_transfer_mib_per_token",
            "best_nll_variant",
            "best_token_nll",
            "best_tps_variant",
            "best_continuation_tokens_per_second",
        ],
        data=[
            [
                row.dataset,
                row.cache_size,
                row.resident_gib_total,
                row.best_transfer_variant,
                row.best_transfer_mib_per_token,
                row.best_nll_variant,
                row.best_token_nll,
                row.best_tps_variant,
                row.best_continuation_tokens_per_second,
            ]
            for row in rows
        ],
    )


def charts(payloads: list[OffloadEvalResult]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        rows = offload_variant_rows(payload)
        for cache_size in sorted({row.cache_size for row in rows}):
            cache_rows = [row for row in rows if row.cache_size == cache_size]
            transfer_nll_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "token_nll"],
                data=[[row.dataset, row.variant, row.transfer_mib_per_token, row.token_nll] for row in cache_rows],
            )
            transfer_tps_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "continuation_tokens_per_second"],
                data=[
                    [row.dataset, row.variant, row.transfer_mib_per_token, row.continuation_tokens_per_second]
                    for row in cache_rows
                ],
            )
            transfer_hit_rate_table = wandb.Table(
                columns=["dataset", "variant", "transfer_mib_per_token", "hit_rate"],
                data=[[row.dataset, row.variant, row.transfer_mib_per_token, row.hit_rate] for row in cache_rows],
            )
            dataset = cache_rows[0].dataset
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
                transfer_hit_rate_table,
                x="transfer_mib_per_token",
                y="hit_rate",
                title=f"{dataset}: hit rate vs transfer (cache={cache_size})",
            )
        resident_transfer_table = wandb.Table(
            columns=["dataset", "variant", "resident_gib_total", "transfer_mib_per_token"],
            data=[[row.dataset, row.variant, row.resident_gib_total, row.transfer_mib_per_token] for row in rows],
        )
        resident_tps_table = wandb.Table(
            columns=["dataset", "variant", "resident_gib_total", "continuation_tokens_per_second"],
            data=[
                [row.dataset, row.variant, row.resident_gib_total, row.continuation_tokens_per_second] for row in rows
            ],
        )
        dataset = rows[0].dataset
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
    payloads = [read_payload(path, OffloadEvalResult) for path in args.offload_results]
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
            "model_repo": first.config.model_repo,
            "seed": first.config.seed,
            "cache_sizes": first.config.cache_sizes,
            "ewma_alphas": first.config.ewma_alphas,
            "max_rows": first.config.max_rows,
            "max_prompts": first.config.max_prompts,
            "max_prompt_tokens": first.config.max_prompt_tokens,
            "max_continuation_tokens": first.config.max_continuation_tokens,
            "device_map_mode": first.config.device_map_mode,
        },
    )
    assert run is not None
    try:
        for payload in payloads:
            rows = offload_variant_rows(payload)
            dataset = rows[0].dataset
            run.summary[f"{dataset}/dataset_rows_total"] = payload.dataset_rows_total
            run.summary[f"{dataset}/assistant_turns_total"] = payload.assistant_turns_total
            for budget in payload.cache_budgets:
                run.summary[f"{dataset}/cache{budget.cache_size}/baseline_transfer_mib_per_token"] = (
                    budget.baseline.transfer.continuation_transfer_mib_per_token
                )
                run.summary[f"{dataset}/cache{budget.cache_size}/baseline_token_nll"] = (
                    budget.baseline.statistics.token_weighted_mean_continuation_nll
                )
        run.log({"offload/variant_table": offload_table(payloads)})
        run.log({"offload/budget_summary_table": budget_summary_table(payloads)})
        run.log(charts(payloads))
        log_artifact(run, args.offload_results)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
