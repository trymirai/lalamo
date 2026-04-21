from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from lalamo.qwen_moe_ewma_eval import EwmaEvalResult
from lalamo.qwen_moe_ewma_study_summary import StudySummary
from lalamo.qwen_moe_payloads import read_payload
from lalamo.qwen_moe_routing import RoutingAnalysisResult
from lalamo.qwen_moe_rows import (
    continuation_agreement_chart_rows as _continuation_agreement_chart_rows,
)
from lalamo.qwen_moe_rows import (
    continuation_cache_chart_rows as _continuation_cache_chart_rows,
)
from lalamo.qwen_moe_rows import (
    continuation_locality_chart_rows as _continuation_locality_chart_rows,
)
from lalamo.qwen_moe_rows import (
    continuation_transfer_chart_rows as _continuation_transfer_chart_rows,
)
from lalamo.qwen_moe_rows import (
    nll_rows as _nll_rows,
)
from lalamo.qwen_moe_rows import (
    quality_transfer_rows as _quality_transfer_rows,
)
from lalamo.qwen_moe_rows import (
    routing_rows as _routing_rows,
)
from lalamo.qwen_moe_rows import (
    study_rows as _study_rows,
)


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


def routing_tables(payloads: list[RoutingAnalysisResult]) -> dict[str, wandb.Table]:
    locality_rows = []
    cache_rows = []
    resident_budget_rows = []
    agreement_rows = []
    for payload in payloads:
        payload_locality, payload_cache, payload_resident_budgets, payload_agreement = _routing_rows(payload)
        locality_rows.extend(payload_locality)
        cache_rows.extend(payload_cache)
        resident_budget_rows.extend(payload_resident_budgets)
        agreement_rows.extend(payload_agreement)
    return {
        "routing/locality_table": wandb.Table(
            columns=[
                "dataset",
                "phase",
                "alpha",
                "window_size",
                "distinct_experts",
                "distinct_ci95",
                "random_baseline",
                "observed_to_random_ratio",
            ],
            data=[
                [
                    row.dataset,
                    row.phase,
                    row.alpha,
                    row.window_size,
                    row.distinct_experts,
                    row.distinct_ci95,
                    row.random_baseline,
                    row.observed_to_random_ratio,
                ]
                for row in locality_rows
            ],
        ),
        "routing/cache_table": wandb.Table(
            columns=["dataset", "phase", "alpha", "window_size", "cache_size", "cache_hit_rate", "cache_hit_ci95"],
            data=[
                [
                    row.dataset,
                    row.phase,
                    row.alpha,
                    row.window_size,
                    row.cache_size,
                    row.cache_hit_rate,
                    row.cache_hit_ci95,
                ]
                for row in cache_rows
            ],
        ),
        "routing/agreement_table": wandb.Table(
            columns=[
                "dataset",
                "phase",
                "alpha",
                "retained_fraction",
                "retained_ci95",
                "exact_match_rate",
                "exact_match_ci95",
            ],
            data=[
                [
                    row.dataset,
                    row.phase,
                    row.alpha,
                    row.retained_fraction,
                    row.retained_ci95,
                    row.exact_match_rate,
                    row.exact_match_ci95,
                ]
                for row in agreement_rows
            ],
        ),
        "routing/resident_budget_table": wandb.Table(
            columns=[
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
            ],
            data=[
                [
                    row.dataset,
                    row.phase,
                    row.alpha,
                    row.cache_size,
                    row.resident_gib_total,
                    row.hit_rate,
                    row.hit_rate_ci95,
                    row.expert_loads_per_token,
                    row.expert_loads_per_token_ci95,
                    row.transfer_mib_per_token,
                    row.transfer_mib_per_token_ci95,
                ]
                for row in resident_budget_rows
            ],
        ),
    }


def routing_charts(payloads: list[RoutingAnalysisResult]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        locality_rows = _continuation_locality_chart_rows(payload)
        cache_rows = _continuation_cache_chart_rows(payload, cache_size=16)
        transfer_rows = _continuation_transfer_chart_rows(payload)
        agreement_rows = _continuation_agreement_chart_rows(payload)
        dataset = locality_rows[0].dataset
        locality_table = wandb.Table(
            columns=["dataset", "alpha", "window_size", "distinct_experts"],
            data=[[row.dataset, row.alpha, row.window_size, row.distinct_experts] for row in locality_rows],
        )
        cache_table = wandb.Table(
            columns=["dataset", "alpha", "window_size", "cache_hit_rate"],
            data=[[row.dataset, row.alpha, row.window_size, row.cache_hit_rate] for row in cache_rows],
        )
        transfer_table = wandb.Table(
            columns=["dataset", "alpha", "cache_size", "resident_gib_total", "transfer_mib_per_token"],
            data=[
                [row.dataset, row.alpha, row.cache_size, row.resident_gib_total, row.transfer_mib_per_token]
                for row in transfer_rows
            ],
        )
        agreement_table = wandb.Table(
            columns=["dataset", "alpha", "retained_fraction", "exact_match_rate"],
            data=[
                [row.dataset, float(row.alpha), row.retained_fraction, row.exact_match_rate] for row in agreement_rows
            ],
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
            ys=[agreement_table.get_column("retained_fraction"), agreement_table.get_column("exact_match_rate")],
            keys=["retained_fraction", "exact_match_rate"],
            title=f"{dataset}: continuation agreement vs alpha",
            xname="alpha",
        )
    return charts


def nll_table(payloads: list[EwmaEvalResult]) -> wandb.Table | None:
    if not payloads:
        return None
    rows = [row for payload in payloads for row in _nll_rows(payload)]
    return wandb.Table(
        columns=[
            "dataset",
            "alpha",
            "token_nll",
            "token_ppl",
            "sequence_nll",
            "sequence_ppl",
            "delta_token_nll_vs_baseline",
            "delta_sequence_nll_vs_baseline",
        ],
        data=[
            [
                row.dataset,
                row.alpha,
                row.token_nll,
                row.token_ppl,
                row.sequence_nll,
                row.sequence_ppl,
                row.delta_token_nll_vs_baseline,
                row.delta_sequence_nll_vs_baseline,
            ]
            for row in rows
        ],
    )


def nll_charts(payloads: list[EwmaEvalResult]) -> dict[str, object]:
    charts: dict[str, object] = {}
    for payload in payloads:
        rows = _nll_rows(payload)
        dataset = rows[0].dataset
        table = wandb.Table(
            columns=["dataset", "variant", "token_nll", "token_ppl"],
            data=[[row.dataset, row.alpha, row.token_nll, row.token_ppl] for row in rows],
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
    routing_payloads: list[RoutingAnalysisResult],
    nll_payloads: list[EwmaEvalResult],
) -> dict[str, object]:
    nll_by_dataset = {payload.config.dataset: payload for payload in nll_payloads}
    if {payload.config.dataset for payload in routing_payloads} != set(nll_by_dataset):
        raise ValueError("Routing and NLL datasets must match exactly for quality/transfer charts.")
    charts: dict[str, object] = {}
    for routing_payload in routing_payloads:
        tradeoff_rows = _quality_transfer_rows(routing_payload, nll_by_dataset[routing_payload.config.dataset])
        dataset = tradeoff_rows[0].dataset
        for cache_size in sorted({row.cache_size for row in tradeoff_rows}):
            tradeoff_table = wandb.Table(
                columns=[
                    "dataset",
                    "alpha",
                    "cache_size",
                    "resident_gib_total",
                    "transfer_mib_per_token",
                    "token_nll",
                ],
                data=[
                    [
                        row.dataset,
                        row.alpha,
                        row.cache_size,
                        row.resident_gib_total,
                        row.transfer_mib_per_token,
                        row.token_nll,
                    ]
                    for row in tradeoff_rows
                    if row.cache_size == cache_size
                ],
            )
            charts[f"charts/{dataset}/quality_vs_transfer_cache{cache_size}"] = wandb.plot.scatter(
                tradeoff_table,
                x="transfer_mib_per_token",
                y="token_nll",
                title=f"{dataset}: token NLL vs transfer (cache={cache_size})",
            )
    return charts


def study_table(payload: StudySummary | None) -> wandb.Table | None:
    if payload is None:
        return None
    rows = _study_rows(payload)
    return wandb.Table(
        columns=[
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
        ],
        data=[
            [
                row.alpha,
                row.dataset,
                row.window_size,
                row.baseline_distinct,
                row.ewma_distinct,
                row.delta_distinct,
                row.baseline_cache_hit_rate,
                row.ewma_cache_hit_rate,
                row.delta_cache_hit_rate,
                row.retained_fraction,
                row.exact_match_rate,
                row.passes_rule,
            ]
            for row in rows
        ],
    )


def log_artifact(run: object, routing_paths: list[Path], nll_paths: list[Path], study_path: Path | None) -> None:
    artifact = wandb.Artifact(name=f"{run.id}-outputs", type="qwen-moe-study")
    for path in routing_paths + nll_paths:
        artifact.add_file(str(path), name=path.name)
    if study_path is not None:
        artifact.add_file(str(study_path), name=study_path.name)
    run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    routing_payloads = [read_payload(path, RoutingAnalysisResult) for path in args.routing_results]
    nll_payloads = [read_payload(path, EwmaEvalResult) for path in args.nll_results]
    study_payload = None if args.study_summary is None else read_payload(args.study_summary, StudySummary)
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
            "model_repo": first.config.model_repo,
            "seed": first.config.seed,
            "window_sizes": first.config.window_sizes,
            "ewma_alphas": first.config.ewma_alphas,
            "max_rows": first.config.max_rows,
            "max_prompts": first.config.max_prompts,
            "max_prompt_tokens": first.config.max_prompt_tokens,
            "max_continuation_tokens": first.config.max_continuation_tokens,
        },
    )
    assert run is not None
    try:
        for payload in routing_payloads:
            dataset = payload.config.dataset
            run.summary[f"{dataset}/prompts_processed"] = payload.prompts_processed
            run.summary[f"{dataset}/dataset_rows_processed"] = payload.dataset_rows_processed
        if study_payload is not None:
            run.summary["study/recommended_alpha"] = study_payload.recommended_alpha
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
