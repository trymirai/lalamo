import json
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from lalamo.distill_runner import DistillConfig, DistillResult, OptimizerName, TrainingMode, distill
from lalamo.modules.common import ParameterLeafInfo, ParameterNorm


class TrainableSubset(StrEnum):
    ALL = "all"
    QUANTIZED_ONLY = "quantized_only"
    QUANTIZED_AND_L_INF = "quantized_and_l_inf"


def _trainable_filter(subset: TrainableSubset) -> Callable[[ParameterLeafInfo], bool]:
    match subset:
        case TrainableSubset.ALL:
            return lambda _info: True
        case TrainableSubset.QUANTIZED_ONLY:
            return lambda info: info.quantized
        case TrainableSubset.QUANTIZED_AND_L_INF:
            return lambda info: info.quantized or info.norm == ParameterNorm.L_INF


def _run(name: str, config: DistillConfig, subset: TrainableSubset) -> DistillResult:
    print(f"\n--- {name} ---")
    result = distill(config, trainable_filter=_trainable_filter(subset))
    print(
        f"KL {result.initial_eval.kl_divergence:.4f} -> {result.final_eval.kl_divergence:.4f}  "
        f"top1 {result.initial_eval.top1_agreement:.4f} -> {result.final_eval.top1_agreement:.4f}  "
        f"sec/step {result.seconds_per_step:.2f}",
    )
    return result


def _result_row(name: str, subset: TrainableSubset, result: DistillResult) -> dict[str, object]:
    return {
        "name": name,
        "trainable_subset": subset.value,
        "initial_kl": result.initial_eval.kl_divergence,
        "final_kl": result.final_eval.kl_divergence,
        "initial_top1": result.initial_eval.top1_agreement,
        "final_top1": result.final_eval.top1_agreement,
        "compilation_seconds": result.compilation_seconds,
        "seconds_per_step": result.seconds_per_step,
        "completed_steps": result.completed_steps,
    }


def main(
    teacher_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    student_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    dataset_path: Annotated[Path, typer.Option(exists=True, dir_okay=False, file_okay=True)],
    output_dir: Annotated[Path, typer.Option()],
    train_examples: Annotated[int, typer.Option()] = 64,
    eval_examples: Annotated[int, typer.Option()] = 32,
    max_sequence_length: Annotated[int, typer.Option()] = 128,
    batch_size: Annotated[int, typer.Option()] = 4,
    num_steps: Annotated[int, typer.Option()] = 25,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    base_config = DistillConfig(
        teacher_path=teacher_path,
        student_path=student_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        training_mode=TrainingMode.ONLINE_EXACT,
        train_examples=train_examples,
        eval_examples=eval_examples,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        num_steps=num_steps,
        learning_rate=3e-6,
        optimizer_name=OptimizerName.ADAMW,
        seed=seed,
        stochastic_rounding=True,
        save_checkpoints=False,
    )

    print("\n=== Optimizer / LR ===")
    best_optimizer = OptimizerName.ADAMW
    best_lr = 3e-6
    best_kl = float("inf")
    for optimizer_name in (OptimizerName.ADAMW, OptimizerName.MUON):
        for learning_rate in (1e-7, 3e-7, 1e-6, 3e-6):
            name = f"{optimizer_name.value}_lr{learning_rate}"
            result = _run(
                name,
                replace(
                    base_config,
                    output_dir=output_dir / name,
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                ),
                TrainableSubset.ALL,
            )
            results.append(_result_row(name, TrainableSubset.ALL, result))
            if result.final_eval.kl_divergence < best_kl:
                best_optimizer = optimizer_name
                best_lr = learning_rate
                best_kl = result.final_eval.kl_divergence

    print(f"\nBest optimizer/lr: {best_optimizer.value} @ {best_lr}")

    print("\n=== Trainable subset ===")
    best_subset = TrainableSubset.ALL
    best_kl = float("inf")
    for subset in TrainableSubset:
        name = f"subset_{subset.value}"
        result = _run(
            name,
            replace(
                base_config,
                output_dir=output_dir / name,
                optimizer_name=best_optimizer,
                learning_rate=best_lr,
            ),
            subset,
        )
        results.append(_result_row(name, subset, result))
        if result.final_eval.kl_divergence < best_kl:
            best_subset = subset
            best_kl = result.final_eval.kl_divergence

    print(f"\nBest trainable subset: {best_subset.value}")

    print("\n=== Stochastic rounding ===")
    best_stochastic = True
    best_kl = float("inf")
    for stochastic_rounding in (True, False):
        name = f"stochastic_{stochastic_rounding}"
        result = _run(
            name,
            replace(
                base_config,
                output_dir=output_dir / name,
                optimizer_name=best_optimizer,
                learning_rate=best_lr,
                stochastic_rounding=stochastic_rounding,
            ),
            best_subset,
        )
        results.append(_result_row(name, best_subset, result))
        if result.final_eval.kl_divergence < best_kl:
            best_stochastic = stochastic_rounding
            best_kl = result.final_eval.kl_divergence

    results_path = output_dir / "sweep_results.json"
    results_path.write_text(json.dumps(results, indent=4))

    print("\n=== Done ===")
    print(
        f"Best recipe: {best_optimizer.value} lr={best_lr} subset={best_subset.value} "
        f"stochastic={best_stochastic}",
    )
    print(f"Results: {results_path}")


if __name__ == "__main__":
    typer.run(main)
