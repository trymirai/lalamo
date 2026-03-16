import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from lalamo.modules.common import ParameterNorm

from .run_distillation_experiment import (
    ExperimentConfig,
    ExperimentResult,
    OptimizerName,
    TrainingMode,
    run_experiment,
)


class TrainableSubset(StrEnum):
    ALL = "all"
    QUANTIZED_ONLY = "quantized_only"
    QUANTIZED_AND_AUX = "quantized_and_aux"


def _make_trainable_filter(
    subset: TrainableSubset,
) -> Callable | None:
    match subset:
        case TrainableSubset.ALL:
            return None
        case TrainableSubset.QUANTIZED_ONLY:
            return lambda info: info.quantized
        case TrainableSubset.QUANTIZED_AND_AUX:
            return lambda info: info.quantized or info.norm == ParameterNorm.L_INF
        case _:
            raise ValueError(f"Unknown trainable subset: {subset}")


@dataclass(frozen=True)
class SweepRun:
    name: str
    config: ExperimentConfig
    trainable_subset: TrainableSubset = TrainableSubset.ALL


@dataclass(frozen=True)
class SweepResult:
    name: str
    trainable_subset: str
    initial_kl: float
    final_kl: float
    initial_top1: float
    final_top1: float
    compilation_seconds: float
    seconds_per_step: float
    completed_steps: int


def _run_sweep_phase(
    phase_name: str,
    runs: list[SweepRun],
) -> list[tuple[SweepRun, ExperimentResult]]:
    results: list[tuple[SweepRun, ExperimentResult]] = []
    print(f"\n{'=' * 60}")
    print(f"Phase: {phase_name} ({len(runs)} runs)")
    print(f"{'=' * 60}")

    for i, run in enumerate(runs, start=1):
        print(f"\n--- Run {i}/{len(runs)}: {run.name} ---")
        trainable_filter = _make_trainable_filter(run.trainable_subset)
        result = run_experiment(run.config, trainable_filter=trainable_filter)
        results.append((run, result))

        print(
            f"  KL: {result.initial_eval.kl_divergence:.4f} -> {result.final_eval.kl_divergence:.4f}  "
            f"top1: {result.initial_eval.top1_agreement:.4f} -> {result.final_eval.top1_agreement:.4f}  "
            f"sec/step: {result.seconds_per_step:.2f}",
        )

    return results


def _best_by_final_kl(
    results: list[tuple[SweepRun, ExperimentResult]],
) -> tuple[SweepRun, ExperimentResult]:
    return min(results, key=lambda pair: pair[1].final_eval.kl_divergence)


def _summarize(run: SweepRun, result: ExperimentResult) -> SweepResult:
    return SweepResult(
        name=run.name,
        trainable_subset=run.trainable_subset.value,
        initial_kl=result.initial_eval.kl_divergence,
        final_kl=result.final_eval.kl_divergence,
        initial_top1=result.initial_eval.top1_agreement,
        final_top1=result.final_eval.top1_agreement,
        compilation_seconds=result.compilation_seconds,
        seconds_per_step=result.seconds_per_step,
        completed_steps=result.completed_steps,
    )


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
    all_results: list[SweepResult] = []

    def make_config(
        name: str,
        *,
        optimizer_name: OptimizerName = OptimizerName.ADAMW,
        learning_rate: float = 3e-6,
        training_mode: TrainingMode = TrainingMode.ONLINE_EXACT,
        stochastic_rounding: bool = True,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            teacher_path=teacher_path,
            student_path=student_path,
            dataset_path=dataset_path,
            output_dir=output_dir / name,
            training_mode=training_mode,
            train_examples=train_examples,
            eval_examples=eval_examples,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            seed=seed,
            save_checkpoints=False,
            stochastic_rounding=stochastic_rounding,
        )

    # Phase 1: Optimizer/LR sweep
    phase1_runs = [
        SweepRun(
            name=f"{opt.value}_lr{lr}",
            config=make_config(f"{opt.value}_lr{lr}", optimizer_name=opt, learning_rate=lr),
        )
        for opt in [OptimizerName.ADAMW, OptimizerName.MUON]
        for lr in [1e-7, 3e-7, 1e-6, 3e-6]
    ]
    phase1_results = _run_sweep_phase("Optimizer/LR", phase1_runs)
    all_results.extend(_summarize(run, result) for run, result in phase1_results)

    best_run, _ = _best_by_final_kl(phase1_results)
    best_optimizer = best_run.config.optimizer_name
    best_lr = best_run.config.learning_rate
    print(f"\nBest optimizer/LR: {best_optimizer.value} @ {best_lr}")

    # Phase 2: Trainable subset sweep
    phase2_runs = [
        SweepRun(
            name=f"subset_{subset.value}",
            config=make_config(
                f"subset_{subset.value}",
                optimizer_name=best_optimizer,
                learning_rate=best_lr,
            ),
            trainable_subset=subset,
        )
        for subset in TrainableSubset
    ]
    phase2_results = _run_sweep_phase("Trainable subset", phase2_runs)
    all_results.extend(_summarize(run, result) for run, result in phase2_results)

    best_subset_run, _ = _best_by_final_kl(phase2_results)
    best_subset = best_subset_run.trainable_subset
    print(f"\nBest trainable subset: {best_subset.value}")

    # Phase 3: Stochastic vs deterministic
    phase3_runs = [
        SweepRun(
            name=f"stochastic_{enabled}",
            config=make_config(
                f"stochastic_{enabled}",
                optimizer_name=best_optimizer,
                learning_rate=best_lr,
                stochastic_rounding=enabled,
            ),
            trainable_subset=best_subset,
        )
        for enabled in [True, False]
    ]
    phase3_results = _run_sweep_phase("Stochastic rounding", phase3_runs)
    all_results.extend(_summarize(run, result) for run, result in phase3_results)

    best_stochastic_run, _ = _best_by_final_kl(phase3_results)
    best_stochastic = best_stochastic_run.config.stochastic_rounding
    print(f"\nBest stochastic rounding: {best_stochastic}")

    # Phase 4: Online vs trace (only if trace dataset exists)
    # Note: trace mode requires a .msp trace file, not the parquet dataset.
    # This phase is skipped if the dataset is parquet (online-only).
    # To run trace comparison, generate traces first, then run this sweep
    # with --dataset-path pointing to the .msp file.

    # Write results
    results_path = output_dir / "sweep_results.json"
    with results_path.open("w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=4)

    print(f"\n{'=' * 60}")
    print("Sweep complete")
    print(f"{'=' * 60}")
    print(f"Best recipe: {best_optimizer.value} lr={best_lr} subset={best_subset.value} stochastic={best_stochastic}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    typer.run(main)
