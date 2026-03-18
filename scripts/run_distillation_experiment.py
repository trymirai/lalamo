import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from lalamo.distill_runner import (
    ComputeDTypeName,
    DistillConfig,
    DistillResult,
    OptimizerName,
    TrainingMode,
    distill,
)
from lalamo.quantization import QuantizationMode

ExperimentConfig = DistillConfig
ExperimentResult = DistillResult
run_experiment = distill


def main(
    teacher_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    student_path: Annotated[Path, typer.Option(exists=True, dir_okay=True, file_okay=False)],
    dataset_path: Annotated[Path, typer.Option(exists=True, dir_okay=False, file_okay=True)],
    output_dir: Annotated[Path, typer.Option()],
    training_mode: Annotated[TrainingMode, typer.Option()] = TrainingMode.ONLINE_EXACT,
    train_examples: Annotated[int, typer.Option()] = 256,
    eval_examples: Annotated[int, typer.Option()] = 64,
    max_sequence_length: Annotated[int, typer.Option()] = 256,
    batch_size: Annotated[int, typer.Option()] = 4,
    num_steps: Annotated[int, typer.Option()] = 25,
    learning_rate: Annotated[float, typer.Option()] = 3e-7,
    warmup_steps: Annotated[int, typer.Option()] = 0,
    gradient_clip_norm: Annotated[float | None, typer.Option()] = None,
    gradient_accumulation_steps: Annotated[int, typer.Option()] = 1,
    optimizer_name: Annotated[OptimizerName, typer.Option()] = OptimizerName.MUON,
    quantization_mode: Annotated[QuantizationMode, typer.Option()] = QuantizationMode.UINT4,
    compute_dtype_name: Annotated[ComputeDTypeName, typer.Option()] = ComputeDTypeName.AUTO,
    eval_every_steps: Annotated[int, typer.Option()] = 0,
    checkpoint_every_steps: Annotated[int, typer.Option()] = 0,
    early_stop_patience: Annotated[int, typer.Option()] = 0,
    resume_from: Annotated[Path | None, typer.Option()] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    result = distill(
        DistillConfig(
            teacher_path=teacher_path,
            student_path=student_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            training_mode=training_mode,
            train_examples=train_examples,
            eval_examples=eval_examples,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            num_steps=num_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            gradient_clip_norm=gradient_clip_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer_name=optimizer_name,
            quantization_mode=quantization_mode,
            compute_dtype_name=compute_dtype_name,
            eval_every_steps=eval_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            early_stop_patience=early_stop_patience,
            resume_from=resume_from,
            seed=seed,
        ),
    )
    print(json.dumps(asdict(result), indent=4))


if __name__ == "__main__":
    typer.run(main)
