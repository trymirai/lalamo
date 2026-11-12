"""Samples the teacher's own responses to the training prompts of the distillation pilot.

The responses are the training data of the supervised baseline: the student is fine-tuned on what the teacher
wrote instead of being corrected on what it wrote itself. One file is written per batch, and running the script
again with the same output directory continues with the first batch that is missing.
"""

# ruff: noqa: I001
# Lalamo configures JAX on import, so it has to be imported before jax.
import time
from pathlib import Path
from typing import Annotated

from lalamo.distillation import sample_responses

import jax
import typer

from lalamo.models import GenerationConfig
from lalamo.module import Keychain
from scripts.on_policy_distillation_pilot import (
    TeacherResponse,
    load_model,
    log,
    math_reward,
    save_teacher_responses,
    select_training_prompts,
    teacher_responses_path,
    training_problems_without_evaluation_overlap,
)


def main(
    output_directory: Annotated[Path, typer.Option(help="Receives one file of responses per batch.")],
    student_repo: str = "Qwen/Qwen3.5-9B",
    teacher_repo: str = "Qwen/Qwen3.8-27B",
    num_prompts: int = 3200,
    max_prompt_length: int = 384,
    max_response_length: int = 4096,
    batch_size: int = 48,
    seed: int = 0,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    teacher_device, *_ = jax.devices()
    cpu_device, *_ = jax.devices("cpu")
    # The student only renders its prompts here, so it stays off the accelerator.
    student = load_model(student_repo, cpu_device)
    teacher = load_model(teacher_repo, teacher_device)
    training_prompts = select_training_prompts(
        student,
        teacher,
        training_problems_without_evaluation_overlap(),
        num_prompts,
        max_prompt_length,
        seed,
    )
    del student
    prompt_capacity = max(len(prompt.pair.teacher) for prompt in training_prompts)
    stop_token_ids = frozenset(teacher.config.generation_config.stop_token_ids)

    for batch_index, batch_start in enumerate(range(0, num_prompts, batch_size)):
        if teacher_responses_path(output_directory, batch_index).exists():
            continue
        start = time.monotonic()
        batch = training_prompts[batch_start : batch_start + batch_size]
        *_, last_prompt = batch
        # A short last batch is filled up with repeats, so that every batch reuses one compiled generation loop.
        filled_batch = [*batch, *([last_prompt] * (batch_size - len(batch)))]
        # The teacher samples with its own default settings: these are the responses it would give a user.
        sampled = sample_responses(
            teacher,
            [prompt.pair.teacher for prompt in filled_batch],
            max_response_length,
            GenerationConfig(),
            prompt_capacity=prompt_capacity,
            keychain=Keychain.init(seed + batch_index, sharding_config=teacher.sharding_config),
        )
        responses = [
            TeacherResponse(index=batch_start + offset, problem=prompt.problem, pair=prompt.pair, response=response)
            for offset, (prompt, response) in enumerate(zip(batch, sampled[: len(batch)], strict=True))
        ]
        save_teacher_responses(output_directory, batch_index, responses)
        num_finished = sum(response.response[-1] in stop_token_ids for response in responses)
        num_correct = sum(math_reward(teacher, response.problem, response.response) for response in responses)
        mean_length = sum(len(response.response) for response in responses) / len(responses)
        log(
            f"batch {batch_index}: prompts {batch_start}-{batch_start + len(batch) - 1} of {num_prompts}, "
            f"finished {num_finished}/{len(responses)}, correct {num_correct:.0f}/{len(responses)}, "
            f"mean length {mean_length:.0f}, {time.monotonic() - start:.0f}s"
        )


if __name__ == "__main__":
    typer.run(main)
