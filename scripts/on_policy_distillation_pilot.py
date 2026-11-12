"""Pilot of on-policy distillation: a large teacher is distilled into a smaller student on math problems.

Accuracy on MATH-500 (the distilled skill) and on MMLU-Pro (a check against forgetting) is measured before and
after training. Everything the run produces is written to the output directory, and restarting the script with
the same output directory resumes the run.
"""

# ruff: noqa: I001
# Lalamo configures JAX on import, so it has to be imported before jax.
import json
import math
import random
import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

from lalamo import LanguageModel, UserMessage

import cattrs
import jax
import numpy as np
import polars as pl
import typer
from huggingface_hub import hf_hub_download

from lalamo.distillation import AdvantageBaseline, sample_responses
from lalamo.distillation.runner import (
    DistillationConfig,
    ExecutionConfig,
    PromptPair,
    StepReport,
    run_on_policy_distillation,
)
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig
from lalamo.module import Keychain, ShardingConfig


@dataclass(frozen=True)
class MathProblem:
    problem: str
    answer: str


@dataclass(frozen=True)
class MultipleChoiceQuestion:
    question: str
    options: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class EvaluationResult:
    num_questions: int
    accuracy: float
    answered_fraction: float
    mean_response_length: float
    seconds: float


@dataclass(frozen=True)
class EvaluationConfig:
    num_math_problems: int
    num_multiple_choice_questions: int
    math_max_response_length: int
    multiple_choice_max_response_length: int
    student_batch_size: int
    teacher_batch_size: int
    seed: int


@dataclass(frozen=True)
class Evaluations:
    math: tuple[MathProblem, ...]
    multiple_choice: tuple[MultipleChoiceQuestion, ...]


@dataclass(frozen=True)
class WandbConfig:
    project: str
    entity: str | None
    run_name: str | None


class WandbRun(Protocol):
    @property
    def id(self) -> str: ...

    def log(self, data: dict[str, Any], step: int | None = None, commit: bool | None = None) -> None: ...

    def define_metric(self, name: str, step_metric: str | None = None) -> object: ...

    def finish(self) -> None: ...


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_math_problems(repo: str, filename: str) -> list[MathProblem]:
    path = Path(hf_hub_download(repo, filename, repo_type="dataset"))
    with path.open() as file:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in file]
        else:
            rows = json.load(file)
    return cattrs.structure(rows, list[MathProblem])


def load_multiple_choice_questions(repo: str, filename: str) -> list[MultipleChoiceQuestion]:
    frame = pl.read_parquet(hf_hub_download(repo, filename, repo_type="dataset"))
    rows = frame.select("question", "options", "answer").iter_rows(named=True)
    return cattrs.structure(list(rows), list[MultipleChoiceQuestion])


def math_prompt(problem: MathProblem) -> str:
    return f"{problem.problem}\n\nPut your final answer within \\boxed{{}}."


def multiple_choice_prompt(question: MultipleChoiceQuestion) -> str:
    options = "\n".join(
        f"{letter}. {option}" for letter, option in zip(string.ascii_uppercase, question.options, strict=False)
    )
    return (
        f"{question.question}\n\n{options}\n\n"
        'Finish your reply with a line of the form "Answer: X", where X is the letter of the correct option.'
    )


def word_ngrams(text: str, size: int) -> frozenset[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return frozenset(tuple(words[start : start + size]) for start in range(len(words) - size + 1))


def without_evaluation_overlap(
    training_problems: Sequence[MathProblem],
    evaluation_problems: Sequence[MathProblem],
) -> list[MathProblem]:
    """Drops training problems that repeat an evaluation problem or share an 8-word phrase with one."""
    evaluation_ngrams = frozenset().union(*(word_ngrams(problem.problem, 8) for problem in evaluation_problems))
    evaluation_texts = frozenset(" ".join(problem.problem.lower().split()) for problem in evaluation_problems)
    return [
        problem
        for problem in training_problems
        if " ".join(problem.problem.lower().split()) not in evaluation_texts
        and word_ngrams(problem.problem, 8).isdisjoint(evaluation_ngrams)
    ]


def render(model: LanguageModel, prompt: str) -> tuple[int, ...]:
    codec = model.token_codec
    return tuple(codec.encode_text(codec.render_request([UserMessage(content=prompt)])))


def select_prompt_pairs(
    student: LanguageModel,
    teacher: LanguageModel,
    problems: Sequence[MathProblem],
    num_pairs: int,
    max_prompt_length: int,
    seed: int,
) -> list[PromptPair]:
    shuffled = random.Random(seed).sample(problems, len(problems))
    pairs = (
        PromptPair(student=render(student, math_prompt(problem)), teacher=render(teacher, math_prompt(problem)))
        for problem in shuffled
    )
    fitting_pairs = (pair for pair in pairs if max(len(pair.student), len(pair.teacher)) <= max_prompt_length)
    selected = [pair for _, pair in zip(range(num_pairs), fitting_pairs, strict=False)]
    if len(selected) < num_pairs:
        raise ValueError(f"Only {len(selected)} of the {num_pairs} required training prompts are available.")
    return selected


def final_reply(response: str) -> str | None:
    """The reply that follows the reasoning; `None` when the response was cut off before the reasoning ended."""
    _, reasoning_end, reply = response.rpartition("</think>")
    if not reasoning_end:
        return None
    return reply


def last_boxed_content(text: str) -> str | None:
    opening = "\\boxed{"
    start = text.rfind(opening)
    if start < 0:
        return None
    depth = 0
    for index in range(start + len(opening) - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(opening) : index]
    return None


def normalize_math_answer(answer: str) -> str:
    """Removes formatting differences between equal answers. It does not recognize mathematically equal forms."""
    answer = re.sub(r"\\(?:text|textbf|mbox|mathrm)\{([^{}]*)\}", r"\1", answer)
    for formatting in (
        "\\left",
        "\\right",
        "\\!",
        "\\,",
        "\\;",
        "\\ ",
        "\\$",
        "$",
        "^\\circ",
        "^{\\circ}",
        "\\%",
        "%",
    ):
        answer = answer.replace(formatting, "")
    answer = answer.replace("dfrac", "frac").replace("tfrac", "frac")
    answer = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", answer)
    answer = re.sub(r"\s+", "", answer).rstrip(".")
    variable, equals_sign, value = answer.partition("=")
    if equals_sign and len(variable) <= 2:
        return value
    return answer


def math_answer(reply: str) -> str | None:
    boxed = last_boxed_content(reply)
    if boxed is None:
        return None
    return normalize_math_answer(boxed)


def multiple_choice_answer(reply: str) -> str | None:
    letters = re.findall(r"Answer:\s*\(?([A-J])\)?", reply)
    if not letters:
        return None
    *_, last_letter = letters
    return last_letter


def evaluate(
    model: LanguageModel,
    prompts: Sequence[str],
    reference_answers: Sequence[str],
    extract_answer: Callable[[str], str | None],
    max_response_length: int,
    batch_size: int,
    seed: int,
) -> EvaluationResult:
    """Accuracy of replies sampled with the model's default settings; a reply that never arrives counts as wrong."""
    start = time.monotonic()
    token_prompts = [render(model, prompt) for prompt in prompts]
    prompt_capacity = max(len(prompt) for prompt in token_prompts)
    stop_token_ids = frozenset(model.config.generation_config.stop_token_ids)
    responses: list[tuple[int, ...]] = []
    for batch_index, batch_start in enumerate(range(0, len(token_prompts), batch_size)):
        batch = token_prompts[batch_start : batch_start + batch_size]
        *_, last_prompt = batch
        # A short last batch is filled up with repeats, so that every batch reuses one compiled generation loop.
        filled_batch = [*batch, *([last_prompt] * (batch_size - len(batch)))]
        sampled = sample_responses(
            model,
            filled_batch,
            max_response_length,
            GenerationConfig(),
            prompt_capacity=prompt_capacity,
            keychain=Keychain.init(seed + batch_index, sharding_config=model.sharding_config),
        )
        responses.extend(sampled[: len(batch)])

    texts = [
        model.token_codec.decode_tokens([token_id for token_id in response if token_id not in stop_token_ids])
        for response in responses
    ]
    replies = [final_reply(text) for text in texts]
    answers = [None if reply is None else extract_answer(reply) for reply in replies]
    num_correct = sum(answer == reference for answer, reference in zip(answers, reference_answers, strict=True))
    return EvaluationResult(
        num_questions=len(prompts),
        accuracy=num_correct / len(prompts),
        answered_fraction=sum(answer is not None for answer in answers) / len(prompts),
        mean_response_length=sum(len(response) for response in responses) / len(responses),
        seconds=time.monotonic() - start,
    )


def evaluate_math(
    model: LanguageModel,
    evaluations: Evaluations,
    config: EvaluationConfig,
    batch_size: int,
) -> EvaluationResult:
    return evaluate(
        model,
        [math_prompt(problem) for problem in evaluations.math],
        [normalize_math_answer(problem.answer) for problem in evaluations.math],
        math_answer,
        config.math_max_response_length,
        batch_size,
        config.seed,
    )


def evaluate_multiple_choice(
    model: LanguageModel,
    evaluations: Evaluations,
    config: EvaluationConfig,
    batch_size: int,
) -> EvaluationResult:
    return evaluate(
        model,
        [multiple_choice_prompt(question) for question in evaluations.multiple_choice],
        [question.answer for question in evaluations.multiple_choice],
        multiple_choice_answer,
        config.multiple_choice_max_response_length,
        batch_size,
        config.seed,
    )


def load_model(repo: str, device: jax.Device) -> LanguageModel:
    start = time.monotonic()
    model = import_model(repo, sharding_config=ShardingConfig.replicated(devices=[device])).model
    if not isinstance(model, LanguageModel):
        raise TypeError(f"{repo} is not a language model.")
    log(f"loaded {repo} on {device} in {time.monotonic() - start:.0f}s")
    return model


def start_wandb_run(
    config: WandbConfig,
    output_directory: Path,
    run_config: dict[str, object],
) -> WandbRun:
    import wandb  # noqa: PLC0415  (optional dependency, only needed when a wandb project is given)

    # The run id is kept next to the checkpoints, so that a resumed run keeps writing to the same wandb run.
    run_id_path = output_directory / "wandb_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=config.run_name,
        id=run_id,
        resume="allow",
        dir=str(output_directory),
        config=run_config,
    )
    run_id_path.write_text(run.id)
    # Evaluations are charted against their own axis, so they can be logged again after a restart: wandb's own
    # step only moves forward and would drop points that belong to an earlier step.
    run.define_metric("evaluation/num_completed_steps")
    run.define_metric("evaluation/*", step_metric="evaluation/num_completed_steps")
    return run


def wandb_step_metrics(
    report: StepReport,
    distillation_config: DistillationConfig,
    reasoning_end_token_id: int | None,
) -> dict[str, Any]:
    """Everything charted for one training step: its scalar metrics and how the per-token gaps are distributed."""
    import wandb  # noqa: PLC0415

    metrics = report.metrics
    scalars = {name: value for name, value in asdict(metrics).items() if isinstance(value, int | float)}
    step_seconds = (
        metrics.sampling_seconds
        + metrics.teacher_scoring_seconds
        + metrics.student_scoring_seconds
        + metrics.gradient_seconds
    )
    position_range_length = math.ceil(distillation_config.max_response_length / len(metrics.reverse_kl_by_position))
    position_range_starts = [index * position_range_length for index in range(len(metrics.reverse_kl_by_position))]
    reverse_kl_by_position = {
        f"reverse_kl_by_position/tokens_{start:05d}_to_{start + position_range_length:05d}": value
        for start, value in zip(position_range_starts, metrics.reverse_kl_by_position, strict=True)
        if value is not None
    }
    reasoning_finished = {}
    if reasoning_end_token_id is not None:
        reasoning_finished = {
            "train/reasoning_finished_fraction": sum(
                reasoning_end_token_id in response for response in report.responses
            )
            / len(report.responses)
        }
    return {
        **{f"train/{name}": value for name, value in scalars.items() if name != "step"},
        **reasoning_finished,
        "train/learning_rate": distillation_config.learning_rate,
        "train/gradient_was_clipped": float(metrics.gradient_norm > distillation_config.gradient_clip_norm),
        "train/step_seconds": step_seconds,
        "train/response_tokens_per_second": metrics.num_response_tokens / step_seconds,
        **{
            f"log_prob_gap/{name}": value
            for name, value in zip(("p01", "p10", "p50", "p90", "p99"), metrics.log_prob_gap_quantiles, strict=True)
        },
        "log_prob_gap/strong_disagreement_fraction": metrics.strong_disagreement_fraction,
        # Clipped so that a few extreme tokens do not flatten the histogram.
        "log_prob_gap/histogram": wandb.Histogram(
            np.clip(np.concatenate(report.log_prob_gaps), -10.0, 5.0).tolist(),
            num_bins=64,
        ),
        **{
            f"rollout_reverse_kl/{name}": value
            for name, value in zip(("min", "median", "max"), metrics.response_reverse_kl_quantiles, strict=True)
        },
        **reverse_kl_by_position,
    }


def wandb_rollout_table(report: StepReport, num_rollouts: int, num_shown_characters: int) -> object:
    """A few rollouts of the step as text, with the reverse KL of each, to read what the student is producing."""
    import wandb  # noqa: PLC0415

    codec = report.student.token_codec
    stop_token_ids = frozenset(report.student.config.generation_config.stop_token_ids)
    rows = [
        [
            codec.decode_tokens(list(pair.student)),
            response_text[:num_shown_characters],
            response_text[-num_shown_characters:],
            len(response),
            response[-1] in stop_token_ids,
            -float(gaps.mean()),
        ]
        for pair, response, gaps in zip(report.prompt_pairs, report.responses, report.log_prob_gaps, strict=True)
        for response_text in [codec.decode_tokens(list(response))]
    ][:num_rollouts]
    return wandb.Table(
        columns=["prompt", "response_start", "response_end", "response_tokens", "finished", "reverse_kl"],
        data=rows,
    )


def main(
    output_directory: Annotated[Path, typer.Option(help="Where metrics, checkpoints and results are written.")],
    student_repo: str = "Qwen/Qwen3.5-9B",
    teacher_repo: str = "Qwen/Qwen3.8-27B",
    num_steps: int = 50,
    prompts_per_step: int = 64,
    max_response_length: int = 4096,
    max_prompt_length: int = 384,
    adapter_rank: int = 64,
    learning_rate: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    gradient_micro_batch_size: int = 1,
    student_scoring_micro_batch_size: int = 2,
    teacher_scoring_micro_batch_size: int = 2,
    checkpoint_every_steps: int = 5,
    evaluate_every_steps: int = 10,
    rollout_table_every_steps: int = 10,
    num_math_problems: int = 200,
    num_multiple_choice_questions: int = 300,
    math_max_response_length: int = 8192,
    multiple_choice_max_response_length: int = 4096,
    student_evaluation_batch_size: int = 100,
    teacher_evaluation_batch_size: int = 50,
    evaluate_teacher: bool = True,
    wandb_project: Annotated[
        str | None, typer.Option(help="Log to this wandb project; wandb is off if unset.")
    ] = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 0,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    distillation_config = DistillationConfig(
        num_steps=num_steps,
        prompts_per_step=prompts_per_step,
        max_response_length=max_response_length,
        adapter_rank=adapter_rank,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
        baseline=AdvantageBaseline.BATCH_MEAN,
        seed=seed,
    )
    execution_config = ExecutionConfig(
        gradient_micro_batch_size=gradient_micro_batch_size,
        student_scoring_micro_batch_size=student_scoring_micro_batch_size,
        teacher_scoring_micro_batch_size=teacher_scoring_micro_batch_size,
        checkpoint_every_steps=checkpoint_every_steps,
    )
    evaluation_config = EvaluationConfig(
        num_math_problems=num_math_problems,
        num_multiple_choice_questions=num_multiple_choice_questions,
        math_max_response_length=math_max_response_length,
        multiple_choice_max_response_length=multiple_choice_max_response_length,
        student_batch_size=student_evaluation_batch_size,
        teacher_batch_size=teacher_evaluation_batch_size,
        seed=seed,
    )
    run_config: dict[str, object] = {
        "student": student_repo,
        "teacher": teacher_repo,
        "max_prompt_length": max_prompt_length,
        "evaluate_every_steps": evaluate_every_steps,
        "distillation": asdict(distillation_config),
        "execution": asdict(execution_config),
        "evaluation": asdict(evaluation_config),
    }
    (output_directory / "config.json").write_text(json.dumps(run_config, indent=2))

    # Started before the models are loaded, so that a login or network problem shows up within seconds.
    wandb_run = None
    if wandb_project is not None:
        wandb_run = start_wandb_run(
            WandbConfig(project=wandb_project, entity=wandb_entity, run_name=wandb_run_name),
            output_directory,
            run_config,
        )

    student_device, *_ = jax.devices()
    *_, teacher_device = jax.devices()
    student = load_model(student_repo, student_device)
    teacher = load_model(teacher_repo, teacher_device)
    reasoning_end_token_ids = student.token_codec.encode_text("</think>")
    reasoning_end_token_id = reasoning_end_token_ids[0] if len(reasoning_end_token_ids) == 1 else None

    math_evaluation_pool = load_math_problems("HuggingFaceH4/MATH-500", "test.jsonl")
    multiple_choice_pool = load_multiple_choice_questions("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")
    evaluations = Evaluations(
        math=tuple(random.Random(seed).sample(math_evaluation_pool, num_math_problems)),
        multiple_choice=tuple(random.Random(seed).sample(multiple_choice_pool, num_multiple_choice_questions)),
    )
    training_problems = without_evaluation_overlap(
        load_math_problems("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json"),
        math_evaluation_pool,
    )
    prompt_pairs = select_prompt_pairs(
        student,
        teacher,
        training_problems,
        num_steps * prompts_per_step,
        max_prompt_length,
        seed,
    )
    log(f"{len(training_problems)} training problems after removing overlap with MATH-500; using {len(prompt_pairs)}")

    results_path = output_directory / "results.json"
    results: dict[str, dict[str, float | int]] = json.loads(results_path.read_text()) if results_path.exists() else {}

    def record(name: str, num_completed_steps: int, compute: Callable[[], EvaluationResult]) -> None:
        """Runs an evaluation once per run: a restarted run reuses the saved result and only charts it again."""
        result_name = f"{name}_step_{num_completed_steps:05d}"
        if result_name in results:
            log(f"{result_name}: already evaluated: {results[result_name]}")
        else:
            results[result_name] = asdict(compute())
            results_path.write_text(json.dumps(results, indent=2))
            log(f"{result_name}: {results[result_name]}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "evaluation/num_completed_steps": num_completed_steps,
                    **{f"evaluation/{name}_{metric}": value for metric, value in results[result_name].items()},
                },
                commit=False,
            )

    def evaluate_student(model: LanguageModel, num_completed_steps: int) -> None:
        record(
            "student_math",
            num_completed_steps,
            lambda: evaluate_math(model, evaluations, evaluation_config, student_evaluation_batch_size),
        )
        record(
            "student_multiple_choice",
            num_completed_steps,
            lambda: evaluate_multiple_choice(model, evaluations, evaluation_config, student_evaluation_batch_size),
        )

    if evaluate_teacher:
        record(
            "teacher_math",
            0,
            lambda: evaluate_math(teacher, evaluations, evaluation_config, teacher_evaluation_batch_size),
        )
    evaluate_student(student, 0)

    def report_step(report: StepReport) -> None:
        metrics = report.metrics
        num_completed_steps = metrics.step + 1
        log(
            f"step {num_completed_steps}/{num_steps}: reverse KL {metrics.reverse_kl:.4f}, "
            f"gradient norm {metrics.gradient_norm:.3f}, adapter norm {metrics.adapter_norm:.3f}, "
            f"mean length {metrics.mean_response_length:.0f}, finished {metrics.finished_fraction:.2f}, "
            f"seconds sampling/teacher/student/gradients {metrics.sampling_seconds:.0f}/"
            f"{metrics.teacher_scoring_seconds:.0f}/{metrics.student_scoring_seconds:.0f}/"
            f"{metrics.gradient_seconds:.0f}, peak GB student/teacher "
            f"{metrics.student_peak_memory_gigabytes}/{metrics.teacher_peak_memory_gigabytes}"
        )
        if wandb_run is not None:
            wandb_run.log(
                wandb_step_metrics(report, distillation_config, reasoning_end_token_id),
                step=num_completed_steps,
            )
            if num_completed_steps % rollout_table_every_steps == 0 or num_completed_steps == 1:
                wandb_run.log({"rollouts/samples": wandb_rollout_table(report, 8, 2000)}, step=num_completed_steps)
        if num_completed_steps % evaluate_every_steps == 0 and num_completed_steps < num_steps:
            evaluate_student(report.student, num_completed_steps)

    distilled_student = run_on_policy_distillation(
        student,
        teacher,
        prompt_pairs,
        distillation_config,
        execution_config,
        output_directory,
        report_step,
    )
    evaluate_student(distilled_student, num_steps)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
