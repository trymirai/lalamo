"""Pilot of distilling a large teacher into a smaller student on math problems.

The method is chosen with `--method`: on-policy distillation (the student is corrected on what it samples itself),
a supervised baseline (the student is fine-tuned on responses the teacher wrote), training on a reward for correct
answers, or on-policy distillation combined with that reward.

Accuracy on MATH-500 (the distilled skill) and on MMLU-Pro (a check against forgetting) is measured before, during
and after training. Everything the run produces is written to the output directory, and restarting the script with
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
from enum import StrEnum
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
    ExecutionConfig,
    OnPolicyConfig,
    PromptPair,
    StepReport,
    SupervisedConfig,
    SupervisedExample,
    SupervisedStepReport,
    run_on_policy_training,
    run_supervised_fine_tuning,
)
from lalamo.model_import.common import import_model
from lalamo.models import GenerationConfig
from lalamo.module import Keychain, ShardingConfig


class Method(StrEnum):
    ON_POLICY_DISTILLATION = "on-policy-distillation"
    SUPERVISED = "supervised"
    REWARD = "reward"
    DISTILLATION_AND_REWARD = "distillation-and-reward"


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
class TrainingPrompt:
    problem: MathProblem
    pair: PromptPair


@dataclass(frozen=True)
class TeacherResponse:
    """A training prompt with the response the teacher sampled for it."""

    index: int
    problem: MathProblem
    pair: PromptPair
    response: tuple[int, ...]


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


def select_training_prompts(
    student: LanguageModel,
    teacher: LanguageModel,
    problems: Sequence[MathProblem],
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
) -> list[TrainingPrompt]:
    shuffled = random.Random(seed).sample(problems, len(problems))
    prompts = (
        TrainingPrompt(
            problem=problem,
            pair=PromptPair(
                student=render(student, math_prompt(problem)),
                teacher=render(teacher, math_prompt(problem)),
            ),
        )
        for problem in shuffled
    )
    fitting_prompts = (
        prompt for prompt in prompts if max(len(prompt.pair.student), len(prompt.pair.teacher)) <= max_prompt_length
    )
    selected = [prompt for _, prompt in zip(range(num_prompts), fitting_prompts, strict=False)]
    if len(selected) < num_prompts:
        raise ValueError(f"Only {len(selected)} of the {num_prompts} required training prompts are available.")
    return selected


def training_problems_without_evaluation_overlap() -> list[MathProblem]:
    return without_evaluation_overlap(
        load_math_problems("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json"),
        load_math_problems("HuggingFaceH4/MATH-500", "test.jsonl"),
    )


def final_reply(response: str) -> str | None:
    """The reply that follows the reasoning; `None` when the response was cut off before the reasoning ended."""
    _, reasoning_end, reply = response.rpartition("</think>")
    if not reasoning_end:
        return None
    return reply


def response_text(model: LanguageModel, response: Sequence[int]) -> str:
    stop_token_ids = frozenset(model.config.generation_config.stop_token_ids)
    return model.token_codec.decode_tokens([token_id for token_id in response if token_id not in stop_token_ids])


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


def math_reward(model: LanguageModel, problem: MathProblem, response: Sequence[int]) -> float:
    """1 if the final reply boxes the reference answer, else 0. A response cut off while reasoning scores 0."""
    reply = final_reply(response_text(model, response))
    if reply is None:
        return 0.0
    return float(math_answer(reply) == normalize_math_answer(problem.answer))


def save_teacher_responses(directory: Path, batch_index: int, responses: Sequence[TeacherResponse]) -> None:
    """A batch file appears only once it is complete, so an interrupted generation redoes the unfinished batch."""
    path = teacher_responses_path(directory, batch_index)
    unfinished_path = path.with_suffix(".unfinished")
    unfinished_path.write_text("".join(json.dumps(cattrs.unstructure(response)) + "\n" for response in responses))
    unfinished_path.replace(path)


def teacher_responses_path(directory: Path, batch_index: int) -> Path:
    return directory / f"batch_{batch_index:05d}.jsonl"


def load_teacher_responses(directory: Path | str) -> list[TeacherResponse]:
    rows = [
        json.loads(line)
        for path in sorted(Path(directory).glob("batch_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    responses = cattrs.structure(rows, list[TeacherResponse])
    if [response.index for response in responses] != list(range(len(responses))):
        raise ValueError(f"The teacher responses in {directory} are not a complete run of batches.")
    return responses


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

    replies = [final_reply(response_text(model, response)) for response in responses]
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
    config: OnPolicyConfig,
    reasoning_end_token_id: int | None,
) -> dict[str, Any]:
    """Everything charted for one on-policy step: its scalar metrics and how the per-token gaps are distributed."""
    import wandb  # noqa: PLC0415

    metrics = report.metrics
    scalars = {name: value for name, value in asdict(metrics).items() if isinstance(value, int | float)}
    step_seconds = (
        metrics.sampling_seconds
        + metrics.teacher_scoring_seconds
        + metrics.student_scoring_seconds
        + metrics.gradient_seconds
    )
    reasoning_finished = {}
    if reasoning_end_token_id is not None:
        reasoning_finished = {
            "train/reasoning_finished_fraction": sum(
                reasoning_end_token_id in response for response in report.responses
            )
            / len(report.responses)
        }
    # Only a run with a teacher has log-probability gaps to chart.
    gap_charts: dict[str, Any] = {}
    if (
        report.log_prob_gaps is not None
        and metrics.log_prob_gap_quantiles is not None
        and metrics.response_reverse_kl_quantiles is not None
        and metrics.reverse_kl_by_position is not None
    ):
        position_range_length = math.ceil(config.max_response_length / len(metrics.reverse_kl_by_position))
        gap_charts = {
            **{
                f"log_prob_gap/{name}": value
                for name, value in zip(
                    ("p01", "p10", "p50", "p90", "p99"), metrics.log_prob_gap_quantiles, strict=True
                )
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
            **{
                f"reverse_kl_by_position/tokens_{start:05d}_to_{start + position_range_length:05d}": value
                for start, value in zip(
                    range(0, config.max_response_length, position_range_length),
                    metrics.reverse_kl_by_position,
                    strict=True,
                )
                if value is not None
            },
        }
    return {
        **{f"train/{name}": value for name, value in scalars.items() if name != "step"},
        **reasoning_finished,
        "train/learning_rate": config.learning_rate,
        "train/gradient_was_clipped": float(metrics.gradient_norm > config.gradient_clip_norm),
        "train/step_seconds": step_seconds,
        "train/response_tokens_per_second": metrics.num_response_tokens / step_seconds,
        **gap_charts,
    }


def wandb_supervised_step_metrics(report: SupervisedStepReport, config: SupervisedConfig) -> dict[str, Any]:
    metrics = report.metrics
    scalars = {name: value for name, value in asdict(metrics).items() if isinstance(value, int | float)}
    step_seconds = metrics.student_scoring_seconds + metrics.gradient_seconds
    return {
        **{f"train/{name}": value for name, value in scalars.items() if name != "step"},
        "train/learning_rate": config.learning_rate,
        "train/gradient_was_clipped": float(metrics.gradient_norm > config.gradient_clip_norm),
        "train/step_seconds": step_seconds,
        "train/response_tokens_per_second": metrics.num_response_tokens / step_seconds,
    }


def wandb_rollout_table(report: StepReport, num_rollouts: int, num_shown_characters: int) -> object:
    """A few rollouts of the step as text, to read what the student is producing.

    The reverse KL of a rollout is empty in a run without a teacher, and its reward in a run without rewards.
    """
    import wandb  # noqa: PLC0415

    codec = report.student.token_codec
    stop_token_ids = frozenset(report.student.config.generation_config.stop_token_ids)
    num_shown = min(num_rollouts, len(report.responses))
    reverse_kls = (
        [None] * num_shown
        if report.log_prob_gaps is None
        else [-float(gaps.mean()) for gaps in report.log_prob_gaps[:num_shown]]
    )
    rewards = [None] * num_shown if report.rewards is None else list(report.rewards[:num_shown])
    rows = [
        [
            codec.decode_tokens(list(pair.student)),
            text[:num_shown_characters],
            text[-num_shown_characters:],
            len(response),
            response[-1] in stop_token_ids,
            reverse_kl,
            reward,
        ]
        for pair, response, reverse_kl, reward in zip(
            report.prompt_pairs[:num_shown],
            report.responses[:num_shown],
            reverse_kls,
            rewards,
            strict=True,
        )
        for text in [codec.decode_tokens(list(response))]
    ]
    return wandb.Table(
        columns=["prompt", "response_start", "response_end", "response_tokens", "finished", "reverse_kl", "reward"],
        data=rows,
    )


def main(
    output_directory: Annotated[Path, typer.Option(help="Where metrics, checkpoints and results are written.")],
    method: Method = Method.ON_POLICY_DISTILLATION,
    teacher_responses: Annotated[
        Path | None,
        typer.Option(
            help="Directory written by generate_teacher_responses.py. The supervised method trains on its responses. "
            "Every method takes its training prompts from it when it is given, so a method that does not use the "
            "teacher does not have to load it to render prompts.",
        ),
    ] = None,
    student_repo: str = "Qwen/Qwen3.5-9B",
    teacher_repo: str = "Qwen/Qwen3.8-27B",
    num_steps: int = 50,
    prompts_per_step: int = 64,
    rollouts_per_prompt: int = 1,
    distillation_weight: float = 1.0,
    reward_weight: float = 1.0,
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
    uses_teacher = method in (Method.ON_POLICY_DISTILLATION, Method.DISTILLATION_AND_REWARD)
    uses_reward = method in (Method.REWARD, Method.DISTILLATION_AND_REWARD)
    training_config: OnPolicyConfig | SupervisedConfig
    if method is Method.SUPERVISED:
        training_config = SupervisedConfig(
            num_steps=num_steps,
            examples_per_step=prompts_per_step,
            adapter_rank=adapter_rank,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            seed=seed,
        )
    else:
        training_config = OnPolicyConfig(
            num_steps=num_steps,
            prompts_per_step=prompts_per_step,
            max_response_length=max_response_length,
            adapter_rank=adapter_rank,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            baseline=AdvantageBaseline.BATCH_MEAN,
            seed=seed,
            rollouts_per_prompt=rollouts_per_prompt,
            distillation_weight=distillation_weight if uses_teacher else 0.0,
            reward_weight=reward_weight if uses_reward else 0.0,
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
        "method": method.value,
        "student": student_repo,
        "teacher": teacher_repo,
        "teacher_responses": None if teacher_responses is None else str(teacher_responses),
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "evaluate_every_steps": evaluate_every_steps,
        "training": asdict(training_config),
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
    teacher = load_model(teacher_repo, teacher_device) if uses_teacher else None
    reasoning_end_token_ids = student.token_codec.encode_text("</think>")
    reasoning_end_token_id = reasoning_end_token_ids[0] if len(reasoning_end_token_ids) == 1 else None

    math_evaluation_pool = load_math_problems("HuggingFaceH4/MATH-500", "test.jsonl")
    multiple_choice_pool = load_multiple_choice_questions("TIGER-Lab/MMLU-Pro", "data/test-00000-of-00001.parquet")
    evaluations = Evaluations(
        math=tuple(random.Random(seed).sample(math_evaluation_pool, num_math_problems)),
        multiple_choice=tuple(random.Random(seed).sample(multiple_choice_pool, num_multiple_choice_questions)),
    )

    num_training_prompts = num_steps * prompts_per_step
    stored_responses = None
    if teacher_responses is not None:
        stored_responses = load_teacher_responses(teacher_responses)
        if len(stored_responses) < num_training_prompts:
            raise ValueError(
                f"The run needs {num_training_prompts} prompts, but {teacher_responses} holds {len(stored_responses)}."
            )
        stored_responses = stored_responses[:num_training_prompts]
        training_prompts = [TrainingPrompt(problem=stored.problem, pair=stored.pair) for stored in stored_responses]
        log(f"using the first {len(training_prompts)} training prompts stored in {teacher_responses}")
    elif teacher is not None:
        training_problems = training_problems_without_evaluation_overlap()
        training_prompts = select_training_prompts(
            student,
            teacher,
            training_problems,
            num_training_prompts,
            max_prompt_length,
            seed,
        )
        log(
            f"{len(training_problems)} training problems after removing overlap with MATH-500; "
            f"using {len(training_prompts)}"
        )
    else:
        raise ValueError(f"The {method.value} method does not load the teacher, so it needs --teacher-responses.")

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

    def evaluate_if_due(model: LanguageModel, num_completed_steps: int) -> None:
        if num_completed_steps % evaluate_every_steps == 0 and num_completed_steps < num_steps:
            evaluate_student(model, num_completed_steps)

    if teacher is not None and evaluate_teacher:
        evaluated_teacher = teacher
        record(
            "teacher_math",
            0,
            lambda: evaluate_math(evaluated_teacher, evaluations, evaluation_config, teacher_evaluation_batch_size),
        )
    evaluate_student(student, 0)

    def report_on_policy_step(report: StepReport) -> None:
        metrics = report.metrics
        num_completed_steps = metrics.step + 1
        reverse_kl = "" if metrics.reverse_kl is None else f"reverse KL {metrics.reverse_kl:.4f}, "
        mean_reward = (
            ""
            if metrics.mean_reward is None
            else f"mean reward {metrics.mean_reward:.3f}, informative groups {metrics.informative_group_fraction}, "
        )
        log(
            f"step {num_completed_steps}/{num_steps}: {reverse_kl}{mean_reward}"
            f"gradient norm {metrics.gradient_norm:.3f}, adapter norm {metrics.adapter_norm:.3f}, "
            f"mean length {metrics.mean_response_length:.0f}, finished {metrics.finished_fraction:.2f}, "
            f"seconds sampling/teacher/student/gradients {metrics.sampling_seconds:.0f}/"
            f"{metrics.teacher_scoring_seconds:.0f}/{metrics.student_scoring_seconds:.0f}/"
            f"{metrics.gradient_seconds:.0f}, peak GB student/teacher "
            f"{metrics.student_peak_memory_gigabytes}/{metrics.teacher_peak_memory_gigabytes}"
        )
        if wandb_run is not None and isinstance(training_config, OnPolicyConfig):
            wandb_run.log(
                wandb_step_metrics(report, training_config, reasoning_end_token_id),
                step=num_completed_steps,
            )
            if num_completed_steps % rollout_table_every_steps == 0 or num_completed_steps == 1:
                wandb_run.log({"rollouts/samples": wandb_rollout_table(report, 8, 2000)}, step=num_completed_steps)
        evaluate_if_due(report.student, num_completed_steps)

    def report_supervised_step(report: SupervisedStepReport) -> None:
        metrics = report.metrics
        num_completed_steps = metrics.step + 1
        log(
            f"step {num_completed_steps}/{num_steps}: cross-entropy {metrics.cross_entropy:.4f}, "
            f"gradient norm {metrics.gradient_norm:.3f}, adapter norm {metrics.adapter_norm:.3f}, "
            f"mean length {metrics.mean_response_length:.0f}, finished {metrics.finished_fraction:.2f}, "
            f"seconds student/gradients {metrics.student_scoring_seconds:.0f}/{metrics.gradient_seconds:.0f}, "
            f"peak GB student {metrics.student_peak_memory_gigabytes}"
        )
        if wandb_run is not None and isinstance(training_config, SupervisedConfig):
            wandb_run.log(wandb_supervised_step_metrics(report, training_config), step=num_completed_steps)
        evaluate_if_due(report.student, num_completed_steps)

    if isinstance(training_config, SupervisedConfig):
        if stored_responses is None:
            raise ValueError("The supervised method trains on teacher responses, so it needs --teacher-responses.")
        trained_student = run_supervised_fine_tuning(
            student,
            [
                SupervisedExample(prompt=stored.pair.student, response=stored.response[:max_response_length])
                for stored in stored_responses
            ],
            training_config,
            execution_config,
            output_directory,
            report_supervised_step,
        )
    else:
        reward_model = student

        def reward(prompt_index: int, response: Sequence[int]) -> float:
            return math_reward(reward_model, training_prompts[prompt_index].problem, response)

        trained_student = run_on_policy_training(
            student,
            teacher,
            [prompt.pair for prompt in training_prompts],
            reward if uses_reward else None,
            training_config,
            execution_config,
            output_directory,
            report_on_policy_step,
        )
    evaluate_student(trained_student, num_steps)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
