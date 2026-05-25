import json
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, StrEnum
from itertools import zip_longest
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import requests
import thefuzz.fuzz
import thefuzz.process
from datasets import load_dataset

from lalamo.data.utils import pad_sequences
from lalamo.model_import import ModelSpec
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FileSpec,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
    import_model,
)
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile
from lalamo.models.chat_codec import AssistantMessage, Message, UserMessage
from lalamo.models.language_model import GenerationConfig, LanguageModel
from lalamo.module import Keychain
from lalamo.speculator.common import load_speculator


@dataclass
class PullCallbacks:
    model_spec: RegistryModel
    output_dir: Path
    overwrite: bool

    def started(self) -> None:
        pass

    def output_dir_exists(self) -> None:
        raise RuntimeError(f"{self.output_dir=} already exists, refusing to overwrite!")

    def downloading(self, file_spec: RegistryModelFile) -> None:
        pass

    def finished_downloading(self, file_spec: RegistryModelFile) -> None:
        pass

    def finished(self) -> None:
        pass


def _download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def _suggest_similar_models(query: str, repo_ids: list[str], limit: int = 3, min_score: int = 70) -> str:
    ranked_matches = thefuzz.process.extract(query, repo_ids, limit=limit, scorer=thefuzz.fuzz.ratio)
    similar_repos = [repo for repo, score in ranked_matches if score >= min_score]
    if not similar_repos:
        return ""
    return "\n\nDid you mean one of these?\n" + "\n".join(f"  - {repo}" for repo in similar_repos)


class EvalDatasetName(StrEnum):
    MTBENCH = "mtbench"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MATH500 = "math500"


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    category: str
    turns: tuple[str, ...]


@dataclass(frozen=True)
class EvalStats:
    count: int
    tokens: int
    steps: int
    acceptance_length_sum: float
    elapsed_seconds: float

    @property
    def tokens_per_step(self) -> float:
        return self.acceptance_length_sum / max(self.count, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / max(self.elapsed_seconds, 1e-9)

    @property
    def mean_draft_accepted(self) -> float:
        return max(self.tokens_per_step - 1.0, 0.0)

    @property
    def speculation_rate(self) -> float:
        return max(self.tokens - self.steps, 0) / max(self.tokens, 1)


@dataclass(frozen=True)
class EvalConfig:
    dataset_names: tuple[EvalDatasetName, ...]
    model_path: Path
    speculator_path: Path | None
    num_questions: int
    batch_size: int
    max_output_length: int
    reasoning: bool
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    padded_length: int
    seed: int
    warmup: bool
    mtbench_cache_path: Path


@dataclass(frozen=True)
class EvalResults:
    config: EvalConfig
    by_category: dict[str, EvalStats]
    elapsed_seconds: float

    @property
    def total_count(self) -> int:
        return sum(stats.count for stats in self.by_category.values())

    @property
    def tokens(self) -> int:
        return sum(stats.tokens for stats in self.by_category.values())

    @property
    def steps(self) -> int:
        return sum(stats.steps for stats in self.by_category.values())

    @property
    def tokens_per_step(self) -> float:
        acceptance_length_sum = sum(stats.acceptance_length_sum for stats in self.by_category.values())
        return acceptance_length_sum / max(self.total_count, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / max(self.elapsed_seconds, 1e-9)

    @property
    def mean_draft_accepted(self) -> float:
        return max(self.tokens_per_step - 1.0, 0.0)

    @property
    def speculation_rate(self) -> float:
        return max(self.tokens - self.steps, 0) / max(self.tokens, 1)


@dataclass(frozen=True)
class EvalCounts:
    count: int = 0
    tokens: int = 0
    steps: int = 0
    acceptance_length_sum: float = 0.0

    def add(self, tokens: int, steps: int) -> "EvalCounts":
        return EvalCounts(
            count=self.count + 1,
            tokens=self.tokens + tokens,
            steps=self.steps + steps,
            acceptance_length_sum=self.acceptance_length_sum + tokens / max(steps, 1),
        )


def load_eval_questions(
    names: tuple[EvalDatasetName, ...],
    num_questions: int | None,
    mtbench_cache_path: Path,
) -> list[EvalQuestion]:
    def mtbench() -> list[EvalQuestion]:
        mtbench_url = (
            "https://raw.githubusercontent.com/lm-sys/FastChat/"
            "587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl"
        )
        if not mtbench_cache_path.exists():
            mtbench_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _download_file(mtbench_url, mtbench_cache_path)
        with mtbench_cache_path.open() as file:
            rows = [json.loads(line) for line in file]
        return [
            EvalQuestion(
                id=int(row["question_id"]),
                category=str(row["category"]),
                turns=tuple(str(turn) for turn in row["turns"]),
            )
            for row in rows
        ]

    def gsm8k() -> list[EvalQuestion]:
        rows = cast("Iterable[dict[str, object]]", load_dataset("openai/gsm8k", "main", split="test"))
        return [EvalQuestion(id=idx, category="math", turns=(str(row["question"]),)) for idx, row in enumerate(rows)]

    def humaneval() -> list[EvalQuestion]:
        rows = cast("Iterable[dict[str, object]]", load_dataset("openai/openai_humaneval", split="test"))
        return [EvalQuestion(id=idx, category="code", turns=(str(row["prompt"]),)) for idx, row in enumerate(rows)]

    def math500() -> list[EvalQuestion]:
        rows = cast("Iterable[dict[str, object]]", load_dataset("HuggingFaceH4/MATH-500", split="test"))
        return [
            EvalQuestion(id=idx, category=str(row["subject"]), turns=(str(row["problem"]),))
            for idx, row in enumerate(rows)
        ]

    loaders = {
        EvalDatasetName.MTBENCH: mtbench,
        EvalDatasetName.GSM8K: gsm8k,
        EvalDatasetName.HUMANEVAL: humaneval,
        EvalDatasetName.MATH500: math500,
    }
    source_questions = tuple((name, loaders[name]()) for name in names)
    groups = (
        tuple(
            EvalQuestion(id=question.id, category=f"{name.value}/{question.category}", turns=question.turns)
            for question in questions
        )
        for name, questions in source_questions
    )
    questions = [
        EvalQuestion(id=idx, category=question.category, turns=question.turns)
        for idx, question in enumerate(
            question for row in zip_longest(*groups) for question in row if question is not None
        )
    ]

    return questions if num_questions is None else questions[:num_questions]


def evaluate_speculator(
    model_path: Path,
    dataset_names: tuple[EvalDatasetName, ...],
    speculator_path: Path | None,
    mtbench_cache_path: Path,
    num_questions: int | None = None,
    batch_size: int = 32,
    max_output_length: int = 2048,
    reasoning: bool = True,
    temperature: float = 1.0,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
    seed: int = 0,
    warmup: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvalResults:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if max_output_length < 1:
        raise ValueError("max_output_length must be at least 1.")

    questions = load_eval_questions(dataset_names, num_questions, mtbench_cache_path)
    if not questions:
        raise ValueError("Evaluation dataset is empty.")
    total_questions = len(questions)
    total_generations = sum(len(question.turns) for question in questions)
    if progress_callback is not None:
        progress_callback(0, total_generations)

    model = LanguageModel.load(model_path)
    speculator = load_speculator(speculator_path, model.decoder) if speculator_path is not None else None
    generation_config = GenerationConfig(
        stop_token_ids=model.config.generation_config.stop_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    max_padded_length = 0
    by_category: dict[str, EvalCounts] = {}

    def response_token_ids(token_ids: list[int]) -> list[int]:
        stop_token_ids = set(generation_config.stop_token_ids)
        if not stop_token_ids:
            return token_ids
        response_length = next(
            (idx for idx, token_id in enumerate(token_ids) if token_id in stop_token_ids),
            len(token_ids),
        )
        return token_ids[:response_length]

    def uses_reasoning(category: str) -> bool:
        return reasoning and not category.startswith(f"{EvalDatasetName.MTBENCH.value}/")

    def run_generation_batch(
        messages: list[list[Message]],
        categories: list[str],
        seed_offset: int,
        *,
        collect_counts: bool,
    ) -> list[AssistantMessage]:
        nonlocal max_padded_length
        tokenized = [
            model.token_codec.encode_request(
                message,
                enable_thinking=uses_reasoning(category),
            )
            for message, category in zip(messages, categories, strict=True)
        ]
        padded_length = max(len(tokens) for tokens in tokenized)
        max_padded_length = max(max_padded_length, padded_length)
        prompt_lengths_without_padding = jnp.asarray([len(tokens) for tokens in tokenized], dtype=jnp.int32)
        prompt_token_ids, _ = pad_sequences(tokenized, pad_token_id=0, padded_length=padded_length)
        batch_results = model.generate_tokens(
            prompt_token_ids,
            generation_config=generation_config,
            prompt_lengths_without_padding=prompt_lengths_without_padding,
            max_output_length=max_output_length,
            speculator=speculator,
            keychain=Keychain.init(seed + seed_offset, shape=(len(messages),)),
        )
        counts = jax.device_get(batch_results.num_tokens_per_step)
        if collect_counts:
            token_counts = counts.sum(axis=0).tolist()
            step_counts = (counts > 0).sum(axis=0).tolist()
            for category, token_count, step_count in zip(categories, token_counts, step_counts, strict=True):
                tokens = int(token_count)
                steps = int(step_count)
                by_category[category] = by_category.get(category, EvalCounts()).add(
                    tokens=tokens,
                    steps=steps,
                )

        responses: list[AssistantMessage] = []
        for token_ids, category in zip(jax.device_get(batch_results.token_ids).tolist(), categories, strict=True):
            response = model.token_codec.decode_response(
                response_token_ids(token_ids),
                expect_thinking=uses_reasoning(category),
            )
            responses.append(response)
        return responses

    def run_question_batch(
        batch_questions: list[EvalQuestion],
        seed_offset: int,
        *,
        collect_counts: bool,
    ) -> int:
        completed_generations = 0
        conversations: list[list[Message]] = [[] for _ in batch_questions]
        max_turns = max(len(question.turns) for question in batch_questions)
        for turn_index in range(max_turns):
            active_indices = [
                question_index
                for question_index, question in enumerate(batch_questions)
                if turn_index < len(question.turns)
            ]
            if not active_indices:
                continue
            active_messages: list[list[Message]] = []
            active_categories: list[str] = []
            for question_index in active_indices:
                question = batch_questions[question_index]
                conversations[question_index].append(UserMessage(question.turns[turn_index]))
                active_messages.append(conversations[question_index])
                active_categories.append(question.category)

            responses = run_generation_batch(
                active_messages,
                active_categories,
                seed_offset + completed_generations,
                collect_counts=collect_counts,
            )
            for question_index, response in zip(active_indices, responses, strict=True):
                conversations[question_index].append(response)
            completed_generations += len(active_indices)
        return completed_generations

    if warmup:
        warmup_questions = questions[: min(batch_size, total_questions)]
        run_question_batch(warmup_questions, total_questions, collect_counts=False)

    started_at = time.perf_counter()
    completed = 0
    for batch_start in range(0, total_questions, batch_size):
        batch_questions = questions[batch_start : batch_start + batch_size]
        completed += run_question_batch(batch_questions, completed, collect_counts=True)
        if progress_callback is not None:
            progress_callback(completed, total_generations)
    elapsed_seconds = time.perf_counter() - started_at

    return EvalResults(
        config=EvalConfig(
            dataset_names=dataset_names,
            model_path=model_path,
            speculator_path=speculator_path,
            num_questions=total_questions,
            batch_size=batch_size,
            max_output_length=max_output_length,
            reasoning=reasoning,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            padded_length=max_padded_length,
            seed=seed,
            warmup=warmup,
            mtbench_cache_path=mtbench_cache_path,
        ),
        by_category={
            category: EvalStats(
                count=counts.count,
                tokens=counts.tokens,
                steps=counts.steps,
                acceptance_length_sum=counts.acceptance_length_sum,
                elapsed_seconds=elapsed_seconds,
            )
            for category, counts in by_category.items()
        },
        elapsed_seconds=elapsed_seconds,
    )


def pull(
    model_spec: RegistryModel,
    output_dir: Path,
    callbacks_type: Callable[
        [
            RegistryModel,
            Path,
            bool,
        ],
        PullCallbacks,
    ] = PullCallbacks,
    overwrite: bool = False,
) -> None:
    callbacks = callbacks_type(model_spec, output_dir, overwrite)

    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for file_spec in model_spec.files:
            callbacks.downloading(file_spec)

            # Security: validate filename to prevent path traversal attacks
            safe_name = Path(file_spec.name).name
            if not safe_name or safe_name != file_spec.name:
                raise RuntimeError(
                    f"Invalid filename from registry: {file_spec.name!r}. "
                    f"Filenames must not contain path separators or traversal sequences.",
                )

            file_path = temp_path / safe_name
            try:
                _download_file(file_spec.url, file_path)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download {safe_name}: {e}") from e

            callbacks.finished_downloading(file_spec)

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in model_spec.files:
            safe_name = Path(file_spec.name).name
            src = temp_path / safe_name
            dst = output_dir / safe_name
            shutil.move(str(src), str(dst))

    callbacks.finished()


class DType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


@dataclass
class ConversionCallbacks:
    model_spec: ModelSpec
    output_dir: Path
    dtype: DType | None
    context_length: int | None

    def started(self) -> None:
        pass

    def output_dir_exists(self) -> None:
        raise RuntimeError(f"{self.output_dir=} already exists, refusing to overwrite!")

    def downloading(self, file_spec: FileSpec) -> None:
        pass

    def finished_downloading(self, file_spec: FileSpec) -> None:
        pass

    def initializing_model(self) -> None:
        pass

    def finished_initializing_model(self) -> None:
        pass

    def saving_model(self) -> None:
        pass

    def finished_saving_model(self) -> None:
        pass


def convert(
    model_spec: ModelSpec,
    output_dir: Path,
    dtype: DType | None = None,
    context_length: int | None = None,
    callbacks_type: Callable[
        [
            ModelSpec,
            Path,
            DType | None,
            int | None,
        ],
        ConversionCallbacks,
    ] = ConversionCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_spec,
        output_dir,
        dtype,
        context_length,
    )

    if output_dir.exists():
        callbacks.output_dir_exists()

    callbacks.started()

    def progress_callback(event: StatusEvent) -> None:
        match event:
            case DownloadingFileEvent(file_spec):
                callbacks.downloading(file_spec)
            case FinishedDownloadingFileEvent(file_spec):
                callbacks.finished_downloading(file_spec)
            case InitializingModelEvent():
                callbacks.initializing_model()
            case FinishedInitializingModelEvent():
                callbacks.finished_initializing_model()

    import_dtype = None
    if dtype is not None:
        import_dtype = jnp.dtype(dtype.value)

    imported_model = import_model(
        model_spec,
        dtype=import_dtype,
        context_length=context_length,
        progress_callback=progress_callback,
    )
    callbacks.saving_model()
    imported_model.model.save(output_dir)

    callbacks.finished_saving_model()
