import dataclasses
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import chain, zip_longest
from pathlib import Path

import jax
import numpy as np
import polars as pl
import requests
import thefuzz.fuzz
import thefuzz.process
from datasets import load_dataset
from jaxtyping import DTypeLike

from lalamo.common import flatten_parameters, get_default_device_bytes
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import iter_completions, save_completions
from lalamo.message_processor import AssistantMessage, Message, UserMessage
from lalamo.model_import import ModelMetadata, ModelSpec, import_model
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FileSpec,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
)
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile
from lalamo.models import GenerationConfig, LanguageModelConfig
from lalamo.models.common import BatchSizesComputedEvent, InferenceConfig
from lalamo.models.completion_feature_extractor import OnlineCompletionFeatureExtractor
from lalamo.models.lm_helpers import estimate_batchsize_from_bytes
from lalamo.modules import config_converter
from lalamo.modules.common import ShardingConfig, use_sharding
from lalamo.safetensors import safe_write
from lalamo.speculator.common import (
    SpeculatorBackend,
    load_speculator,
)
from lalamo.speculator.inference import CollectTracesEvent, inference_collect_traces
from lalamo.speculator.training import SpeculatorTrainingConfig, SpeculatorTrainingEvent, train_speculator


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


class EvalDatasetName(str, Enum):
    MTBENCH = "mtbench"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"
    MATH500 = "math500"
    MERGED = "merged"


@dataclass(frozen=True)
class EvalQuestion:
    id: int
    category: str
    prompt: str


@dataclass(frozen=True)
class EvalStats:
    count: int
    tokens: int
    steps: int
    elapsed_seconds: float

    @property
    def tokens_per_step(self) -> float:
        return self.tokens / max(self.steps, 1)

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
    dataset_name: EvalDatasetName
    model_path: Path
    speculator_path: Path | None
    num_questions: int
    batch_size: int
    max_output_length: int
    padded_length: int
    seed: int
    warmup: bool
    reasoning: bool
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
        return self.tokens / max(self.steps, 1)

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

    def add(self, tokens: int, steps: int) -> "EvalCounts":
        return EvalCounts(
            count=self.count + 1,
            tokens=self.tokens + tokens,
            steps=self.steps + steps,
        )


def load_eval_questions(
    name: EvalDatasetName,
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
                prompt=str(row["turns"][0]),
            )
            for row in rows
        ]

    def gsm8k() -> list[EvalQuestion]:
        return [
            EvalQuestion(id=idx, category="math", prompt=str(row["question"]))
            for idx, row in enumerate(load_dataset("openai/gsm8k", "main", split="test"))
        ]

    def humaneval() -> list[EvalQuestion]:
        return [
            EvalQuestion(id=idx, category="code", prompt=str(row["prompt"]))
            for idx, row in enumerate(load_dataset("openai/openai_humaneval", split="test"))
        ]

    def math500() -> list[EvalQuestion]:
        return [
            EvalQuestion(id=idx, category=str(row["subject"]), prompt=str(row["problem"]))
            for idx, row in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        ]

    def merged() -> list[EvalQuestion]:
        sources = (
            ("gsm8k", gsm8k()),
            ("mtbench", mtbench()),
            ("math500", math500()),
        )
        groups = (
            tuple(
                EvalQuestion(id=question.id, category=f"{source}/{question.category}", prompt=question.prompt)
                for question in questions
            )
            for source, questions in sources
        )
        return [
            EvalQuestion(id=idx, category=question.category, prompt=question.prompt)
            for idx, question in enumerate(
                question for row in zip_longest(*groups) for question in row if question is not None
            )
        ]

    match name:
        case EvalDatasetName.MTBENCH:
            questions = mtbench()
        case EvalDatasetName.GSM8K:
            questions = gsm8k()
        case EvalDatasetName.HUMANEVAL:
            questions = humaneval()
        case EvalDatasetName.MATH500:
            questions = math500()
        case EvalDatasetName.MERGED:
            questions = merged()

    return questions if num_questions is None else questions[:num_questions]


def eval_padded_length(tokenized: list[list[int]]) -> int:
    max_prompt_length = max(len(tokens) for tokens in tokenized)
    lengths = tuple(256 * 2**i for i in range(12))
    for length in lengths:
        if max_prompt_length <= length:
            return length
    raise ValueError(f"Prompt length {max_prompt_length} exceeds largest supported bucket {lengths[-1]}.")


def evaluate_speculator(
    model_path: Path,
    dataset_name: EvalDatasetName,
    speculator_path: Path | None,
    mtbench_cache_path: Path,
    num_questions: int | None = None,
    batch_size: int = 32,
    max_output_length: int = 4096,
    seed: int = 0,
    warmup: bool = True,
    reasoning: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvalResults:
    questions = load_eval_questions(dataset_name, num_questions, mtbench_cache_path)
    if not questions:
        raise ValueError("Evaluation dataset is empty.")
    total_questions = len(questions)
    if progress_callback is not None:
        progress_callback(0, total_questions)

    sharding_config = ShardingConfig.build()
    with use_sharding(sharding_config):
        model = LanguageModelConfig.load_model(model_path)
        speculator = load_speculator(speculator_path, model.model) if speculator_path is not None else None

    tokenized = model.message_processor.tokenize_requests(
        ([UserMessage(question.prompt)] for question in questions),
        enable_thinking=reasoning,
    )
    padded_length = eval_padded_length(tokenized)
    inference_config = InferenceConfig(
        max_output_length=max_output_length,
        padded_length=padded_length,
        batch_size=batch_size,
    )
    keys = jax.random.split(jax.random.key(seed), total_questions)
    by_category: dict[str, EvalCounts] = {}

    with use_sharding(sharding_config):
        if warmup:
            warmup_results = model.generate_tokens_many(
                tokenized[:1],
                inference_config=inference_config,
                speculator=speculator,
                sharding_config=sharding_config,
                keys=keys[:1],
            )
            for result in warmup_results:
                jax.device_get(result.tokens_per_step)

        started_at = time.perf_counter()
        results = model.generate_tokens_many(
            tokenized,
            inference_config=inference_config,
            speculator=speculator,
            sharding_config=sharding_config,
            keys=keys,
        )
        for completed, (question, result) in enumerate(zip(questions, results, strict=True), start=1):
            tokens_per_step = np.asarray(jax.device_get(result.tokens_per_step))
            num_tokens = int(tokens_per_step.sum())
            num_steps = int(np.sum(tokens_per_step > 0))
            by_category[question.category] = by_category.get(question.category, EvalCounts()).add(
                tokens=num_tokens,
                steps=num_steps,
            )
            if progress_callback is not None:
                progress_callback(completed, total_questions)
        elapsed_seconds = time.perf_counter() - started_at

    return EvalResults(
        config=EvalConfig(
            dataset_name=dataset_name,
            model_path=model_path,
            speculator_path=speculator_path,
            num_questions=total_questions,
            batch_size=batch_size,
            max_output_length=max_output_length,
            padded_length=padded_length,
            seed=seed,
            warmup=warmup,
            reasoning=reasoning,
            mtbench_cache_path=mtbench_cache_path,
        ),
        by_category={
            category: EvalStats(
                count=counts.count,
                tokens=counts.tokens,
                steps=counts.steps,
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


class Precision(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


@dataclass
class ConversionCallbacks:
    model_spec: ModelSpec
    output_dir: Path
    precision: Precision | None
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
    precision: Precision | None = None,
    context_length: int | None = None,
    callbacks_type: Callable[
        [
            ModelSpec,
            Path,
            Precision | None,
            int | None,
        ],
        ConversionCallbacks,
    ] = ConversionCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_spec,
        output_dir,
        precision,
        context_length,
    )

    if precision is not None:
        precision_dtype = config_converter.structure(precision.value, DTypeLike)  # type: ignore
    else:
        precision_dtype = None

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

    model, metadata = import_model(
        model_spec,
        precision=precision_dtype,
        context_length=context_length,
        progress_callback=progress_callback,
    )
    callbacks.saving_model()
    output_dir.mkdir(parents=True, exist_ok=True)

    model.message_processor.tokenizer.save(str(output_dir / "tokenizer.json"))
    weights = flatten_parameters(model.export_weights())
    del model

    with Path(output_dir / "model.safetensors").open("wb") as fd:
        safe_write(fd, weights)

    config_json = config_converter.unstructure(metadata, ModelMetadata)
    with open(output_dir / "config.json", "w") as file:
        json.dump(config_json, file, indent=4)

    callbacks.finished_saving_model()


@dataclass
class TraceCallbacks:
    model_path: Path
    output_path: Path
    messages: Iterable[Message] | None

    def output_exists(self) -> None:
        raise RuntimeError(f"{self.output_path=} already exists, refusing to overwrite!")

    def started(self) -> None:
        pass

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def tracing_model(self) -> None:
        pass

    def finished_tracing_model(self) -> None:
        pass

    def saving_trace(self) -> None:
        pass

    def finished_saving_trace(self) -> None:
        pass


def trace(
    model_path: Path,
    output_path: Path,
    messages: Iterable[Message] | None = None,
    callbacks_type: Callable[
        [
            Path,
            Path,
            Iterable[Message] | None,
        ],
        TraceCallbacks,
    ] = TraceCallbacks,
) -> None:
    callbacks = callbacks_type(model_path, output_path, messages)

    if output_path.exists():
        callbacks.output_exists()

    callbacks.started()

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    callbacks.tracing_model()
    result = model.record_trace(messages)
    callbacks.finished_tracing_model()

    callbacks.saving_trace()
    traces = flatten_parameters(result.export())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("wb") as fd:
        safe_write(fd, traces)
    callbacks.finished_saving_trace()


@dataclass
class EstimateBatchsizeCallbacks:
    model_path: Path
    max_input_length: int
    max_output_length: int
    mem: int

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def estimating_batchsize(self, lo: int, hi: int | None) -> None:
        pass

    def finished_estimating_batchsize(self, batchsize: int) -> None:
        pass


def estimate_batchsize(
    model_path: Path,
    mem: int,
    max_input_length: int = 1024,
    max_output_length: int = 4096,
    callbacks_type: Callable[
        [
            Path,
            int,
            int,
            int,
        ],
        EstimateBatchsizeCallbacks,
    ] = EstimateBatchsizeCallbacks,
) -> int:
    callbacks = callbacks_type(model_path, max_input_length, max_output_length, mem)

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    def memory_per_batchsize(batch_size: int) -> int:
        inference_config = InferenceConfig(
            max_output_length=max_output_length,
            padded_length=max_input_length,
            batch_size=batch_size,
        )
        return model.estimate_memory_consumption(
            inference_config=inference_config,
        )

    bs = estimate_batchsize_from_bytes(
        memory_per_batchsize,
        mem,
        lambda event: callbacks.estimating_batchsize(event.lo, event.hi),
    )

    callbacks.finished_estimating_batchsize(bs)
    return bs


@dataclass
class CollectTracesCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path
    max_input_length: int
    max_output_length: int
    batch_size: int
    num_tokens_to_generate: int | None

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def inference_progress(self, event: CollectTracesEvent) -> None:
        pass

    def finished_inference(self) -> None:
        pass


def collect_traces(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    max_input_length: int = 1024,
    max_output_length: int = 4096,
    batch_size: int = 1,
    num_tokens_to_generate: int | None = None,
    callbacks_type: Callable[..., CollectTracesCallbacks] = CollectTracesCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        max_input_length,
        max_output_length,
        batch_size,
        num_tokens_to_generate,
    )

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path))
    conversations = dataframe.get_column("conversation")
    dataset = iter(
        [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
    )
    dataset = chain([next(dataset)], dataset)  # iterator is lazy, force it to actually open the file
    callbacks.finished_loading_dataset()

    def progress_callback(event: CollectTracesEvent) -> None:
        callbacks.inference_progress(event)

    traces = inference_collect_traces(
        model,
        dataset,
        batch_size,
        max_input_length,
        max_output_length,
        num_tokens_to_generate,
        progress_callback,
    )

    if output_path.exists():
        raise RuntimeError(f"{output_path} must not exist.")
    save_completions(output_path, traces)

    callbacks.finished_inference()


@dataclass
class TrainCallbacks:
    backend_name: str
    model_path: Path
    train_path: Path
    eval_path: Path | None
    output_path: Path
    training_config: SpeculatorTrainingConfig

    def started(self) -> None:
        pass

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def training_progress(self, event: SpeculatorTrainingEvent) -> None:
        pass

    def finished_training(self) -> None:
        pass


def train[ConfigT](
    backend: type[SpeculatorBackend[ConfigT]],
    backend_config: ConfigT,
    model_path: Path,
    train_path: Path,
    output_path: Path,
    eval_path: Path | None = None,
    feature_device_id: int = 0,
    training_device_id: int = 0,
    prompt_padding_multiple: int = 128,
    generation_padding_multiple: int = 512,
    training_config: SpeculatorTrainingConfig = SpeculatorTrainingConfig(),  # noqa: B008
    callbacks_type: Callable[
        [
            str,
            Path,
            Path,
            Path | None,
            Path,
            SpeculatorTrainingConfig,
        ],
        TrainCallbacks,
    ] = TrainCallbacks,
) -> None:
    callbacks = callbacks_type(
        backend.name,
        model_path,
        train_path,
        eval_path,
        output_path,
        training_config,
    )

    callbacks.started()

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    trainer = backend.create_trainer(
        backend_config,
        output_path,
        model.model,
    )
    extractor = OnlineCompletionFeatureExtractor(
        model=model,
        device_id=feature_device_id,
        prompt_padding_multiple=prompt_padding_multiple,
        generation_padding_multiple=generation_padding_multiple,
    )

    if eval_path is not None:
        eval_completions_path = eval_path

        def eval_completions() -> Iterable:
            return iter_completions(eval_completions_path)

    else:
        eval_completions = None

    train_speculator(
        trainer=trainer,
        extractor=extractor,
        train_completions=lambda: iter_completions(train_path),
        eval_completions=eval_completions,
        config=training_config,
        training_device_id=training_device_id,
        progress_callback=callbacks.training_progress,
    )

    callbacks.finished_training()


@dataclass
class GenerateRepliesCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path
    max_vram: int | None
    batch_size: int | None
    total_rows: int

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def estimating_batchsize(self, sequence_length: int, lo: int, hi: int | None) -> None:
        pass

    def batch_sizes_estimated(self) -> None:
        pass

    def batch_sizes_computed(self, event: BatchSizesComputedEvent) -> None:
        pass

    def generation_progress(self, rows_processed: int) -> None:
        pass

    def finished_generation(self) -> None:
        pass


def generate_replies(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    max_vram: int | None,
    max_output_length: int = 8192,
    batch_size: int | None = None,
    generation_config_override: GenerationConfig | None = None,
    speculator_path: Path | None = None,
    callbacks_type: Callable[
        [
            Path,
            Path,
            Path,
            int | None,
            int | None,
            int,
        ],
        GenerateRepliesCallbacks,
    ] = GenerateRepliesCallbacks,
) -> None:
    """Generate replies for every conversation in a dataset.

    Loads a locally-converted model from ``model_path``, reads a Parquet dataset from ``dataset_path`` (expected to
    have a ``conversation`` column of HuggingFace-style message lists), and writes a Parquet file to ``output_path``
    with ``response`` and ``chain_of_thought`` columns.

    Exactly one of ``max_vram`` or ``batch_size`` should be provided.  When ``max_vram`` is given (in bytes), batch
    sizes are estimated automatically per sequence length.  If neither is set, an estimate for the device memory is
    used.  Prefer setting ``max_output_length`` to a value no larger than you actually need, since it directly affects
    memory consumption and therefore the batch sizes that fit in VRAM.

    If ``generation_config_override`` is provided it replaces the model's default generation config entirely
    (temperature, top-k, etc.).  Do not set ``stop_token_ids`` on it: the model's own stop tokens are always
    injected automatically, and providing them will throw a ValueError.

    ``callbacks_type`` is used internally (cli) for progress visualisation.
    """
    sharding_config = ShardingConfig.build()

    # figure out max_vram if neither batch_size nor max_vram is set
    if max_vram is None and batch_size is None:
        max_vram = get_default_device_bytes()
        if max_vram is None:
            raise ValueError(
                "Unable to determine default device memory capacity; please specify either --vram-gb or --batch-size",
            )

    # Count rows without loading full dataset
    total_rows: int = pl.scan_parquet(dataset_path).select(pl.len()).collect().item()

    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        max_vram,
        batch_size,
        total_rows,
    )

    callbacks.loading_model()
    with use_sharding(sharding_config):
        model = LanguageModelConfig.load_model(model_path)
        speculator = load_speculator(speculator_path, model.model) if speculator_path is not None else None
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataframe: pl.DataFrame = load_hf_parquet(dataset_path).collect()
    conversations = dataframe.get_column("conversation")
    dataset = iter(
        [HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations
    )
    try:
        first_row = next(dataset)
    except StopIteration:
        callbacks.finished_loading_dataset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"response": [], "chain_of_thought": []}).write_parquet(output_path)
        return
    dataset = chain([first_row], dataset)  # iterator is lazy, force it to actually open the file
    callbacks.finished_loading_dataset()

    inference_config = InferenceConfig(max_output_length=max_output_length, batch_size=batch_size)

    callbacks.batch_sizes_estimated()

    generation_config = None
    if generation_config_override is not None:
        if generation_config_override.stop_token_ids:
            raise ValueError(
                "Do not set generation_config.stop_token_ids for this command; "
                "the model's configured stop tokens are always used instead.",
            )

        generation_config = dataclasses.replace(
            generation_config_override,
            stop_token_ids=model.config.generation_config.stop_token_ids,
        )

    with use_sharding(sharding_config):
        replies: list[tuple[int, AssistantMessage]] = []
        for rows_processed, (idx, reply) in enumerate(
            model.reply_many(
                dataset,
                generation_config=generation_config,
                inference_config=inference_config,
                vram_bytes=max_vram,
                speculator=speculator,
                batch_sizes_callback=callbacks.batch_sizes_computed,
            ),
        ):
            replies.append((idx, reply))
            callbacks.generation_progress(rows_processed)

    # Sort by original index to restore input order
    replies.sort(key=lambda x: x[0])

    df = pl.DataFrame(
        {
            "response": [reply.response for _, reply in replies],
            "chain_of_thought": [reply.chain_of_thought for _, reply in replies],
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    callbacks.finished_generation()
