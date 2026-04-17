import dataclasses
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import ClassVar, Self

import cattrs
import polars as pl
import requests
import thefuzz.fuzz
import thefuzz.process
from datasets import load_dataset
from jaxtyping import DTypeLike

from lalamo.common import flatten_parameters, get_default_device_bytes
from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion, save_completions
from lalamo.message_processor import AssistantMessage, Message
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
from lalamo.models import GenerationConfig, GenerationTraceConfig, LanguageModelConfig
from lalamo.models.common import BatchSizesComputedEvent, InferenceConfig
from lalamo.models.lm_helpers import estimate_batchsize_from_bytes
from lalamo.modules import config_converter
from lalamo.modules.common import ShardingConfig, use_sharding
from lalamo.safetensors import safe_write
from lalamo.speculator.common import SamplerConfig, Speculator
from lalamo.speculator.eval import EvalQuestion, EvalResults, run_eval
from lalamo.speculator.inference import CollectTracesEvent, inference_collect_traces
from lalamo.speculator.speculate import SpeculativeDecodingResult


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


def download_file(url: str, dest_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=(10, 60))
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
                download_file(file_spec.url, file_path)
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
    num_logits_per_token: int
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
    max_output_length: int = 1024,
    num_logits_per_token: int = 1024,
    callbacks_type: Callable[
        [
            Path,
            int,
            int,
            int,
            int,
        ],
        EstimateBatchsizeCallbacks,
    ] = EstimateBatchsizeCallbacks,
) -> int:
    callbacks = callbacks_type(model_path, max_input_length, max_output_length, num_logits_per_token, mem)

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
            generation_trace_config=GenerationTraceConfig(
                num_logits_per_token=num_logits_per_token,
            ),
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
    num_logits_per_token: int
    trace_layers: tuple[int, ...]
    max_input_length: int
    max_output_length: int
    batch_size: int
    shard_size: int
    num_tokens_to_generate: int | None

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def inference_progress(self, tokens_generated: int) -> None:
        pass

    def finished_inference(self) -> None:
        pass


def collect_traces(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    num_logits_per_token: int = 1024,
    trace_layers: tuple[int, ...] = tuple(),
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    batch_size: int = 1,
    shard_size: int = 64,
    num_tokens_to_generate: int | None = None,
    callbacks_type: Callable[..., CollectTracesCallbacks] = CollectTracesCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        num_logits_per_token,
        trace_layers,
        max_input_length,
        max_output_length,
        batch_size,
        shard_size,
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
        callbacks.inference_progress(event.tokens_generated)

    traces = inference_collect_traces(
        model,
        dataset,
        num_logits_per_token,
        trace_layers,
        batch_size,
        max_input_length,
        max_output_length,
        num_tokens_to_generate,
        progress_callback,
    )

    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise RuntimeError(f"{output_path} must be an empty directory or not exist.")
    output_path.mkdir(parents=True, exist_ok=True)
    shard: list[LalamoCompletion] = []
    shard_idx = 0
    for trace in traces:
        shard.append(trace)
        if len(shard) >= shard_size:
            save_completions(output_path / f"part-{shard_idx:05d}.safetensors", shard)
            shard.clear()
            shard_idx += 1
    if shard:
        save_completions(output_path / f"part-{shard_idx:05d}.safetensors", shard)

    callbacks.finished_inference()


@dataclass
class TrainCallbacks:
    output_path: Path
    subsample_size: int | None

    def training_progress(self, trained_tokens: int) -> None:
        pass

    def finished_training(self) -> None:
        pass

    def saving(self) -> None:
        pass

    def finished_saving(self) -> None:
        pass

    def save_drafter(self, drafter_bytes: bytes) -> None:
        self.saving()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(drafter_bytes)
        self.finished_saving()


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
    total_rows: int = pl.scan_parquet(dataset_path).select(pl.len()).collect().item()  # type: ignore[possibly-missing-attribute]

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
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataframe: pl.DataFrame = load_hf_parquet(dataset_path).collect()  # type: ignore[possibly-missing-attribute]
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


class EvalDatasetName(str, Enum):
    MTBENCH = "mtbench"
    GSM8K = "gsm8k"
    HUMANEVAL = "humaneval"


@dataclass(frozen=True)
class MTBenchRow:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    question_id: int
    category: str
    turns: list[str]

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> Self:
        return cls._converter.structure(dict(obj), cls)


@dataclass(frozen=True)
class GSM8KRow:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    question: str
    answer: str

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> Self:
        return cls._converter.structure(dict(obj), cls)


@dataclass(frozen=True)
class HumanEvalRow:
    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    task_id: str
    prompt: str

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> Self:
        return cls._converter.structure(dict(obj), cls)


def load_mtbench(cache_path: Path) -> list[EvalQuestion]:
    # Pinned commit so upstream branch movement cannot silently 404 the eval.
    mtbench_url = (
        "https://raw.githubusercontent.com/lm-sys/FastChat/"
        "587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl"
    )
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(mtbench_url, cache_path)
    with open(cache_path) as f:
        rows = [MTBenchRow.from_dict(json.loads(line)) for line in f]
    return [EvalQuestion(id=row.question_id, category=row.category, prompt=row.turns[0]) for row in rows]


def load_gsm8k() -> list[EvalQuestion]:
    rows = [GSM8KRow.from_dict(row) for row in load_dataset("openai/gsm8k", "main", split="test")]
    return [EvalQuestion(id=idx, category="math", prompt=row.question) for idx, row in enumerate(rows)]


def load_humaneval() -> list[EvalQuestion]:
    rows = [HumanEvalRow.from_dict(row) for row in load_dataset("openai/openai_humaneval", split="test")]
    return [EvalQuestion(id=idx, category="code", prompt=row.prompt) for idx, row in enumerate(rows)]


def load_eval_questions(
    name: EvalDatasetName,
    num_questions: int | None,
    mtbench_cache_path: Path,
) -> list[EvalQuestion]:
    match name:
        case EvalDatasetName.MTBENCH:
            questions = load_mtbench(mtbench_cache_path)
        case EvalDatasetName.GSM8K:
            questions = load_gsm8k()
        case EvalDatasetName.HUMANEVAL:
            questions = load_humaneval()
    if num_questions is not None:
        questions = questions[:num_questions]
    return questions


@dataclass
class SpeculatorEvalCallbacks:
    model_path: Path
    speculator_path: Path
    dataset_name: EvalDatasetName

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_drafter(self) -> None:
        pass

    def finished_loading_drafter(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self, num_questions: int) -> None:
        pass

    def eval_started(self, num_questions: int) -> None:
        pass

    def eval_progress(
        self,
        question_idx: int,
        question: EvalQuestion,
        result: SpeculativeDecodingResult,
        elapsed_s: float,
    ) -> None:
        pass

    def finished_eval(self) -> None:
        pass


def speculator_eval(
    *,
    model_path: Path,
    speculator_path: Path,
    dataset_name: EvalDatasetName,
    num_questions: int | None,
    mtbench_cache_path: Path,
    sampler_config: SamplerConfig,
    drafter_name: str,
    callbacks_type: Callable[..., SpeculatorEvalCallbacks] = SpeculatorEvalCallbacks,
) -> EvalResults:
    callbacks = callbacks_type(model_path, speculator_path, dataset_name)

    callbacks.loading_model()
    llm = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    callbacks.loading_drafter()
    with open(speculator_path, "rb") as fd:
        speculator = Speculator.deserialize(
            drafter_name,
            fd.read(),
            decoder=llm.model,
            config=sampler_config,
            eos_set=frozenset(int(e) for e in llm.stop_token_ids),
            width=sampler_config.width,
            depth=sampler_config.K,
        )
    callbacks.finished_loading_drafter()

    callbacks.loading_dataset()
    questions = load_eval_questions(dataset_name, num_questions, mtbench_cache_path)
    callbacks.finished_loading_dataset(len(questions))

    callbacks.eval_started(len(questions))
    results = run_eval(
        speculator,
        llm.message_processor,
        questions,
        on_question=callbacks.eval_progress,
    )
    callbacks.finished_eval()

    return results
