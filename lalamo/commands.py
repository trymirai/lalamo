import json
import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path

import requests
import thefuzz.process
from jaxtyping import DTypeLike

from lalamo.common import flatten_parameters
from lalamo.data import import_hf_parquet
from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.message_processor import Message
from lalamo.model_import import ModelMetadata, ModelSpec, import_model
from lalamo.model_import.common import (
    DownloadingFileEvent,
    FileSpec,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
)
from lalamo.model_import.remote_registry import RemoteFileSpec, RemoteModelSpec, fetch_available_models
from lalamo.models import LanguageModelConfig
from lalamo.modules import config_converter
from lalamo.safetensors import safe_write
from lalamo.speculator.estimator import EstimateBatchsizeFromMemoryEvent, estimate_batchsize_from_memory
from lalamo.speculator.inference import CollectTracesEvent, inference_collect_traces
from lalamo.speculator.ngram import NGramSpeculator
from lalamo.speculator.utils import SpeculatorTrainingEvent, train_speculator


@dataclass
class PullCallbacks:
    model_spec: RemoteModelSpec
    output_dir: Path
    overwrite: bool

    def started(self) -> None:
        pass

    def output_dir_exists(self) -> None:
        raise RuntimeError(f"{self.output_dir=} already exists, refusing to overwrite!")

    def downloading(self, file_spec: RemoteFileSpec) -> None:
        pass

    def finished_downloading(self, file_spec: RemoteFileSpec) -> None:
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


def _match_model(query: str, available_models: list[RemoteModelSpec]) -> RemoteModelSpec | None:
    """Match user query to a model using exact and fuzzy matching. """
    # Try exact match on repo_id
    for model in available_models:
        if model.repo_id == query:
            return model

    # Try exact match on name
    for model in available_models:
        if model.name == query:
            return model

    # Try fuzzy matching on repo_id
    repo_ids = [m.repo_id for m in available_models]
    matches = thefuzz.process.extract(query, repo_ids, limit=1)
    if matches and matches[0][1] >= 80:
        matched_repo_id = matches[0][0]
        for model in available_models:
            if model.repo_id == matched_repo_id:
                return model

    return None


def _suggest_similar_models(query: str, available_models: list[RemoteModelSpec], limit: int = 3) -> list[str]:
    repo_ids = [m.repo_id for m in available_models]
    matches = thefuzz.process.extract(query, repo_ids, limit=limit)
    return [match[0] for match in matches if match[1] >= 50]


def pull(
    model_spec: RemoteModelSpec,
    output_dir: Path,
    callbacks_type: Callable[
        [RemoteModelSpec, Path, bool],
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

            file_path = temp_path / file_spec.name
            try:
                _download_file(file_spec.url, file_path)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download {file_spec.name}: {e}") from e

            callbacks.finished_downloading(file_spec)

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in model_spec.files:
            src = temp_path / file_spec.name
            dst = output_dir / file_spec.name
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
    num_logits_per_token: int = 8,
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

    def progress_callback(event: EstimateBatchsizeFromMemoryEvent) -> None:
        callbacks.estimating_batchsize(event.lo, event.hi)

    bs = estimate_batchsize_from_memory(
        model,
        max_input_length,
        max_output_length,
        num_logits_per_token,
        mem,
        progress_callback,
    )

    callbacks.finished_estimating_batchsize(bs)
    return bs


@dataclass
class CollectTracesCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path
    num_logits_per_token: int
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

    def inference_progress(self, tokens_generated: int) -> None:
        pass

    def finished_inference(self) -> None:
        pass


def collect_traces(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    num_logits_per_token: int = 8,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    batch_size: int = 1,
    num_tokens_to_generate: int | None = None,
    callbacks_type: Callable[
        [
            Path,
            Path,
            Path,
            int,
            int,
            int,
            int,
            int | None,
        ],
        CollectTracesCallbacks,
    ] = CollectTracesCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        num_logits_per_token,
        max_input_length,
        max_output_length,
        batch_size,
        num_tokens_to_generate,
    )

    callbacks.loading_model()
    model = LanguageModelConfig.load_model(model_path)
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataset = iter(import_hf_parquet(dataset_path))
    dataset = chain([next(dataset)], dataset)  # iterator is lazy, force it to actually open the file
    callbacks.finished_loading_dataset()

    def progress_callback(event: CollectTracesEvent) -> None:
        callbacks.inference_progress(event.tokens_generated)

    traces = inference_collect_traces(
        model,
        dataset,
        num_logits_per_token,
        batch_size,
        max_input_length,
        max_output_length,
        num_tokens_to_generate,
        progress_callback,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as output_fd:
        for trace in traces:
            blob = trace.serialize()
            output_fd.write(blob)

    callbacks.finished_inference()


@dataclass
class TrainCallbacks:
    trace_path: Path
    output_path: Path
    hashtable_size: int
    num_logits_per_token: int
    ngram_size: int
    subsample_size: int | None

    def started(self) -> None:
        pass

    def training_progress(self, trained_tokens: int) -> None:
        pass

    def finished_training(self) -> None:
        pass

    def saving_speculator(self) -> None:
        pass

    def finished_saving_speculator(self) -> None:
        pass


def train(
    trace_path: Path,
    output_path: Path,
    hashtable_size: int = 65536,
    num_logits_per_token: int = 8,
    ngram_size: int = 2,
    subsample_size: int | None = None,
    callbacks_type: Callable[
        [
            Path,
            Path,
            int,
            int,
            int,
            int | None,
        ],
        TrainCallbacks,
    ] = TrainCallbacks,
) -> None:
    callbacks = callbacks_type(
        trace_path,
        output_path,
        hashtable_size,
        num_logits_per_token,
        ngram_size,
        subsample_size,
    )

    callbacks.started()

    with open(trace_path, "rb") as trace_fd:
        traces = LalamoCompletion.deserialize_many(trace_fd)
        speculator = NGramSpeculator.new(hashtable_size, num_logits_per_token, ngram_size)

        def progress_callback(event: SpeculatorTrainingEvent) -> None:
            callbacks.training_progress(event.trained_tokens)

        train_speculator(speculator, traces, subsample_size, progress_callback)

    callbacks.finished_training()

    callbacks.saving_speculator()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fd:
        fd.write(speculator.serialize())
    callbacks.finished_saving_speculator()
