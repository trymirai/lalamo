import shutil
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import jax.numpy as jnp
import requests
import thefuzz.fuzz
import thefuzz.process

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
from lalamo.models.chat_codec import Message
from lalamo.safetensors import safe_write
from lalamo.trace_comparator import TraceComparison, compare_trace_files
from lalamo.tracer import (
    export_trace_result,
    load_traceable_model,
    load_traceable_token_codec,
    record_message_trace_with_tokenization,
    record_saved_token_trace,
    record_tokenization_trace,
)
from lalamo.utils.sharding import ShardingConfig


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
        sharding_config=ShardingConfig.replicated(),
        dtype=import_dtype,
        context_length=context_length,
        progress_callback=progress_callback,
    )
    callbacks.saving_model()
    imported_model.model.save(output_dir)

    callbacks.finished_saving_model()


def trace(
    model_path: Path,
    output_path: Path,
    messages: Iterable[Message],
    input_trace_path: Path | None = None,
) -> None:
    if output_path.exists():
        raise RuntimeError(f"{output_path=} already exists, refusing to overwrite!")

    messages = tuple(messages)
    model = load_traceable_model(model_path)
    if input_trace_path is None:
        trace_result, tokenization_trace = record_message_trace_with_tokenization(model, messages)
        metadata = tokenization_trace.metadata
    else:
        if messages:
            raise ValueError("messages cannot be used with input_trace_path")
        trace_result, metadata = record_saved_token_trace(model, input_trace_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fd:
        safe_write(fd, export_trace_result(trace_result), metadata=metadata)


def trace_tokenization(
    model_path: Path,
    output_path: Path,
    messages: Iterable[Message],
) -> None:
    if output_path.exists():
        raise RuntimeError(f"{output_path=} already exists, refusing to overwrite!")

    token_codec = load_traceable_token_codec(model_path)
    tokenization_trace = record_tokenization_trace(token_codec, messages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fd:
        safe_write(fd, tokenization_trace.arrays, metadata=tokenization_trace.metadata)


def compare_traces(reference_path: Path, result_path: Path) -> TraceComparison:
    return compare_trace_files(reference_path, result_path)
