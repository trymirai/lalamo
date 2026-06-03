import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from itertools import batched
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import requests
import thefuzz.fuzz
import thefuzz.process

from lalamo.data import load_hf_parquet, shuffle_dataset
from lalamo.data.huggingface_message import HFMessage
from lalamo.data.lalamo_completions import LalamoCompletion, save_completions
from lalamo.data.utils import get_prompt_ending_in_user_message, pad_sequences
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
from lalamo.models import LanguageModel
from lalamo.models.chat_codec import Message
from lalamo.module import Keychain
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


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    total_sequences: int | None
    tokens_generated: int


def iter_hf_conversations(conversations: Iterable[Iterable[dict[str, str]]]) -> Iterator[list[Message]]:
    return ([HFMessage.from_dict(message).as_message() for message in conversation] for conversation in conversations)


def iter_collectable_prompt_token_ids(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    max_input_length: int,
) -> Iterator[list[int]]:
    prefixes = (
        prefix
        for conversation in conversations
        if (prefix := get_prompt_ending_in_user_message(conversation)) is not None
    )
    tokenized_prefixes = map(model.token_codec.encode_request, prefixes)
    return filter(lambda token_ids: len(token_ids) <= max_input_length, tokenized_prefixes)


def count_collectable_prompt_token_ids(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    max_input_length: int,
) -> int:
    return sum(1 for _ in iter_collectable_prompt_token_ids(model, conversations, max_input_length))


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    batch_size: int = 1,
    max_input_length: int = 2048,
    max_output_length: int = 4096,
    total_sequences: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
) -> Iterable[LalamoCompletion]:
    tokens_generated = 0
    sequences_processed = 0
    key = jax.random.key(0)

    for prefix_batch in batched(iter_collectable_prompt_token_ids(model, conversations, max_input_length), batch_size):
        next_key, batch_key = jax.random.split(key)
        key = next_key
        padded_prefixes, _ = pad_sequences(prefix_batch, 0, max_input_length)
        prefix_lengths = jnp.asarray([len(prefix) for prefix in prefix_batch], dtype=jnp.int32)
        generated_batch = model.generate_tokens(
            padded_prefixes,
            prompt_lengths_without_padding=prefix_lengths,
            max_output_length=max_output_length,
            keychain=Keychain(vmapped_keys=batch_key, batch_key=batch_key, sharding_config=model.sharding_config),
        )
        for row_index, prefix_token_ids in enumerate(prefix_batch):
            token_ids = model.trim_at_eos(generated_batch.token_ids[row_index].tolist())
            seqlen = len(token_ids)

            tokens_generated += seqlen
            sequences_processed += 1
            token_ids = token_ids[:seqlen]

            yield LalamoCompletion(
                prefix_token_ids=prefix_token_ids,
                completion_token_ids=token_ids,
            )

            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, total_sequences, tokens_generated))


@dataclass
class CollectTracesCallbacks:
    model_path: Path
    dataset_path: Path
    output_path: Path
    max_input_length: int
    max_output_length: int
    batch_size: int

    def loading_model(self) -> None:
        pass

    def finished_loading_model(self) -> None:
        pass

    def loading_dataset(self) -> None:
        pass

    def finished_loading_dataset(self) -> None:
        pass

    def counting_prompts(self) -> None:
        pass

    def finished_counting_prompts(self, _total_sequences: int) -> None:
        pass

    def starting_inference(self, total_sequences: int) -> None:
        pass

    def inference_progress(self, event: CollectTracesEvent) -> None:
        pass

    def finished_inference(self) -> None:
        pass


def collect_traces(
    model_path: Path,
    dataset_path: Path,
    output_path: Path,
    max_input_length: int = 2048,
    max_output_length: int = 4096,
    batch_size: int = 1,
    callbacks_type: Callable[..., CollectTracesCallbacks] = CollectTracesCallbacks,
) -> None:
    callbacks = callbacks_type(
        model_path,
        dataset_path,
        output_path,
        max_input_length,
        max_output_length,
        batch_size,
    )

    callbacks.loading_model()
    model = LanguageModel.load(model_path, ShardingConfig.replicated())
    callbacks.finished_loading_model()

    callbacks.loading_dataset()
    dataframe = shuffle_dataset(load_hf_parquet(dataset_path))
    conversations = dataframe.get_column("conversation")
    callbacks.finished_loading_dataset()

    callbacks.counting_prompts()
    total_sequences = count_collectable_prompt_token_ids(
        model,
        iter_hf_conversations(conversations),
        max_input_length,
    )
    callbacks.finished_counting_prompts(total_sequences)
    callbacks.starting_inference(total_sequences)

    def progress_callback(event: CollectTracesEvent) -> None:
        callbacks.inference_progress(event)

    traces = inference_collect_traces(
        model,
        iter_hf_conversations(conversations),
        batch_size,
        max_input_length,
        max_output_length,
        total_sequences,
        progress_callback,
    )

    if output_path.exists():
        raise RuntimeError(f"{output_path} must not exist.")
    save_completions(output_path, traces)

    callbacks.finished_inference()
