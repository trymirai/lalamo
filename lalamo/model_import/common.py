import importlib.metadata
import json
from collections import ChainMap
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import huggingface_hub
import jax.numpy as jnp
from jax import Array
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.message_processor import MessageProcessor, MessageProcessorConfig
from lalamo.models import ClassifierModel, ClassifierModelConfig, GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.modules import Classifier, Decoder, LalamoModule
from lalamo.quantization import QuantizationMode
from lalamo.utils import process_chat_template

from .decoder_configs import ForeignClassifierConfig, ForeignConfig, ForeignLMConfig
from .huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config
from .huggingface_tokenizer_config import HFTokenizerConfig
from .model_specs import REPO_TO_MODEL, FileSpec, ModelSpec, ModelType, UseCase
from .model_specs.common import JSONFieldSpec

__all__ = [
    "REPO_TO_MODEL",
    "DownloadingFileEvent",
    "FinishedDownloadingFileEvent",
    "InitializingModelEvent",
    "ModelMetadata",
    "ModelSpec",
    "ModelType",
    "StatusEvent",
    "download_file",
    "import_model",
]


LALAMO_VERSION = importlib.metadata.version("lalamo")


class DownloadingFileEvent(NamedTuple):
    file: FileSpec


class FinishedDownloadingFileEvent(NamedTuple):
    file: FileSpec


class InitializingModelEvent(NamedTuple):
    pass


class FinishedInitializingModelEvent(NamedTuple):
    pass


type StatusEvent = (
    DownloadingFileEvent | FinishedDownloadingFileEvent | InitializingModelEvent | FinishedInitializingModelEvent
)


@dataclass(frozen=True)
class ModelMetadata:
    toolchain_version: str
    vendor: str
    family: str
    name: str
    size: str
    quantization: QuantizationMode | None
    repo: str
    use_cases: tuple[UseCase, ...]
    model_type: ModelType
    model_config: LanguageModelConfig | ClassifierModelConfig
    grammar_start_tokens: tuple[str, ...]


def download_file(
    file_spec: FileSpec,
    model_repo: str,
    output_dir: Path | str | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Path:
    if progress_callback is not None:
        progress_callback(DownloadingFileEvent(file_spec))
    result = huggingface_hub.hf_hub_download(
        repo_id=file_spec.repo or model_repo,
        local_dir=output_dir,
        filename=file_spec.filename,
    )
    if progress_callback is not None:
        progress_callback(FinishedDownloadingFileEvent(file_spec))
    return Path(result)


def list_weight_files(model_repo: str) -> list[FileSpec]:
    all_files = huggingface_hub.list_repo_files(model_repo)
    return [FileSpec(filename) for filename in all_files if filename.endswith(".safetensors")]


def download_weights(
    model_spec: ModelSpec,
    output_dir: Path | str | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> list[Path]:
    return [
        download_file(file_spec, model_spec.repo, output_dir, progress_callback)
        for file_spec in list_weight_files(model_spec.repo)
    ]


def download_config_file(
    model_spec: ModelSpec,
    output_dir: Path | str | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Path:
    return download_file(model_spec.configs.model_config, model_spec.repo, output_dir, progress_callback)


class ImportResults(NamedTuple):
    model: LanguageModel | ClassifierModel
    metadata: ModelMetadata


def import_message_processor(
    model_spec: ModelSpec,
    output_dir: Path | str | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> MessageProcessor:
    tokenizer_file = download_file(model_spec.configs.tokenizer, model_spec.repo, output_dir, progress_callback)
    tokenizer_config_file = download_file(
        model_spec.configs.tokenizer_config,
        model_spec.repo,
        output_dir,
        progress_callback,
    )
    tokenizer_config = HFTokenizerConfig.from_json(tokenizer_config_file)
    if tokenizer_config.chat_template is None:
        match model_spec.configs.chat_template:
            case JSONFieldSpec(file_spec, field_name):
                json_file = download_file(file_spec, model_spec.repo, output_dir)
                with open(json_file) as file:
                    json_dict = json.load(file)
                prompt_template = json_dict[field_name]
            case FileSpec(_) as file_spec:
                chat_template_file = download_file(file_spec, model_spec.repo, output_dir)
                prompt_template = chat_template_file.read_text()
            case str() as template_string:
                prompt_template = template_string
            case None:
                raise ValueError("No chat template specified.")
    else:
        if model_spec.configs.chat_template is not None:
            raise ValueError("Conflicting chat template specifications.")
        prompt_template = tokenizer_config.chat_template
    prompt_template = process_chat_template(prompt_template)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))

    added_tokens = tokenizer_config.added_tokens()
    added_special_tokens = [token for token in added_tokens if token.special]
    added_not_special_tokens = [token for token in added_tokens if not token.special]
    tokenizer.add_special_tokens(added_special_tokens)
    tokenizer.add_tokens(added_not_special_tokens)

    message_processor_config = MessageProcessorConfig(
        prompt_template=prompt_template,
        output_parser_regex=model_spec.output_parser_regex,
        system_role_name=model_spec.system_role_name,
        user_role_name=model_spec.user_role_name,
        assistant_role_name=model_spec.assistant_role_name,
        bos_token=tokenizer_config.bos_token,
    )
    return MessageProcessor(config=message_processor_config, tokenizer=tokenizer)


def _load_main_processing_module(
    model_spec: ModelSpec,
    precision: DTypeLike,
    foreign_config: ForeignConfig,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    context_length: int | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
) -> LalamoModule:
    weights_paths = download_weights(model_spec, progress_callback=progress_callback)
    with ExitStack() as stack:
        weights_shards = []
        metadata_shards = []
        for weights_path in weights_paths:
            weights_shard, metadata_shard = stack.enter_context(model_spec.weights_type.load(weights_path, precision))
            weights_shards.append(weights_shard)
            metadata_shards.append(metadata_shard)
        weights_dict: ChainMap[str, Array] = ChainMap(*weights_shards)
        metadata_dict: ChainMap[str, str] = ChainMap(*metadata_shards)

        if progress_callback is not None:
            progress_callback(InitializingModelEvent())

        processing_module = foreign_config.load(
            context_length,
            precision,
            accumulation_precision,
            weights_dict,
            metadata_dict,
        )

    return processing_module


def _import_language_model(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[LanguageModel, LanguageModelConfig]:
    foreign_decoder_config_file = download_config_file(model_spec)
    foreign_decoder_config = model_spec.config_type.from_json(foreign_decoder_config_file)
    assert isinstance(foreign_decoder_config, ForeignLMConfig)

    if precision is None:
        precision = foreign_decoder_config.default_precision
    decoder = _load_main_processing_module(
        model_spec,
        precision,
        foreign_decoder_config,
        progress_callback,
        context_length,
        accumulation_precision,
    )
    assert isinstance(decoder, Decoder)

    if progress_callback is not None:
        progress_callback(FinishedInitializingModelEvent())

    message_processor = import_message_processor(model_spec)

    stop_token_ids = tuple(foreign_decoder_config.eos_token_ids)

    if isinstance(model_spec.configs.generation_config, GenerationConfig):
        generation_config = replace(model_spec.configs.generation_config, stop_token_ids=stop_token_ids)
    elif isinstance(model_spec.configs.generation_config, FileSpec):
        hf_generation_config_file = download_file(model_spec.configs.generation_config, model_spec.repo)
        hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
        generation_config = _policy_from_hf_config(hf_generation_config, stop_token_ids)
    else:
        generation_config = GenerationConfig(stop_token_ids)

    language_model_config = LanguageModelConfig(
        model_config=decoder.config,
        message_processor_config=message_processor.config,
        generation_config=generation_config,
    )

    language_model = LanguageModel(language_model_config, decoder, message_processor)
    return language_model, language_model_config


def _import_classifier(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[ClassifierModel, ClassifierModelConfig]:
    foreign_classifier_config_file = download_config_file(model_spec)
    foreign_classifier_config = model_spec.config_type.from_json(foreign_classifier_config_file)
    assert isinstance(foreign_classifier_config, ForeignClassifierConfig)

    if precision is None:
        precision = foreign_classifier_config.default_precision

    classifier = _load_main_processing_module(
        model_spec,
        precision,
        foreign_classifier_config,
        progress_callback,
        context_length,
        accumulation_precision,
    )
    assert isinstance(classifier, Classifier)

    if progress_callback is not None:
        progress_callback(FinishedInitializingModelEvent())

    message_processor = import_message_processor(model_spec)

    classifier_model_config = ClassifierModelConfig(
        model_config=classifier.config,
        message_processor_config=message_processor.config,
    )
    classifier_model = ClassifierModel(classifier_model_config, classifier, message_processor)
    return classifier_model, classifier_model_config


def import_model(
    model_spec: ModelSpec | str,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> ImportResults:
    if isinstance(model_spec, str):
        try:
            model_spec = REPO_TO_MODEL[model_spec]
        except KeyError as e:
            raise ValueError(f"Unknown model: {model_spec}") from e

    match model_spec.model_type:
        case ModelType.LANGUAGE_MODEL:
            model, config = _import_language_model(
                model_spec,
                context_length=context_length,
                precision=precision,
                accumulation_precision=accumulation_precision,
                progress_callback=progress_callback,
            )
        case ModelType.CLASSIFIER_MODEL:
            model, config = _import_classifier(
                model_spec,
                context_length=context_length,
                precision=precision,
                accumulation_precision=accumulation_precision,
                progress_callback=progress_callback,
            )

    metadata = ModelMetadata(
        toolchain_version=LALAMO_VERSION,
        vendor=model_spec.vendor,
        family=model_spec.family,
        name=model_spec.name,
        size=model_spec.size,
        quantization=model_spec.quantization,
        repo=model_spec.repo,
        use_cases=model_spec.use_cases,
        model_type=model_spec.model_type,
        model_config=config,
        grammar_start_tokens=model_spec.grammar_start_tokens,
    )
    return ImportResults(model, metadata)
