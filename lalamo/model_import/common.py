import importlib.metadata
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import DTypeLike
from tokenizers import Tokenizer

from lalamo.audio.tts_message_processor import TTSMessageProcessor, TTSMessageProcessorConfig
from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.common import WeightShard
from lalamo.message_processor import MessageProcessor, MessageProcessorConfig
from lalamo.model_import.model_configs.huggingface.fishaudio import FishAudioConfig
from lalamo.model_import.model_configs.nanocodec import NanoCodecForeignConfig
from lalamo.model_registry import ModelRegistry
from lalamo.models import (
    ClassifierModel,
    ClassifierModelConfig,
    GenerationConfig,
    LanguageModel,
    LanguageModelConfig,
    LatentTTSGenerator,
    LatentTTSGeneratorConfig,
    TTSGenerator,
    TTSGeneratorConfig,
)
from lalamo.modules import Classifier, Decoder, LalamoModule, LatentTTSModel, TTSModel
from lalamo.modules.common import ShardingConfig, use_sharding
from lalamo.quantization import QuantizationMode
from lalamo.utils import process_chat_template

from .huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config, merge_token_ids
from .huggingface_tokenizer_config import HFTokenizerConfig
from .model_configs import ForeignClassifierConfig, ForeignConfig, ForeignLMConfig
from .model_specs import FileSpec, ModelSpec, ModelType, UseCase
from .model_specs.common import JSONFieldSpec
from .model_specs.origins import (
    DownloadingFileEvent,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    StatusEvent,
)

__all__ = [
    "DownloadingFileEvent",
    "FinishedDownloadingFileEvent",
    "InitializingModelEvent",
    "ModelMetadata",
    "ModelSpec",
    "ModelType",
    "StatusEvent",
    "import_model",
]


LALAMO_VERSION = importlib.metadata.version("lalamo")


@dataclass(frozen=True)
class ModelMetadata:
    toolchain_version: str
    vendor: str
    family: str
    name: str
    size: str
    quantization: QuantizationMode | None
    source_description: str
    use_cases: tuple[UseCase, ...]
    model_type: ModelType
    model_config: LanguageModelConfig | ClassifierModelConfig | TTSGeneratorConfig | LatentTTSGeneratorConfig
    grammar_start_tokens: tuple[str, ...]


class ImportResults(NamedTuple):
    model: LanguageModel | ClassifierModel | TTSGenerator | LatentTTSGenerator
    metadata: ModelMetadata


def token_ids_to_text(tokenizer: Tokenizer, token_ids: int | list[int] | None) -> str | None:
    match token_ids:
        case int(tid):
            return tokenizer.decode([tid], skip_special_tokens=False)
        case [int(), *_] as ids if all(isinstance(i, int) for i in ids):
            return tokenizer.decode([ids[0]], skip_special_tokens=False)
        case None:
            return None
        case _:
            raise ValueError(f"Expected int, list[int], or None, got {token_ids!r}")


def _instantiate_tokenizer_from_model_spec(
    model_spec: ModelSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Tokenizer:
    match model_spec.configs.tokenizer:
        case FileSpec() as file_spec:
            tokenizer_file = model_spec.origin.resolve_file(file_spec, progress_callback)
            return Tokenizer.from_file(str(tokenizer_file))
        case None if model_spec.config_type is NanoCodecForeignConfig:
            return Tokenizer.from_str(dummy_char_level_tokenizer_config())
        case None:
            raise ValueError(f"Model {model_spec.name} has no tokenizer configured but is not a NanoCodec model.")
        case _:
            raise ValueError(f"Expected FileSpec or None for tokenizer, got {type(model_spec.configs.tokenizer)}")


def import_message_processor(
    model_spec: ModelSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> MessageProcessor:
    tokenizer_config_file = model_spec.origin.resolve_file(model_spec.configs.tokenizer_config, progress_callback)
    tokenizer_config = HFTokenizerConfig.from_json(tokenizer_config_file)

    if tokenizer_config.chat_template is None:
        match model_spec.configs.chat_template:
            case JSONFieldSpec(file_spec, field_name):
                json_file = model_spec.origin.resolve_file(file_spec, progress_callback)
                with open(json_file) as file:
                    json_dict = json.load(file)
                prompt_template = json_dict[field_name]
            case FileSpec(_) as file_spec:
                chat_template_file = model_spec.origin.resolve_file(file_spec, progress_callback)
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
    tokenizer = _instantiate_tokenizer_from_model_spec(model_spec, progress_callback)

    added_tokens = tokenizer_config.added_tokens()
    added_special_tokens = [token for token in added_tokens if token.special]
    added_not_special_tokens = [token for token in added_tokens if not token.special]
    tokenizer.add_special_tokens(added_special_tokens)
    tokenizer.add_tokens(added_not_special_tokens)

    bos_token = tokenizer_config.bos_token
    eos_token = tokenizer_config.eos_token

    if eos_token is None or bos_token is None:
        foreign_decoder_config_file = model_spec.origin.resolve_file(
            model_spec.configs.model_config, progress_callback
        )
        with open(foreign_decoder_config_file) as foreign_decoder_file:
            foreign_decoder_json = json.load(foreign_decoder_file)

        if bos_token is None:
            bos_token_id: int | list[int] | None = foreign_decoder_json.get("bos_token_id")
            bos_token = token_ids_to_text(tokenizer, bos_token_id)
        if eos_token is None:
            eos_token_id: int | list[int] | None = foreign_decoder_json.get("eos_token_id")
            eos_token = token_ids_to_text(tokenizer, eos_token_id)

    system_prompt_text = None
    match model_spec.configs.system_prompt:
        case FileSpec(_) as file_spec:
            system_prompt_file = model_spec.origin.resolve_file(file_spec, progress_callback)
            system_prompt_text = system_prompt_file.read_text()
        case str() as sp:
            system_prompt_text = sp
        case None:
            pass
        case _:
            raise ValueError(f"Unexpected system_prompt type: {type(model_spec.configs.system_prompt)}")

    message_processor_config = MessageProcessorConfig(
        prompt_template=prompt_template,
        output_parser_regex=model_spec.output_parser_regex,
        system_role_name=model_spec.system_role_name,
        user_role_name=model_spec.user_role_name,
        assistant_role_name=model_spec.assistant_role_name,
        bos_token=bos_token,
        eos_token=eos_token,
        default_system_prompt=system_prompt_text,
    )
    return MessageProcessor(config=message_processor_config, tokenizer=tokenizer)


def _resolve_configs(
    model_spec: ModelSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    origin = model_spec.origin
    config_path = origin.resolve_file(model_spec.configs.model_config, progress_callback)
    extra_config_paths = tuple(origin.resolve_file(ec, progress_callback) for ec in model_spec.configs.extra_configs)
    return (config_path, extra_config_paths)


def _load_main_processing_module(
    weight_shards: Sequence[WeightShard],
    precision: DTypeLike,
    foreign_config: ForeignConfig,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    context_length: int | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
) -> LalamoModule:
    if progress_callback is not None:
        progress_callback(InitializingModelEvent())

    return foreign_config.load(
        context_length,
        precision,
        accumulation_precision,
        weight_shards,
    )


def _import_language_model(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[LanguageModel, LanguageModelConfig]:
    config_path, extra_config_paths = _resolve_configs(model_spec, progress_callback)
    foreign_decoder_config = model_spec.config_type.from_json(config_path, extra_config_paths)
    assert isinstance(foreign_decoder_config, ForeignLMConfig)

    if precision is None:
        precision = foreign_decoder_config.default_precision
    weight_shards = model_spec.origin.get_weights(precision, progress_callback)
    decoder = _load_main_processing_module(
        weight_shards,
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

    stop_token_ids = merge_token_ids(foreign_decoder_config.eos_token_ids)

    if isinstance(model_spec.configs.generation_config, GenerationConfig):
        candidate_generation_config = model_spec.configs.generation_config
        stop_token_ids = merge_token_ids(candidate_generation_config.stop_token_ids, stop_token_ids)
        generation_config = replace(candidate_generation_config, stop_token_ids=stop_token_ids)
    elif isinstance(model_spec.configs.generation_config, FileSpec):
        hf_generation_config_file = model_spec.origin.resolve_file(model_spec.configs.generation_config)
        hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
        stop_token_ids = merge_token_ids(stop_token_ids, hf_generation_config.eos_token_id)
        generation_config = _policy_from_hf_config(hf_generation_config, stop_token_ids=stop_token_ids)
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
    config_path, extra_config_paths = _resolve_configs(model_spec, progress_callback)
    foreign_classifier_config = model_spec.config_type.from_json(config_path, extra_config_paths)
    assert isinstance(foreign_classifier_config, ForeignClassifierConfig)

    if precision is None:
        precision = foreign_classifier_config.default_precision

    weight_shards = model_spec.origin.get_weights(precision, progress_callback)
    classifier = _load_main_processing_module(
        weight_shards,
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


def _import_tts_model(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[TTSGenerator, TTSGeneratorConfig]:
    config_path, extra_config_paths = _resolve_configs(model_spec, progress_callback)
    foreign_tts_config = model_spec.config_type.from_json(config_path, extra_config_paths)
    if precision is None:
        precision = foreign_tts_config.default_precision
    # TODO @knyazer: transition to tokenizer enum so this FishAudio special-case goes away
    if isinstance(foreign_tts_config, FishAudioConfig):
        foreign_tts_config, tokenizer = foreign_tts_config.prepare_tokenizer(model_spec, progress_callback)
    else:
        tokenizer = _instantiate_tokenizer_from_model_spec(model_spec, progress_callback)

    weight_shards = model_spec.origin.get_weights(precision, progress_callback)
    tts_model = _load_main_processing_module(
        weight_shards,
        precision,
        foreign_tts_config,
        progress_callback,
        context_length,
        accumulation_precision,
    )

    assert isinstance(tts_model, TTSModel)
    if progress_callback is not None:
        progress_callback(FinishedInitializingModelEvent())

    assert isinstance(model_spec.configs.chat_template, str)
    tts_request_factory_config = TTSMessageProcessorConfig(
        prompt_template=model_spec.configs.chat_template,
    )
    message_processor = TTSMessageProcessor(tts_request_factory_config, tokenizer)

    tts_generator_config = TTSGeneratorConfig(
        tts_config=foreign_tts_config.to_lalamo_config(
            context_length=context_length,
            activation_precision=precision,
            accumulation_precision=accumulation_precision,
            metadata_dict={},
        ),
        message_processor_config=message_processor.config,
    )
    tts_generator = TTSGenerator(tts_generator_config, tts_model, message_processor)

    return (tts_generator, tts_generator_config)


def _import_latent_tts_model(
    model_spec: ModelSpec,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[LatentTTSGenerator, LatentTTSGeneratorConfig]:
    config_path, extra_config_paths = _resolve_configs(model_spec, progress_callback)
    foreign_config = model_spec.config_type.from_json(config_path, extra_config_paths)
    if precision is None:
        precision = foreign_config.default_precision

    tokenizer = _instantiate_tokenizer_from_model_spec(model_spec, progress_callback)

    weight_shards = model_spec.origin.get_weights(precision, progress_callback)
    latent_tts_model = _load_main_processing_module(
        weight_shards,
        precision,
        foreign_config,
        progress_callback,
        context_length,
        accumulation_precision,
    )

    assert isinstance(latent_tts_model, LatentTTSModel)
    if progress_callback is not None:
        progress_callback(FinishedInitializingModelEvent())

    assert isinstance(model_spec.configs.chat_template, str)
    tts_request_factory_config = TTSMessageProcessorConfig(
        prompt_template=model_spec.configs.chat_template,
    )

    latent_tts_config = foreign_config.to_lalamo_config(
        context_length=context_length,
        activation_precision=precision,
        accumulation_precision=accumulation_precision,
        metadata_dict={},
    )

    message_processor = latent_tts_config.create_message_processor(tts_request_factory_config, tokenizer)

    generation_config = latent_tts_config.default_generation_config()
    generator_config = LatentTTSGeneratorConfig(
        latent_tts_config=latent_tts_config,
        message_processor_config=message_processor.config,
        generation_config=generation_config,
    )

    generator = LatentTTSGenerator(
        config=generator_config,
        latent_tts_model=latent_tts_model,
        message_processor=message_processor,
    )

    return (generator, generator_config)


def import_model(
    model_spec: ModelSpec | str,
    *,
    context_length: int | None = None,
    precision: DTypeLike | None = None,
    accumulation_precision: DTypeLike = jnp.float32,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    sharding_config: ShardingConfig | None = None,
) -> ImportResults:
    if isinstance(model_spec, str):
        model_spec = ModelRegistry.build().repo_to_model[model_spec]

    with use_sharding(sharding_config):
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
            case ModelType.TTS_MODEL:
                model, config = _import_tts_model(
                    model_spec,
                    context_length=context_length,
                    precision=precision,
                    accumulation_precision=accumulation_precision,
                    progress_callback=progress_callback,
                )
            case ModelType.LATENT_TTS_MODEL:
                model, config = _import_latent_tts_model(
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
        source_description=model_spec.origin.description,
        use_cases=model_spec.use_cases,
        model_type=model_spec.model_type,
        model_config=config,
        grammar_start_tokens=model_spec.grammar_start_tokens,
    )
    return ImportResults(model, metadata)
