import importlib.metadata
import json
from collections import ChainMap
from collections.abc import Callable, Generator, Mapping, MutableMapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple, cast

from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.model import Model, ModelConfig
from lalamo.model_import.loaders.fishaudio_loaders import load_tokenizer_from_fishaudio_tiktoken
from lalamo.model_import.model_configs.huggingface.fishaudio import FishAudioConfig
from lalamo.model_registry import ModelRegistry
from lalamo.models import (
    ClassifierModel,
    ClassifierModelConfig,
    GenerationConfig,
    LanguageModel,
    TTSModel,
    TTSModelConfig,
)
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.tts_codec import TTSCodecConfig
from lalamo.utils.template_hacking import fix_chat_template
from lalamo.weight_matrix import CompressionImplementation

from .huggingface_generation_config import HFGenerationConfig, _policy_from_hf_config, merge_token_ids
from .huggingface_tokenizer_config import HFTokenizerConfig
from .model_configs.foreign_config import ForeignConfig, ForeignLMConfig
from .model_spec import ClassifierModelSpec, FileSpec, JSONFieldSpec, LanguageModelSpec, ModelSpec, TTSModelSpec
from .origins import (
    DownloadingFileEvent,
    FinishedDownloadingFileEvent,
    FinishedInitializingModelEvent,
    InitializingModelEvent,
    Origin,
    StatusEvent,
    WeightShard,
    report_status,
)

__all__ = [
    "DownloadingFileEvent",
    "FinishedDownloadingFileEvent",
    "FinishedInitializingModelEvent",
    "ImportResults",
    "InitializingModelEvent",
    "ModelMetadata",
    "ModelSpec",
    "StatusEvent",
    "import_model",
]


@dataclass(frozen=True)
class ModelMetadata:
    toolchain_version: str
    vendor: str
    family: str
    name: str
    size: str
    origin: str
    model_config: ModelConfig
    grammar_start_tokens: tuple[str, ...]


class ImportResults(NamedTuple):
    model: Model
    metadata: ModelMetadata


class Checkpoint(NamedTuple):
    weights: Mapping[str, Array]
    metadata: Mapping[str, str]


def token_ids_to_text(tokenizer: Tokenizer, token_ids: int | list[int] | None) -> str | None:
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    if not isinstance(token_ids, list) or any((not isinstance(el, int)) for el in token_ids):
        return None

    return tokenizer.decode(token_ids[:1], skip_special_tokens=False)


def _instantiate_tokenizer(
    origin: Origin,
    tokenizer_spec: FileSpec | None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Tokenizer:
    if tokenizer_spec is None:
        # NOTE: tokenizer-less TTS specs currently use a dummy tokenizer for stub text decoding.
        tokenizer = Tokenizer.from_str(dummy_char_level_tokenizer_config())
    else:
        tokenizer_file = origin.resolve_file(tokenizer_spec, progress_callback)
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    return tokenizer


def _read_text_spec(
    origin: Origin,
    text_spec: FileSpec | str | None,
    progress_callback: Callable[[StatusEvent], None] | None,
) -> str | None:
    match text_spec:
        case FileSpec() as file_spec:
            return origin.resolve_file(file_spec, progress_callback).read_text()
        case str() as text:
            return text
        case None:
            return None


def _read_chat_template(
    origin: Origin,
    template_spec: FileSpec | JSONFieldSpec | str | None,
    progress_callback: Callable[[StatusEvent], None] | None,
) -> str | None:
    match template_spec:
        case JSONFieldSpec(file_spec, field_name):
            with origin.resolve_file(file_spec, progress_callback).open() as file:
                return json.load(file)[field_name]
        case FileSpec() | str() | None:
            return _read_text_spec(origin, template_spec, progress_callback)


def _import_chat_codec(
    model_spec: LanguageModelSpec | ClassifierModelSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> tuple[Tokenizer, ChatCodecConfig]:
    origin = model_spec.origin
    tokenizer_config_file = origin.resolve_file(model_spec.configs.tokenizer_config, progress_callback)
    tokenizer_config = HFTokenizerConfig.from_json(tokenizer_config_file)

    if tokenizer_config.chat_template is None:
        prompt_template = _read_chat_template(origin, model_spec.configs.chat_template, progress_callback)
        if prompt_template is None:
            raise ValueError("No chat template specified.")
    else:
        if model_spec.configs.chat_template is not None:
            raise ValueError("Conflicting chat template specifications.")
        prompt_template = tokenizer_config.chat_template

    prompt_template = fix_chat_template(prompt_template)
    tokenizer = _instantiate_tokenizer(origin, model_spec.configs.tokenizer, progress_callback)

    added_tokens = tokenizer_config.added_tokens()
    added_special_tokens = [token for token in added_tokens if token.special]
    added_not_special_tokens = [token for token in added_tokens if not token.special]
    tokenizer.add_special_tokens(added_special_tokens)
    tokenizer.add_tokens(added_not_special_tokens)

    bos_token = getattr(tokenizer_config, "bos_token", None)
    eos_token = getattr(tokenizer_config, "eos_token", None)

    # If we were not able to identify bos/eos - they are probably somewhere else, so we check config.json
    if eos_token is None or bos_token is None:
        foreign_decoder_config_file = origin.resolve_file(model_spec.configs.model_config, progress_callback)
        with open(foreign_decoder_config_file) as foreign_decoder_file:
            foreign_decoder_json = json.load(foreign_decoder_file)

        if bos_token is None:
            bos_token_id: int | list[int] | None = foreign_decoder_json.get("bos_token_id")
            bos_token = token_ids_to_text(tokenizer, bos_token_id)
        if eos_token is None:
            eos_token_id: int | list[int] | None = foreign_decoder_json.get("eos_token_id")
            eos_token = token_ids_to_text(tokenizer, eos_token_id)

    system_prompt_text = _read_text_spec(origin, model_spec.configs.system_prompt, progress_callback)

    return (
        tokenizer,
        ChatCodecConfig(
            prompt_template=prompt_template,
            output_parser_regex=model_spec.output_parser_regex,
            system_role_name=model_spec.system_role_name,
            user_role_name=model_spec.user_role_name,
            assistant_role_name=model_spec.assistant_role_name,
            bos_token=bos_token,
            eos_token=eos_token,
            default_system_prompt=system_prompt_text,
        ),
    )


def _combine_weight_shards(
    weight_shards: Sequence[WeightShard],
) -> Checkpoint:
    return Checkpoint(
        weights=ChainMap(*(cast("MutableMapping[str, Array]", weights) for weights, _ in weight_shards)),
        metadata=ChainMap(*(cast("MutableMapping[str, str]", metadata) for _, metadata in weight_shards)),
    )


@contextmanager
def _load_checkpoint(
    model_spec: ModelSpec,
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> Generator[Checkpoint]:
    with ExitStack() as stack:
        weight_shards = tuple(
            stack.enter_context(weight_shard)
            for weight_shard in model_spec.origin.get_weights(progress_callback=progress_callback)
        )
        yield _combine_weight_shards(weight_shards)


def _load_foreign_config[ForeignConfigT: ForeignConfig](
    model_spec: ModelSpec[ForeignConfigT],
    progress_callback: Callable[[StatusEvent], None] | None = None,
) -> ForeignConfigT:
    config_path = model_spec.origin.resolve_file(model_spec.configs.model_config, progress_callback)
    return model_spec.config_type.from_json(config_path)


def _dtype_or_default(dtype: DTypeLike | None, foreign_config: ForeignConfig) -> DTypeLike:
    return foreign_config.default_dtype if dtype is None else dtype


def _load_model[ModelT: Model](
    expected_model_type: type[ModelT],
    foreign_config: ForeignConfig,
    model_config: ModelConfig,
    tokenizer: Tokenizer,
    dtype: DTypeLike,
    weights_dict: Mapping[str, Array],
    progress_callback: Callable[[StatusEvent], None] | None = None,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> ModelT:
    report_status(progress_callback, InitializingModelEvent())

    model = foreign_config.load(
        config=model_config,
        tokenizer=tokenizer,
        dtype=dtype,
        weights_dict=weights_dict,
        implementation=implementation,
    )
    assert isinstance(model, expected_model_type)
    report_status(progress_callback, FinishedInitializingModelEvent())
    return model


def _import_generation_config(
    model_spec: LanguageModelSpec,
    foreign_decoder_config: ForeignLMConfig,
    progress_callback: Callable[[StatusEvent], None] | None,
) -> GenerationConfig:
    stop_token_ids = merge_token_ids(foreign_decoder_config.eos_token_ids)
    match model_spec.configs.generation_config:
        case GenerationConfig() as generation_config:
            stop_token_ids = merge_token_ids(generation_config.stop_token_ids, stop_token_ids)
            return replace(generation_config, stop_token_ids=stop_token_ids)
        case FileSpec() as file_spec:
            hf_generation_config_file = model_spec.origin.resolve_file(file_spec, progress_callback)
            hf_generation_config = HFGenerationConfig.from_json(hf_generation_config_file)
            stop_token_ids = merge_token_ids(stop_token_ids, hf_generation_config.eos_token_id)
            return _policy_from_hf_config(hf_generation_config, stop_token_ids=stop_token_ids)
        case None:
            return GenerationConfig(stop_token_ids)


def _import_language_model(
    model_spec: LanguageModelSpec,
    *,
    context_length: int | None = None,
    dtype: DTypeLike | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> LanguageModel:
    foreign_decoder_config = _load_foreign_config(model_spec, progress_callback=progress_callback)
    dtype = _dtype_or_default(dtype, foreign_decoder_config)

    tokenizer, token_codec_config = _import_chat_codec(model_spec, progress_callback=progress_callback)
    generation_config = _import_generation_config(model_spec, foreign_decoder_config, progress_callback)

    with _load_checkpoint(model_spec, progress_callback) as checkpoint:
        model_config = foreign_decoder_config.to_lalamo_config(
            context_length=context_length,
            metadata_dict=checkpoint.metadata,
            token_codec_config=token_codec_config,
            generation_config=generation_config,
        )
        return _load_model(
            LanguageModel,
            foreign_config=foreign_decoder_config,
            model_config=model_config,
            tokenizer=tokenizer,
            dtype=dtype,
            weights_dict=checkpoint.weights,
            progress_callback=progress_callback,
            implementation=implementation,
        )


def _import_classifier(
    model_spec: ClassifierModelSpec,
    *,
    context_length: int | None = None,
    dtype: DTypeLike | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> ClassifierModel:
    foreign_classifier_config = _load_foreign_config(model_spec, progress_callback=progress_callback)
    dtype = _dtype_or_default(dtype, foreign_classifier_config)

    tokenizer, token_codec_config = _import_chat_codec(model_spec, progress_callback=progress_callback)
    classifier_config = foreign_classifier_config.to_classifier_config(context_length)
    model_config = ClassifierModelConfig(
        token_codec_config=token_codec_config,
        classifier_config=classifier_config,
        output_labels=classifier_config.output_labels,
    )
    with _load_checkpoint(model_spec, progress_callback) as checkpoint:
        return _load_model(
            ClassifierModel,
            foreign_config=foreign_classifier_config,
            model_config=model_config,
            tokenizer=tokenizer,
            dtype=dtype,
            weights_dict=checkpoint.weights,
            progress_callback=progress_callback,
            implementation=implementation,
        )


def _import_tts_model(
    model_spec: TTSModelSpec,
    *,
    context_length: int | None = None,
    dtype: DTypeLike | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> TTSModel:
    foreign_tts_config = _load_foreign_config(model_spec, progress_callback=progress_callback)
    dtype = _dtype_or_default(dtype, foreign_tts_config)
    if isinstance(foreign_tts_config, FishAudioConfig):
        assert isinstance(model_spec.configs.tokenizer, FileSpec)
        tokenizer_path = model_spec.origin.resolve_file(model_spec.configs.tokenizer, progress_callback)
        tokenizer_special_tokens_path = model_spec.origin.resolve_file(
            FileSpec(filename="special_tokens.json"),
            progress_callback,
        )
        tokenizer, special_inference_tokens = load_tokenizer_from_fishaudio_tiktoken(
            tokenizer_path,
            tokenizer_special_tokens_path,
        )
        foreign_tts_config = replace(
            foreign_tts_config,
            semantic_token_begin_id=special_inference_tokens.semantic_begin_id,
            semantic_token_end_id=special_inference_tokens.semantic_end_id,
            im_end_token_id=special_inference_tokens.im_end_token_id,
        )
    else:
        tokenizer = _instantiate_tokenizer(
            model_spec.origin,
            model_spec.configs.tokenizer,
            progress_callback=progress_callback,
        )

    assert isinstance(model_spec.configs.chat_template, str)
    token_codec_config = TTSCodecConfig(
        prompt_template=model_spec.configs.chat_template,
    )
    model_config = TTSModelConfig(
        token_codec_config=token_codec_config,
        tts_config=foreign_tts_config.to_tts_config(context_length),
    )
    with _load_checkpoint(model_spec, progress_callback) as checkpoint:
        return _load_model(
            TTSModel,
            foreign_config=foreign_tts_config,
            model_config=model_config,
            tokenizer=tokenizer,
            dtype=dtype,
            weights_dict=checkpoint.weights,
            progress_callback=progress_callback,
            implementation=implementation,
        )


def import_model(
    model_spec: ModelSpec | str,
    *,
    context_length: int | None = None,
    dtype: DTypeLike | None = None,
    progress_callback: Callable[[StatusEvent], None] | None = None,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    to_full_precision: bool = False,
) -> ImportResults:
    if isinstance(model_spec, str):
        registry_model_spec = ModelRegistry.build().repo_to_model.get(model_spec)
        if registry_model_spec is not None:
            model_spec = registry_model_spec
        else:
            model_path = Path(model_spec)
            if not model_path.exists():
                raise ValueError(f"Unknown model and local path does not exist: {model_spec}")

            if dtype is None:
                model = Model.load(model_path)
            else:
                model = Model.load(model_path, dtype=dtype)

            model_metadata = ModelMetadata(
                toolchain_version=importlib.metadata.version("lalamo"),
                vendor="local",
                family="unknown",
                name=model_path.name,
                size="unknown",
                origin=str(model_path.resolve()),
                model_config=model.config,
                grammar_start_tokens=(),
            )
            return ImportResults(model, model_metadata)

    match model_spec:
        case LanguageModelSpec():
            model = _import_language_model(
                model_spec,
                context_length=context_length,
                dtype=dtype,
                progress_callback=progress_callback,
                implementation=implementation,
            )
        case ClassifierModelSpec():
            model = _import_classifier(
                model_spec,
                context_length=context_length,
                dtype=dtype,
                progress_callback=progress_callback,
                implementation=implementation,
            )
        case TTSModelSpec():
            model = _import_tts_model(
                model_spec,
                context_length=context_length,
                dtype=dtype,
                progress_callback=progress_callback,
                implementation=implementation,
            )
        case _:
            raise TypeError(f"Unsupported model spec type: {type(model_spec).__name__}")

    if to_full_precision:
        model = model.to_full_precision()

    metadata = ModelMetadata(
        toolchain_version=importlib.metadata.version("lalamo"),
        vendor=model_spec.vendor,
        family=model_spec.family,
        name=model_spec.name,
        size=model_spec.size,
        origin=model_spec.origin.description,
        model_config=model.config,
        grammar_start_tokens=model_spec.grammar_start_tokens if isinstance(model_spec, LanguageModelSpec) else (),
    )
    return ImportResults(model, metadata)
