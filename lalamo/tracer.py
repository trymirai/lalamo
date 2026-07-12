import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict, cast

import jax
import jax.numpy as jnp
from jax.experimental.checkify import checkify, div_checks, nan_checks, user_checks
from jax.sharding import AxisType
from jaxtyping import Array, Int
from tokenizers import Tokenizer

from lalamo.exportable import ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.model import Model, ModelConfig
from lalamo.models import ClassifierModel, LanguageModel
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig, Message
from lalamo.models.language_model import GenerationConfig, LanguageModelConfig
from lalamo.modules import (
    ClassifierForwardPassConfig,
    ClassifierResult,
    DecoderConfig,
    DecoderForwardPassConfig,
    DecoderResult,
    Keychain,
)
from lalamo.safetensors import safe_read
from lalamo.utils.json import JSON
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import FullPrecisionSpec, Layout

type TraceableModel = LanguageModel | ClassifierModel
type TraceResult = DecoderResult | ClassifierResult
TRACE_TOKEN_IDS_PATH = "activation_trace.token_ids"
TRACE_TOKEN_POSITIONS_PATH = "activation_trace.token_positions"


class TokenizationTrace(NamedTuple):
    arrays: Mapping[str, Array]
    metadata: dict[str, str]


class UzuGenerationConfig(TypedDict):
    stop_token_ids: list[int]
    temperature: float | None
    top_k: int | None
    top_p: float | None
    min_p: float | None
    banned_tokens: list[int] | None


class UzuMessageProcessorConfig(TypedDict):
    prompt_template: str
    output_parser_regex: str | None
    system_role_name: str
    user_role_name: str
    assistant_role_name: str
    eos_token: str | None
    bos_token: str | None
    default_system_prompt: str | None


class UzuLanguageModelConfig(TypedDict):
    model_config: JSON
    message_processor_config: UzuMessageProcessorConfig
    generation_config: UzuGenerationConfig


class UzuModelMetadata(TypedDict):
    model_type: NotRequired[str]
    model_config: UzuLanguageModelConfig


def trace_sharding_config() -> ShardingConfig:
    devices = jax.devices()
    first_device, *_ = devices
    if first_device.platform == "cpu":
        devices = devices[:1]
    mesh = jax.make_mesh(
        (len(devices),),
        ("replica",),
        axis_types=(AxisType.Auto,),
        devices=devices,
    )
    return ShardingConfig(mesh=mesh)


def load_traceable_model(model_path: Path) -> TraceableModel:
    with (model_path / "config.json").open() as fd:
        config = json.load(fd)

    if "type" in config:
        model = Model.load(model_path, ShardingConfig.replicated())
        if isinstance(model, (LanguageModel, ClassifierModel)):
            return model
        raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")

    if config["model_type"] == "language_model":
        return _load_uzu_language_model(model_path, cast("UzuModelMetadata", config))

    raise TypeError(f"Unsupported model type for tracing: {config['model_type']}")


def load_traceable_token_codec(model_path: Path) -> ChatCodec:
    with (model_path / "config.json").open() as fd:
        config = json.load(fd)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))

    if "type" in config:
        model_config = ModelConfig.from_json(config)
        token_codec_config = model_config.token_codec_config
        if isinstance(token_codec_config, ChatCodecConfig):
            return token_codec_config.init(tokenizer)
        raise TypeError(f"Unsupported token codec for tracing: {type(token_codec_config).__name__}")

    if config["model_type"] == "language_model":
        return _uzu_language_model_config(cast("UzuModelMetadata", config)).token_codec_config.init(tokenizer)

    raise TypeError(f"Unsupported model type for tracing: {config['model_type']}")


def _load_uzu_language_model(model_path: Path, metadata: UzuModelMetadata) -> LanguageModel:
    config = _uzu_language_model_config(metadata)
    with (model_path / "model.safetensors").open("rb") as fd:
        _, uzu_arrays = safe_read(fd)
        arrays: dict[str, Array] = {}
        export_metadata = {}
        for path in uzu_arrays:
            mapped_path, spec_path, spec = _map_uzu_export_path(path, uzu_arrays[path])
            arrays[mapped_path] = uzu_arrays[path]
            if spec_path is not None:
                export_metadata[spec_path] = spec

        weight_dtype = arrays["decoder.embedding.embedding.weights"].dtype
        template = config.init(
            Tokenizer.from_file(str(model_path / "tokenizer.json")),
            EmptyInitializer(weight_dtype, ShardingConfig.replicated()),
        )
        result = template.load_exported(
            ExportResults(arrays=arrays, metadata=export_metadata),
        )

    assert isinstance(result, LanguageModel)
    return result


def _uzu_language_model_config(metadata: UzuModelMetadata) -> LanguageModelConfig:
    model_config = metadata["model_config"]
    message_config = model_config["message_processor_config"]
    generation_config = model_config["generation_config"]
    banned_tokens = generation_config["banned_tokens"]
    return LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template=message_config["prompt_template"],
            output_parser_regex=message_config["output_parser_regex"],
            system_role_name=message_config["system_role_name"],
            user_role_name=message_config["user_role_name"],
            assistant_role_name=message_config["assistant_role_name"],
            eos_token=message_config["eos_token"],
            bos_token=message_config["bos_token"],
            default_system_prompt=message_config["default_system_prompt"],
        ),
        decoder_config=DecoderConfig.from_json(model_config["model_config"]),
        generation_config=GenerationConfig(
            stop_token_ids=tuple(generation_config["stop_token_ids"]),
            temperature=generation_config["temperature"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
            min_p=generation_config["min_p"],
            banned_tokens=None if banned_tokens is None else tuple(banned_tokens),
        ),
    )


def _map_uzu_export_path(path: str, array: Array) -> tuple[str, str | None, JSON | None]:
    if path == "embedding.weights":
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Only full-precision uzu exports can be traced.")
        return (
            "decoder.embedding.embedding.weights",
            "decoder.embedding.embedding.spec",
            FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).to_json(),
        )
    if path.startswith("transformer.global_rope."):
        return ("decoder.transformer.ropes.0." + path.removeprefix("transformer.global_rope."), None, None)
    if path.endswith(".weights"):
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Only full-precision uzu exports can be traced.")
        mapped_path = f"decoder.{path}.weights"
        return (
            mapped_path,
            mapped_path.removesuffix(".weights") + ".spec",
            FullPrecisionSpec().to_json(),
        )
    return (f"decoder.{path}", None, None)


def record_message_trace(model: TraceableModel, messages: Iterable[Message]) -> TraceResult:
    result, _ = record_message_trace_with_tokenization(model, messages)
    return result


def record_message_trace_with_tokenization(
    model: TraceableModel,
    messages: Iterable[Message],
) -> tuple[TraceResult, TokenizationTrace]:
    tokenization_trace = record_tokenization_trace(model.token_codec, messages)
    token_ids = tokenization_trace.arrays[TRACE_TOKEN_IDS_PATH]
    token_positions = tokenization_trace.arrays[TRACE_TOKEN_POSITIONS_PATH]
    _check_token_trace_inputs(model, token_ids, token_positions)
    token_sharding = model.sharding_config.make_sharding((None, None))
    result = record_token_trace(
        model=model,
        token_ids=jax.device_put(token_ids, token_sharding),
        token_positions=jax.device_put(token_positions, token_sharding),
    )
    return result, tokenization_trace


def record_saved_token_trace(model: TraceableModel, trace_path: Path) -> tuple[TraceResult, dict[str, str] | None]:
    with trace_path.open("rb") as fd:
        metadata, arrays = safe_read(fd)
        token_ids = arrays[TRACE_TOKEN_IDS_PATH]
        token_positions = arrays[TRACE_TOKEN_POSITIONS_PATH]
        _check_token_trace_inputs(model, token_ids, token_positions)

    token_sharding = model.sharding_config.make_sharding((None, None))
    return (
        record_token_trace(
            model=model,
            token_ids=jax.device_put(token_ids, token_sharding),
            token_positions=jax.device_put(token_positions, token_sharding),
        ),
        metadata,
    )


def _check_token_trace_inputs(model: TraceableModel, token_ids: Array, token_positions: Array) -> None:
    if token_ids.dtype != jnp.int32:
        raise TypeError(f"{TRACE_TOKEN_IDS_PATH} must be int32, got {token_ids.dtype}")
    if token_positions.dtype != jnp.int32:
        raise TypeError(f"{TRACE_TOKEN_POSITIONS_PATH} must be int32, got {token_positions.dtype}")
    if token_ids.ndim != 2:
        raise ValueError(f"{TRACE_TOKEN_IDS_PATH} must have shape [1, suffix_tokens], got {token_ids.shape}")
    batch_size, suffix_tokens = token_ids.shape
    if batch_size != 1 or suffix_tokens == 0:
        raise ValueError(f"{TRACE_TOKEN_IDS_PATH} must have shape [1, suffix_tokens], got {token_ids.shape}")
    if token_positions.shape != token_ids.shape:
        raise ValueError(
            f"{TRACE_TOKEN_POSITIONS_PATH} shape {token_positions.shape} must match"
            f" {TRACE_TOKEN_IDS_PATH} {token_ids.shape}"
        )

    token_ids = jnp.asarray(jax.device_get(token_ids))
    token_positions = jnp.asarray(jax.device_get(token_positions))
    if isinstance(model, LanguageModel):
        vocab_size = model.decoder.vocab_size
        if model.decoder.per_layer_embedding is not None:
            vocab_size = min(vocab_size, model.decoder.per_layer_embedding.token_embedding.shape[0])
        ropes = model.decoder.transformer.ropes
    elif isinstance(model, ClassifierModel):
        vocab_size = model.classifier.embedding.vocab_size
        ropes = model.classifier.transformer.ropes
    else:
        raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")

    if bool(jnp.any(token_ids < 0).item()) or bool(jnp.any(token_ids >= vocab_size).item()):
        raise ValueError(f"{TRACE_TOKEN_IDS_PATH} must contain token ids in [0, {vocab_size})")
    if bool(jnp.any(token_positions < 0).item()):
        raise ValueError(f"{TRACE_TOKEN_POSITIONS_PATH} must contain nonnegative positions")

    if ropes:
        position_capacity = min(rope.config.max_sequence_length for rope in ropes)
        if bool(jnp.any(token_positions >= position_capacity).item()):
            raise ValueError(f"{TRACE_TOKEN_POSITIONS_PATH} must contain positions in [0, {position_capacity})")


def record_tokenization_trace(token_codec: ChatCodec, messages: Iterable[Message]) -> TokenizationTrace:
    messages = tuple(messages)
    request = token_codec.request_to_dict(messages, enable_thinking=True)
    rendered_request = token_codec.render_request(messages)
    encoding = token_codec.tokenizer.encode(rendered_request, add_special_tokens=False)
    token_ids = jnp.asarray(encoding.ids, dtype=jnp.int32)[None, :]
    batch, tokens = token_ids.shape
    assert batch == 1
    if tokens == 0:
        raise ValueError("Tokenization trace must contain at least one token.")
    return TokenizationTrace(
        arrays={
            TRACE_TOKEN_IDS_PATH: token_ids,
            TRACE_TOKEN_POSITIONS_PATH: jnp.arange(tokens, dtype=jnp.int32)[None, :],
        },
        metadata={
            "add_special_tokens": "false",
            "prompt_template": token_codec.config.prompt_template,
            "rendered_request": rendered_request,
            "request": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            "tokens": json.dumps(encoding.tokens, ensure_ascii=False, separators=(",", ":")),
        },
    )


def export_trace_result(result: TraceResult) -> Mapping[str, Array]:
    arrays = dict(result.export().arrays)
    if result.activation_trace is not None:
        trace_dtype = result.activation_trace.output_norm.dtype
        arrays = {
            path: array.astype(trace_dtype)
            if path.endswith("logits") and jnp.issubdtype(array.dtype, jnp.floating)
            else array
            for path, array in arrays.items()
        }
    return arrays


def record_token_trace(
    model: TraceableModel,
    token_ids: Int[Array, "batch suffix_tokens"],
    token_positions: Int[Array, "batch suffix_tokens"],
) -> TraceResult:
    keychain = Keychain.init(0, sharding_config=model.sharding_config)

    with jax.set_mesh(model.sharding_config.mesh), jax.disable_jit():
        if isinstance(model, LanguageModel):
            return model.decoder(
                token_ids=token_ids,
                token_positions=token_positions,
                return_updated_state=True,
                return_activation_trace=True,
                forward_pass_config=DecoderForwardPassConfig.for_inference(),
                keychain=keychain,
            )

        if isinstance(model, ClassifierModel):
            return model.classifier(
                token_ids=token_ids,
                token_positions=token_positions,
                return_activation_trace=True,
                forward_pass_config=ClassifierForwardPassConfig.for_inference(),
                keychain=keychain,
            )

    raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")


def record_checked_token_trace(
    model: TraceableModel,
    token_ids: Int[Array, "batch suffix_tokens"],
    token_positions: Int[Array, "batch suffix_tokens"],
) -> TraceResult:
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    errors = nan_checks | div_checks | user_checks

    with jax.set_mesh(model.sharding_config.mesh), jax.disable_jit():
        if isinstance(model, LanguageModel):
            err, result = checkify(model.decoder.__call__, errors=errors)(
                token_ids=token_ids,
                token_positions=token_positions,
                return_updated_state=True,
                return_activation_trace=True,
                forward_pass_config=DecoderForwardPassConfig.for_tracer_tests(),
                keychain=keychain,
            )
            err.throw()
            assert isinstance(result, DecoderResult)
            return result

        if isinstance(model, ClassifierModel):
            err, result = checkify(model.classifier.__call__, errors=errors)(
                token_ids=token_ids,
                token_positions=token_positions,
                return_activation_trace=True,
                forward_pass_config=ClassifierForwardPassConfig.for_tracer_tests(),
                keychain=keychain,
            )
            err.throw()
            assert isinstance(result, ClassifierResult)
            return result

    raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")
