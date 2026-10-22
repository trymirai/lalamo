import json
from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.checkify import checkify, div_checks, nan_checks, user_checks
from jaxtyping import Array, Int
from tokenizers import Tokenizer

from lalamo.model import Model, ModelConfig
from lalamo.models import ClassifierModel, LanguageModel
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig, Message
from lalamo.modules import (
    ClassifierForwardPassConfig,
    ClassifierResult,
    DecoderForwardPassConfig,
    DecoderResult,
    Keychain,
)
from lalamo.safetensors import safe_read
from lalamo.utils.sharding import ShardingConfig

type TraceableModel = LanguageModel | ClassifierModel
type TraceResult = DecoderResult | ClassifierResult
TRACE_TOKEN_IDS_PATH = "activation_trace.token_ids"
TRACE_TOKEN_POSITIONS_PATH = "activation_trace.token_positions"


class TokenizationTrace(NamedTuple):
    arrays: Mapping[str, Array]
    metadata: dict[str, str]


def load_traceable_model(model_path: Path) -> TraceableModel:
    model = Model.load(model_path, ShardingConfig.replicated())
    if isinstance(model, (LanguageModel, ClassifierModel)):
        return model
    raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")


def load_traceable_token_codec(model_path: Path) -> ChatCodec:
    with (model_path / "config.json").open() as fd:
        config = json.load(fd)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))

    token_codec_config = ModelConfig.from_json(config).token_codec_config
    if isinstance(token_codec_config, ChatCodecConfig):
        return token_codec_config.init(tokenizer)
    raise TypeError(f"Unsupported token codec for tracing: {type(token_codec_config).__name__}")


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
        checked=False,
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
            checked=False,
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
    context = token_codec.template_context(messages)
    rendered_request = token_codec.prompt_template.render(context)
    request = {name: value for name, value in context.items() if not callable(value)}
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
    *,
    checked: bool,
) -> TraceResult:
    """checked: run eagerly under checkify (NaN, division and user checks) with the tracer-test forward pass config."""
    if isinstance(model, LanguageModel):
        if checked:
            decoder_config = DecoderForwardPassConfig.for_tracer_tests()
        else:
            decoder_config = DecoderForwardPassConfig.for_inference()
        forward = partial(model.decoder.__call__, return_updated_state=True, forward_pass_config=decoder_config)
    elif isinstance(model, ClassifierModel):
        if checked:
            classifier_config = ClassifierForwardPassConfig.for_tracer_tests()
        else:
            classifier_config = ClassifierForwardPassConfig.for_inference()
        forward = partial(model.classifier.__call__, forward_pass_config=classifier_config)
    else:
        raise TypeError(f"Unsupported model type for tracing: {type(model).__name__}")
    forward = partial(
        forward,
        token_ids=token_ids,
        token_positions=token_positions,
        return_activation_trace=True,
        keychain=Keychain.init(0, sharding_config=model.sharding_config),
    )

    with jax.set_mesh(model.sharding_config.mesh):
        if not checked:
            return forward()
        with jax.disable_jit():
            err, result = checkify(forward, errors=nan_checks | div_checks | user_checks)()
        err.throw()
        return result
