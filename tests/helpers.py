import math
from collections import defaultdict
from collections.abc import Sequence
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, NamedSharding

from lalamo.initializer import RandomInitializer
from lalamo.models import LanguageModel
from lalamo.module import ForwardPassMode, Keychain
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DenseMLPConfig,
    Identity,
    LinearConfig,
    ModelForwardPassConfig,
    NormalizationConfig,
    SequenceConfig,
    StepConfig,
    TiedEmbeddingConfig,
    UnscaledRoPEConfig,
    UpcastMode,
    residual,
)
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.sharding import LogicalAxis, ShardingConfig

UNITS = ["", "K", "M", "G", "T", "P", "E"]


def dense_log_softmax_rows(model: LanguageModel, prompt: tuple[int, ...], token_ids: list[int]) -> list[np.ndarray]:
    """Teacher-forced dense reference before each token and after the last."""
    keychain = Keychain.init(0, sharding_config=model.sharding_config)
    prefilled = model.prefill_tokens(jnp.asarray([prompt]), len(prompt) + len(token_ids), keychain=keychain)
    state, logits = prefilled.state, prefilled.last_token_logits
    rows = []
    for position, token_id in enumerate(token_ids, start=len(prompt)):
        rows.append(np.asarray(jax.nn.log_softmax(logits[0].astype(jnp.float32))))
        decoded = model.decoder(
            jnp.asarray([[token_id]]),
            jnp.asarray([[position]]),
            state=state,
            return_updated_state=True,
            forward_pass_config=ModelForwardPassConfig.for_inference(ForwardPassMode.SINGLE_TOKEN),
            keychain=keychain,
        )
        assert decoded.updated_state is not None
        state, logits = decoded.updated_state, decoded.logits[:, 0]
    return [*rows, np.asarray(jax.nn.log_softmax(logits[0].astype(jnp.float32)))]


def si(x: int, base: int = 1024, units: Sequence[str] = UNITS) -> str:
    precision = math.ceil(math.log10(base))
    power = min(math.trunc(math.log(math.fabs(x), base)), len(units) - 1) if x != 0 else 0
    return f"{x / (base**power):.{precision}f} {units[power]}"


def unsi(x: str, base: int = 1024, units: Sequence[str] = UNITS) -> int:
    val, unit, *_ = [*x.split(" ", 1), ""]
    return int(float(val) * (base ** units.index(unit)))


@cache
def make_test_sharding_config() -> ShardingConfig:
    mesh = jax.make_mesh(
        (2, 2, 2),
        (LogicalAxis.BATCH.value, LogicalAxis.MATRIX.value, LogicalAxis.MIXTURE.value),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=jax.devices("cpu")[:8],
    )
    return ShardingConfig(
        mesh=mesh,
        logical_to_physical=defaultdict(
            lambda: None,
            {
                LogicalAxis.BATCH: LogicalAxis.BATCH.value,
                LogicalAxis.MATRIX: LogicalAxis.MATRIX.value,
                LogicalAxis.MIXTURE: LogicalAxis.MIXTURE.value,
            },
        ),
    )


def make_sharding(logical_axes: tuple[LogicalAxis | None, ...]) -> NamedSharding:
    sharding_config = make_test_sharding_config()
    return sharding_config.resolve_sharding(logical_axes)


def tiny_norm_config() -> NormalizationConfig:
    return NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )


def tiny_mlp_config() -> DenseMLPConfig:
    return DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
        hidden_dim=16,
    )


def tiny_attention_config(kv_source: AttentionConfig | None = None) -> AttentionConfig:
    return AttentionConfig(
        qkvg_projection_config=LinearConfig(),
        out_projection_config=LinearConfig(),
        query_norm_config=None,
        key_norm_config=None,
        num_heads=2,
        num_groups=2,
        head_dim=4,
        is_causal=True,
        scale=None,
        sliding_window_size=None,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkvg_biases=False,
        has_out_biases=False,
        kv_source=kv_source,
        rope_config=UnscaledRoPEConfig(base=10_000.0, max_sequence_length=16, head_dim=4),
    )


def build_tiny_decoder_config(mixers: Sequence[StepConfig]) -> DecoderConfig:
    """Eight-dimensional pre-norm decoder with one layer per mixer."""
    steps = [
        step
        for mixer in mixers
        for step in (residual(tiny_norm_config(), mixer), residual(tiny_norm_config(), tiny_mlp_config()))
    ]
    return DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        program=SequenceConfig((*steps, tiny_norm_config())),
        model_dim=8,
        vocab_size=32,
    )


def build_tiny_attention_decoder_config(kv_source_layer_indices: tuple[int | None, ...]) -> DecoderConfig:
    """Layer `i` borrows the KV cache of layer `kv_source_layer_indices[i]`."""

    @cache
    def layer_attention(index: int) -> AttentionConfig:
        kv_source = kv_source_layer_indices[index]
        return tiny_attention_config(None if kv_source is None else layer_attention(kv_source))

    return build_tiny_decoder_config([layer_attention(index) for index in range(len(kv_source_layer_indices))])


def build_tiny_attention_decoder(kv_source_layer_indices: tuple[int | None, ...]) -> Decoder:
    return build_tiny_attention_decoder_config(kv_source_layer_indices).init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(4),
        ),
    )


__all__ = [
    "UNITS",
    "build_tiny_attention_decoder",
    "build_tiny_attention_decoder_config",
    "build_tiny_decoder_config",
    "make_sharding",
    "make_test_sharding_config",
    "si",
    "tiny_attention_config",
    "tiny_mlp_config",
    "tiny_norm_config",
    "unsi",
]
