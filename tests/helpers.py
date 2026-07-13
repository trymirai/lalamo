import math
from collections import defaultdict
from collections.abc import Sequence
from functools import cache

import jax
import jax.numpy as jnp
from frozendict import frozendict
from jax.sharding import AxisType, NamedSharding

from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DenseMLPConfig,
    Identity,
    LinearConfig,
    NormalizationConfig,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.token_mixers.attention import AttentionConfig, AttentionProjectionMode
from lalamo.utils.sharding import LogicalAxis, ShardingConfig

UNITS = ["", "K", "M", "G", "T", "P", "E"]


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


def build_tiny_attention_decoder(num_layers: int, kv_reuse_map: frozendict[int, int]) -> Decoder:
    model_dim = 8
    hidden_dim = 16
    vocab_size = 32
    context_length = 16

    norm_config = NormalizationConfig(
        epsilon=1e-5,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    mlp_config = DenseMLPConfig(
        linear_config=LinearConfig(),
        activation=Identity(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    rope_config = UnscaledRoPEConfig(
        base=10_000.0,
        max_sequence_length=context_length,
        head_dim=4,
    )

    layer_configs = []
    for _ in range(num_layers):
        attention_config = AttentionConfig(
            qkv_projection_config=LinearConfig(),
            out_projection_config=LinearConfig(),
            query_norm_config=None,
            key_norm_config=None,
            rope_config=rope_config,
            num_heads=2,
            num_groups=2,
            head_dim=4,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
            projection_mode=AttentionProjectionMode.QKV,
        )
        layer_configs.append(
            TransformerLayerConfig(
                pre_mixer_norm_config=norm_config,
                mixer_config=attention_config,
                post_mixer_norm_config=None,
                pre_mlp_norm_config=norm_config,
                mlp_config=mlp_config,
                post_mlp_norm_config=None,
            ),
        )

    transformer_config = TransformerConfig(
        layer_configs=tuple(layer_configs),
        output_norm_config=norm_config,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        kv_reuse_map=kv_reuse_map,
    )
    decoder_config = DecoderConfig(
        embedding_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        transformer_config=transformer_config,
        vocab_size=vocab_size,
    )
    return decoder_config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(4),
        ),
    )


__all__ = [
    "UNITS",
    "build_tiny_attention_decoder",
    "make_sharding",
    "make_test_sharding_config",
    "si",
    "unsi",
]
