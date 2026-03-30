from dataclasses import dataclass
from enum import Enum

from jax import numpy as jnp
from jaxtyping import DTypeLike

__all__ = [
    "AttentionForwardPassConfig",
    "AttentionImplementation",
    "DecoderForwardPassConfig",
    "ForwardPassConfig",
    "MLPForwardPassConfig",
    "MixerForwardPassConfig",
    "TransformerForwardPassConfig",
    "TransformerLayerForwardPassConfig",
]


class AttentionImplementation(Enum):
    STABLE_REDUCTION = "stable_reduction"
    STANDARD = "standard"
    CUDNN = "cudnn"
    TOKAMAX = "tokamax"


@dataclass(frozen=True)
class AttentionForwardPassConfig:
    implementation: AttentionImplementation = AttentionImplementation.STABLE_REDUCTION
    upcast_dtype: DTypeLike | None = jnp.float32


MixerForwardPassConfig = AttentionForwardPassConfig


@dataclass(frozen=True)
class MLPForwardPassConfig:
    moe_chunk_size_ratio: float = 0.2


@dataclass(frozen=True)
class TransformerLayerForwardPassConfig:
    mixer: MixerForwardPassConfig = AttentionForwardPassConfig()
    mlp: MLPForwardPassConfig = MLPForwardPassConfig()


@dataclass(frozen=True)
class TransformerForwardPassConfig:
    layer: TransformerLayerForwardPassConfig = TransformerLayerForwardPassConfig()


@dataclass(frozen=True)
class DecoderForwardPassConfig:
    transformer: TransformerForwardPassConfig = TransformerForwardPassConfig()


@dataclass(frozen=True)
class ForwardPassConfig:
    decoder: DecoderForwardPassConfig = DecoderForwardPassConfig()
    deterministic_ops: bool = False
    xla_autotune_level: int = 0

    @staticmethod
    def init(
        *,
        attention_implementation: AttentionImplementation = AttentionImplementation.STABLE_REDUCTION,
        attention_upcast_dtype: DTypeLike | None = jnp.float32,
        moe_chunk_size_ratio: float = 0.2,
        deterministic_ops: bool = False,
        xla_autotune_level: int = 0,
    ) -> "ForwardPassConfig":
        return ForwardPassConfig(
            decoder=DecoderForwardPassConfig(
                transformer=TransformerForwardPassConfig(
                    layer=TransformerLayerForwardPassConfig(
                        mixer=AttentionForwardPassConfig(
                            implementation=attention_implementation,
                            upcast_dtype=attention_upcast_dtype,
                        ),
                        mlp=MLPForwardPassConfig(
                            moe_chunk_size_ratio=moe_chunk_size_ratio,
                        ),
                    ),
                ),
            ),
            deterministic_ops=deterministic_ops,
            xla_autotune_level=xla_autotune_level,
        )
