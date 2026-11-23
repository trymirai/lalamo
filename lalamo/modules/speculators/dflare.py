from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.residual import AdditiveResidualConfig
from lalamo.modules.rope import RoPE, RoPEConfig
from lalamo.modules.speculator import Speculator, SpeculatorConfig
from lalamo.modules.speculators.dflash import _layer_attention_config
from lalamo.modules.speculators.dspark import DSparkHead, DSparkHeadConfig
from lalamo.modules.transformer_layer import TransformerLayer, TransformerLayerConfig

__all__ = [
    "DFlareDraftConfig",
    "DFlareDraftModel",
    "DFlareLayerContext",
    "DFlareSpeculator",
    "DFlareSpeculatorConfig",
]


class DFlareLayerContext(NamedTuple):
    tap_weights: Float[Array, "taps channels"]
    norm: Normalization
    key_value_projection: Linear
    key_norm: Normalization


@dataclass(frozen=True)
class DFlareDraftConfig(LalamoConfig):
    model_dim: int
    hidden_dim: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    vocab_size: int
    context_projection_config: LinearConfig
    context_norm_config: NormalizationConfig
    rope_config: RoPEConfig
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig

    def init(self, initializer: Initializer) -> "DFlareDraftModel":
        attention_configs = tuple(_layer_attention_config(layer_config) for layer_config in self.layer_configs)
        return DFlareDraftModel(
            config=self,
            sharding_config=initializer.sharding_config,
            mask_embedding=initializer.zeros((self.model_dim,)),
            contexts=tuple(
                DFlareLayerContext(
                    tap_weights=initializer.zeros((len(self.target_layer_ids), self.model_dim), dtype=jnp.float32),
                    norm=self.context_norm_config.init(initializer, self.model_dim),
                    key_value_projection=self.context_projection_config.init(
                        initializer,
                        self.model_dim,
                        (attention_config.num_groups * attention_config.head_dim,) * 2,
                        has_biases=False,
                    ),
                    key_norm=self.context_norm_config.init(initializer, attention_config.head_dim),
                )
                for attention_config in attention_configs
            ),
            rope=self.rope_config.init(initializer),
            layers=tuple(
                layer_config.init(initializer, AdditiveResidualConfig(), self.model_dim, self.hidden_dim)
                for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
        )


class DFlareDraftModel(LalamoModule[DFlareDraftConfig]):
    mask_embedding: Float[Array, " channels"]
    contexts: tuple[DFlareLayerContext, ...]
    rope: RoPE
    layers: tuple[TransformerLayer, ...]
    output_norm: Normalization


@dataclass(frozen=True)
class DFlareSpeculatorConfig(SpeculatorConfig):
    draft_config: DFlareDraftConfig
    dspark_config: DSparkHeadConfig

    def init(self, initializer: Initializer) -> "DFlareSpeculator":
        return DFlareSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_model=self.draft_config.init(initializer),
            dspark_head=self.dspark_config.init(
                initializer,
                self.draft_config.model_dim,
                self.draft_config.vocab_size,
                len(self.draft_config.target_layer_ids),
            ),
        )


class DFlareSpeculator(Speculator[DFlareSpeculatorConfig]):
    draft_model: DFlareDraftModel
    dspark_head: DSparkHead
