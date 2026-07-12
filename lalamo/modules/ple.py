import math
from dataclasses import dataclass

import jax
from einops import rearrange
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.sharding import lookup_sharded_indices

from .activations import Activation
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped_twice

__all__ = [
    "PLELayer",
    "PLELayerConfig",
    "PerLayerEmbedding",
    "PerLayerEmbeddingConfig",
]


@dataclass(frozen=True)
class PerLayerEmbeddingConfig(LalamoConfig):
    num_ple_channels: int
    num_layers: int
    ple_vocab_size: int
    ple_embed_scale: float
    model_projection_scale: float
    input_scale: float
    linear_config: LinearConfig
    norm_config: NormalizationConfig

    def init(self, initializer: Initializer, model_dim: int) -> "PerLayerEmbedding":
        total_ple_channels = self.num_layers * self.num_ple_channels
        return PerLayerEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            token_embedding=initializer.normal(
                1 / math.sqrt(self.num_ple_channels),
                (self.ple_vocab_size, total_ple_channels),
            ),
            model_projection=self.linear_config.init(
                initializer,
                input_dim=model_dim,
                output_dims=(total_ple_channels,),
                has_biases=False,
            ),
            projection_norm=self.norm_config.init(initializer, self.num_ple_channels),
        )


class PerLayerEmbedding(LalamoModule[PerLayerEmbeddingConfig]):
    token_embedding: Float[Array, "vocab ple_total_dim"]
    model_projection: Linear
    projection_norm: Normalization

    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        inner_features: Float[Array, "batch suffix_tokens channels"],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens ple_channels"], ...]:
        config = self.config
        token_ple = lookup_sharded_indices(self.token_embedding, token_ids) * config.ple_embed_scale
        token_ple = rearrange(
            token_ple,
            "batch tokens (layers ple_channels) -> batch tokens layers ple_channels",
            layers=config.num_layers,
            ple_channels=config.num_ple_channels,
        )
        (model_ple,) = call_vmapped_twice(
            self.model_projection,
            inner_features,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        model_ple = model_ple * config.model_projection_scale
        model_ple = rearrange(
            model_ple,
            "batch tokens (layers ple_channels) -> batch tokens layers ple_channels",
            layers=config.num_layers,
            ple_channels=config.num_ple_channels,
        )
        model_ple = jax.vmap(jax.vmap(jax.vmap(self.projection_norm)))(model_ple)
        token_ple = token_ple.astype(model_ple.dtype)
        combined = (model_ple + token_ple) * config.input_scale
        return tuple(combined[:, :, layer_index, :] for layer_index in range(config.num_layers))


@dataclass(frozen=True)
class PLELayerConfig(LalamoConfig):
    linear_config: LinearConfig
    ple_channels: int
    activation: Activation

    def init(self, initializer: Initializer, model_dim: int) -> "PLELayer":
        gate = self.linear_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(self.ple_channels,),
            has_biases=False,
        )
        projection = self.linear_config.init(
            initializer,
            input_dim=self.ple_channels,
            output_dims=(model_dim,),
            has_biases=False,
        )
        return PLELayer(
            config=self,
            sharding_config=initializer.sharding_config,
            gate=gate,
            projection=projection,
        )


class PLELayer(LalamoModule[PLELayerConfig]):
    gate: Linear
    projection: Linear

    def __call__(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        per_layer_input: Float[Array, "batch suffix_tokens ple_channels"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        gate_keychain, projection_keychain = keychain.split()
        (ple_gated,) = call_vmapped_twice(
            self.gate,
            outputs,
            keychain=gate_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        ple_gated = self.config.activation(ple_gated) * per_layer_input
        (ple_projected,) = call_vmapped_twice(
            self.projection,
            ple_gated,
            keychain=projection_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        return ple_projected
