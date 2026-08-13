from dataclasses import dataclass

import jax
from einops import rearrange
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.sharding import lookup_sharded_indices
from lalamo.weight_matrix import EmbeddingMatrix

from .activations import Activation
from .linear import Linear, LinearConfig
from .utils import call_vmapped_twice

__all__ = [
    "PLELayer",
    "PLELayerConfig",
    "PerLayerEmbedding",
    "PerLayerEmbeddingConfig",
]


@dataclass(frozen=True)
class PerLayerEmbeddingConfig(LalamoConfig):
    num_layers: int
    num_channels_per_layer: int

    def init(self, initializer: Initializer, vocabulary_size: int) -> "PerLayerEmbedding":
        total_ple_channels = self.num_layers * self.num_channels_per_layer
        return PerLayerEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding=initializer.embedding_matrix(vocabulary_size, total_ple_channels),
        )


class PerLayerEmbedding(LalamoModule[PerLayerEmbeddingConfig]):
    embedding: EmbeddingMatrix

    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens ple_channels"], ...]:
        self.embedding.lookup_embedding()
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
class PLEModulatorConfig(LalamoConfig):
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
        return PLEModulator(
            config=self,
            sharding_config=initializer.sharding_config,
            gate=gate,
            projection=projection,
        )


class PLEModulator(LalamoModule[PLEModulatorConfig]):
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
