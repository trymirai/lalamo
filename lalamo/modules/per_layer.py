import math
from dataclasses import dataclass

import jax
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoModule, LogicalAxis
from lalamo.weight_matrix import EmbeddingMatrix

from .flow import Context, Op, StepConfig, Value
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped_twice

__all__ = [
    "MultiplyByLayerInput",
    "PerLayerInputs",
    "PerLayerInputsConfig",
    "Projection",
    "ProjectionConfig",
    "Scale",
    "ScaleConfig",
]


@dataclass(frozen=True)
class PerLayerInputsConfig(StepConfig):
    """Gemma's per-layer inputs: a token embedding plus a projection of the hidden state, one slice per layer."""

    ple_dim: int
    num_layers: int
    ple_vocab_size: int
    ple_embed_scale: float
    model_projection_scale: float
    input_scale: float
    linear_config: LinearConfig
    norm_config: NormalizationConfig

    def init(self, initializer: Initializer, model_dim: int) -> "PerLayerInputs":
        total_ple_dim = self.num_layers * self.ple_dim
        return PerLayerInputs(
            config=self,
            sharding_config=initializer.sharding_config,
            token_embedding=initializer.embedding_matrix(
                self.ple_vocab_size,
                total_ple_dim,
                standard_deviation=1 / math.sqrt(self.ple_dim),
            ),
            model_projection=self.linear_config.init(initializer, model_dim, (total_ple_dim,), has_biases=False),
            projection_norm=self.norm_config.init(initializer, self.ple_dim),
        )


class PerLayerInputs(LalamoModule[PerLayerInputsConfig]):
    token_embedding: EmbeddingMatrix
    model_projection: Linear
    projection_norm: Normalization

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Array,
        ctx: Context,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens layers ple_channels"]:
        () = operands
        if ctx.token_ids is None:
            raise ValueError("Per-layer inputs require raw token IDs.")
        config = self.config
        matmul_config = ctx.forward_pass_config.matmul_config
        embedding_keychain, projection_keychain = keychain.split()
        token_inputs = self.token_embedding.lookup_embedding(
            ctx.token_ids,
            dtype=x.dtype,
            keychain=embedding_keychain,
            forward_pass_config=matmul_config,
        )
        (model_inputs,) = call_vmapped_twice(
            self.model_projection,
            x,
            forward_pass_config=matmul_config,
            keychain=projection_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        token_inputs, model_inputs = (
            rearrange(
                inputs,
                "batch tokens (layers ple_channels) -> batch tokens layers ple_channels",
                layers=config.num_layers,
            )
            for inputs in (token_inputs * config.ple_embed_scale, model_inputs * config.model_projection_scale)
        )
        model_inputs = jax.vmap(jax.vmap(jax.vmap(self.projection_norm)))(model_inputs)
        return (model_inputs + token_inputs) * config.input_scale


@dataclass(frozen=True)
class ProjectionConfig(StepConfig):
    """A linear map of the stream; a missing dimension stands for the model dimension."""

    linear_config: LinearConfig
    input_dim: int | None = None
    output_dim: int | None = None

    def init(self, initializer: Initializer, model_dim: int) -> "Projection":
        return Projection(
            config=self,
            sharding_config=initializer.sharding_config,
            linear=self.linear_config.init(
                initializer,
                self.input_dim or model_dim,
                (self.output_dim or model_dim,),
                has_biases=False,
            ),
        )


class Projection(LalamoModule[ProjectionConfig]):
    linear: Linear

    def forward(
        self,
        x: Float[Array, "batch tokens in_channels"],
        *operands: Array,
        ctx: Context,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens out_channels"]:
        () = operands
        (outputs,) = call_vmapped_twice(
            self.linear,
            x,
            forward_pass_config=ctx.forward_pass_config.matmul_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        return outputs


@dataclass(frozen=True)
class MultiplyByLayerInput(Op):
    """Multiplies the stream by one layer's slice of the emitted per-layer inputs."""

    layer_index: int

    def forward(self, x: Value, *operands: Value, ctx: Context, keychain: Keychain) -> Value:  # noqa: ARG002
        (per_layer_inputs,) = operands
        return x * per_layer_inputs[:, :, self.layer_index]


@dataclass(frozen=True)
class ScaleConfig(StepConfig):
    def init(self, initializer: Initializer, model_dim: int) -> "Scale":  # noqa: ARG002
        return Scale(config=self, sharding_config=initializer.sharding_config, scale=initializer.ones((1,)))


class Scale(LalamoModule[ScaleConfig]):
    scale: Float[Array, "1"]

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Array,
        ctx: Context,  # noqa: ARG002
        keychain: Keychain,  # noqa: ARG002
    ) -> Float[Array, "batch tokens channels"]:
        () = operands
        return x * self.scale.astype(x.dtype)
