from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import prod
from typing import Self

import jax.numpy as jnp
from jax import lax, nn
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.common import LalamoModule

__all__ = [
    "NeuCodecConv1d",
    "NeuCodecConv1dConfig",
    "NeuCodecFSQ",
    "NeuCodecFSQConfig",
    "NeuCodecGroupNorm",
    "NeuCodecGroupNormConfig",
    "NeuCodecLinear",
    "NeuCodecLinearConfig",
    "NeuCodecResidualFSQ",
    "NeuCodecResidualFSQConfig",
    "NeuCodecResnetBlock",
    "NeuCodecResnetBlockConfig",
]


@dataclass(frozen=True)
class NeuCodecFSQConfig:
    levels: tuple[int, ...]
    precision: DTypeLike

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("NeuCodec FSQ levels must be non-empty.")
        if any(level <= 1 for level in self.levels):
            raise ValueError("NeuCodec FSQ levels must all be greater than 1.")

    @property
    def codebook_dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return prod(self.levels)

    def empty(self) -> "NeuCodecFSQ":
        levels = jnp.asarray(self.levels, dtype=jnp.int32)
        basis = jnp.cumprod(jnp.concatenate([jnp.asarray([1], dtype=jnp.int32), levels[:-1]]))
        return NeuCodecFSQ(
            config=self,
            levels=levels,
            basis=basis,
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecFSQ":
        return self.empty()


class NeuCodecFSQ(LalamoModule[NeuCodecFSQConfig]):
    levels: Int[Array, " codebook_dim"]
    basis: Int[Array, " codebook_dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def indices_to_level_indices(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Int[Array, "batch tokens codebook_dim"]:
        return (indices[..., None] // self.basis) % self.levels

    def indices_to_codes(
        self,
        indices: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens codebook_dim"]:
        level_indices = self.indices_to_level_indices(indices)
        half_width = self.levels // 2
        return (level_indices - half_width).astype(self.config.precision) / half_width.astype(self.config.precision)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "levels": self.levels,
            "basis": self.basis,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            levels=require_array(weights["levels"]),
            basis=require_array(weights["basis"]),
        )


@dataclass(frozen=True)
class NeuCodecLinearConfig:
    input_dim: int
    output_dim: int
    precision: DTypeLike
    has_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("NeuCodec linear input_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("NeuCodec linear output_dim must be positive.")

    def empty(self) -> "NeuCodecLinear":
        weights = jnp.zeros((self.output_dim, self.input_dim), dtype=self.precision)
        biases = jnp.zeros((self.output_dim,), dtype=self.precision) if self.has_bias else None
        return NeuCodecLinear(config=self, weights=weights, biases=biases)

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecLinear":
        return self.empty()


class NeuCodecLinear(LalamoModule[NeuCodecLinearConfig]):
    weights: Float[Array, "output_dim input_dim"]
    biases: Float[Array, " output_dim"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "*batch input_dim"],
    ) -> Float[Array, "*batch output_dim"]:
        output = jnp.einsum("...i,oi->...o", inputs, self.weights)
        if self.biases is not None:
            output = output + self.biases
        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]) if self.biases is not None else None,
        )


@dataclass(frozen=True)
class NeuCodecConv1dConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    precision: DTypeLike
    has_bias: bool = True

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("NeuCodec Conv1d in_channels must be positive.")
        if self.out_channels <= 0:
            raise ValueError("NeuCodec Conv1d out_channels must be positive.")
        if self.kernel_size <= 0:
            raise ValueError("NeuCodec Conv1d kernel_size must be positive.")
        if self.padding < 0:
            raise ValueError("NeuCodec Conv1d padding must be non-negative.")

    def empty(self) -> "NeuCodecConv1d":
        weights = jnp.zeros((self.out_channels, self.in_channels, self.kernel_size), dtype=self.precision)
        biases = jnp.zeros((self.out_channels,), dtype=self.precision) if self.has_bias else None
        return NeuCodecConv1d(config=self, weights=weights, biases=biases)

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecConv1d":
        return self.empty()


class NeuCodecConv1d(LalamoModule[NeuCodecConv1dConfig]):
    weights: Float[Array, "out_channels in_channels kernel_size"]
    biases: Float[Array, " out_channels"] | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        output = lax.conv_general_dilated(
            inputs,
            self.weights,
            window_strides=(1,),
            padding=((self.config.padding, self.config.padding),),
            dimension_numbers=("NHC", "OIH", "NHC"),
        )
        if self.biases is not None:
            output = output + self.biases[None, None, :]
        return output

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, Array] = {"weights": self.weights}
        if self.biases is not None:
            result["biases"] = self.biases
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]) if self.biases is not None else None,
        )


@dataclass(frozen=True)
class NeuCodecGroupNormConfig:
    channels: int
    precision: DTypeLike
    num_groups: int = 32
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("NeuCodec GroupNorm channels must be positive.")
        if self.num_groups <= 0:
            raise ValueError("NeuCodec GroupNorm num_groups must be positive.")
        if self.channels % self.num_groups != 0:
            raise ValueError("NeuCodec GroupNorm channels must be divisible by num_groups.")
        if self.eps <= 0:
            raise ValueError("NeuCodec GroupNorm eps must be positive.")

    def empty(self) -> "NeuCodecGroupNorm":
        return NeuCodecGroupNorm(
            config=self,
            weights=jnp.ones((self.channels,), dtype=self.precision),
            biases=jnp.zeros((self.channels,), dtype=self.precision),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecGroupNorm":
        return self.empty()


class NeuCodecGroupNorm(LalamoModule[NeuCodecGroupNormConfig]):
    weights: Float[Array, " channels"]
    biases: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        batch_size, sequence_length, channels = inputs.shape
        channels_per_group = channels // self.config.num_groups
        grouped_inputs = inputs.reshape(batch_size, sequence_length, self.config.num_groups, channels_per_group)
        mean = jnp.mean(grouped_inputs, axis=(1, 3), keepdims=True)
        variance = jnp.mean(jnp.square(grouped_inputs - mean), axis=(1, 3), keepdims=True)
        normalized = (grouped_inputs - mean) * lax.rsqrt(variance + self.config.eps)
        return normalized.reshape(inputs.shape) * self.weights[None, None, :] + self.biases[None, None, :]

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            weights=require_array(weights["weights"]),
            biases=require_array(weights["biases"]),
        )


@dataclass(frozen=True)
class NeuCodecResnetBlockConfig:
    channels: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("NeuCodec ResnetBlock channels must be positive.")
        NeuCodecGroupNormConfig(channels=self.channels, precision=self.precision)

    def empty(self) -> "NeuCodecResnetBlock":
        norm_config = NeuCodecGroupNormConfig(channels=self.channels, precision=self.precision)
        conv_config = NeuCodecConv1dConfig(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            precision=self.precision,
        )
        return NeuCodecResnetBlock(
            config=self,
            norm1=norm_config.empty(),
            conv1=conv_config.empty(),
            norm2=norm_config.empty(),
            conv2=conv_config.empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecResnetBlock":
        return self.empty()


class NeuCodecResnetBlock(LalamoModule[NeuCodecResnetBlockConfig]):
    norm1: NeuCodecGroupNorm
    conv1: NeuCodecConv1d
    norm2: NeuCodecGroupNorm
    conv2: NeuCodecConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        inputs: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        hidden_states = self.conv1(_swish(self.norm1(inputs)))
        hidden_states = self.conv2(_swish(self.norm2(hidden_states)))
        return inputs + hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "norm1": self.norm1.export_weights(),
            "conv1": self.conv1.export_weights(),
            "norm2": self.norm2.export_weights(),
            "conv2": self.conv2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            norm1=self.norm1.import_weights(require_tree(weights["norm1"])),
            conv1=self.conv1.import_weights(require_tree(weights["conv1"])),
            norm2=self.norm2.import_weights(require_tree(weights["norm2"])),
            conv2=self.conv2.import_weights(require_tree(weights["conv2"])),
        )


def _swish(inputs: Float[Array, "*batch"]) -> Float[Array, "*batch"]:
    return inputs * nn.sigmoid(inputs)


@dataclass(frozen=True)
class NeuCodecResidualFSQConfig:
    levels: tuple[int, ...]
    num_quantizers: int
    output_dim: int
    precision: DTypeLike

    def __post_init__(self) -> None:
        if self.num_quantizers <= 0:
            raise ValueError("NeuCodec residual FSQ num_quantizers must be positive.")
        if self.num_quantizers != 1:
            raise ValueError("NeuCodec residual FSQ currently supports num_quantizers == 1 only.")
        if self.output_dim <= 0:
            raise ValueError("NeuCodec residual FSQ output_dim must be positive.")
        NeuCodecFSQConfig(levels=self.levels, precision=self.precision)

    @property
    def codebook_dim(self) -> int:
        return len(self.levels)

    def empty(self) -> "NeuCodecResidualFSQ":
        return NeuCodecResidualFSQ(
            config=self,
            fsq=NeuCodecFSQConfig(levels=self.levels, precision=self.precision).empty(),
            project_out=NeuCodecLinearConfig(
                input_dim=self.codebook_dim,
                output_dim=self.output_dim,
                precision=self.precision,
            ).empty(),
        )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,  # noqa: ARG002
    ) -> "NeuCodecResidualFSQ":
        return self.empty()


class NeuCodecResidualFSQ(LalamoModule[NeuCodecResidualFSQConfig]):
    fsq: NeuCodecFSQ
    project_out: NeuCodecLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def get_output_from_indices(
        self,
        indices: Int[Array, "batch tokens num_quantizers"],
    ) -> Float[Array, "batch tokens output_dim"]:
        if indices.ndim != 3:
            raise ValueError(f"NeuCodec residual FSQ indices must have 3 dimensions; got {indices.shape}.")
        if indices.shape[-1] != self.config.num_quantizers:
            raise ValueError(
                "NeuCodec residual FSQ indices last dimension must match num_quantizers"
                f" ({self.config.num_quantizers}); got {indices.shape[-1]}.",
            )
        codes = self.fsq.indices_to_codes(indices[..., 0])
        return self.project_out(codes)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "fsq": self.fsq.export_weights(),
            "project_out": self.project_out.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            fsq=self.fsq.import_weights(require_tree(weights["fsq"])),
            project_out=self.project_out.import_weights(require_tree(weights["project_out"])),
        )
