from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import prod
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.common import LalamoModule

__all__ = [
    "NeuCodecFSQ",
    "NeuCodecFSQConfig",
    "NeuCodecLinear",
    "NeuCodecLinearConfig",
    "NeuCodecResidualFSQ",
    "NeuCodecResidualFSQConfig",
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
