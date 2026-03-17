from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.audio.common_modules import (
    DecoderBlock,
    DecoderBlockConfig,
    ResidualUnit,
    ResidualUnitConfig,
    SnakeBeta,
    SnakeBetaConfig,
)
from lalamo.modules.common import LalamoModule
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig

__all__ = [
    "DecoderBlock",
    "DecoderBlockConfig",
    "Qwen3TTSEuclideanCodebook",
    "Qwen3TTSEuclideanCodebookConfig",
    "Qwen3TTSResidualVectorQuantization",
    "Qwen3TTSResidualVectorQuantizationConfig",
    "Qwen3TTSResidualVectorQuantizer",
    "Qwen3TTSResidualVectorQuantizerConfig",
    "Qwen3TTSSplitResidualVectorQuantizer",
    "Qwen3TTSSplitResidualVectorQuantizerConfig",
    "Qwen3TTSVectorQuantization",
    "Qwen3TTSVectorQuantizationConfig",
    "ResidualUnit",
    "ResidualUnitConfig",
    "SnakeBeta",
    "SnakeBetaConfig",
]


@dataclass(frozen=True)
class Qwen3TTSEuclideanCodebookConfig:
    precision: DTypeLike
    epsilon: float = 1e-5

    def empty(self, dim: int, codebook_size: int) -> "Qwen3TTSEuclideanCodebook":
        return Qwen3TTSEuclideanCodebook(
            config=self,
            cluster_usage=jnp.ones((codebook_size,), dtype=self.precision),
            embedding_sum=jnp.zeros((codebook_size, dim), dtype=self.precision),
        )

    def random_init(self, dim: int, codebook_size: int, *, key: PRNGKeyArray) -> "Qwen3TTSEuclideanCodebook":
        return Qwen3TTSEuclideanCodebook(
            config=self,
            cluster_usage=jnp.ones((codebook_size,), dtype=self.precision),
            embedding_sum=jax.random.normal(key, (codebook_size, dim), dtype=self.precision),
        )


class Qwen3TTSEuclideanCodebook(LalamoModule[Qwen3TTSEuclideanCodebookConfig]):
    cluster_usage: Float[Array, " codebook_size"]
    embedding_sum: Float[Array, "codebook_size dim"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def codebook_size(self) -> int:
        (codebook_size,) = self.cluster_usage.shape
        return codebook_size

    def decode(
        self,
        codes: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens dim"]:
        embedding = self.embedding_sum / jnp.clip(self.cluster_usage, min=self.config.epsilon)[:, None]
        return jnp.take(embedding, codes, axis=0)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "cluster_usage": self.cluster_usage,
            "embedding_sum": self.embedding_sum,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            cluster_usage=require_array(weights["cluster_usage"]),
            embedding_sum=require_array(weights["embedding_sum"]),
        )


@dataclass(frozen=True)
class Qwen3TTSVectorQuantizationConfig:
    precision: DTypeLike
    codebook_config: Qwen3TTSEuclideanCodebookConfig
    project_out_config: FullPrecisionLinearConfig

    def empty(self, dim: int, codebook_size: int, codebook_dim: int | None = None) -> "Qwen3TTSVectorQuantization":
        codebook_dim = dim if codebook_dim is None else codebook_dim
        codebook = self.codebook_config.empty(dim=codebook_dim, codebook_size=codebook_size)
        if codebook_dim == dim:
            project_out = None
        else:
            project_out = self.project_out_config.empty(
                input_dim=codebook_dim,
                output_dims=(dim,),
                has_biases=True,
            )
        return Qwen3TTSVectorQuantization(
            config=self,
            codebook=codebook,
            project_out=project_out,
            output_dim=dim,
        )

    def random_init(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSVectorQuantization":
        key_codebook, key_project = jax.random.split(key)
        codebook_dim = dim if codebook_dim is None else codebook_dim
        codebook = self.codebook_config.random_init(dim=codebook_dim, codebook_size=codebook_size, key=key_codebook)
        if codebook_dim == dim:
            project_out = None
        else:
            project_out = self.project_out_config.random_init(
                input_dim=codebook_dim,
                output_dims=(dim,),
                has_biases=True,
                key=key_project,
            )
        return Qwen3TTSVectorQuantization(
            config=self,
            codebook=codebook,
            project_out=project_out,
            output_dim=dim,
        )


class Qwen3TTSVectorQuantization(LalamoModule[Qwen3TTSVectorQuantizationConfig]):
    codebook: Qwen3TTSEuclideanCodebook
    project_out: FullPrecisionLinear | None
    output_dim: int

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.codebook.decode(codes)
        if self.project_out is not None:
            (quantized,) = vmap(vmap(self.project_out))(quantized)
        quantized = rearrange(quantized, "batch tokens channels -> batch channels tokens")
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        project_out_weights: ParameterTree[Array]
        if self.project_out is None:
            project_out_weights = {}
        else:
            project_out_weights = self.project_out.export_weights()
        return {
            "codebook": self.codebook.export_weights(),
            "project_out": project_out_weights,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        if self.project_out is None:
            project_out = None
        else:
            project_out = self.project_out.import_weights(require_tree(weights["project_out"]))
        return replace(
            self,
            codebook=self.codebook.import_weights(require_tree(weights["codebook"])),
            project_out=project_out,
        )


@dataclass(frozen=True)
class Qwen3TTSResidualVectorQuantizationConfig:
    precision: DTypeLike
    vector_quantization_config: Qwen3TTSVectorQuantizationConfig

    def empty(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
    ) -> "Qwen3TTSResidualVectorQuantization":
        layers = tuple(
            self.vector_quantization_config.empty(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        )
        return Qwen3TTSResidualVectorQuantization(config=self, layers=layers)

    def random_init(
        self,
        num_quantizers: int,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSResidualVectorQuantization":
        keys = jax.random.split(key, num_quantizers)
        layers = tuple(
            self.vector_quantization_config.random_init(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                key=layer_key,
            )
            for layer_key in keys
        )
        return Qwen3TTSResidualVectorQuantization(config=self, layers=layers)


class Qwen3TTSResidualVectorQuantization(LalamoModule[Qwen3TTSResidualVectorQuantizationConfig]):
    layers: tuple[Qwen3TTSVectorQuantization, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "num_quantizers batch tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        layer_pairs = zip(self.layers, codes, strict=True)
        first_layer, first_codes = next(layer_pairs)
        quantized = first_layer.decode(first_codes)
        for layer, layer_codes in layer_pairs:
            quantized = quantized + layer.decode(layer_codes)
        return quantized

    def __call__(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        return self.decode(rearrange(codes, "batch num_quantizers tokens -> num_quantizers batch tokens"))

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "layers": [layer.export_weights() for layer in self.layers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        layers_weights = weights["layers"]
        assert isinstance(layers_weights, Sequence)
        return replace(
            self,
            layers=tuple(
                layer.import_weights(require_tree(layer_weights))
                for layer, layer_weights in zip(self.layers, layers_weights, strict=True)
            ),
        )


@dataclass(frozen=True)
class Qwen3TTSResidualVectorQuantizerConfig:
    precision: DTypeLike
    rvq_config: Qwen3TTSResidualVectorQuantizationConfig
    output_projection_config: FullPrecisionLinearConfig

    def empty(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        bins: int,
        force_projection: bool = False,
    ) -> "Qwen3TTSResidualVectorQuantizer":
        if input_dimension != dimension and not force_projection:
            raise ValueError("Decode path requires force_projection when input_dimension differs from dimension.")
        if output_dimension == dimension and not force_projection:
            output_projection = None
        else:
            output_projection = self.output_projection_config.empty(
                input_dim=dimension,
                output_dims=(output_dimension,),
                has_biases=False,
            )

        return Qwen3TTSResidualVectorQuantizer(
            config=self,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q,
            bins=bins,
            rvq=self.rvq_config.empty(
                num_quantizers=n_q,
                dim=dimension,
                codebook_size=bins,
                codebook_dim=dimension,
            ),
            output_projection=output_projection,
        )

    def random_init(
        self,
        dimension: int,
        input_dimension: int,
        output_dimension: int,
        n_q: int,
        bins: int,
        *,
        key: PRNGKeyArray,
        force_projection: bool = False,
    ) -> "Qwen3TTSResidualVectorQuantizer":
        key_rvq, key_out = jax.random.split(key)
        if input_dimension != dimension and not force_projection:
            raise ValueError("Decode path requires force_projection when input_dimension differs from dimension.")
        if output_dimension == dimension and not force_projection:
            output_projection = None
        else:
            output_projection = self.output_projection_config.random_init(
                input_dim=dimension,
                output_dims=(output_dimension,),
                has_biases=False,
                key=key_out,
            )

        return Qwen3TTSResidualVectorQuantizer(
            config=self,
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q,
            bins=bins,
            rvq=self.rvq_config.random_init(
                num_quantizers=n_q,
                dim=dimension,
                codebook_size=bins,
                codebook_dim=dimension,
                key=key_rvq,
            ),
            output_projection=output_projection,
        )


class Qwen3TTSResidualVectorQuantizer(LalamoModule[Qwen3TTSResidualVectorQuantizerConfig]):
    dimension: int
    input_dimension: int
    output_dimension: int
    n_q: int
    bins: int

    rvq: Qwen3TTSResidualVectorQuantization
    output_projection: FullPrecisionLinear | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.rvq.decode(rearrange(codes, "batch num_quantizers tokens -> num_quantizers batch tokens"))
        if self.output_projection is not None:
            quantized_nsc = rearrange(quantized, "batch channels tokens -> batch tokens channels")
            (quantized_nsc,) = vmap(vmap(self.output_projection))(quantized_nsc)
            quantized = rearrange(quantized_nsc, "batch tokens channels -> batch channels tokens")
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        if self.output_projection is None:
            output_projection = {}
        else:
            output_projection = self.output_projection.export_weights()
        return {
            "rvq": self.rvq.export_weights(),
            "output_projection": output_projection,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        if self.output_projection is None:
            output_projection = None
        else:
            output_projection = self.output_projection.import_weights(require_tree(weights["output_projection"]))
        return replace(
            self,
            rvq=self.rvq.import_weights(require_tree(weights["rvq"])),
            output_projection=output_projection,
        )


@dataclass(frozen=True)
class Qwen3TTSSplitResidualVectorQuantizerConfig:
    precision: DTypeLike
    residual_vector_quantizer_config: Qwen3TTSResidualVectorQuantizerConfig
    n_q_semantic: int = 1

    def empty(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        input_dimension: int,
        output_dimension: int,
    ) -> "Qwen3TTSSplitResidualVectorQuantizer":
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        rvq_first = self.residual_vector_quantizer_config.empty(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=self.n_q_semantic,
            bins=bins,
            force_projection=True,
        )
        rvq_rest = self.residual_vector_quantizer_config.empty(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - self.n_q_semantic,
            bins=bins,
            force_projection=True,
        )

        return Qwen3TTSSplitResidualVectorQuantizer(
            config=self,
            n_q=n_q,
            n_q_semantic=self.n_q_semantic,
            rvq_first=rvq_first,
            rvq_rest=rvq_rest,
        )

    def random_init(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        input_dimension: int,
        output_dimension: int,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSSplitResidualVectorQuantizer":
        key_first, key_rest = jax.random.split(key)
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        rvq_first = self.residual_vector_quantizer_config.random_init(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=self.n_q_semantic,
            bins=bins,
            key=key_first,
            force_projection=True,
        )
        rvq_rest = self.residual_vector_quantizer_config.random_init(
            dimension=dimension,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_q=n_q - self.n_q_semantic,
            bins=bins,
            key=key_rest,
            force_projection=True,
        )

        return Qwen3TTSSplitResidualVectorQuantizer(
            config=self,
            n_q=n_q,
            n_q_semantic=self.n_q_semantic,
            rvq_first=rvq_first,
            rvq_rest=rvq_rest,
        )


class Qwen3TTSSplitResidualVectorQuantizer(LalamoModule[Qwen3TTSSplitResidualVectorQuantizerConfig]):
    n_q: int
    n_q_semantic: int
    rvq_first: Qwen3TTSResidualVectorQuantizer
    rvq_rest: Qwen3TTSResidualVectorQuantizer

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def decode(
        self,
        codes: Int[Array, "batch num_quantizers tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic, :])
        _, num_quantizers, _ = codes.shape
        if num_quantizers > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :, :])
        return quantized

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "rvq_first": self.rvq_first.export_weights(),
            "rvq_rest": self.rvq_rest.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            rvq_first=self.rvq_first.import_weights(require_tree(weights["rvq_first"])),
            rvq_rest=self.rvq_rest.import_weights(require_tree(weights["rvq_rest"])),
        )
