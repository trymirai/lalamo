from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_mapping, require_tree
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
    "EuclideanCodebook",
    "EuclideanCodebookConfig",
    "ResidualUnit",
    "ResidualUnitConfig",
    "ResidualVectorQuantizer",
    "ResidualVectorQuantizerConfig",
    "SnakeBeta",
    "SnakeBetaConfig",
    "VectorQuantization",
    "VectorQuantizationConfig",
]


@dataclass(frozen=True)
class EuclideanCodebookConfig:
    precision: DTypeLike
    epsilon: float = 1e-5

    def empty(self, dim: int, codebook_size: int) -> "EuclideanCodebook":
        return EuclideanCodebook(
            config=self,
            cluster_usage=jnp.ones((codebook_size,), dtype=self.precision),
            embedding_sum=jnp.zeros((codebook_size, dim), dtype=self.precision),
        )

    def random_init(self, dim: int, codebook_size: int, *, key: PRNGKeyArray) -> "EuclideanCodebook":
        return EuclideanCodebook(
            config=self,
            cluster_usage=jnp.ones((codebook_size,), dtype=self.precision),
            embedding_sum=jax.random.normal(key, (codebook_size, dim), dtype=self.precision),
        )


class EuclideanCodebook(LalamoModule[EuclideanCodebookConfig]):
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
        weights = require_mapping(weights)
        return replace(
            self,
            cluster_usage=require_array(weights["cluster_usage"]),
            embedding_sum=require_array(weights["embedding_sum"]),
        )


@dataclass(frozen=True)
class VectorQuantizationConfig:
    precision: DTypeLike
    codebook_config: EuclideanCodebookConfig
    project_out_config: FullPrecisionLinearConfig

    def empty(self, dim: int, codebook_size: int, codebook_dim: int | None = None) -> "VectorQuantization":
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
        return VectorQuantization(
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
    ) -> "VectorQuantization":
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
        return VectorQuantization(
            config=self,
            codebook=codebook,
            project_out=project_out,
            output_dim=dim,
        )


class VectorQuantization(LalamoModule[VectorQuantizationConfig]):
    codebook: EuclideanCodebook
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
        weights = require_mapping(weights)
        if self.project_out is None:
            project_out = None
        else:
            project_out = self.project_out.import_weights(require_tree(weights["project_out"]))
        return replace(
            self,
            codebook=self.codebook.import_weights(require_tree(weights["codebook"])),
            project_out=project_out,
        )


def _decode_group(
    layers: tuple[VectorQuantization, ...],
    codes: Int[Array, "batch num_quantizers tokens"],
) -> Float[Array, "batch channels tokens"]:
    codes_qbt = rearrange(codes, "batch num_quantizers tokens -> num_quantizers batch tokens")
    layer_pairs = zip(layers, codes_qbt, strict=True)
    first_layer, first_codes = next(layer_pairs)
    quantized = first_layer.decode(first_codes)
    for layer, layer_codes in layer_pairs:
        quantized = quantized + layer.decode(layer_codes)
    return quantized


def _apply_projection(
    quantized: Float[Array, "batch channels tokens"],
    projection: FullPrecisionLinear,
) -> Float[Array, "batch channels tokens"]:
    quantized_nsc = rearrange(quantized, "batch channels tokens -> batch tokens channels")
    (quantized_nsc,) = vmap(vmap(projection))(quantized_nsc)
    return rearrange(quantized_nsc, "batch tokens channels -> batch channels tokens")


def _make_layers(
    vq_config: VectorQuantizationConfig,
    num_quantizers: int,
    dim: int,
    codebook_size: int,
) -> tuple[VectorQuantization, ...]:
    return tuple(
        vq_config.empty(dim=dim, codebook_size=codebook_size, codebook_dim=dim) for _ in range(num_quantizers)
    )


def _make_layers_random(
    vq_config: VectorQuantizationConfig,
    num_quantizers: int,
    dim: int,
    codebook_size: int,
    key: PRNGKeyArray,
) -> tuple[VectorQuantization, ...]:
    keys = jax.random.split(key, num_quantizers)
    return tuple(vq_config.random_init(dim=dim, codebook_size=codebook_size, codebook_dim=dim, key=k) for k in keys)


@dataclass(frozen=True)
class ResidualVectorQuantizerConfig:
    precision: DTypeLike
    vector_quantization_config: VectorQuantizationConfig
    output_projection_config: FullPrecisionLinearConfig
    n_q_semantic: int

    def empty(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        output_dimension: int,
    ) -> "ResidualVectorQuantizer":
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        vq = self.vector_quantization_config
        semantic_layers = _make_layers(vq, self.n_q_semantic, dimension, bins)
        acoustic_layers = _make_layers(vq, n_q - self.n_q_semantic, dimension, bins)
        semantic_projection = self.output_projection_config.empty(
            input_dim=dimension,
            output_dims=(output_dimension,),
            has_biases=False,
        )
        acoustic_projection = self.output_projection_config.empty(
            input_dim=dimension,
            output_dims=(output_dimension,),
            has_biases=False,
        )

        return ResidualVectorQuantizer(
            config=self,
            semantic_layers=semantic_layers,
            acoustic_layers=acoustic_layers,
            semantic_projection=semantic_projection,
            acoustic_projection=acoustic_projection,
        )

    def random_init(
        self,
        dimension: int,
        n_q: int,
        bins: int,
        output_dimension: int,
        *,
        key: PRNGKeyArray,
    ) -> "ResidualVectorQuantizer":
        if n_q <= self.n_q_semantic:
            raise ValueError(f"n_q must be > n_q_semantic ({self.n_q_semantic}), got {n_q}")

        key_sem, key_aco, key_sem_proj, key_aco_proj = jax.random.split(key, 4)
        vq = self.vector_quantization_config
        semantic_layers = _make_layers_random(vq, self.n_q_semantic, dimension, bins, key_sem)
        acoustic_layers = _make_layers_random(vq, n_q - self.n_q_semantic, dimension, bins, key_aco)
        semantic_projection = self.output_projection_config.random_init(
            input_dim=dimension,
            output_dims=(output_dimension,),
            has_biases=False,
            key=key_sem_proj,
        )
        acoustic_projection = self.output_projection_config.random_init(
            input_dim=dimension,
            output_dims=(output_dimension,),
            has_biases=False,
            key=key_aco_proj,
        )

        return ResidualVectorQuantizer(
            config=self,
            semantic_layers=semantic_layers,
            acoustic_layers=acoustic_layers,
            semantic_projection=semantic_projection,
            acoustic_projection=acoustic_projection,
        )


class ResidualVectorQuantizer(LalamoModule[ResidualVectorQuantizerConfig]):
    semantic_layers: tuple[VectorQuantization, ...]
    acoustic_layers: tuple[VectorQuantization, ...]
    semantic_projection: FullPrecisionLinear
    acoustic_projection: FullPrecisionLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def n_q_semantic(self) -> int:
        return self.config.n_q_semantic

    def decode(
        self,
        semantic_codes: Int[Array, "batch n_q_semantic tokens"],
        acoustic_codes: Int[Array, "batch n_q_acoustic tokens"],
    ) -> Float[Array, "batch channels tokens"]:
        quant_semantic = _decode_group(self.semantic_layers, semantic_codes)
        quant_semantic = _apply_projection(quant_semantic, self.semantic_projection)
        quant_acoustic = _decode_group(self.acoustic_layers, acoustic_codes)
        quant_acoustic = _apply_projection(quant_acoustic, self.acoustic_projection)
        return quant_semantic + quant_acoustic

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "rvq_first": {
                "rvq": {"layers": [layer.export_weights() for layer in self.semantic_layers]},
                "output_projection": self.semantic_projection.export_weights(),
            },
            "rvq_rest": {
                "rvq": {"layers": [layer.export_weights() for layer in self.acoustic_layers]},
                "output_projection": self.acoustic_projection.export_weights(),
            },
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        rvq_first = require_mapping(weights["rvq_first"])
        rvq_rest = require_mapping(weights["rvq_rest"])

        first_layers = require_mapping(rvq_first["rvq"])
        first_layer_weights = first_layers["layers"]

        rest_layers = require_mapping(rvq_rest["rvq"])
        rest_layer_weights = rest_layers["layers"]

        return replace(
            self,
            semantic_layers=tuple(
                layer.import_weights(require_tree(lw))
                for layer, lw in zip(self.semantic_layers, first_layer_weights, strict=True)
            ),
            acoustic_layers=tuple(
                layer.import_weights(require_tree(lw))
                for layer, lw in zip(self.acoustic_layers, rest_layer_weights, strict=True)
            ),
            semantic_projection=self.semantic_projection.import_weights(require_tree(rvq_first["output_projection"])),
            acoustic_projection=self.acoustic_projection.import_weights(require_tree(rvq_rest["output_projection"])),
        )
