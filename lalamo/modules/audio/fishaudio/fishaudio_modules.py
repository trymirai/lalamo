from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.common_modules import (
    ConvNeXtSpatialParams,
    TransposeConvSpatialParams,
    UpsamplingBlock,
    UpsamplingBlockConfig,
)
from lalamo.modules.common import ForwardPassMode, LalamoModule
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.transformer import Transformer, TransformerConfig


@dataclass(frozen=True)
class UpsamplerConfig:
    block_configs: tuple[UpsamplingBlockConfig, ...]

    def empty(
        self,
        trans_conv_params_per_block: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
    ) -> "Upsampler":
        assert len(self.block_configs) == len(trans_conv_params_per_block), (
            f"Number of block configs ({len(self.block_configs)}) must match "
            f"number of block params ({len(trans_conv_params_per_block)})"
        )

        blocks = []
        for config, trans_conv_params in zip(self.block_configs, trans_conv_params_per_block, strict=True):
            block = config.empty(
                trans_conv_params=trans_conv_params,
                convnext_spatial_params=convnext_spatial_params,
            )
            blocks.append(block)

        return Upsampler(config=self, blocks=tuple(blocks))

    def random_init(
        self,
        trans_conv_params_per_block: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "Upsampler":
        assert len(self.block_configs) == len(trans_conv_params_per_block)

        blocks = []
        keys = jax.random.split(key, len(self.block_configs))
        for config, trans_conv_params, k in zip(self.block_configs, trans_conv_params_per_block, keys, strict=True):
            block = config.random_init(
                trans_conv_params=trans_conv_params,
                convnext_spatial_params=convnext_spatial_params,
                key=k,
            )
            blocks.append(block)

        return Upsampler(config=self, blocks=tuple(blocks))


class Upsampler(LalamoModule[UpsamplerConfig]):
    """Full upsampler module consisting of multiple UpsamplingBlocks.

    This module sequentially applies a series of upsampling blocks to progressively
    increase the temporal resolution of the input while transforming channel dimensions.

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, channels) - NSC format (JAX convention)
    """

    blocks: tuple[UpsamplingBlock, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        if len(self.blocks) > 0:
            return self.blocks[0].activation_precision
        raise ValueError("Upsampler has no blocks")

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        for block in self.blocks:
            x = block(x)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "blocks": [block.export_weights() for block in self.blocks],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "Upsampler":
        assert isinstance(weights, Mapping)
        block_weights = weights["blocks"]
        new_blocks = []
        for block, w in zip(self.blocks, block_weights, strict=True):
            assert isinstance(w, Mapping)
            new_blocks.append(block.import_weights(w))

        return replace(self, blocks=tuple(new_blocks))


@dataclass(frozen=True)
class VectorQuantizerParams:
    input_dim: int
    codebook_size: int
    codebook_dim: int | list[int]


@dataclass(frozen=True)
class VectorQuantizeConfig:
    precision: DTypeLike
    codebook_config: TiedEmbeddingConfig
    out_proj_config: FullPrecisionLinearConfig

    def empty(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | list[int],
    ) -> "VectorQuantize":
        codebook_dims = [codebook_dim] if isinstance(codebook_dim, int) else list(codebook_dim)
        codebooks = tuple(self.codebook_config.empty(codebook_size, dim) for dim in codebook_dims)
        out_projs = tuple(
            self.out_proj_config.empty(input_dim=dim, output_dims=(input_dim,), has_biases=True)
            for dim in codebook_dims
        )
        return VectorQuantize(config=self, codebooks=codebooks, out_projs=out_projs)

    def random_init(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | list[int],
        key: PRNGKeyArray,
    ) -> "VectorQuantize":
        codebook_dims = [codebook_dim] if isinstance(codebook_dim, int) else list(codebook_dim)
        codebook_keys = jax.random.split(key, 2 * len(codebook_dims))
        codebooks = tuple(
            self.codebook_config.random_init(codebook_size, dim, key=k)
            for dim, k in zip(codebook_dims, codebook_keys[: len(codebook_dims)], strict=True)
        )
        out_projs = tuple(
            self.out_proj_config.random_init(input_dim=dim, output_dims=(input_dim,), has_biases=True, key=k)
            for dim, k in zip(codebook_dims, codebook_keys[len(codebook_dims) :], strict=True)
        )
        return VectorQuantize(config=self, codebooks=codebooks, out_projs=out_projs)


class VectorQuantize(LalamoModule[VectorQuantizeConfig]):
    """Residual vector quantizer (decoding path). A single codebook is just len(codebooks) == 1."""

    codebooks: tuple[TiedEmbedding, ...]
    out_projs: tuple[FullPrecisionLinear, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    def _decode_layer(
        self,
        codebook: TiedEmbedding,
        out_proj: FullPrecisionLinear,
        embed_id: Int[Array, " tokens"],
    ) -> Float[Array, "tokens code_size"]:
        z_p = codebook.embed(embed_id)
        (z_q,) = vmap(out_proj)(z_p)
        return z_q

    def from_codes(self, codes: Int[Array, "num_codebooks tokens"]) -> Float[Array, "tokens code_size"]:
        decoded = [
            self._decode_layer(codebook, out_proj, layer_codes)
            for codebook, out_proj, layer_codes in zip(self.codebooks, self.out_projs, codes, strict=True)
        ]
        return jnp.sum(jnp.stack(decoded, axis=0), axis=0)

    def __call__(self, codes: Int[Array, "batch num_codebooks tokens"]) -> Float[Array, "batch tokens code_size"]:
        return vmap(self.from_codes)(codes)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizers": [
                {"codebook": codebook.export_weights(), "out_proj": out_proj.export_weights()}
                for codebook, out_proj in zip(self.codebooks, self.out_projs, strict=True)
            ],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        layer_weights = weights["quantizers"]
        new_codebooks: list[TiedEmbedding] = []
        new_out_projs: list[FullPrecisionLinear] = []
        for codebook, out_proj, w in zip(self.codebooks, self.out_projs, layer_weights, strict=True):
            assert isinstance(w, Mapping)
            new_codebooks.append(codebook.import_weights(require_tree(w["codebook"])))
            new_out_projs.append(out_proj.import_weights(require_tree(w["out_proj"])))
        return replace(self, codebooks=tuple(new_codebooks), out_projs=tuple(new_out_projs))


@dataclass(frozen=True)
class DownsampleResidualVectorQuantizeConfig:
    precision: DTypeLike
    semantic_quantizer_config: VectorQuantizeConfig
    quantizer_config: VectorQuantizeConfig
    post_module_config: TransformerConfig
    upsampler_config: UpsamplerConfig

    def empty(
        self,
        upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_quantizer_params: VectorQuantizerParams,
        quantizer_params: VectorQuantizerParams,
    ) -> "DownsampleResidualVectorQuantize":
        semantic_quantizer = self.semantic_quantizer_config.empty(
            input_dim=semantic_quantizer_params.input_dim,
            codebook_size=semantic_quantizer_params.codebook_size,
            codebook_dim=semantic_quantizer_params.codebook_dim,
        )
        quantizer = self.quantizer_config.empty(
            input_dim=quantizer_params.input_dim,
            codebook_size=quantizer_params.codebook_size,
            codebook_dim=quantizer_params.codebook_dim,
        )
        post_module = self.post_module_config.empty()
        upsampler = self.upsampler_config.empty(
            trans_conv_params_per_block=upsampler_trans_conv_params,
            convnext_spatial_params=convnext_spatial_params,
        )

        return DownsampleResidualVectorQuantize(
            config=self,
            semantic_quantizer=semantic_quantizer,
            quantizer=quantizer,
            post_module=post_module,
            upsampler=upsampler,
        )

    def random_init(
        self,
        upsampler_trans_conv_params: tuple[TransposeConvSpatialParams, ...],
        convnext_spatial_params: ConvNeXtSpatialParams,
        semantic_quantizer_params: VectorQuantizerParams,
        quantizer_params: VectorQuantizerParams,
        *,
        key: PRNGKeyArray,
    ) -> "DownsampleResidualVectorQuantize":
        key1, key2, key3, key4 = jax.random.split(key, 4)

        semantic_quantizer = self.semantic_quantizer_config.random_init(
            input_dim=semantic_quantizer_params.input_dim,
            codebook_size=semantic_quantizer_params.codebook_size,
            codebook_dim=semantic_quantizer_params.codebook_dim,
            key=key1,
        )
        quantizer = self.quantizer_config.random_init(
            input_dim=quantizer_params.input_dim,
            codebook_size=quantizer_params.codebook_size,
            codebook_dim=quantizer_params.codebook_dim,
            key=key2,
        )
        post_module = self.post_module_config.random_init(key=key3)
        upsampler = self.upsampler_config.random_init(
            trans_conv_params_per_block=upsampler_trans_conv_params,
            convnext_spatial_params=convnext_spatial_params,
            key=key4,
        )

        return DownsampleResidualVectorQuantize(
            config=self,
            semantic_quantizer=semantic_quantizer,
            quantizer=quantizer,
            post_module=post_module,
            upsampler=upsampler,
        )


class DownsampleResidualVectorQuantize(LalamoModule[DownsampleResidualVectorQuantizeConfig]):
    """Downsampled Residual Vector Quantization decoder module.

    This module decodes audio codes by:
    1. Decoding semantic codes through the semantic quantizer
    2. Decoding residual codes through the residual quantizer
    3. Summing the semantic and residual representations
    4. Processing through a transformer post-module
    5. Upsampling to the target temporal resolution

    Input: Integer codes with shape (batch, n_codebooks, tokens)
           where the first codebook row contains semantic codes
           and remaining rows contain residual codes.
    Output: Continuous audio features with shape (batch, upsampled_tokens, channels)
    """

    semantic_quantizer: VectorQuantize
    quantizer: VectorQuantize
    post_module: Transformer
    upsampler: Upsampler

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def semantic_codebook_size(self) -> int:
        return self.semantic_quantizer.codebooks[0].vocab_size

    @property
    def quantizer_codebook_size(self) -> int:
        return self.quantizer.codebooks[0].vocab_size

    def decode(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch upsampled_tokens channels"]:
        semantic_indices = jnp.clip(indices[:, :1], 0, self.semantic_codebook_size - 1)
        residual_indices = jnp.clip(indices[:, 1:], 0, self.quantizer_codebook_size - 1)

        z_q_semantic = vmap(self.semantic_quantizer.from_codes)(semantic_indices)
        z_q_residual = vmap(self.quantizer.from_codes)(residual_indices)

        z_q = z_q_semantic + z_q_residual

        batch_size, seq_length, _ = z_q.shape
        token_positions = jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))

        post_result = self.post_module(
            inner_features=z_q,
            token_positions=token_positions,
            state=None,
            return_updated_state=False,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        z_q = post_result.outputs

        z_q = self.upsampler(z_q)

        return z_q

    def __call__(
        self,
        indices: Int[Array, "batch n_codebooks tokens"],
    ) -> Float[Array, "batch upsampled_tokens channels"]:
        return self.decode(indices)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "semantic_quantizer": self.semantic_quantizer.export_weights(),
            "quantizer": self.quantizer.export_weights(),
            "post_module": self.post_module.export_weights(),
            "upsampler": self.upsampler.export_weights(),
        }

    def import_weights(self, weights: ParameterTree) -> Self:
        assert isinstance(weights, Mapping)

        semantic_quantizer_weights = weights["semantic_quantizer"]
        quantizer_weights = weights["quantizer"]
        post_module_weights = weights["post_module"]
        upsampler_weights = weights["upsampler"]

        assert isinstance(semantic_quantizer_weights, Mapping)
        assert isinstance(quantizer_weights, Mapping)
        assert isinstance(post_module_weights, Mapping)
        assert isinstance(upsampler_weights, Mapping)

        return replace(
            self,
            semantic_quantizer=self.semantic_quantizer.import_weights(semantic_quantizer_weights),
            quantizer=self.quantizer.import_weights(quantizer_weights),
            post_module=self.post_module.import_weights(post_module_weights),
            upsampler=self.upsampler.import_weights(upsampler_weights),
        )
