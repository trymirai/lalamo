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
class VectorQuantizeConfig:
    precision: DTypeLike
    codebook_config: TiedEmbeddingConfig
    out_proj_config: FullPrecisionLinearConfig

    def empty(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> "VectorQuantize":
        codebook = self.codebook_config.empty(codebook_size, codebook_dim)
        assert isinstance(codebook, TiedEmbedding)

        out_proj = self.out_proj_config.empty(
            input_dim=codebook_dim,
            output_dims=(input_dim,),
            has_biases=True,
        )
        assert isinstance(out_proj, FullPrecisionLinear)

        return VectorQuantize(
            config=self,
            codebook=codebook,
            out_proj=out_proj,
        )

    def random_init(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        key: PRNGKeyArray,
    ) -> "VectorQuantize":
        codebook_key, proj_key = jax.random.split(key)

        codebook = self.codebook_config.random_init(codebook_size, codebook_dim, key=codebook_key)
        assert isinstance(codebook, TiedEmbedding)

        out_proj = self.out_proj_config.random_init(
            input_dim=codebook_dim,
            output_dims=(input_dim,),
            has_biases=True,
            key=proj_key,
        )
        assert isinstance(out_proj, FullPrecisionLinear)

        return VectorQuantize(
            config=self,
            codebook=codebook,
            out_proj=out_proj,
        )


class VectorQuantize(LalamoModule[VectorQuantizeConfig]):
    """Vector Quantization module (decoding path only).

    Decodes codebook indices back to input space by:
    1. Looking up codebook vectors
    2. Projecting from codebook_dim to input_dim via out_proj
    """

    codebook: TiedEmbedding
    out_proj: FullPrecisionLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def codebook_size(self) -> int:
        return self.codebook.vocab_size

    @property
    def codebook_dim(self) -> int:
        return self.codebook.model_dim

    def decode_code(self, embed_id: Int[Array, " tokens"]) -> Float[Array, "tokens code_size"]:
        z_p = self.codebook.embed(embed_id)
        (z_q,) = vmap(self.out_proj)(z_p)
        return z_q

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "codebook": self.codebook.export_weights(),
            "out_proj": self.out_proj.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        codebook_weights = weights["codebook"]
        out_proj_weights = weights["out_proj"]
        assert isinstance(codebook_weights, Mapping)
        assert isinstance(out_proj_weights, Mapping)
        return replace(
            self,
            codebook=self.codebook.import_weights(require_tree(codebook_weights)),
            out_proj=self.out_proj.import_weights(require_tree(out_proj_weights)),
        )


@dataclass(frozen=True)
class VectorQuantizerParams:
    input_dim: int
    codebook_size: int
    codebook_dim: int | list[int]


@dataclass(frozen=True)
class ResidualVectorQuantizeConfig:
    precision: DTypeLike
    vq_config: VectorQuantizeConfig

    def empty(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | list[int],
    ) -> "ResidualVectorQuantize":
        if isinstance(codebook_dim, int):
            codebook_dims = [codebook_dim]
        else:
            codebook_dims = list(codebook_dim)

        quantizers = [
            self.vq_config.empty(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=dim,
            )
            for dim in codebook_dims
        ]

        return ResidualVectorQuantize(
            config=self,
            quantizers=tuple(quantizers),
        )

    def random_init(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | list[int],
        key: PRNGKeyArray,
    ) -> "ResidualVectorQuantize":
        if isinstance(codebook_dim, int):
            codebook_dims = [codebook_dim]
        else:
            codebook_dims = list(codebook_dim)

        quantizers = [
            self.vq_config.random_init(input_dim=input_dim, codebook_size=codebook_size, codebook_dim=dim, key=key)
            for dim in codebook_dims
        ]

        return ResidualVectorQuantize(
            config=self,
            quantizers=tuple(quantizers),
        )


class ResidualVectorQuantize(LalamoModule[ResidualVectorQuantizeConfig]):
    """Residual Vector Quantization module (decoding path only).
    Decodes codes from multiple codebooks by summing their decoded outputs.
    """

    quantizers: tuple[VectorQuantize, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def n_codebooks(self) -> int:
        return len(self.quantizers)

    def from_codes(self, codes: Int[Array, "n_codebooks tokens"]) -> Float[Array, "tokens code_size"]:
        n_codebooks = codes.shape[0]
        z_q = self.quantizers[0].decode_code(codes[0])
        for i in range(1, n_codebooks):
            z_q = z_q + self.quantizers[i].decode_code(codes[i])
        return z_q

    def __call__(self, codes: Int[Array, "batch n_codebooks tokens"]) -> Float[Array, "batch tokens code_size"]:
        return vmap(self.from_codes)(codes)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizers": [q.export_weights() for q in self.quantizers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        quantizer_weights = weights["quantizers"]
        new_quantizers = []
        for q, w in zip(self.quantizers, quantizer_weights, strict=True):
            assert isinstance(w, Mapping)
            new_quantizers.append(q.import_weights(w))
        return replace(self, quantizers=tuple(new_quantizers))


@dataclass(frozen=True)
class DownsampleResidualVectorQuantizeConfig:
    precision: DTypeLike
    semantic_quantizer_config: ResidualVectorQuantizeConfig
    quantizer_config: ResidualVectorQuantizeConfig
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

    semantic_quantizer: ResidualVectorQuantize
    quantizer: ResidualVectorQuantize
    post_module: Transformer
    upsampler: Upsampler

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def semantic_codebook_size(self) -> int:
        return self.semantic_quantizer.quantizers[0].codebook_size

    @property
    def quantizer_codebook_size(self) -> int:
        return self.quantizer.quantizers[0].codebook_size

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
