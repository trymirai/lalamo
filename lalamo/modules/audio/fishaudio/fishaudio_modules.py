import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import lax, vmap
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array, require_tree
from lalamo.modules.activations import Activation
from lalamo.modules.audio.common_modules import (
    CausalConv1d,
    CausalConv1dConfig,
    CausalTransposeConv1d,
    CausalTransposeConv1dConfig,
    Snake1d,
    Snake1dConfig,
)
from lalamo.modules.common import ForwardPassMode, LalamoModule
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.transformer import Transformer, TransformerConfig


@dataclass(frozen=True)
class ConvNeXtSpatialParams:
    """Spatial parameters for ConvNeXt blocks.

    These parameters control the spatial convolution and MLP expansion in ConvNeXt blocks.
    """

    # NOTE: default values are taken from the code and they do not seem to be present in
    # the config at all
    layer_scale_init_value: float = 1e-6
    mlp_ratio: float = 4.0
    kernel_size: int = 7
    dilation: int = 1


@dataclass(frozen=True)
class ConvNeXtBlockConfig:
    precision: DTypeLike
    activation: Activation
    dwconv_config: CausalConv1dConfig
    norm_config: NormalizationConfig
    pwconv_config: FullPrecisionLinearConfig

    def random_init(
        self,
        dim: int,
        spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "ConvNeXtBlock":
        key1, key2, key3 = jax.random.split(key, 3)

        dwconv = self.dwconv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=dim,
            key=key1,
        )

        norm = self.norm_config.init(dim)

        hidden_dim = int(spatial_params.mlp_ratio * dim)
        pwconv1 = self.pwconv_config.random_init(dim, (hidden_dim,), has_biases=True, key=key2)
        pwconv2 = self.pwconv_config.random_init(hidden_dim, (dim,), has_biases=True, key=key3)

        return ConvNeXtBlock(
            config=self,
            depthwise_conv=dwconv,
            norm=norm,
            pointwise_conv_step1=pwconv1,
            pointwise_conv_step2=pwconv2,
        )

    def empty(
        self,
        dim: int,
        spatial_params: ConvNeXtSpatialParams,
    ) -> "ConvNeXtBlock":
        dwconv = self.dwconv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=dim,
        )

        norm = self.norm_config.empty(dim)

        hidden_dim = int(spatial_params.mlp_ratio * dim)
        pwconv1 = self.pwconv_config.empty(dim, (hidden_dim,), has_biases=True)
        pwconv2 = self.pwconv_config.empty(hidden_dim, (dim,), has_biases=True)

        return ConvNeXtBlock(
            config=self,
            depthwise_conv=dwconv,
            norm=norm,
            pointwise_conv_step1=pwconv1,
            pointwise_conv_step2=pwconv2,
        )


class ConvNeXtBlock(LalamoModule[ConvNeXtBlockConfig]):
    """ConvNeXt block implementation.

    Architecture:
    1. DwConv (depthwise causal conv)
    2. LayerNorm
    3. Pointwise conv 1 (expand)
    4. GELU
    5. Pointwise conv 2 (project)
    6. Layer scale (gamma)
    7. Residual connection

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
    """

    depthwise_conv: CausalConv1d
    norm: Normalization
    pointwise_conv_step1: FullPrecisionLinear
    pointwise_conv_step2: FullPrecisionLinear

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def dim(self) -> int:
        return self.depthwise_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
        apply_residual: bool = True,
    ) -> Float[Array, "batch sequence channels"]:
        residual = x

        x = self.depthwise_conv(x)
        x = jax.vmap(jax.vmap(self.norm))(x)
        (x,) = jax.vmap(jax.vmap(self.pointwise_conv_step1))(x)
        x = jax.vmap(jax.vmap(self.config.activation))(x)
        (x,) = jax.vmap(jax.vmap(self.pointwise_conv_step2))(x)
        if apply_residual:
            x = residual + x

        return x

    def export_weights(self) -> ParameterTree[Array]:
        result: dict[str, ParameterTree[Array]] = {
            "dwconv": self.depthwise_conv.export_weights(),
            "norm": self.norm.export_weights(),
            "pwconv1": self.pointwise_conv_step1.export_weights(),
            "pwconv2": self.pointwise_conv_step2.export_weights(),
        }
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> "ConvNeXtBlock":
        assert isinstance(weights, Mapping)
        dwconv_weights = weights["dwconv"]
        norm_weights = weights["norm"]
        pwconv1_weights = weights["pwconv1"]
        pwconv2_weights = weights["pwconv2"]
        assert isinstance(dwconv_weights, Mapping)
        assert isinstance(norm_weights, Mapping)
        assert isinstance(pwconv1_weights, Mapping)
        assert isinstance(pwconv2_weights, Mapping)

        return replace(
            self,
            depthwise_conv=self.depthwise_conv.import_weights(require_tree(dwconv_weights)),
            norm=self.norm.import_weights(require_tree(norm_weights)),
            pointwise_conv_step1=self.pointwise_conv_step1.import_weights(require_tree(pwconv1_weights)),
            pointwise_conv_step2=self.pointwise_conv_step2.import_weights(require_tree(pwconv2_weights)),
        )


@dataclass(frozen=True)
class TransposeConvSpatialParams:
    in_channels: int
    out_channels: int
    upsample_kernel_size: int
    upsample_stride: int


@dataclass(frozen=True)
class UpsamplingBlockConfig:
    precision: DTypeLike
    trans_conv_config: CausalTransposeConv1dConfig
    convnext_config: ConvNeXtBlockConfig

    def random_init(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "UpsamplingBlock":
        key1, key2 = jax.random.split(key)

        trans_conv = self.trans_conv_config.random_init(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
            key=key1,
        )

        convnext = self.convnext_config.random_init(
            dim=trans_conv_params.out_channels,
            spatial_params=convnext_spatial_params,
            key=key2,
        )

        return UpsamplingBlock(
            config=self,
            trans_conv=trans_conv,
            convnext=convnext,
        )

    def empty(
        self,
        trans_conv_params: TransposeConvSpatialParams,
        convnext_spatial_params: ConvNeXtSpatialParams,
    ) -> "UpsamplingBlock":
        trans_conv = self.trans_conv_config.empty(
            in_channels=trans_conv_params.in_channels,
            out_channels=trans_conv_params.out_channels,
            kernel_size=trans_conv_params.upsample_kernel_size,
            stride=trans_conv_params.upsample_stride,
        )

        convnext = self.convnext_config.empty(
            dim=trans_conv_params.out_channels,
            spatial_params=convnext_spatial_params,
        )

        return UpsamplingBlock(
            config=self,
            trans_conv=trans_conv,
            convnext=convnext,
        )


class UpsamplingBlock(LalamoModule[UpsamplingBlockConfig]):
    """Upsampling block consisting of transposed convolution followed by ConvNeXt block.

    Architecture:
    1. CausalTransposeConv1d (upsample)
    2. ConvNeXtBlock (refine)

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, channels) - NSC format (JAX convention)
    """

    trans_conv: CausalTransposeConv1d
    convnext: ConvNeXtBlock

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def in_channels(self) -> int:
        return self.trans_conv.in_channels

    @property
    def out_channels(self) -> int:
        return self.trans_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.trans_conv(x)
        x = self.convnext(x)

        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "trans_conv": self.trans_conv.export_weights(),
            "convnext": self.convnext.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "UpsamplingBlock":
        assert isinstance(weights, Mapping)
        trans_conv_weights = weights["trans_conv"]
        convnext_weights = weights["convnext"]
        assert isinstance(trans_conv_weights, Mapping)
        assert isinstance(convnext_weights, Mapping)

        return replace(
            self,
            trans_conv=self.trans_conv.import_weights(require_tree(trans_conv_weights)),
            convnext=self.convnext.import_weights(require_tree(convnext_weights)),
        )


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


@dataclass(frozen=True)
class ResidualUnitSpatialParams:
    dilation: int = 1
    kernel_size: int = 7


@dataclass(frozen=True)
class ResidualUnitConfig:
    precision: DTypeLike
    snake_config: Snake1dConfig
    conv_config: CausalConv1dConfig
    causal: bool = True

    def empty(
        self,
        dim: int,
        spatial_params: ResidualUnitSpatialParams,
    ) -> "ResidualUnit":
        if not self.causal:
            raise NotImplementedError("Non-causal ResidualUnit is not implemented")

        snake1 = self.snake_config.empty(dim)

        conv1 = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=1,
        )

        snake2 = self.snake_config.empty(dim)

        conv2 = self.conv_config.empty(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
        )

        return ResidualUnit(
            config=self,
            snake1=snake1,
            conv1=conv1,
            snake2=snake2,
            conv2=conv2,
        )

    def random_init(
        self,
        dim: int,
        spatial_params: ResidualUnitSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "ResidualUnit":
        if not self.causal:
            raise NotImplementedError("Non-causal ResidualUnit is not implemented")

        key1, key2 = jax.random.split(key, 2)

        snake1 = self.snake_config.random_init(dim)

        conv1 = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=spatial_params.kernel_size,
            stride=1,
            dilation=spatial_params.dilation,
            groups=1,
            key=key1,
        )

        snake2 = self.snake_config.random_init(dim)

        conv2 = self.conv_config.random_init(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            key=key2,
        )

        return ResidualUnit(
            config=self,
            snake1=snake1,
            conv1=conv1,
            snake2=snake2,
            conv2=conv2,
        )


class ResidualUnit(LalamoModule[ResidualUnitConfig]):
    """ResidualUnit module.

    Architecture:
    1. Snake1d activation
    2. Conv1d (kernel_size=7, with dilation)
    3. Snake1d activation
    4. Conv1d (kernel_size=1)
    5. Residual connection

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence, channels) - NSC format (JAX convention)
    """

    snake1: Snake1d
    conv1: CausalConv1d
    snake2: Snake1d
    conv2: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def dim(self) -> int:
        return self.snake1.channels

    def __call__(
        self,
        x: Float[Array, "batch sequence channels"],
    ) -> Float[Array, "batch sequence channels"]:
        """Apply ResidualUnit.

        Args:
            x: Input tensor of shape (batch, sequence, channels)

        Returns:
            Output tensor of shape (batch, sequence, channels)
        """
        # Forward through the block
        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)

        # Handle padding difference for residual connection
        # In causal mode, output may be shorter than input due to causal padding
        pad = x.shape[1] - y.shape[1]
        if pad > 0:
            # For causal: trim from the end of x
            x = x[:, :-pad, :]

        return x + y

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "snake1": self.snake1.export_weights(),
            "conv1": self.conv1.export_weights(),
            "snake2": self.snake2.export_weights(),
            "conv2": self.conv2.export_weights(),
        }

    def import_weights(self, weights: ParameterTree) -> "ResidualUnit":
        assert isinstance(weights, Mapping)
        snake1_weights = weights["snake1"]
        conv1_weights = weights["conv1"]
        snake2_weights = weights["snake2"]
        conv2_weights = weights["conv2"]
        assert isinstance(snake1_weights, Mapping)
        assert isinstance(conv1_weights, Mapping)
        assert isinstance(snake2_weights, Mapping)
        assert isinstance(conv2_weights, Mapping)

        return replace(
            self,
            snake1=self.snake1.import_weights(snake1_weights),
            conv1=self.conv1.import_weights(conv1_weights),
            snake2=self.snake2.import_weights(snake2_weights),
            conv2=self.conv2.import_weights(conv2_weights),
        )


@dataclass(frozen=True)
class AudioDecoderBlockSpatialParams:
    input_dim: int
    output_dim: int
    stride: int


@dataclass(frozen=True)
class DACDecoderBlockConfig:
    precision: DTypeLike
    snake_config: Snake1dConfig
    trans_conv_config: CausalTransposeConv1dConfig
    res_unit_config: ResidualUnitConfig
    causal: bool = True

    def empty(
        self,
        spatial_params: AudioDecoderBlockSpatialParams,
    ) -> "DACDecoderBlock":
        input_dim = spatial_params.input_dim
        output_dim = spatial_params.output_dim
        stride = spatial_params.stride

        snake = self.snake_config.empty(input_dim)

        trans_conv = self.trans_conv_config.empty(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * stride,
            stride=stride,
        )

        res_unit1 = self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=1))
        res_unit2 = self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=3))
        res_unit3 = self.res_unit_config.empty(output_dim, ResidualUnitSpatialParams(dilation=9))

        return DACDecoderBlock(
            config=self,
            snake=snake,
            trans_conv=trans_conv,
            res_unit1=res_unit1,
            res_unit2=res_unit2,
            res_unit3=res_unit3,
        )

    def random_init(
        self,
        spatial_params: AudioDecoderBlockSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "DACDecoderBlock":
        input_dim = spatial_params.input_dim
        output_dim = spatial_params.output_dim
        stride = spatial_params.stride

        key1, key2, key3, key4 = jax.random.split(key, 4)

        snake = self.snake_config.random_init(input_dim)

        trans_conv = self.trans_conv_config.random_init(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=2 * stride,
            stride=stride,
            key=key1,
        )

        res_unit1 = self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=1), key=key2)
        res_unit2 = self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=3), key=key3)
        res_unit3 = self.res_unit_config.random_init(output_dim, ResidualUnitSpatialParams(dilation=9), key=key4)

        return DACDecoderBlock(
            config=self,
            snake=snake,
            trans_conv=trans_conv,
            res_unit1=res_unit1,
            res_unit2=res_unit2,
            res_unit3=res_unit3,
        )


class DACDecoderBlock(LalamoModule[DACDecoderBlockConfig]):
    """DACDecoderBlock module for audio decoding.

    Architecture:
    1. Snake1d activation
    2. CausalTransposeConv1d (upsample)
    3. ResidualUnit (dilation=1)
    4. ResidualUnit (dilation=3)
    5. ResidualUnit (dilation=9)

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, channels) - NSC format (JAX convention)
    """

    snake: Snake1d
    trans_conv: CausalTransposeConv1d
    res_unit1: ResidualUnit
    res_unit2: ResidualUnit
    res_unit3: ResidualUnit

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_dim(self) -> int:
        return self.snake.channels

    @property
    def output_dim(self) -> int:
        return self.trans_conv.out_channels

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.snake(x)
        x = self.trans_conv(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "snake": self.snake.export_weights(),
            "trans_conv": self.trans_conv.export_weights(),
            "res_unit1": self.res_unit1.export_weights(),
            "res_unit2": self.res_unit2.export_weights(),
            "res_unit3": self.res_unit3.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "DACDecoderBlock":
        assert isinstance(weights, Mapping)
        snake_weights = weights["snake"]
        trans_conv_weights = weights["trans_conv"]
        res_unit1_weights = weights["res_unit1"]
        res_unit2_weights = weights["res_unit2"]
        res_unit3_weights = weights["res_unit3"]

        assert isinstance(snake_weights, Mapping)
        assert isinstance(trans_conv_weights, Mapping)
        assert isinstance(res_unit1_weights, Mapping)
        assert isinstance(res_unit2_weights, Mapping)
        assert isinstance(res_unit3_weights, Mapping)

        return replace(
            self,
            snake=self.snake.import_weights(require_tree(snake_weights)),
            trans_conv=self.trans_conv.import_weights(require_tree(trans_conv_weights)),
            res_unit1=self.res_unit1.import_weights(require_tree(res_unit1_weights)),
            res_unit2=self.res_unit2.import_weights(require_tree(res_unit2_weights)),
            res_unit3=self.res_unit3.import_weights(require_tree(res_unit3_weights)),
        )


@dataclass(frozen=True)
class DACDecoderSpatialParams:
    input_channel: int  # Input channels (from quantizer output)
    channels: int  # Initial channel width after first conv
    rates: tuple[int, ...]  # Upsampling rates for each DecoderBlock
    d_out: int = 1  # Output channels (1 for mono audio)


@dataclass(frozen=True)
class DACDecoderConfig:
    precision: DTypeLike
    conv_config: CausalConv1dConfig
    snake_config: Snake1dConfig
    decoder_block_config: DACDecoderBlockConfig
    causal: bool = True

    def empty(
        self,
        spatial_params: DACDecoderSpatialParams,
    ) -> "DACDecoder":
        if not self.causal:
            raise NotImplementedError("Non-causal AudioDecoder is not implemented")

        input_channel = spatial_params.input_channel
        channels = spatial_params.channels
        rates = spatial_params.rates
        d_out = spatial_params.d_out

        first_conv = self.conv_config.empty(
            in_channels=input_channel,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        decoder_blocks: list[DACDecoderBlock] = []
        for i, stride in enumerate(rates):
            block_input_dim = channels // (2**i)
            block_output_dim = channels // (2 ** (i + 1))

            block_spatial = AudioDecoderBlockSpatialParams(
                input_dim=block_input_dim,
                output_dim=block_output_dim,
                stride=stride,
            )
            block = self.decoder_block_config.empty(spatial_params=block_spatial)
            decoder_blocks.append(block)

        # Final output dimension after all decoder blocks
        final_dim = channels // (2 ** len(rates))

        final_snake = self.snake_config.empty(final_dim)

        final_conv = self.conv_config.empty(
            in_channels=final_dim,
            out_channels=d_out,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        return DACDecoder(
            config=self,
            first_conv=first_conv,
            decoder_blocks=tuple(decoder_blocks),
            final_snake=final_snake,
            final_conv=final_conv,
        )

    def random_init(
        self,
        spatial_params: DACDecoderSpatialParams,
        *,
        key: PRNGKeyArray,
    ) -> "DACDecoder":
        if not self.causal:
            raise NotImplementedError("Non-causal AudioDecoder is not implemented")

        input_channel = spatial_params.input_channel
        channels = spatial_params.channels
        rates = spatial_params.rates
        d_out = spatial_params.d_out

        num_keys = 2 + len(rates)  # first_conv + blocks + final_conv
        keys = jax.random.split(key, num_keys)

        first_conv = self.conv_config.random_init(
            in_channels=input_channel,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=keys[0],
        )

        decoder_blocks: list[DACDecoderBlock] = []
        for i, stride in enumerate(rates):
            block_input_dim = channels // (2**i)
            block_output_dim = channels // (2 ** (i + 1))

            block_spatial = AudioDecoderBlockSpatialParams(
                input_dim=block_input_dim,
                output_dim=block_output_dim,
                stride=stride,
            )
            block = self.decoder_block_config.random_init(spatial_params=block_spatial, key=keys[1 + i])
            decoder_blocks.append(block)

        # Final dimension
        final_dim = channels // (2 ** len(rates))

        final_snake = self.snake_config.random_init(final_dim)

        final_conv = self.conv_config.random_init(
            in_channels=final_dim,
            out_channels=d_out,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=keys[-1],
        )

        return DACDecoder(
            config=self,
            first_conv=first_conv,
            decoder_blocks=tuple(decoder_blocks),
            final_snake=final_snake,
            final_conv=final_conv,
        )


class DACDecoder(LalamoModule[DACDecoderConfig]):
    """Decoder module used in DAC for decoding audio from latent representations.

    Architecture:
    1. CausalConv1d (input_channel -> channels)
    2. Multiple DecoderBlocks (upsampling with residual refinement)
    3. Snake1d activation
    4. CausalConv1d (final_dim -> d_out)
    5. Tanh activation

    Input format: (batch, sequence, channels) - NSC format (JAX convention)
    Output format: (batch, sequence_upsampled, d_out) - NSC format (JAX convention)
    """

    first_conv: CausalConv1d
    decoder_blocks: tuple[DACDecoderBlock, ...]
    final_snake: Snake1d
    final_conv: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def input_channels(self) -> int:
        return self.first_conv.in_channels

    @property
    def output_channels(self) -> int:
        return self.final_conv.out_channels

    @property
    def num_blocks(self) -> int:
        return len(self.decoder_blocks)

    def __call__(
        self,
        x: Float[Array, "batch sequence in_channels"],
    ) -> Float[Array, "batch sequence_out out_channels"]:
        x = self.first_conv(x)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.final_snake(x)
        x = self.final_conv(x)

        # Tanh to constrain output to [-1, 1]
        x = jnp.tanh(x)

        return x

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "first_conv": self.first_conv.export_weights(),
            "decoder_blocks": [block.export_weights() for block in self.decoder_blocks],
            "final_snake": self.final_snake.export_weights(),
            "final_conv": self.final_conv.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> "DACDecoder":
        assert isinstance(weights, Mapping)

        first_conv_weights = weights["first_conv"]
        block_weights = weights["decoder_blocks"]
        final_snake_weights = weights["final_snake"]
        final_conv_weights = weights["final_conv"]

        assert isinstance(first_conv_weights, Mapping)
        assert isinstance(final_snake_weights, Mapping)
        assert isinstance(final_conv_weights, Mapping)

        new_blocks = []
        for block, w in zip(self.decoder_blocks, block_weights, strict=True):
            assert isinstance(w, Mapping)
            new_blocks.append(block.import_weights(w))

        return replace(
            self,
            first_conv=self.first_conv.import_weights(require_tree(first_conv_weights)),
            decoder_blocks=tuple(new_blocks),
            final_snake=self.final_snake.import_weights(require_tree(final_snake_weights)),
            final_conv=self.final_conv.import_weights(require_tree(final_conv_weights)),
        )
