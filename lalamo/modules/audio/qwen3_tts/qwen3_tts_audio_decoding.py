import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_mapping, require_tree
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase
from lalamo.modules.audio.common_modules import (
    Conv1d,
    Conv1dConfig,
    ConvNeXtSpatialParams,
    DACDecoder,
    DACDecoderConfig,
    DACDecoderSpatialParams,
    TransposeConvSpatialParams,
    UpsamplingBlock,
    UpsamplingBlockConfig,
)
from lalamo.modules.audio.text_decoder import CodebookCodes
from lalamo.modules.common import ForwardPassMode, LalamoModule
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import RoPE, UnscaledRoPEConfig
from lalamo.modules.token_mixers import Attention, AttentionConfig
from lalamo.modules.utils import vmap_twice

from .qwen3_tts_modules import (
    ResidualVectorQuantizer,
    ResidualVectorQuantizerConfig,
)

__all__ = [
    "Qwen3TTSAudioDecoder",
    "Qwen3TTSAudioDecoderConfig",
    "Qwen3TTSPreTransformer",
    "Qwen3TTSPreTransformerConfig",
    "Qwen3TTSPreTransformerLayer",
    "Qwen3TTSPreTransformerLayerConfig",
]


@dataclass(frozen=True)
class Qwen3TTSPreTransformerLayerConfig:
    precision: DTypeLike
    attention_config: AttentionConfig
    mlp_config: DenseMLPConfig
    norm_config: NormalizationConfig
    layer_scale_initial_scale: float

    def empty(self, hidden_size: int, intermediate_size: int) -> "Qwen3TTSPreTransformerLayer":
        return Qwen3TTSPreTransformerLayer(
            config=self,
            self_attn=self.attention_config.empty(model_dim=hidden_size),
            mlp=self.mlp_config.empty(model_dim=hidden_size, hidden_dim=intermediate_size),
            input_layernorm=self.norm_config.empty(hidden_size),
            post_attention_layernorm=self.norm_config.empty(hidden_size),
            self_attn_layer_scale=jnp.full((hidden_size,), self.layer_scale_initial_scale, dtype=self.precision),
            mlp_layer_scale=jnp.full((hidden_size,), self.layer_scale_initial_scale, dtype=self.precision),
        )

    def random_init(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        key: PRNGKeyArray,
    ) -> "Qwen3TTSPreTransformerLayer":
        key_attn, key_mlp = jax.random.split(key)
        return Qwen3TTSPreTransformerLayer(
            config=self,
            self_attn=self.attention_config.random_init(model_dim=hidden_size, key=key_attn),
            mlp=self.mlp_config.random_init(model_dim=hidden_size, hidden_dim=intermediate_size, key=key_mlp),
            input_layernorm=self.norm_config.init(hidden_size),
            post_attention_layernorm=self.norm_config.init(hidden_size),
            self_attn_layer_scale=jnp.full((hidden_size,), self.layer_scale_initial_scale, dtype=self.precision),
            mlp_layer_scale=jnp.full((hidden_size,), self.layer_scale_initial_scale, dtype=self.precision),
        )


class Qwen3TTSPreTransformerLayer(LalamoModule[Qwen3TTSPreTransformerLayerConfig]):
    self_attn: Attention
    mlp: DenseMLP
    input_layernorm: Normalization
    post_attention_layernorm: Normalization
    self_attn_layer_scale: Float[Array, " channels"]
    mlp_layer_scale: Float[Array, " channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        hidden_states: Float[Array, "batch tokens channels"],
        position_embeddings: object,
    ) -> Float[Array, "batch tokens channels"]:
        residual = hidden_states
        hidden_states = vmap_twice(self.input_layernorm)(hidden_states)
        batched_attention_fn = vmap(partial(self.self_attn, return_updated_state=False))
        attention_outputs, _ = batched_attention_fn(
            hidden_states,
            position_embeddings,
            state=None,
            length_without_padding=None,
        )
        hidden_states = residual + attention_outputs * self.self_attn_layer_scale[None, None, :]

        residual = hidden_states
        hidden_states = vmap_twice(self.post_attention_layernorm)(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        hidden_states = residual + hidden_states * self.mlp_layer_scale[None, None, :]

        return hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "self_attn": self.self_attn.export_weights(),
            "mlp": self.mlp.export_weights(),
            "input_layernorm": self.input_layernorm.export_weights(),
            "post_attention_layernorm": self.post_attention_layernorm.export_weights(),
            "self_attn_layer_scale": self.self_attn_layer_scale,
            "mlp_layer_scale": self.mlp_layer_scale,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            self_attn=self.self_attn.import_weights(require_tree(weights["self_attn"])),
            mlp=self.mlp.import_weights(require_tree(weights["mlp"])),
            input_layernorm=self.input_layernorm.import_weights(require_tree(weights["input_layernorm"])),
            post_attention_layernorm=self.post_attention_layernorm.import_weights(
                require_tree(weights["post_attention_layernorm"]),
            ),
            self_attn_layer_scale=require_array(weights["self_attn_layer_scale"]),
            mlp_layer_scale=require_array(weights["mlp_layer_scale"]),
        )


@dataclass(frozen=True)
class Qwen3TTSPreTransformerConfig:
    precision: DTypeLike
    input_projection_config: FullPrecisionLinearConfig
    output_projection_config: FullPrecisionLinearConfig
    output_norm_config: NormalizationConfig
    rope_config: UnscaledRoPEConfig
    layer_config: Qwen3TTSPreTransformerLayerConfig

    hidden_size: int
    latent_dim: int
    intermediate_size: int
    num_hidden_layers: int
    max_position_embeddings: int

    def empty(self) -> "Qwen3TTSPreTransformer":
        input_projection = self.input_projection_config.empty(
            input_dim=self.latent_dim,
            output_dims=(self.hidden_size,),
            has_biases=True,
        )
        output_projection = self.output_projection_config.empty(
            input_dim=self.hidden_size,
            output_dims=(self.latent_dim,),
            has_biases=True,
        )
        output_norm = self.output_norm_config.empty(self.hidden_size)
        rope_dim = self.layer_config.attention_config.rope_dim
        if rope_dim is None:
            raise ValueError("Qwen3 TTS pre-transformer requires RoPE")
        rope = self.rope_config.init(
            head_dim=rope_dim,
            num_timesteps=self.max_position_embeddings,
        )
        layers = tuple(
            self.layer_config.empty(self.hidden_size, self.intermediate_size) for _ in range(self.num_hidden_layers)
        )
        return Qwen3TTSPreTransformer(
            config=self,
            input_projection=input_projection,
            output_projection=output_projection,
            output_norm=output_norm,
            rope=rope,
            layers=layers,
        )

    def random_init(self, *, key: PRNGKeyArray) -> "Qwen3TTSPreTransformer":
        key_input_projection, key_output_projection, key_hidden_layers = jax.random.split(key, 3)
        input_projection = self.input_projection_config.random_init(
            input_dim=self.latent_dim,
            output_dims=(self.hidden_size,),
            has_biases=True,
            key=key_input_projection,
        )
        output_projection = self.output_projection_config.random_init(
            input_dim=self.hidden_size,
            output_dims=(self.latent_dim,),
            has_biases=True,
            key=key_output_projection,
        )
        output_norm = self.output_norm_config.init(self.hidden_size)
        rope_dim = self.layer_config.attention_config.rope_dim
        if rope_dim is None:
            raise ValueError("Qwen3 TTS pre-transformer requires RoPE")
        rope = self.rope_config.init(
            head_dim=rope_dim,
            num_timesteps=self.max_position_embeddings,
        )
        layer_keys = jax.random.split(key_hidden_layers, self.num_hidden_layers)
        layers = tuple(
            self.layer_config.random_init(self.hidden_size, self.intermediate_size, key=layer_key)
            for layer_key in layer_keys
        )
        return Qwen3TTSPreTransformer(
            config=self,
            input_projection=input_projection,
            output_projection=output_projection,
            output_norm=output_norm,
            rope=rope,
            layers=layers,
        )


class Qwen3TTSPreTransformer(LalamoModule[Qwen3TTSPreTransformerConfig]):
    input_projection: FullPrecisionLinear
    output_projection: FullPrecisionLinear
    output_norm: Normalization
    rope: RoPE
    layers: tuple[Qwen3TTSPreTransformerLayer, ...]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        hidden_states: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        (hidden_states,) = vmap_twice(self.input_projection)(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape
        token_positions = jnp.broadcast_to(jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, seq_length))
        position_embeddings = vmap(self.rope)(token_positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = vmap_twice(self.output_norm)(hidden_states)
        (hidden_states,) = vmap_twice(self.output_projection)(hidden_states)
        return hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "input_projection": self.input_projection.export_weights(),
            "output_projection": self.output_projection.export_weights(),
            "output_norm": self.output_norm.export_weights(),
            "layers": [layer.export_weights() for layer in self.layers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        layers_weights = weights["layers"]
        return replace(
            self,
            input_projection=self.input_projection.import_weights(require_tree(weights["input_projection"])),
            output_projection=self.output_projection.import_weights(require_tree(weights["output_projection"])),
            output_norm=self.output_norm.import_weights(require_tree(weights["output_norm"])),
            layers=tuple(
                layer.import_weights(require_tree(layer_weights))
                for layer, layer_weights in zip(self.layers, layers_weights, strict=True)
            ),
        )


@dataclass(frozen=True)
class Qwen3TTSAudioDecoderConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike

    quantizer_config: ResidualVectorQuantizerConfig
    pre_conv_config: Conv1dConfig
    pre_transformer_config: Qwen3TTSPreTransformerConfig
    upsample_block_config: UpsamplingBlockConfig
    dac_decoder_config: DACDecoderConfig

    samplerate: int
    decode_upsample_rate: int

    num_quantizers: int
    codebook_size: int
    codebook_dim: int

    latent_dim: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    decoder_dim: int

    def empty(self) -> "Qwen3TTSAudioDecoder":
        quantizer = self.quantizer_config.empty(
            dimension=self.codebook_dim // 2,
            num_quantizers=self.num_quantizers,
            bins=self.codebook_size,
            output_dimension=self.codebook_dim,
        )

        pre_conv = self.pre_conv_config.empty(
            in_channels=self.codebook_dim,
            out_channels=self.latent_dim,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
        )

        pre_transformer = self.pre_transformer_config.empty()

        upsample_blocks = tuple(
            self.upsample_block_config.empty(
                TransposeConvSpatialParams(
                    in_channels=self.latent_dim,
                    out_channels=self.latent_dim,
                    upsample_kernel_size=factor,
                    upsample_stride=factor,
                ),
                ConvNeXtSpatialParams(mlp_ratio=4.0, kernel_size=7, dilation=1),
            )
            for factor in self.upsampling_ratios
        )

        dac_params = DACDecoderSpatialParams(
            input_channel=self.latent_dim,
            channels=self.decoder_dim,
            rates=self.upsample_rates,
            out_channels=1,
        )
        dac_decoder = self.dac_decoder_config.empty(dac_params)

        total_upsample = math.prod((*self.upsample_rates, *self.upsampling_ratios))

        return Qwen3TTSAudioDecoder(
            config=self,
            total_upsample=total_upsample,
            quantizer=quantizer,
            pre_conv=pre_conv,
            pre_transformer=pre_transformer,
            upsample_blocks=upsample_blocks,
            dac_decoder=dac_decoder,
        )

    def random_init(self, *, key: PRNGKeyArray) -> "Qwen3TTSAudioDecoder":
        (
            quantizer_key,
            pre_conv_key,
            pre_transformer_key,
            upsample_blocks_key,
            dac_decoder_key,
        ) = jax.random.split(key, 5)

        quantizer = self.quantizer_config.random_init(
            dimension=self.codebook_dim // 2,
            num_quantizers=self.num_quantizers,
            bins=self.codebook_size,
            output_dimension=self.codebook_dim,
            key=quantizer_key,
        )

        pre_conv = self.pre_conv_config.random_init(
            in_channels=self.codebook_dim,
            out_channels=self.latent_dim,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            key=pre_conv_key,
        )

        pre_transformer = self.pre_transformer_config.random_init(key=pre_transformer_key)

        upsample_keys = jax.random.split(upsample_blocks_key, len(self.upsampling_ratios))
        upsample_blocks = tuple(
            self.upsample_block_config.random_init(
                TransposeConvSpatialParams(
                    in_channels=self.latent_dim,
                    out_channels=self.latent_dim,
                    upsample_kernel_size=factor,
                    upsample_stride=factor,
                ),
                ConvNeXtSpatialParams(mlp_ratio=4.0, kernel_size=7, dilation=1),
                key=block_key,
            )
            for factor, block_key in zip(self.upsampling_ratios, upsample_keys, strict=True)
        )

        dac_params = DACDecoderSpatialParams(
            input_channel=self.latent_dim,
            channels=self.decoder_dim,
            rates=self.upsample_rates,
            out_channels=1,
        )
        dac_decoder = self.dac_decoder_config.random_init(
            dac_params,
            key=dac_decoder_key,
        )

        total_upsample = math.prod((*self.upsample_rates, *self.upsampling_ratios))

        return Qwen3TTSAudioDecoder(
            config=self,
            total_upsample=total_upsample,
            quantizer=quantizer,
            pre_conv=pre_conv,
            pre_transformer=pre_transformer,
            upsample_blocks=upsample_blocks,
            dac_decoder=dac_decoder,
        )


class Qwen3TTSAudioDecoder(TTSAudioDecoder[Qwen3TTSAudioDecoderConfig]):
    total_upsample: int

    quantizer: ResidualVectorQuantizer
    pre_conv: Conv1d
    pre_transformer: Qwen3TTSPreTransformer
    upsample_blocks: tuple[UpsamplingBlock, ...]

    dac_decoder: DACDecoder

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    def __call__(
        self,
        codes: CodebookCodes,
    ) -> Float[Array, "batch samples 1"]:
        hidden = self.quantizer.decode(codes.semantic, codes.acoustic)
        hidden = rearrange(hidden, "batch channels tokens -> batch tokens channels")
        hidden = self.pre_conv(hidden)
        hidden = self.pre_transformer(hidden)

        for upsample_block in self.upsample_blocks:
            hidden = upsample_block(hidden)

        hidden = self.dac_decoder(hidden)

        return jnp.clip(hidden, min=-1.0, max=1.0)

    def chunked_decode(
        self,
        codes: CodebookCodes,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> Float[Array, "batch samples 1"]:
        wav_chunks: list[Float[Array, "batch samples 1"]] = []
        start_index = 0
        total_tokens = int(codes.semantic.shape[-1])

        while start_index < total_tokens:
            end_index = min(start_index + chunk_size, total_tokens)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = CodebookCodes(
                semantic=codes.semantic[..., start_index - context_size : end_index],
                acoustic=codes.acoustic[..., start_index - context_size : end_index],
            )
            wav_chunk = self(codes_chunk)
            wav_chunks.append(wav_chunk[:, context_size * self.total_upsample :, :])
            start_index = end_index

        return jnp.concatenate(wav_chunks, axis=1)

    def audio_from_codes(
        self,
        codes: CodebookCodes,
    ) -> Float[Array, " samples"]:
        assert codes.semantic.ndim == 3
        assert codes.acoustic.ndim == 3

        wav = self.chunked_decode(codes)
        (first_wav,) = wav
        return jnp.squeeze(first_wav, axis=-1)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizer": self.quantizer.export_weights(),
            "pre_conv": self.pre_conv.export_weights(),
            "pre_transformer": self.pre_transformer.export_weights(),
            "upsample_blocks": [upsample_block.export_weights() for upsample_block in self.upsample_blocks],
            "dac_decoder": self.dac_decoder.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        upsample_block_weights = weights["upsample_blocks"]
        assert isinstance(upsample_block_weights, Sequence)

        return replace(
            self,
            quantizer=self.quantizer.import_weights(require_tree(weights["quantizer"])),
            pre_conv=self.pre_conv.import_weights(require_tree(weights["pre_conv"])),
            pre_transformer=self.pre_transformer.import_weights(require_tree(weights["pre_transformer"])),
            upsample_blocks=tuple(
                upsample_block.import_weights(require_tree(block_weights))
                for upsample_block, block_weights in zip(self.upsample_blocks, upsample_block_weights, strict=True)
            ),
            dac_decoder=self.dac_decoder.import_weights(require_tree(weights["dac_decoder"])),
        )
