import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_array, require_tree
from lalamo.modules.activations import SiLU
from lalamo.modules.audio.audio_decoder import TTSAudioDecoder, TTSAudioDecoderConfigBase
from lalamo.modules.audio.common_modules import (
    CausalConv1d,
    CausalConv1dConfig,
)
from lalamo.modules.common import ForwardPassMode, LalamoModule
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLP, DenseMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig, UpcastMode
from lalamo.modules.rope import RoPE, UnscaledRoPEConfig
from lalamo.modules.token_mixers import Attention, AttentionConfig

from .qwen3_tts_modules import (
    Qwen3TTSCausalTransposeConv1d,
    Qwen3TTSCausalTransposeConv1dConfig,
    Qwen3TTSConvNeXtBlock,
    Qwen3TTSConvNeXtBlockConfig,
    Qwen3TTSDecoderBlock,
    Qwen3TTSDecoderBlockConfig,
    Qwen3TTSEuclideanCodebookConfig,
    Qwen3TTSResidualUnitConfig,
    Qwen3TTSResidualVectorQuantizationConfig,
    Qwen3TTSResidualVectorQuantizerConfig,
    Qwen3TTSSnakeBeta,
    Qwen3TTSSnakeBetaConfig,
    Qwen3TTSSplitResidualVectorQuantizer,
    Qwen3TTSSplitResidualVectorQuantizerConfig,
    Qwen3TTSVectorQuantizationConfig,
)

__all__ = [
    "QWEN3_TTS_AUDIO_DECODER_CHUNK_SIZE_DEFAULT",
    "QWEN3_TTS_AUDIO_DECODER_LEFT_CONTEXT_SIZE_DEFAULT",
    "Qwen3TTSAudioDecoder",
    "Qwen3TTSAudioDecoderConfig",
    "Qwen3TTSPreTransformer",
    "Qwen3TTSPreTransformerConfig",
    "Qwen3TTSPreTransformerLayer",
    "Qwen3TTSPreTransformerLayerConfig",
    "Qwen3TTSUpsampleBlock",
    "Qwen3TTSUpsampleBlockConfig",
    "default_qwen3_tts_audio_decoder_config",
]

QWEN3_TTS_AUDIO_DECODER_CHUNK_SIZE_DEFAULT = 300
QWEN3_TTS_AUDIO_DECODER_LEFT_CONTEXT_SIZE_DEFAULT = 25


def _debug_tensor(name: str, value: Array, *, enabled: bool) -> None:
    if not enabled:
        return
    jax.debug.print(
        "[qwen3_tts] {name}: shape={shape} min={min_val:.5f} max={max_val:.5f}",
        name=name,
        shape=value.shape,
        min_val=jnp.min(value),
        max_val=jnp.max(value),
    )


@dataclass(frozen=True)
class Qwen3TTSPreTransformerLayerConfig:
    precision: DTypeLike
    attention_config: AttentionConfig
    mlp_config: DenseMLPConfig
    norm_config: NormalizationConfig
    layer_scale_initial_scale: float
    enable_debug: bool = True

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
        _debug_tensor("pre_transformer_layer.input", hidden_states, enabled=self.config.enable_debug)

        residual = hidden_states
        hidden_states = vmap(vmap(self.input_layernorm))(hidden_states)
        batched_attention_fn = vmap(partial(self.self_attn, return_updated_state=False))
        attention_outputs, _ = batched_attention_fn(
            hidden_states,
            position_embeddings,
            state=None,
            length_without_padding=None,
        )
        hidden_states = residual + attention_outputs * self.self_attn_layer_scale[None, None, :]

        residual = hidden_states
        hidden_states = vmap(vmap(self.post_attention_layernorm))(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        hidden_states = residual + hidden_states * self.mlp_layer_scale[None, None, :]

        _debug_tensor("pre_transformer_layer.output", hidden_states, enabled=self.config.enable_debug)
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
        assert isinstance(weights, Mapping)
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
    enable_debug: bool = True

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
        rope = self.rope_config.init(
            head_dim=self.layer_config.attention_config.rope_dim,
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
        rope = self.rope_config.init(
            head_dim=self.layer_config.attention_config.rope_dim,
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
        _debug_tensor("pre_transformer.input", hidden_states, enabled=self.config.enable_debug)
        (hidden_states,) = vmap(vmap(self.input_projection))(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape
        token_positions = jnp.broadcast_to(jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, seq_length))
        position_embeddings = vmap(self.rope)(token_positions)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = vmap(vmap(self.output_norm))(hidden_states)
        (hidden_states,) = vmap(vmap(self.output_projection))(hidden_states)
        _debug_tensor("pre_transformer.output", hidden_states, enabled=self.config.enable_debug)
        return hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "input_projection": self.input_projection.export_weights(),
            "output_projection": self.output_projection.export_weights(),
            "output_norm": self.output_norm.export_weights(),
            "layers": [layer.export_weights() for layer in self.layers],
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        layers_weights = weights["layers"]
        assert isinstance(layers_weights, Sequence)
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
class Qwen3TTSUpsampleBlockConfig:
    precision: DTypeLike
    transposed_conv_config: Qwen3TTSCausalTransposeConv1dConfig
    convnext_config: Qwen3TTSConvNeXtBlockConfig
    enable_debug: bool = True

    def empty(self, latent_dim: int, factor: int) -> "Qwen3TTSUpsampleBlock":
        return Qwen3TTSUpsampleBlock(
            config=self,
            transposed_conv=self.transposed_conv_config.empty(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=factor,
                stride=factor,
                groups=1,
            ),
            convnext=self.convnext_config.empty(latent_dim),
        )

    def random_init(self, latent_dim: int, factor: int, *, key: PRNGKeyArray) -> "Qwen3TTSUpsampleBlock":
        key_transposed, key_convnext = jax.random.split(key)
        return Qwen3TTSUpsampleBlock(
            config=self,
            transposed_conv=self.transposed_conv_config.random_init(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=factor,
                stride=factor,
                groups=1,
                key=key_transposed,
            ),
            convnext=self.convnext_config.random_init(latent_dim, key=key_convnext),
        )


class Qwen3TTSUpsampleBlock(LalamoModule[Qwen3TTSUpsampleBlockConfig]):
    transposed_conv: Qwen3TTSCausalTransposeConv1d
    convnext: Qwen3TTSConvNeXtBlock

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __call__(
        self,
        hidden_states: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens channels"]:
        hidden_states = self.transposed_conv(hidden_states)
        hidden_states = self.convnext(hidden_states)
        return hidden_states

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "transposed_conv": self.transposed_conv.export_weights(),
            "convnext": self.convnext.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            transposed_conv=self.transposed_conv.import_weights(require_tree(weights["transposed_conv"])),
            convnext=self.convnext.import_weights(require_tree(weights["convnext"])),
        )


@dataclass(frozen=True)
class Qwen3TTSAudioDecoderConfig(TTSAudioDecoderConfigBase):
    precision: DTypeLike

    quantizer_config: Qwen3TTSSplitResidualVectorQuantizerConfig
    pre_conv_config: CausalConv1dConfig
    pre_transformer_config: Qwen3TTSPreTransformerConfig
    upsample_block_config: Qwen3TTSUpsampleBlockConfig

    decoder_input_conv_config: CausalConv1dConfig
    decoder_block_config: Qwen3TTSDecoderBlockConfig
    final_snake_config: Qwen3TTSSnakeBetaConfig
    final_conv_config: CausalConv1dConfig

    samplerate: int
    decode_upsample_rate: int

    num_quantizers: int
    codebook_size: int
    codebook_dim: int

    latent_dim: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    decoder_dim: int

    chunk_size: int
    left_context_size: int
    enable_debug: bool = True

    def empty(self) -> "Qwen3TTSAudioDecoder":
        quantizer = self.quantizer_config.empty(
            dimension=self.codebook_dim // 2,
            n_q=self.num_quantizers,
            bins=self.codebook_size,
            input_dimension=self.codebook_dim,
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
            self.upsample_block_config.empty(self.latent_dim, factor) for factor in self.upsampling_ratios
        )

        decoder_input_conv = self.decoder_input_conv_config.empty(
            in_channels=self.latent_dim,
            out_channels=self.decoder_dim,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        decoder_blocks = tuple(
            self.decoder_block_config.empty(
                in_dim=self.decoder_dim // (2**i),
                out_dim=self.decoder_dim // (2 ** (i + 1)),
                upsample_rate=rate,
            )
            for i, rate in enumerate(self.upsample_rates)
        )

        output_dim = self.decoder_dim // (2 ** len(self.upsample_rates))
        final_snake = self.final_snake_config.empty(output_dim)
        final_conv = self.final_conv_config.empty(
            in_channels=output_dim,
            out_channels=1,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
        )

        total_upsample = math.prod((*self.upsample_rates, *self.upsampling_ratios))

        return Qwen3TTSAudioDecoder(
            config=self,
            total_upsample=total_upsample,
            quantizer=quantizer,
            pre_conv=pre_conv,
            pre_transformer=pre_transformer,
            upsample_blocks=upsample_blocks,
            decoder_input_conv=decoder_input_conv,
            decoder_blocks=decoder_blocks,
            final_snake=final_snake,
            final_conv=final_conv,
        )

    def random_init(self, *, key: PRNGKeyArray) -> "Qwen3TTSAudioDecoder":
        (
            quantizer_key,
            pre_conv_key,
            pre_transformer_key,
            upsample_blocks_key,
            decoder_input_conv_key,
            decoder_blocks_key,
            final_snake_key,
            final_conv_key,
        ) = jax.random.split(key, 8)

        quantizer = self.quantizer_config.random_init(
            dimension=self.codebook_dim // 2,
            n_q=self.num_quantizers,
            bins=self.codebook_size,
            input_dimension=self.codebook_dim,
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
            self.upsample_block_config.random_init(self.latent_dim, factor, key=block_key)
            for factor, block_key in zip(self.upsampling_ratios, upsample_keys, strict=True)
        )

        decoder_input_conv = self.decoder_input_conv_config.random_init(
            in_channels=self.latent_dim,
            out_channels=self.decoder_dim,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=decoder_input_conv_key,
        )

        decoder_block_keys = jax.random.split(decoder_blocks_key, len(self.upsample_rates))
        decoder_blocks = tuple(
            self.decoder_block_config.random_init(
                in_dim=self.decoder_dim // (2**i),
                out_dim=self.decoder_dim // (2 ** (i + 1)),
                upsample_rate=rate,
                key=block_key,
            )
            for (i, rate), block_key in zip(enumerate(self.upsample_rates), decoder_block_keys, strict=True)
        )

        output_dim = self.decoder_dim // (2 ** len(self.upsample_rates))
        final_snake = self.final_snake_config.random_init(output_dim, key=final_snake_key)
        final_conv = self.final_conv_config.random_init(
            in_channels=output_dim,
            out_channels=1,
            kernel_size=7,
            stride=1,
            dilation=1,
            groups=1,
            key=final_conv_key,
        )

        total_upsample = math.prod((*self.upsample_rates, *self.upsampling_ratios))

        return Qwen3TTSAudioDecoder(
            config=self,
            total_upsample=total_upsample,
            quantizer=quantizer,
            pre_conv=pre_conv,
            pre_transformer=pre_transformer,
            upsample_blocks=upsample_blocks,
            decoder_input_conv=decoder_input_conv,
            decoder_blocks=decoder_blocks,
            final_snake=final_snake,
            final_conv=final_conv,
        )


class Qwen3TTSAudioDecoder(TTSAudioDecoder[Qwen3TTSAudioDecoderConfig]):
    total_upsample: int

    quantizer: Qwen3TTSSplitResidualVectorQuantizer
    pre_conv: CausalConv1d
    pre_transformer: Qwen3TTSPreTransformer
    upsample_blocks: tuple[Qwen3TTSUpsampleBlock, ...]

    decoder_input_conv: CausalConv1d
    decoder_blocks: tuple[Qwen3TTSDecoderBlock, ...]
    final_snake: Qwen3TTSSnakeBeta
    final_conv: CausalConv1d

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def samplerate(self) -> int:
        return self.config.samplerate

    def __call__(
        self,
        codes: Int[Array, "batch codebooks tokens"],
    ) -> Float[Array, "batch samples 1"]:
        _, num_codebooks, _ = codes.shape
        if num_codebooks != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} codebooks, got {num_codebooks}",
            )

        _debug_tensor("audio_decoder.codes", codes, enabled=self.config.enable_debug)

        hidden = self.quantizer.decode(codes)
        hidden = rearrange(hidden, "batch channels tokens -> batch tokens channels")
        hidden = self.pre_conv(hidden)
        hidden = self.pre_transformer(hidden)

        for upsample_block in self.upsample_blocks:
            hidden = upsample_block(hidden)

        hidden = self.decoder_input_conv(hidden)
        for decoder_block in self.decoder_blocks:
            hidden = decoder_block(hidden)
        hidden = self.final_snake(hidden)
        hidden = self.final_conv(hidden)

        wav = jnp.clip(hidden, min=-1.0, max=1.0)
        _debug_tensor("audio_decoder.wav", wav, enabled=self.config.enable_debug)
        return wav

    def chunked_decode(
        self,
        codes: Int[Array, "batch codebooks tokens"],
        chunk_size: int | None = None,
        left_context_size: int | None = None,
    ) -> Float[Array, "batch samples 1"]:
        chunk_size = self.config.chunk_size if chunk_size is None else chunk_size
        left_context_size = self.config.left_context_size if left_context_size is None else left_context_size

        wav_chunks: list[Float[Array, "batch samples 1"]] = []
        start_index = 0
        total_tokens = int(codes.shape[-1])

        while start_index < total_tokens:
            end_index = min(start_index + chunk_size, total_tokens)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wav_chunks.append(wav_chunk[:, context_size * self.total_upsample :, :])
            start_index = end_index

        return jnp.concatenate(wav_chunks, axis=1)

    def audio_from_codes(
        self,
        indices: Int[Array, "batch codebooks tokens"] | Int[Array, "codebooks tokens"],
    ) -> Float[Array, "samples"]:
        if indices.ndim == 2:
            indices = rearrange(indices, "codebooks tokens -> 1 codebooks tokens")

        wav = self.chunked_decode(indices)
        (first_wav,) = wav
        return jnp.squeeze(first_wav, axis=-1)

    def export_weights(self) -> ParameterTree[Array]:
        return {
            "quantizer": self.quantizer.export_weights(),
            "pre_conv": self.pre_conv.export_weights(),
            "pre_transformer": self.pre_transformer.export_weights(),
            "upsample_blocks": [upsample_block.export_weights() for upsample_block in self.upsample_blocks],
            "decoder_input_conv": self.decoder_input_conv.export_weights(),
            "decoder_blocks": [decoder_block.export_weights() for decoder_block in self.decoder_blocks],
            "final_snake": self.final_snake.export_weights(),
            "final_conv": self.final_conv.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)

        upsample_block_weights = weights["upsample_blocks"]
        decoder_block_weights = weights["decoder_blocks"]

        assert isinstance(upsample_block_weights, Sequence)
        assert isinstance(decoder_block_weights, Sequence)

        return replace(
            self,
            quantizer=self.quantizer.import_weights(require_tree(weights["quantizer"])),
            pre_conv=self.pre_conv.import_weights(require_tree(weights["pre_conv"])),
            pre_transformer=self.pre_transformer.import_weights(require_tree(weights["pre_transformer"])),
            upsample_blocks=tuple(
                upsample_block.import_weights(require_tree(block_weights))
                for upsample_block, block_weights in zip(self.upsample_blocks, upsample_block_weights, strict=True)
            ),
            decoder_input_conv=self.decoder_input_conv.import_weights(require_tree(weights["decoder_input_conv"])),
            decoder_blocks=tuple(
                decoder_block.import_weights(require_tree(block_weights))
                for decoder_block, block_weights in zip(self.decoder_blocks, decoder_block_weights, strict=True)
            ),
            final_snake=self.final_snake.import_weights(require_tree(weights["final_snake"])),
            final_conv=self.final_conv.import_weights(require_tree(weights["final_conv"])),
        )


def default_qwen3_tts_audio_decoder_config(
    *,
    precision: DTypeLike = jnp.float32,
    samplerate: int = 24000,
    decode_upsample_rate: int = 1920,
    codebook_size: int = 2048,
    codebook_dim: int = 2048,
    hidden_size: int = 1024,
    latent_dim: int = 1024,
    max_position_embeddings: int = 8000,
    rope_theta: float = 10000,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 16,
    head_dim: int,
    attention_bias: bool = False,
    sliding_window: int = 72,
    intermediate_size: int = 3072,
    layer_scale_initial_scale: float = 0.01,
    rms_norm_eps: float = 1e-5,
    num_hidden_layers: int = 8,
    num_quantizers: int = 16,
    num_semantic_quantizers: int,
    upsample_rates: tuple[int, ...] = (8, 5, 4, 3),
    upsampling_ratios: tuple[int, ...] = (2, 2),
    decoder_dim: int = 1536,
    enable_debug: bool = True,
) -> Qwen3TTSAudioDecoderConfig:
    linear_config = FullPrecisionLinearConfig(precision=precision)

    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )

    attention_config = AttentionConfig(
        qkv_projection_config=linear_config,
        out_projection_config=linear_config,
        query_norm_config=None,
        key_norm_config=None,
        num_heads=num_attention_heads,
        num_groups=num_key_value_heads,
        head_dim=head_dim,
        is_causal=True,
        scale=None,
        sliding_window_size=sliding_window,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=attention_bias,
        has_out_biases=attention_bias,
    )

    mlp_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )

    rope_config = UnscaledRoPEConfig(
        precision=precision,
        base=rope_theta,
        max_sequence_length=max_position_embeddings,
    )

    pre_transformer_layer_config = Qwen3TTSPreTransformerLayerConfig(
        precision=precision,
        attention_config=attention_config,
        mlp_config=mlp_config,
        norm_config=norm_config,
        layer_scale_initial_scale=layer_scale_initial_scale,
        enable_debug=enable_debug,
    )

    pre_transformer_config = Qwen3TTSPreTransformerConfig(
        precision=precision,
        input_projection_config=linear_config,
        output_projection_config=linear_config,
        output_norm_config=norm_config,
        rope_config=rope_config,
        layer_config=pre_transformer_layer_config,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings=max_position_embeddings,
        enable_debug=enable_debug,
    )

    snake_config = Qwen3TTSSnakeBetaConfig(precision=precision, enable_debug=enable_debug)
    convnext_config = Qwen3TTSConvNeXtBlockConfig(
        precision=precision,
        conv_config=CausalConv1dConfig(precision=precision, has_biases=True),
        norm_config=NormalizationConfig(
            scale_precision=precision,
            accumulation_precision=precision,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        ),
        linear_config=linear_config,
        gamma_init=1e-6,
        enable_debug=enable_debug,
    )

    quantizer_config = Qwen3TTSSplitResidualVectorQuantizerConfig(
        precision=precision,
        residual_vector_quantizer_config=Qwen3TTSResidualVectorQuantizerConfig(
            precision=precision,
            rvq_config=Qwen3TTSResidualVectorQuantizationConfig(
                precision=precision,
                vector_quantization_config=Qwen3TTSVectorQuantizationConfig(
                    precision=precision,
                    codebook_config=Qwen3TTSEuclideanCodebookConfig(precision=precision),
                    project_out_config=linear_config,
                    enable_debug=enable_debug,
                ),
                enable_debug=enable_debug,
            ),
            output_projection_config=linear_config,
            enable_debug=enable_debug,
        ),
        n_q_semantic=num_semantic_quantizers,
        enable_debug=enable_debug,
    )

    return Qwen3TTSAudioDecoderConfig(
        precision=precision,
        quantizer_config=quantizer_config,
        pre_conv_config=CausalConv1dConfig(precision=precision, has_biases=True),
        pre_transformer_config=pre_transformer_config,
        upsample_block_config=Qwen3TTSUpsampleBlockConfig(
            precision=precision,
            transposed_conv_config=Qwen3TTSCausalTransposeConv1dConfig(precision=precision, has_biases=True),
            convnext_config=convnext_config,
            enable_debug=enable_debug,
        ),
        decoder_input_conv_config=CausalConv1dConfig(precision=precision, has_biases=True),
        decoder_block_config=Qwen3TTSDecoderBlockConfig(
            precision=precision,
            snake_config=snake_config,
            transposed_conv_config=Qwen3TTSCausalTransposeConv1dConfig(precision=precision, has_biases=True),
            residual_unit_config=Qwen3TTSResidualUnitConfig(
                precision=precision,
                snake_config=snake_config,
                conv_config=CausalConv1dConfig(precision=precision, has_biases=True),
                enable_debug=enable_debug,
            ),
            enable_debug=enable_debug,
        ),
        final_snake_config=snake_config,
        final_conv_config=CausalConv1dConfig(precision=precision, has_biases=True),
        samplerate=samplerate,
        decode_upsample_rate=decode_upsample_rate,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        latent_dim=latent_dim,
        upsample_rates=upsample_rates,
        upsampling_ratios=upsampling_ratios,
        decoder_dim=decoder_dim,
        chunk_size=QWEN3_TTS_AUDIO_DECODER_CHUNK_SIZE_DEFAULT,
        left_context_size=QWEN3_TTS_AUDIO_DECODER_LEFT_CONTEXT_SIZE_DEFAULT,
        enable_debug=enable_debug,
    )
