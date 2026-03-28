import math
from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.common import load_parameters
from lalamo.model_import.loaders.qwen3_tts_loaders import (
    load_qwen3_tts_audio_decoder,
    load_qwen3_tts_text_decoder,
)
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import (
    AttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    LalamoModule,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TTSConfig,
    TTSModel,
    UnscaledRoPEConfig,
    UpcastMode,
    VocoderConfig,
)
from lalamo.modules.activations import GELU
from lalamo.modules.audio.common_modules import (
    CausalTransposeConv1dConfig,
    Conv1dConfig,
    ConvNeXtBlockConfig,
    DACDecoderConfig,
    DecoderBlockConfig,
    ResidualUnitConfig,
    SnakeBetaConfig,
    UpsamplingBlockConfig,
)
from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder, StubTextDecoderConfig
from lalamo.modules.audio.qwen3_tts.qwen3_tts_audio_decoding import (
    Qwen3TTSAudioDecoder,
    Qwen3TTSAudioDecoderConfig,
    Qwen3TTSPreTransformerConfig,
    Qwen3TTSPreTransformerLayerConfig,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_modules import (
    EuclideanCodebookConfig,
    ResidualVectorQuantizerConfig,
    VectorQuantizationConfig,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_text_decoding import (
    Qwen3TTSTextDecoder,
    Qwen3TTSTextDecoderConfig,
    build_transformer_config,
)

__all__ = ["Qwen3TTSTokenizer12HzConfig"]


@dataclass(frozen=True)
class Qwen3TTSEncoderConfig:
    dtype: str


@dataclass(frozen=True)
class Qwen3TTSTokenizer12HzDecoderConfig:
    attention_dropout: float
    attention_bias: bool
    codebook_dim: int
    codebook_size: int
    decoder_dim: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    latent_dim: int
    layer_scale_initial_scale: float
    max_position_embeddings: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    num_quantizers: int
    num_semantic_quantizers: int
    rms_norm_eps: float
    semantic_codebook_size: int
    rope_theta: float
    sliding_window: int
    upsample_rates: tuple[int, ...]
    upsampling_ratios: tuple[int, ...]
    vector_quantization_hidden_dimension: int


@dataclass(frozen=True)
class Qwen3TTSTalkerCodePredictorConfig:
    attention_bias: bool
    attention_dropout: float
    head_dim: int
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    num_attention_heads: int
    num_code_groups: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int | None
    use_sliding_window: bool
    vocab_size: int
    layer_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Qwen3TTSTalkerConfig:
    attention_bias: bool
    attention_dropout: float
    codec_bos_id: int
    codec_eos_token_id: int
    codec_nothink_id: int
    codec_pad_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_think_id: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    num_attention_heads: int
    num_code_groups: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int | None
    text_hidden_size: int
    text_vocab_size: int
    use_sliding_window: bool
    vocab_size: int
    code_predictor_config: Qwen3TTSTalkerCodePredictorConfig
    spk_id: dict[str, int]
    codec_language_id: dict[str, int]
    layer_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Qwen3TTSTokenizer12HzConfig(ForeignTTSConfig):
    decoder_config: Qwen3TTSTokenizer12HzDecoderConfig
    decode_upsample_rate: int
    encode_downsample_rate: int
    encoder_valid_num_quantizers: int
    input_sample_rate: int
    encoder_config: Qwen3TTSEncoderConfig
    model_type: str
    output_sample_rate: int

    talker_config: Qwen3TTSTalkerConfig | None = None
    tts_pad_token_id: int | None = None
    tts_bos_token_id: int | None = None
    tts_eos_token_id: int | None = None

    def to_tts_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,  # noqa: ARG002
    ) -> TTSConfig:
        dc = self.decoder_config
        assert dc.hidden_act == "silu", (
            f"Only `hidden_act=silu` is supported for Qwen3-TTS decoder, got {dc.hidden_act!r}."
        )
        assert self.encoder_valid_num_quantizers == dc.num_quantizers, (
            f"Mismatch between top-level and decoder quantizer counts:"
            f" encoder_valid_num_quantizers={self.encoder_valid_num_quantizers},"
            f" decoder.num_quantizers={dc.num_quantizers}."
        )
        upsample_rate = math.prod((*dc.upsample_rates, *dc.upsampling_ratios))
        assert self.decode_upsample_rate == upsample_rate, (
            f"decode_upsample_rate does not match decoder upsampling factors:"
            f" decode_upsample_rate={self.decode_upsample_rate},"
            f" product(upsample_rates, upsampling_ratios)={upsample_rate}."
        )

        audio_decoder_config = _build_audio_decoder_config(activation_precision, dc, self)

        if self.talker_config is None:
            text_decoder_config = StubTextDecoderConfig(
                num_codebooks=self.encoder_valid_num_quantizers,
                codebook_size=dc.codebook_size,
                precision=activation_precision,
            )
        else:
            text_decoder_config = _build_text_decoder_config(
                activation_precision,
                self.talker_config,
                self,
                dc,
                context_length,
            )

        return TTSConfig(
            text_decoder_config=text_decoder_config,
            audio_decoder_config=audio_decoder_config,
            vocoder_config=VocoderConfig(),
            activation_precision=activation_precision,
        )

    def _load_weights(
        self,
        model: LalamoModule,
        weights_dict: Mapping[str, Array],
    ) -> LalamoModule:
        assert isinstance(model, TTSModel)
        assert isinstance(model.audio_decoder, Qwen3TTSAudioDecoder)

        decoder_path = _detect_decoder_path(weights_dict)
        loaded_audio_decoder = load_qwen3_tts_audio_decoder(model.audio_decoder, weights_dict, decoder_path)

        if isinstance(model.text_decoder, Qwen3TTSTextDecoder):
            talker_path = _detect_talker_path(weights_dict)
            loaded_text_decoder = load_qwen3_tts_text_decoder(model.text_decoder, weights_dict, talker_path)
        elif isinstance(model.text_decoder, StubTextDecoder):
            loaded_text_decoder = model.text_decoder
        else:
            raise TypeError(f"Unsupported Qwen3-TTS text decoder type: {type(model.text_decoder)!r}")

        return load_parameters(
            lambda m: (m.text_decoder, m.audio_decoder),
            model,
            (loaded_text_decoder, loaded_audio_decoder),
        )

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(self.encoder_config.dtype)


def _build_sliding_window_sizes(
    *,
    num_hidden_layers: int,
    use_sliding_window: bool,
    sliding_window: int | None,
    max_window_layers: int | None = None,
    layer_types: tuple[str, ...] | None = None,
) -> tuple[int | None, ...]:
    if layer_types is not None:
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(layer_types)} does not match num_hidden_layers={num_hidden_layers}",
            )
        return tuple(sliding_window if layer_type == "sliding_attention" else None for layer_type in layer_types)

    if not use_sliding_window:
        return tuple(None for _ in range(num_hidden_layers))

    if max_window_layers is None:
        return tuple(sliding_window for _ in range(num_hidden_layers))

    return tuple(sliding_window if layer_idx >= max_window_layers else None for layer_idx in range(num_hidden_layers))


def _detect_decoder_path(weights_dict: Mapping[str, Array]) -> ParameterPath:
    path = ParameterPath("decoder")
    assert path / "pre_transformer" / "input_proj" / "weight" in weights_dict, (
        f"Expected decoder weights under 'decoder.*' prefix. First keys: {sorted(weights_dict)[:8]}"
    )
    return path


def _detect_talker_path(weights_dict: Mapping[str, Array]) -> ParameterPath:
    path = ParameterPath("")
    assert path / "talker" / "model" / "codec_embedding" / "weight" in weights_dict, (
        f"Expected talker weights under 'talker.*' prefix. First keys: {sorted(weights_dict)[:8]}"
    )
    return path


def _clamp_context(context_length: int | None, max_pos: int) -> int:
    return min(context_length, max_pos) if context_length else max_pos


def _build_audio_decoder_config(
    precision: DTypeLike,
    dc: Qwen3TTSTokenizer12HzDecoderConfig,
    top_level_config: Qwen3TTSTokenizer12HzConfig,
) -> Qwen3TTSAudioDecoderConfig:
    linear_config = FullPrecisionLinearConfig(precision=precision)
    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=dc.rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    attention_config = AttentionConfig(
        qkv_projection_config=linear_config,
        out_projection_config=linear_config,
        query_norm_config=None,
        key_norm_config=None,
        num_heads=dc.num_attention_heads,
        num_groups=dc.num_key_value_heads,
        head_dim=dc.head_dim,
        is_causal=True,
        scale=None,
        sliding_window_size=dc.sliding_window,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=dc.attention_bias,
        has_out_biases=dc.attention_bias,
    )
    mlp_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    pre_transformer_config = Qwen3TTSPreTransformerConfig(
        precision=precision,
        input_projection_config=linear_config,
        output_projection_config=linear_config,
        output_norm_config=norm_config,
        rope_config=UnscaledRoPEConfig(
            precision=precision,
            base=dc.rope_theta,
            max_sequence_length=dc.max_position_embeddings,
        ),
        layer_config=Qwen3TTSPreTransformerLayerConfig(
            precision=precision,
            attention_config=attention_config,
            mlp_config=mlp_config,
            norm_config=norm_config,
            layer_scale_initial_scale=dc.layer_scale_initial_scale,
        ),
        hidden_size=dc.hidden_size,
        latent_dim=dc.latent_dim,
        intermediate_size=dc.intermediate_size,
        num_hidden_layers=dc.num_hidden_layers,
        max_position_embeddings=dc.max_position_embeddings,
    )

    snake_config = SnakeBetaConfig(precision=precision)
    convnext_config = ConvNeXtBlockConfig(
        precision=precision,
        activation=GELU(approximate=False),
        conv_config=Conv1dConfig(precision=precision, has_biases=True),
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
    )
    quantizer_config = ResidualVectorQuantizerConfig(
        precision=precision,
        vector_quantization_config=VectorQuantizationConfig(
            precision=precision,
            codebook_config=EuclideanCodebookConfig(precision=precision),
            project_out_config=linear_config,
        ),
        output_projection_config=linear_config,
        n_q_semantic=dc.num_semantic_quantizers,
    )

    conv_config = Conv1dConfig(precision=precision, has_biases=True)
    transpose_conv_config = CausalTransposeConv1dConfig(precision=precision, has_biases=True)
    decoder_block_config = DecoderBlockConfig(
        precision=precision,
        snake_config=snake_config,
        trans_conv_config=transpose_conv_config,
        res_unit_config=ResidualUnitConfig(
            precision=precision,
            snake_config=snake_config,
            conv_config=conv_config,
        ),
    )
    return Qwen3TTSAudioDecoderConfig(
        precision=precision,
        quantizer_config=quantizer_config,
        pre_conv_config=conv_config,
        pre_transformer_config=pre_transformer_config,
        upsample_block_config=UpsamplingBlockConfig(
            precision=precision,
            trans_conv_config=transpose_conv_config,
            convnext_config=convnext_config,
        ),
        dac_decoder_config=DACDecoderConfig(
            precision=precision,
            conv_config=conv_config,
            snake_config=snake_config,
            decoder_block_config=decoder_block_config,
        ),
        samplerate=top_level_config.output_sample_rate,
        decode_upsample_rate=top_level_config.decode_upsample_rate,
        num_quantizers=dc.num_quantizers,
        codebook_size=dc.codebook_size,
        codebook_dim=dc.codebook_dim,
        latent_dim=dc.latent_dim,
        upsample_rates=dc.upsample_rates,
        upsampling_ratios=dc.upsampling_ratios,
        decoder_dim=dc.decoder_dim,
    )


def _build_text_decoder_config(
    precision: DTypeLike,
    talker: Qwen3TTSTalkerConfig,
    top_level_config: Qwen3TTSTokenizer12HzConfig,
    decoder_config: Qwen3TTSTokenizer12HzDecoderConfig,
    context_length: int | None,
) -> Qwen3TTSTextDecoderConfig:
    predictor = talker.code_predictor_config

    if talker.hidden_act != "silu":
        raise ValueError(f"Only talker hidden_act=silu is supported, got {talker.hidden_act!r}.")
    if predictor.hidden_act != "silu":
        raise ValueError(f"Only predictor hidden_act=silu is supported, got {predictor.hidden_act!r}.")
    if talker.attention_dropout != 0.0:
        raise ValueError(
            f"Talker attention dropout is not implemented; expected 0.0, got {talker.attention_dropout}.",
        )
    if predictor.attention_dropout != 0.0:
        raise ValueError(
            f"Predictor attention dropout is not implemented; expected 0.0, got {predictor.attention_dropout}.",
        )

    assert top_level_config.tts_pad_token_id is not None, "tts_pad_token_id is required for talker config"
    assert top_level_config.tts_bos_token_id is not None, "tts_bos_token_id is required for talker config"
    assert top_level_config.tts_eos_token_id is not None, "tts_eos_token_id is required for talker config"

    linear_config = FullPrecisionLinearConfig(precision=precision)
    embedding_config = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)

    talker_transformer_config = build_transformer_config(
        precision=precision,
        hidden_size=talker.hidden_size,
        intermediate_size=talker.intermediate_size,
        num_hidden_layers=talker.num_hidden_layers,
        num_attention_heads=talker.num_attention_heads,
        num_key_value_heads=talker.num_key_value_heads,
        head_dim=talker.head_dim,
        max_position_embeddings=_clamp_context(context_length, talker.max_position_embeddings),
        rope_theta=talker.rope_theta,
        rms_norm_eps=talker.rms_norm_eps,
        attention_bias=talker.attention_bias,
        sliding_window_sizes=_build_sliding_window_sizes(
            num_hidden_layers=talker.num_hidden_layers,
            use_sliding_window=talker.use_sliding_window,
            sliding_window=talker.sliding_window,
            layer_types=talker.layer_types,
        ),
    )
    predictor_transformer_config = build_transformer_config(
        precision=precision,
        hidden_size=predictor.hidden_size,
        intermediate_size=predictor.intermediate_size,
        num_hidden_layers=predictor.num_hidden_layers,
        num_attention_heads=predictor.num_attention_heads,
        num_key_value_heads=predictor.num_key_value_heads,
        head_dim=predictor.head_dim,
        max_position_embeddings=_clamp_context(context_length, predictor.max_position_embeddings),
        rope_theta=predictor.rope_theta,
        rms_norm_eps=predictor.rms_norm_eps,
        attention_bias=predictor.attention_bias,
        sliding_window_sizes=_build_sliding_window_sizes(
            num_hidden_layers=predictor.num_hidden_layers,
            use_sliding_window=predictor.use_sliding_window,
            sliding_window=predictor.sliding_window,
            max_window_layers=predictor.max_window_layers,
            layer_types=predictor.layer_types,
        ),
    )

    return Qwen3TTSTextDecoderConfig(
        default_speaker="aiden",
        default_language="english",
        precision=precision,
        codec_embedding_config=embedding_config,
        text_embedding_config=embedding_config,
        predictor_embedding_config=embedding_config,
        linear_config=linear_config,
        talker_transformer_config=talker_transformer_config,
        predictor_transformer_config=predictor_transformer_config,
        talker_vocab_size=talker.vocab_size,
        text_vocab_size=talker.text_vocab_size,
        talker_hidden_size=talker.hidden_size,
        text_hidden_size=talker.text_hidden_size,
        predictor_hidden_size=predictor.hidden_size,
        predictor_vocab_size=predictor.vocab_size,
        num_code_groups=talker.num_code_groups,
        n_q_semantic=decoder_config.num_semantic_quantizers,
        max_new_tokens=_clamp_context(context_length, talker.max_position_embeddings),
        codec_bos_id=talker.codec_bos_id,
        codec_eos_token_id=talker.codec_eos_token_id,
        codec_pad_id=talker.codec_pad_id,
        codec_think_id=talker.codec_think_id,
        codec_nothink_id=talker.codec_nothink_id,
        codec_think_bos_id=talker.codec_think_bos_id,
        codec_think_eos_id=talker.codec_think_eos_id,
        tts_bos_token_id=top_level_config.tts_bos_token_id,
        tts_eos_token_id=top_level_config.tts_eos_token_id,
        tts_pad_token_id=top_level_config.tts_pad_token_id,
        speaker_id=talker.spk_id,
        language_id=talker.codec_language_id,
    )
