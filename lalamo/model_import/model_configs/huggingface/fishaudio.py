import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.common import load_parameters
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_fishaudio_audio_decoder,
    load_fishaudio_text_decoder,
)
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import (
    GELU,
    AttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinearConfig,
    LalamoModule,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    TTSConfig,
    TTSModel,
    UnscaledRoPEConfig,
    UpcastMode,
    VocoderConfig,
)
from lalamo.modules.audio.common_modules import (
    CausalConv1dConfig,
)
from lalamo.modules.audio.fishaudio import (
    DescriptAudioCodec,
    DescriptAudioCodecConfig,
    FishAudioTextDecoder,
    FishAudioTextDecoderConfig,
)
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    CausalTransposeConv1dConfig,
    ConvNeXtBlockConfig,
    DACDecoderBlockConfig,
    DACDecoderConfig,
    DownsampleResidualVectorQuantizeConfig,
    ResidualUnitConfig,
    ResidualVectorQuantizeConfig,
    Snake1dConfig,
    UpsamplerConfig,
    UpsamplingBlockConfig,
    VectorQuantizeConfig,
)

__all__ = ["FishAudioConfig"]


def lalamo_transformer_cfg_from_fish_audio_codec_cfg(
    config: Mapping[Any, Any],
    precision: DTypeLike,
    window_size: int,
    input_dim: int,
) -> TransformerConfig:
    # NOTE: this condifion is from post_init() for the post-module config object
    n_local_heads = config["n_head"] if config["n_local_heads"] == -1 else config["n_local_heads"]

    global_rope_config = UnscaledRoPEConfig(
        precision=precision,
        base=config["rope_base"],
        max_sequence_length=config["block_size"],
    )
    local_rope_config = None

    norm_config_pre = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=config["norm_eps"],
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )

    qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
    out_projection_config = FullPrecisionLinearConfig(precision=precision)
    mixer_config = AttentionConfig(
        qkv_projection_config=qkv_projection_config,
        out_projection_config=out_projection_config,
        query_norm_config=None,
        key_norm_config=None,
        num_heads=config["n_head"],
        num_groups=n_local_heads,
        head_dim=config["head_dim"],
        is_causal=True,
        scale=None,
        sliding_window_size=window_size,
        logit_soft_cap=None,
        has_sinks=False,
        has_qkv_biases=False,
        has_out_biases=False,
    )

    mlp_linear_config = FullPrecisionLinearConfig(precision=precision)
    mlp_use_up_biases = False
    mlp_use_down_biases = False
    mlp_config = DenseMLPConfig(
        linear_config=mlp_linear_config,
        activation=SiLU(),
        has_up_biases=mlp_use_up_biases,
        has_down_biases=mlp_use_down_biases,
        gate_clipping=None,
        up_clipping=None,
    )

    pre_mixer_norm_config = norm_config_pre
    post_mixer_norm_config = None
    pre_mlp_norm_config = norm_config_pre
    post_mlp_norm_config = None

    layer_config = TransformerLayerConfig(
        pre_mixer_norm_config=pre_mixer_norm_config,
        mixer_config=mixer_config,
        post_mixer_norm_config=post_mixer_norm_config,
        pre_mlp_norm_config=pre_mlp_norm_config,
        mlp_config=mlp_config,
        post_mlp_norm_config=post_mlp_norm_config,
    )
    hidden_dim = config["intermediate_size"]
    context_length = config["block_size"]

    transformer_cfg = TransformerConfig(
        global_rope_config=global_rope_config,
        local_rope_config=local_rope_config,
        layer_configs=tuple([layer_config] * config["n_layer"]),
        output_norm_config=norm_config_pre,
        model_dim=input_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
    )

    return transformer_cfg


def instantiate_dac_config_from_fishaudio_config(
    fish_dac_config: Mapping[Any, Any],
) -> DescriptAudioCodecConfig:
    precision = jnp.float32

    samplerate = fish_dac_config["sample_rate"]
    fish_quantizer_config = fish_dac_config["quantizer"]

    input_dim = fish_quantizer_config["input_dim"]
    downsample_factor = fish_quantizer_config["downsample_factor"]
    post_module_config_dict = fish_quantizer_config["post_module"]
    encoder_dim = fish_dac_config["encoder_dim"]
    encoder_rates = fish_dac_config["encoder_rates"]
    decoder_dim = fish_dac_config["decoder_dim"]
    decoder_rates = fish_dac_config["decoder_rates"]
    fish_quantizer_config = fish_dac_config["quantizer"]
    input_dim = fish_quantizer_config["input_dim"]
    n_codebooks = fish_quantizer_config["n_codebooks"]
    codebook_dim = fish_quantizer_config["codebook_dim"]
    downsample_factor = fish_quantizer_config["downsample_factor"]
    codebook_size = fish_quantizer_config["codebook_size"]
    semantic_codebook_size = fish_quantizer_config["semantic_codebook_size"]

    convnext_config = ConvNeXtBlockConfig(
        precision=jnp.float32,
        activation=GELU(approximate=False),
        dwconv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        norm_config=NormalizationConfig(
            scale_precision=jnp.float32,
            accumulation_precision=jnp.float32,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        ),
        pwconv_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    upsampling_block_config = UpsamplingBlockConfig(
        precision=jnp.float32,
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        convnext_config=convnext_config,
    )
    num_blocks = len(downsample_factor)
    block_configs = tuple(upsampling_block_config for _ in range(num_blocks))
    upsampler_config = UpsamplerConfig(block_configs=block_configs)

    post_module_transformer_foreign = post_module_config_dict["config"]
    post_module_config = lalamo_transformer_cfg_from_fish_audio_codec_cfg(
        post_module_transformer_foreign,
        precision,
        window_size=post_module_config_dict["window_size"],
        input_dim=post_module_config_dict["input_dim"],
    )

    vq_config = VectorQuantizeConfig(
        precision=jnp.float32,
        codebook_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=jnp.float32,
        ),
        out_proj_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    lalamo_rvq_config = ResidualVectorQuantizeConfig(
        precision=jnp.float32,
        vq_config=vq_config,
    )

    quantizer_full_config = DownsampleResidualVectorQuantizeConfig(
        precision=precision,
        semantic_quantizer_config=lalamo_rvq_config,
        quantizer_config=lalamo_rvq_config,
        post_module_config=post_module_config,
        upsampler_config=upsampler_config,
    )
    res_unit_config = ResidualUnitConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        causal=True,
    )
    decoder_block_config = DACDecoderBlockConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        res_unit_config=res_unit_config,
        causal=True,
    )
    decoder_config = DACDecoderConfig(
        precision=jnp.float32,
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        snake_config=Snake1dConfig(precision=jnp.float32),
        decoder_block_config=decoder_block_config,
        causal=True,
    )

    return DescriptAudioCodecConfig(
        precision=precision,
        quantizer_config=quantizer_full_config,
        decoder_config=decoder_config,
        samplerate=samplerate,
        encoder_dim=encoder_dim,
        encoder_rates=encoder_rates,
        decoder_dim=decoder_dim,
        decoder_rates=decoder_rates,
        input_dim=input_dim,
        n_codebooks=n_codebooks,
        codebook_dim=codebook_dim,
        downsample_factor=downsample_factor,
        codebook_size=codebook_size,
        semantic_codebook_size=semantic_codebook_size,
    )


@dataclass(frozen=True)
class FishAudioConfig(ForeignTTSConfig):
    attention_o_bias: bool
    attention_qk_norm: bool
    attention_qkv_bias: bool
    codebook_size: int
    dim: int
    dropout: float
    fast_attention_o_bias: bool
    fast_attention_qk_norm: bool
    fast_attention_qkv_bias: bool
    fast_dim: int
    fast_head_dim: int
    fast_intermediate_size: int
    fast_n_head: int
    fast_n_local_heads: int
    head_dim: int
    initializer_range: int
    intermediate_size: int
    max_seq_len: int
    model_type: str
    n_fast_layer: int
    n_head: int
    n_layer: int
    n_local_heads: int
    norm_eps: float
    num_codebooks: int
    rope_base: int
    scale_codebook_embeddings: bool
    tie_word_embeddings: bool
    use_gradient_checkpointing: bool
    vocab_size: int

    # NOTE: these fields are used during inference but must be retrieved from
    # tokenizer config files
    semantic_token_begin_id: int = -1
    semantic_token_end_id: int = -1
    im_end_token_id: int = -1

    def extract_textual_transformer_configs(
        self,
        precision: DTypeLike,
        fast_module: bool = False,
    ) -> tuple[TransformerConfig, FullPrecisionLinearConfig]:
        n_layer = self.n_fast_layer if fast_module else self.n_layer
        n_head = self.fast_n_head if fast_module else self.n_head
        dim = self.fast_dim if fast_module else self.dim
        intermediate_size = self.fast_intermediate_size if fast_module else self.intermediate_size
        n_local_heads = self.fast_n_local_heads if fast_module else self.n_local_heads
        head_dim = self.fast_head_dim if fast_module else self.head_dim
        attention_qk_norm = self.fast_attention_qk_norm if fast_module else self.attention_qk_norm

        global_rope_config = UnscaledRoPEConfig(
            precision=precision,
            base=self.rope_base,
            max_sequence_length=self.max_seq_len,
        )
        local_rope_config = None

        norm_config = NormalizationConfig(
            scale_precision=precision,
            accumulation_precision=precision,
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
        out_projection_config = FullPrecisionLinearConfig(precision=precision)
        mixer_config = AttentionConfig(
            qkv_projection_config=qkv_projection_config,
            out_projection_config=out_projection_config,
            query_norm_config=norm_config if attention_qk_norm else None,
            key_norm_config=norm_config if attention_qk_norm else None,
            num_heads=n_head,
            num_groups=n_local_heads,
            head_dim=head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
        )

        mlp_linear_config = FullPrecisionLinearConfig(precision=precision)
        mlp_use_up_biases = False
        mlp_use_down_biases = False
        mlp_config = DenseMLPConfig(
            linear_config=mlp_linear_config,
            activation=SiLU(),
            has_up_biases=mlp_use_up_biases,
            has_down_biases=mlp_use_down_biases,
            gate_clipping=None,
            up_clipping=None,
        )

        pre_mixer_norm_config = norm_config
        post_mixer_norm_config = None
        pre_mlp_norm_config = norm_config
        post_mlp_norm_config = None

        layer_config = TransformerLayerConfig(
            pre_mixer_norm_config=pre_mixer_norm_config,
            mixer_config=mixer_config,
            post_mixer_norm_config=post_mixer_norm_config,
            pre_mlp_norm_config=pre_mlp_norm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=post_mlp_norm_config,
        )
        model_dim = dim
        hidden_dim = intermediate_size
        context_length = self.max_seq_len

        transformer_cfg = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple([layer_config] * n_layer),
            output_norm_config=norm_config,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )
        linear_out_cfg = FullPrecisionLinearConfig(precision=precision)

        return (transformer_cfg, linear_out_cfg)

    def to_tts_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,  # noqa: ARG002
    ) -> TTSConfig:
        audio_decoder_config = instantiate_dac_config_from_fishaudio_config(
            fish_dac_config=get_default_fishaudio_dac_config(),
        )

        slow_transformer_cfg, slow_readout_cfg = self.extract_textual_transformer_configs(
            precision=activation_precision,
            fast_module=False,
        )
        fast_transformer_cfg, fast_readout_cfg = self.extract_textual_transformer_configs(
            precision=activation_precision,
            fast_module=True,
        )
        slow_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=activation_precision)
        fast_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=activation_precision)

        codebook_embeddings_cfg = TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=activation_precision,
        )
        if self.dim == self.fast_dim:
            fast_model_projection_config = None
        else:
            fast_model_projection_config = FullPrecisionLinearConfig(activation_precision)
        text_decoder_config = FishAudioTextDecoderConfig(
            slow_embeddings_config=slow_embedding_cfg,
            slow_model_config=slow_transformer_cfg,
            slow_readout_config=slow_readout_cfg,
            fast_embeddings_config=fast_embedding_cfg,
            fast_model_config=fast_transformer_cfg,
            fast_readout_config=fast_readout_cfg,
            codebook_embeddings_config=codebook_embeddings_cfg,
            fast_model_projection_config=fast_model_projection_config,
            semantic_token_begin_id=self.semantic_token_begin_id,
            semantic_token_end_id=self.semantic_token_end_id,
            im_end_token_id=self.im_end_token_id,
            codebook_size=self.codebook_size,
            vocab_size=self.vocab_size,
            slow_model_dim=self.dim,
            fast_model_dim=self.fast_dim,
            num_codebooks=self.num_codebooks,
            max_seq_len=min(context_length, self.max_seq_len) if context_length else self.max_seq_len,
            scale_codebook_embeddings=self.scale_codebook_embeddings,
            precision=activation_precision,
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

        assert isinstance(model.text_decoder, FishAudioTextDecoder)
        loaded_text_decoder = load_fishaudio_text_decoder(model.text_decoder, weights_dict, ParameterPath())

        assert isinstance(model.audio_decoder, DescriptAudioCodec)
        loaded_audio_decoder = load_fishaudio_audio_decoder(model.audio_decoder, weights_dict, ParameterPath())

        return load_parameters(
            lambda m: (
                m.text_decoder,
                m.audio_decoder,
            ),
            model,
            (loaded_text_decoder, loaded_audio_decoder),
        )

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

    @property
    def default_precision(self) -> DTypeLike:
        # NOTE: in reality FishAudio text-decoder is bf16 while audio-decoder if fp32.
        # Currently lalamo weight manipulation pipeline does not suport such
        # mixed-model-mixed-weight configuration so we upcast everything to fp32
        # as temporary solution
        return jnp.dtype(getattr(self, "torch_dtype", "float32"))
