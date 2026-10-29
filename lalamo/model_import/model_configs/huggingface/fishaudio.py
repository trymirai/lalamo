import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Self

from jaxtyping import Array

from lalamo.model import Model
from lalamo.model_import.loaders.binding import Binder
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_fish_attention,
    load_fish_mlp,
    load_fishaudio_audio_decoder,
    load_fishaudio_text_decoder,
)
from lalamo.model_import.loaders.huggingface import huggingface_loader
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.models import TTSConfig, TTSModel
from lalamo.modules.activations import GELU, SiLU
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
from lalamo.modules.audio.vocoders import NoopVocoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import EmitConfig, SequenceConfig, StepConfig, residual
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation

__all__ = ["FishAudioConfig"]


def _tts_decoders(model: TTSModel) -> tuple[object, object]:
    return model.text_decoder, model.audio_decoder


def fish_transformer_program(
    at: Binder,
    path: ParameterPath,
    *,
    num_layers: int,
    attention: AttentionConfig,
    mlp: DenseMLPConfig,
    norm: NormalizationConfig,
    layers_name: str = "layers",
    output_norm_name: str = "norm",
) -> tuple[SequenceConfig, EmitConfig]:
    """FishAudio's pre-norm transformer; also returns a tap on the hidden state before the output norm."""
    steps: list[StepConfig] = []
    for index in range(num_layers):
        layer_path = path / layers_name / index
        load_attention = partial(load_fish_attention, layer_scale_path=layer_path / "attention_layer_scale" / "gamma")
        load_mlp = partial(load_fish_mlp, layer_scale_path=layer_path / "ffn_layer_scale" / "gamma")
        steps += [
            residual(
                at(layer_path / "attention_norm", norm),
                at(layer_path / "attention", attention, load=load_attention),
            ),
            residual(
                at(layer_path / "ffn_norm", norm),
                at(layer_path / "feed_forward", mlp, load=load_mlp),
            ),
        ]
    hidden_states = EmitConfig("hidden_states")
    return SequenceConfig((*steps, hidden_states, at(path / output_norm_name, norm))), hidden_states


def fish_audio_codec_post_module(
    at: Binder,
    config: Mapping[Any, Any],
    window_size: int,
) -> SequenceConfig:
    # FishAudio post-init treats -1 as n_head.
    n_local_heads = config["n_head"] if config["n_local_heads"] == -1 else config["n_local_heads"]
    norm_config = NormalizationConfig(
        epsilon=config["norm_eps"],
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    linear_config = LinearConfig()
    program, _ = fish_transformer_program(
        at,
        ParameterPath("quantizer.post_module"),
        num_layers=config["n_layer"],
        attention=AttentionConfig(
            qkvg_projection_config=linear_config,
            out_projection_config=linear_config,
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
            has_qkvg_biases=False,
            has_out_biases=False,
            rope_config=UnscaledRoPEConfig(
                base=float(config["rope_base"]),
                max_sequence_length=config["block_size"],
                head_dim=config["head_dim"],
            ),
        ),
        mlp=DenseMLPConfig(
            linear_config=linear_config,
            activation=SiLU(),
            has_up_biases=False,
            has_down_biases=False,
            gate_clipping=None,
            up_clipping=None,
            hidden_dim=config["intermediate_size"],
        ),
        norm=norm_config,
    )
    return program


def instantiate_dac_config_from_fishaudio_config(
    fish_dac_config: Mapping[Any, Any],
    at: Binder,
) -> DescriptAudioCodecConfig:
    samplerate = fish_dac_config["sample_rate"]
    fish_quantizer_config = fish_dac_config["quantizer"]

    input_dim = fish_quantizer_config["input_dim"]
    downsample_factor = tuple(fish_quantizer_config["downsample_factor"])
    post_module_config_dict = fish_quantizer_config["post_module"]
    encoder_dim = fish_dac_config["encoder_dim"]
    encoder_rates = tuple(fish_dac_config["encoder_rates"])
    decoder_dim = fish_dac_config["decoder_dim"]
    decoder_rates = tuple(fish_dac_config["decoder_rates"])
    n_codebooks = fish_quantizer_config["n_codebooks"]
    codebook_dim = fish_quantizer_config["codebook_dim"]
    codebook_size = fish_quantizer_config["codebook_size"]
    semantic_codebook_size = fish_quantizer_config["semantic_codebook_size"]

    convnext_config = ConvNeXtBlockConfig(
        activation=GELU(approximate=False),
        dwconv_config=CausalConv1dConfig(has_biases=True),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            has_biases=True,
        ),
        pwconv_config=LinearConfig(),
    )
    upsampling_block_config = UpsamplingBlockConfig(
        trans_conv_config=CausalTransposeConv1dConfig(has_biases=True),
        convnext_config=convnext_config,
    )
    num_blocks = len(downsample_factor)
    block_configs = tuple(upsampling_block_config for _ in range(num_blocks))
    upsampler_config = UpsamplerConfig(block_configs=block_configs)

    post_module_config = fish_audio_codec_post_module(
        at,
        post_module_config_dict["config"],
        window_size=post_module_config_dict["window_size"],
    )

    vq_config = VectorQuantizeConfig(
        codebook_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
        ),
        out_proj_config=LinearConfig(),
    )
    lalamo_rvq_config = ResidualVectorQuantizeConfig(
        vq_config=vq_config,
    )

    quantizer_full_config = DownsampleResidualVectorQuantizeConfig(
        semantic_quantizer_config=lalamo_rvq_config,
        quantizer_config=lalamo_rvq_config,
        post_module_config=post_module_config,
        post_module_dim=post_module_config_dict["input_dim"],
        upsampler_config=upsampler_config,
    )
    res_unit_config = ResidualUnitConfig(
        snake_config=Snake1dConfig(),
        conv_config=CausalConv1dConfig(has_biases=True),
        causal=True,
    )
    decoder_block_config = DACDecoderBlockConfig(
        snake_config=Snake1dConfig(),
        trans_conv_config=CausalTransposeConv1dConfig(has_biases=True),
        res_unit_config=res_unit_config,
        causal=True,
    )
    decoder_config = DACDecoderConfig(
        conv_config=CausalConv1dConfig(has_biases=True),
        snake_config=Snake1dConfig(),
        decoder_block_config=decoder_block_config,
        causal=True,
    )

    return DescriptAudioCodecConfig(
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
    initializer_range: float
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

    def textual_program(self, at: Binder, *, fast: bool) -> tuple[SequenceConfig, EmitConfig]:
        n_layer = self.n_fast_layer if fast else self.n_layer
        n_head = self.fast_n_head if fast else self.n_head
        intermediate_size = self.fast_intermediate_size if fast else self.intermediate_size
        n_local_heads = self.fast_n_local_heads if fast else self.n_local_heads
        head_dim = self.fast_head_dim if fast else self.head_dim
        attention_qk_norm = self.fast_attention_qk_norm if fast else self.attention_qk_norm

        norm_config = NormalizationConfig(
            epsilon=self.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )
        linear_config = LinearConfig()
        return fish_transformer_program(
            at,
            ParameterPath(),
            num_layers=n_layer,
            attention=AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
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
                has_qkvg_biases=False,
                has_out_biases=False,
                rope_config=UnscaledRoPEConfig(
                    base=float(self.rope_base),
                    max_sequence_length=self.max_seq_len,
                    head_dim=head_dim,
                ),
            ),
            mlp=DenseMLPConfig(
                linear_config=linear_config,
                activation=SiLU(),
                has_up_biases=False,
                has_down_biases=False,
                gate_clipping=None,
                up_clipping=None,
                hidden_dim=intermediate_size,
            ),
            norm=norm_config,
            layers_name="fast_layers" if fast else "layers",
            output_norm_name="fast_norm" if fast else "norm",
        )

    def to_tts_config(self) -> TTSConfig:
        return self.build_tts_config(Binder())

    def build_tts_config(self, at: Binder) -> TTSConfig:
        audio_decoder_config = instantiate_dac_config_from_fishaudio_config(
            fish_dac_config=get_default_fishaudio_dac_config(),
            at=at,
        )
        slow_program, slow_hidden_states = self.textual_program(at, fast=False)
        fast_program, _ = self.textual_program(at, fast=True)

        if self.dim == self.fast_dim:
            fast_model_projection_config = None
        else:
            fast_model_projection_config = LinearConfig()
        text_decoder_config = FishAudioTextDecoderConfig(
            slow_embeddings_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            slow_model_config=slow_program,
            slow_hidden_states=slow_hidden_states,
            slow_readout_config=LinearConfig(),
            fast_embeddings_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            fast_model_config=fast_program,
            fast_readout_config=LinearConfig(),
            codebook_embeddings_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
            fast_model_projection_config=fast_model_projection_config,
            semantic_token_begin_id=self.semantic_token_begin_id,
            semantic_token_end_id=self.semantic_token_end_id,
            im_end_token_id=self.im_end_token_id,
            codebook_size=self.codebook_size,
            vocab_size=self.vocab_size,
            slow_model_dim=self.dim,
            fast_model_dim=self.fast_dim,
            num_codebooks=self.num_codebooks,
            max_seq_len=self.max_seq_len,
            scale_codebook_embeddings=self.scale_codebook_embeddings,
        )
        return TTSConfig(
            text_decoder_config=text_decoder_config,
            audio_decoder_config=audio_decoder_config,
            vocoder_config=NoopVocoderConfig(),
        )

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG002
    ) -> Model:
        assert isinstance(model, TTSModel)
        tts_model: TTSModel = model

        binder = Binder(huggingface_loader)
        tts_config = self.build_tts_config(binder)
        assert isinstance(tts_config.text_decoder_config, FishAudioTextDecoderConfig)
        assert isinstance(tts_config.audio_decoder_config, DescriptAudioCodecConfig)

        assert isinstance(tts_model.text_decoder, FishAudioTextDecoder)
        loaded_text_decoder = load_fishaudio_text_decoder(
            tts_model.text_decoder,
            tts_config.text_decoder_config,
            binder,
            weights_dict,
            ParameterPath(),
        )

        assert isinstance(tts_model.audio_decoder, DescriptAudioCodec)
        loaded_audio_decoder = load_fishaudio_audio_decoder(
            tts_model.audio_decoder,
            tts_config.audio_decoder_config,
            binder,
            weights_dict,
            ParameterPath(),
        )

        return load_as_at(
            _tts_decoders,
            tts_model,
            (loaded_text_decoder, loaded_audio_decoder),
        )

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)
