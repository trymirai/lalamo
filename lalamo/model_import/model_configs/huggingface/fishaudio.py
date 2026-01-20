import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.common import load_parameters
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_audio_decoder,
    load_downsample_rvq,
    load_fish_audio_text_decoding_modules,
)
from lalamo.model_import.loaders.huggingface import load_linear, load_tied_embedding
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import (
    AttentionConfig,
    DenseMLPConfig,
    FullPrecisionLinear,
    FullPrecisionLinearConfig,
    Identity,
    LalamoModule,
    NormalizationConfig,
    SiLU,
    TiedEmbeddingConfig,
    TransformerConfig,
    TransformerLayerConfig,
    TTSConfig,
    TTSModel,
    UpcastMode,
    VocoderConfig,
)
from lalamo.modules.audio.fishaudio import (
    DescriptAudioCodec,
    DescriptAudioCodecConfig,
    FishAudioTextDecoder,
    FishAudioTextDecoderConfig,
)
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.rope import RoPEConfigCis

__all__ = ["FishAudioConfig"]


def load_fishaudio_text_decoder(
    module: FishAudioTextDecoder,
    weights_dict: Mapping[str, Array],
    decoder_path: ParameterPath | None = None,
) -> FishAudioTextDecoder:
    basepath = ParameterPath() if decoder_path is None else decoder_path
    transformer_slow, readout_slow = load_fish_audio_text_decoding_modules(
        module.transformer_slow,
        module.readout_slow,
        weights_dict,
        fast=False,
    )
    transformer_fast, readout_fast = load_fish_audio_text_decoding_modules(
        module.transformer_fast,
        module.readout_fast,
        weights_dict,
        fast=True,
    )
    embeddings_slow = load_tied_embedding(
        module.embeddings_slow,
        weights_dict,
        basepath / "embeddings",
    )
    embeddings_fast = load_tied_embedding(
        module.embeddings_fast,
        weights_dict,
        basepath / "fast_embeddings",
    )

    codebook_embeddings = load_tied_embedding(
        module.codebook_embeddings,
        weights_dict,
        basepath / "codebook_embeddings",
    )

    if isinstance(module.fast_model_projection, FullPrecisionLinear):
        fast_model_projection = load_linear(
            module.fast_model_projection,
            weights_dict,
            basepath / "fast_project_in",
        )
        assert isinstance(fast_model_projection, FullPrecisionLinear)
    else:
        fast_model_projection = Identity()

    return load_parameters(
        lambda m: (
            m.embeddings_slow,
            m.transformer_slow,
            m.readout_slow,
            m.embeddings_fast,
            m.transformer_fast,
            m.readout_fast,
            m.codebook_embeddings,
            m.fast_model_projection,
        ),
        module,
        (
            embeddings_slow,
            transformer_slow,
            readout_slow,
            embeddings_fast,
            transformer_fast,
            readout_fast,
            codebook_embeddings,
            fast_model_projection,
        ),
    )


def load_fishaudio_audio_decoder(
    module: DescriptAudioCodec,
    weights_dict: Mapping[str, Array],
    base_path: ParameterPath,
) -> DescriptAudioCodec:
    loaded_quantizer = load_downsample_rvq(module.quantizer, weights_dict, base_path / "quantizer")
    loaded_decoder = load_audio_decoder(module.decoder, weights_dict, base_path / "decoder")

    return load_parameters(lambda m: (m.quantizer, m.decoder), module, (loaded_quantizer, loaded_decoder))


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

    def extract_transformer_configs(
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

        global_rope_config = RoPEConfigCis(precision=precision, base=self.rope_base)
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
        audio_decoder_config = DescriptAudioCodecConfig.instantiate_config_from_fishaudio_config(
            fish_dac_config=get_default_fishaudio_dac_config(),
        )

        slow_transformer_cfg, slow_readout_cfg = self.extract_transformer_configs(
            precision=activation_precision,
            fast_module=False,
        )
        fast_transformer_cfg, fast_readout_cfg = self.extract_transformer_configs(
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
