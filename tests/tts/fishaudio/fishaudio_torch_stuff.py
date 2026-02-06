import shutil
import tempfile
from pathlib import Path
from typing import Optional

import huggingface_hub
import torch
from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    DualARModelArgs,
)
from fish_speech.tokenizer import FishTokenizer
from jax import numpy as jnp
from jaxtyping import DTypeLike
from tokenizers import Tokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from lalamo.models.tts_model import FishAudioTTSGenerator, TTSGenerator, TTSGeneratorConfig
from lalamo.modules import (
    AttentionConfig,
    DenseMLPConfig,
    NormalizationConfig,
    TransformerConfig,
    TransformerLayerConfig,
    UnscaledRoPEConfig,
    UpcastMode,
)
from lalamo.modules.activations import SiLU
from lalamo.modules.audio.fishaudio.fishaudio_consts import IM_END_TOKEN
from lalamo.modules.audio.fishaudio.fishaudio_text_decoding import FishAudioTextDecoderConfig
from lalamo.modules.audio.text_to_speech import TTSConfig, TTSMessageProcessor, TTSMessageProcessorConfig, TTSModel
from lalamo.modules.audio.vocoders import NoopVocoder, VocoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.torch_interop import DTypeConvert, torch_to_jax

from .fishaudio_thin_wrapper import FishAudioTextDecoderConfig_Foreign


def from_fish_audio_config(
    fish_audio_cfg: DualARModelArgs,
    tokenizer: FishTokenizer,
    precision: DTypeLike,
) -> "FishAudioTextDecoderConfig":
    slow_transformer_cfg, slow_readout_cfg = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(
        fish_audio_cfg, precision
    )
    fast_fish_cfg = ConfigMapping.extract_fast_transformer_params(fish_audio_cfg)
    fast_transformer_cfg, fast_readout_cfg = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(
        fast_fish_cfg, precision
    )

    slow_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
    fast_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)

    codebook_embeddings_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
    if fish_audio_cfg.dim == fish_audio_cfg.fast_dim:
        fast_model_projection_config = None
    else:
        fast_model_projection_config = FullPrecisionLinearConfig(precision)

    assert fish_audio_cfg.fast_dim is not None
    return FishAudioTextDecoderConfig(
        slow_embeddings_config=slow_embedding_cfg,
        slow_model_config=slow_transformer_cfg,
        slow_readout_config=slow_readout_cfg,
        fast_embeddings_config=fast_embedding_cfg,
        fast_model_config=fast_transformer_cfg,
        fast_readout_config=fast_readout_cfg,
        codebook_embeddings_config=codebook_embeddings_cfg,
        fast_model_projection_config=fast_model_projection_config,
        semantic_token_begin_id=tokenizer.semantic_begin_id,
        semantic_token_end_id=tokenizer.semantic_end_id,
        im_end_token_id=tokenizer.get_token_id(IM_END_TOKEN),
        codebook_size=fish_audio_cfg.codebook_size,
        precision=precision,
        vocab_size=fish_audio_cfg.vocab_size,
        slow_model_dim=fish_audio_cfg.dim,
        fast_model_dim=fish_audio_cfg.fast_dim,
        num_codebooks=fish_audio_cfg.num_codebooks,
        max_seq_len=fish_audio_cfg.max_seq_len,
        scale_codebook_embeddings=fish_audio_cfg.scale_codebook_embeddings,
    )


class ConfigMapping:
    @staticmethod
    def extract_fast_transformer_params(fish_transformer_config: DualARModelArgs) -> BaseModelArgs:
        return BaseModelArgs(
            model_type=fish_transformer_config.model_type,
            vocab_size=fish_transformer_config.vocab_size,
            n_layer=fish_transformer_config.n_fast_layer,
            n_head=fish_transformer_config.fast_n_head,
            dim=fish_transformer_config.fast_dim,
            intermediate_size=fish_transformer_config.fast_intermediate_size,
            n_local_heads=fish_transformer_config.fast_n_local_heads,
            head_dim=fish_transformer_config.fast_head_dim,
            rope_base=fish_transformer_config.rope_base,
            norm_eps=fish_transformer_config.norm_eps,
            max_seq_len=fish_transformer_config.max_seq_len,
            dropout=fish_transformer_config.dropout,
            tie_word_embeddings=fish_transformer_config.tie_word_embeddings,
            attention_qkv_bias=fish_transformer_config.fast_attention_qkv_bias,
            attention_o_bias=fish_transformer_config.fast_attention_o_bias,
            attention_qk_norm=fish_transformer_config.fast_attention_qk_norm,
            codebook_size=fish_transformer_config.codebook_size,
            num_codebooks=fish_transformer_config.num_codebooks,
        )

    @staticmethod
    def lalamo_transformer_cfg_from_fish_text_decoder_cfg(
        config: BaseModelArgs, precision: DTypeLike
    ) -> tuple[TransformerConfig, FullPrecisionLinearConfig]:
        global_rope_config = UnscaledRoPEConfig(
            precision=precision,
            base=config.rope_base,
            max_sequence_length=config.max_seq_len,
        )
        local_rope_config = None

        norm_config = NormalizationConfig(
            scale_precision=precision,
            accumulation_precision=precision,
            epsilon=config.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
        out_projection_config = FullPrecisionLinearConfig(precision=precision)
        mixer_config = AttentionConfig(
            qkv_projection_config=qkv_projection_config,
            out_projection_config=out_projection_config,
            query_norm_config=norm_config if config.attention_qk_norm else None,
            key_norm_config=norm_config if config.attention_qk_norm else None,
            num_heads=config.n_head,
            num_groups=config.n_local_heads,
            head_dim=config.head_dim,
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
        model_dim = config.dim
        hidden_dim = config.intermediate_size
        context_length = config.max_seq_len

        transformer_cfg = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple([layer_config] * config.n_layer),
            output_norm_config=norm_config,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )
        linear_out_cfg = FullPrecisionLinearConfig(precision=precision)

        return (transformer_cfg, linear_out_cfg)


class FishAudioFromTorch:
    @staticmethod
    def load_tokenizer_from_fish_audio(path_to_chkpt: str) -> Tokenizer:
        output_temp_dir = tempfile.mkdtemp()
        try:
            fishspeech_tokenizer = FishTokenizer.from_pretrained(path_to_chkpt)

            convert_tiktoken_to_fast(fishspeech_tokenizer.tkt_model, output_temp_dir)
            tokenizer = Tokenizer.from_file(output_temp_dir + "/tokenizer.json")
            return tokenizer
        finally:
            shutil.rmtree(output_temp_dir)

    @staticmethod
    def build_foreign_fish_audio_tts_generator(
        path_to_checkpoints: Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
    ) -> "TTSGenerator":
        from .fishaudio_thin_wrapper import FishAudioTextDecoder_Foreign, load_fish_audio_audio_decoder

        text_decoder_config = FishAudioTextDecoderConfig_Foreign.from_config_file(path_to_checkpoints / "config.json")
        text_decoder: FishAudioTextDecoder_Foreign = text_decoder_config.load_model(
            path_to_checkpoints, device=device, precision=precision
        )
        audio_decoder = load_fish_audio_audio_decoder(path_to_checkpoints)

        tokenizer = FishAudioFromTorch.load_tokenizer_from_fish_audio(str(path_to_checkpoints))

        prompt_template = """
    {% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
    """

        tts_request_factory_config = TTSMessageProcessorConfig(
            prompt_template=prompt_template,
        )

        message_processor = TTSMessageProcessor(tts_request_factory_config, tokenizer)

        tts_config = TTSConfig(
            text_decoder.config,
            audio_decoder.config,
            VocoderConfig(),
            activation_precision=DTypeConvert.to_jax(precision),
        )

        tts_model = TTSModel(
            config=tts_config,
            text_decoder=text_decoder,
            audio_decoder=audio_decoder,
            vocoder=NoopVocoder(tts_config.vocoder_config),
        )
        return FishAudioTTSGenerator(
            config=TTSGeneratorConfig(tts_config=tts_config, message_processor_config=message_processor.config),
            tts_model=tts_model,
            message_processor=message_processor,
        )


def prepare_state_dict_for_lalamo_loaders(
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> dict[str, jnp.ndarray]:
    """Convert PyTorch state_dict to JAX arrays with optional key prefix for Lalamo loaders."""
    result = {}
    for key, tensor in state_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        result[full_key] = torch_to_jax(tensor.detach())
    return result
