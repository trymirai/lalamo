from tokenizers import Tokenizer
import shutil
import tempfile
from fish_speech.tokenizer import FishTokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast


def load_fishaudio_text_decoder(
    fish_model_or_path: Path | torch.nn.Module, precision: DTypeLike
) -> FishAudioTextDecoder:
    from pathlib import Path

    from fish_speech.models.text2semantic.llama import (
        BaseModelArgs,
        DualARModelArgs,
        DualARTransformer,
        NaiveModelArgs,
    )
    from fish_speech.tokenizer import FishTokenizer

    from lalamo.modules.audio.fishaudio.fishaudio_thin_wrapper import FishAudioTextDecoderConfig_Foreign

    def cast_if_float(array: Array, cast_to: DTypeLike) -> Array:
        if array.dtype in [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64]:
            return array.astype(cast_to)
        return array

    if isinstance(fish_model_or_path, Path):
        fish_tokenizer = FishTokenizer.from_pretrained(str(fish_model_or_path))
        fish_model_cfg: DualARModelArgs | NaiveModelArgs = BaseModelArgs.from_pretrained(
            str(fish_model_or_path / "config.json")
        )
        assert isinstance(fish_model_cfg, DualARModelArgs)

        fish_model = FishAudioTextDecoderConfig_Foreign._load_fish_model(fish_model_or_path, fish_model_cfg)
        fish_model_dict = fish_model.state_dict()
    else:
        assert isinstance(fish_model_or_path, DualARTransformer)
        fish_tokenizer = fish_model_or_path.tokenizer
        fish_model_cfg = fish_model_or_path.config
        fish_model_dict = fish_model_or_path.state_dict()

    assert isinstance(fish_model_cfg, DualARModelArgs)
    config = FishAudioTextDecoderConfig.from_fish_audio_config(fish_model_cfg, fish_tokenizer, precision)
    weights_mapping = dict(MapDictValues(lambda v: cast_if_float(torch_to_jax(v), precision), fish_model_dict))

    transformer_slow, readout_slow = load_fish_audio_text_decoding_modules(
        config.slow_model_config.empty(),
        config.slow_readout_config.empty(
            input_dim=config.slow_model_dim,
            output_dims=(config.vocab_size,),
            has_biases=False,
        ),
        weights_mapping,
        fast=False,
    )
    transformer_fast, readout_fast = load_fish_audio_text_decoding_modules(
        config.fast_model_config.empty(),
        config.fast_readout_config.empty(
            input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
        ),
        weights_mapping,
        fast=True,
    )
    embeddings_slow = load_tied_embedding(
        config.slow_embeddings_config.empty(config.vocab_size, config.slow_model_dim),
        weights_mapping,
        ParameterPath("embeddings"),
    )
    embeddings_fast = load_tied_embedding(
        config.fast_embeddings_config.empty(config.codebook_size, config.fast_model_dim),
        weights_mapping,
        ParameterPath("fast_embeddings"),
    )

    codebook_embeddings = load_tied_embedding(
        config.codebook_embeddings_config.empty(config.codebook_size * config.num_codebooks, config.slow_model_dim),
        weights_mapping,
        ParameterPath("codebook_embeddings"),
    )

    if config.fast_model_projection_config is not None:
        fast_model_projection = load_linear(
            config.fast_model_projection_config.empty(
                input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
            ),
            weights_mapping,
            ParameterPath("fast_project_in"),
        )
        assert isinstance(fast_model_projection, FullPrecisionLinear)
    else:
        fast_model_projection = Identity()

    return FishAudioTextDecoder(
        config=config,
        embeddings_slow=embeddings_slow,
        transformer_slow=transformer_slow,
        readout_slow=readout_slow,
        embeddings_fast=embeddings_fast,
        transformer_fast=transformer_fast,
        readout_fast=readout_fast,
        codebook_embeddings=codebook_embeddings,
        fast_model_projection=fast_model_projection,
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
        global_rope_config = RoPEConfigCis(precision=precision, base=config.rope_base)
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


# @classmethod from FishAudioTextDecoder
def from_fish_audio_config(
    cls,
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

    def _build_foreign_fish_audio_tts_generator(
        path_to_checkpoints: Path, device: str = "cpu", precision: torch.dtype = torch.bfloat16
    ) -> "TTSGenerator":
        text_decoder_config = FishAudioTextDecoderConfig_Foreign.from_config_file(path_to_checkpoints / "config.json")
        text_decoder: FishAudioTextDecoder_Foreign = text_decoder_config.load_model(
            path_to_checkpoints, device=device, precision=precision
        )
        audio_decoder = load_fish_audio_audio_decoder(path_to_checkpoints)

        tokenizer = FishAudioFromTorch.load_tokenizer_from_fish_audio(str(path_to_checkpoints))

        prompt_template = """
    {% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
    """

        tts_request_factory_config = TTSRequestFactoryConfig(
            prompt_template=prompt_template,
        )

        message_processor = TTSRequestFactory(tts_request_factory_config, tokenizer)

        tts_config = TTSConfig(
            text_decoder.config,
            audio_decoder.config,
            VocoderConfig(),
            activation_precision=DTypeConvert.to_jax(precision),
        )

        audio_renderer_config = AudioRenderingConfig(44100, 1, 16, AudioEncoding.pcm)
        tts_model = TTSModel(
            config=tts_config,
            text_decoder=text_decoder,
            audio_decoder=audio_decoder,
            vocoder=NoopVocoder(tts_config.vocoder_config),
        )
        return FishAudioTTSGenerator(
            config=FishAudioGeneratorConfig(tts_config=tts_config, message_processor_config=message_processor.config),
            tts_model=tts_model,
            message_processor=message_processor,
            audio_renderer=AudioRenderer(audio_renderer_config),
        )

    def _build_lalamo_fish_audio_tts_generator_from_checkpoint(
        path_to_checkpoints: Path, device="cpu", precision: torch.dtype = torch.bfloat16
    ) -> "TTSGenerator":
        path_to_audio_model = path_to_checkpoints / "codec.pth"
        text_decoder = FishAudioModeling.text_decoder_from_foreign_model(path_to_checkpoints, jnp.bfloat16)
        audio_decoder = FishAudioModeling.dac_from_foreign_model(path_to_audio_model, jnp.float32)

        tokenizer = FishAudioFromTorch.load_tokenizer_from_fish_audio(str(path_to_checkpoints))

        prompt_template = """
    {% for message in messages %}<|{{message.style}}|><|{{message.speaker_id}}|>{{message.content}}{% endfor %}
    """

        tts_request_factory_config = TTSRequestFactoryConfig(
            prompt_template=prompt_template,
        )

        message_processor = TTSRequestFactory(tts_request_factory_config, tokenizer)

        tts_config = TTSConfig(
            text_decoder.config,
            audio_decoder.config,
            VocoderConfig(),
            activation_precision=DTypeConvert.to_jax(precision),
        )
        audio_renderer_config = AudioRenderingConfig(44100, 1, 16, AudioEncoding.pcm)
        tts_model = TTSModel(
            config=tts_config,
            text_decoder=text_decoder,
            audio_decoder=audio_decoder,
            vocoder=NoopVocoder(tts_config.vocoder_config),
        )
        return FishAudioTTSGenerator(
            config=FishAudioGeneratorConfig(tts_config=tts_config, message_processor_config=message_processor.config),
            message_processor=message_processor,
            tts_model=tts_model,
            audio_renderer=AudioRenderer(audio_renderer_config),
        )


class FishAudioModeling:
    from fish_speech.models.text2semantic.llama import DualARTransformer

    @staticmethod
    def dac_from_foreign_model(audio_chkpt_path: Path, precision: DTypeLike) -> DescriptAudioCodec:
        from fish_speech.models.dac.inference import load_model
        from fish_speech.models.dac.modded_dac import DAC

        fish_dac = load_model("modded_dac_vq", audio_chkpt_path, device="cpu")
        assert isinstance(fish_dac, DAC)

        dac_weights = dict(
            MapDictValues(
                lambda v: cast_if_float(torch_to_jax(v), precision),
                fish_dac.state_dict(),
            )
        )

        return load_descript_audio_codec(dac_weights)


class TTSLoader:
    def text_decoder_from_foreign_model(
        fish_model_or_path: Path | DualARTransformer, precision: DTypeLike
    ) -> FishAudioTextDecoder:
        from lalamo.model_import.loaders.fishaudio_loaders import load_fishaudio_text_decoder

        return load_fishaudio_text_decoder(fish_model_or_path, precision)

    @staticmethod
    def load_model_from_foreign_model_preset(preset: ForeignTTSModelType, path_to_checkpoints: Path) -> "TTSGenerator":
        match preset:
            case ForeignTTSModelType.FISH_AUDIO:
                return _build_foreign_fish_audio_tts_generator(path_to_checkpoints)
            case ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return _build_lalamo_fish_audio_tts_generator_from_checkpoint(path_to_checkpoints)

    @staticmethod
    def try_locate_audio_model_path(preset: ForeignTTSModelType) -> Path | None:
        match preset:
            case ForeignTTSModelType.FISH_AUDIO | ForeignTTSModelType.FISH_AUDIO_LALAMO:
                return try_locate_fish_audio_model_path()
