import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import soundfile as sf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.audio.neutts import (
    build_neutts_prompt_text,
    load_voice_prompt,
    parse_neutts_speech_tokens,
    speech_tokens_to_codebook_codes,
)
from lalamo.audio.tts_message_processor import TTSMessageProcessor, TTSMessageProcessorConfig, VoicePrompt
from lalamo.model_import.model_configs.huggingface.neutts import HFNeuTTSConfig
from lalamo.model_import.model_specs.common import ModelType
from lalamo.model_registry import ModelRegistry
from lalamo.models.tts_model import NeuTTSGenerator, TTSGeneratorConfig
from lalamo.modules.audio.neutts.audio_decoding import NeuCodecAudioDecoder, NeuCodecAudioDecoderConfig
from lalamo.modules.audio.neutts.text_decoding import NeuTTSTextDecoderConfig
from lalamo.modules.rope import LinearScalingRoPEConfig


def _write_neutts_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 128000,
                "dtype": "float32",
                "eos_token_id": 128261,
                "head_dim": 64,
                "hidden_act": "silu",
                "hidden_size": 576,
                "initializer_range": 0.02,
                "intermediate_size": 2304,
                "max_position_embeddings": 2048,
                "mlp_bias": False,
                "model_type": "llama",
                "num_attention_heads": 9,
                "num_hidden_layers": 24,
                "num_key_value_heads": 3,
                "pad_token_id": 128001,
                "pretraining_tp": 1,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                    "factor": 32.0,
                    "rope_type": "linear",
                    "type": "linear",
                },
                "rope_theta": 500000,
                "tie_word_embeddings": True,
                "transformers_version": "4.57.6",
                "use_cache": True,
                "vocab_size": 194256,
            },
        ),
    )


def test_build_neutts_prompt_text_includes_reference_text_input_text_and_reference_codes() -> None:
    prompt = build_neutts_prompt_text(
        input_phones="HH AH L OW",
        reference_phones="JH OW",
        reference_codes=(12, 34),
    )

    assert prompt == (
        "user: Convert the text to speech:<|TEXT_PROMPT_START|>JH OW HH AH L OW"
        "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|><|speech_12|><|speech_34|>"
    )


def test_parse_neutts_speech_tokens() -> None:
    assert parse_neutts_speech_tokens("<|speech_12|>ignored<|speech_34|>") == (12, 34)


def test_speech_tokens_to_codebook_codes_returns_single_semantic_codebook() -> None:
    codes = speech_tokens_to_codebook_codes((12, 34))

    assert codes.acoustic is None
    assert codes.semantic.shape == (1, 1, 2)
    assert codes.semantic.tolist() == [[[12, 34]]]


def test_neucodec_normalize_codes_rejects_multiple_batch_items() -> None:
    decoder = NeuCodecAudioDecoder(config=NeuCodecAudioDecoderConfig(precision=jnp.float32))

    with pytest.raises(ValueError, match="single batch item"):
        decoder._normalize_codes(jnp.zeros((2, 1, 3), dtype=jnp.int32))  # noqa: SLF001


@pytest.mark.xfail(reason="Runtime NeuCodec bridge still exists until native decoder replacement lands.", strict=True)
def test_neucodec_runtime_decode_bridge_is_absent() -> None:
    source = Path("lalamo/modules/audio/neutts/audio_decoding.py").read_text()

    assert "from neucodec import" not in source
    assert "_load_neucodec" not in source
    assert "codec.decode_code" not in source


def test_parse_neutts_speech_tokens_requires_at_least_one_token() -> None:
    with pytest.raises(ValueError, match="No valid speech tokens"):
        parse_neutts_speech_tokens("no speech here")


def test_tts_cli_reference_options_load_voice_prompt(tmp_path: Path) -> None:
    reference_audio_path = tmp_path / "reference.wav"
    reference_audio = np.asarray([0.0, 0.25, -0.25], dtype=np.float32)
    sf.write(reference_audio_path, reference_audio, 16_000)

    assert load_voice_prompt(None, None) is None
    voice_prompt = load_voice_prompt(reference_audio_path, "hello")

    assert voice_prompt is not None
    assert voice_prompt.reference_samplerate == 16_000
    np.testing.assert_allclose(voice_prompt.reference_audio, reference_audio, atol=1e-4)
    assert voice_prompt.reference_text == "hello"
    assert isinstance(voice_prompt, VoicePrompt)

    with pytest.raises(ValueError, match="--ref-audio and --ref-text must be provided together"):
        load_voice_prompt(reference_audio_path, None)

    with pytest.raises(ValueError, match="--ref-audio and --ref-text must be provided together"):
        load_voice_prompt(None, "hello")


def test_neutts_config_delegates_backbone_to_llama_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_neutts_config(config_path)

    config = HFNeuTTSConfig.from_json(config_path)
    tts_config = config.to_tts_config(
        context_length=None,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )

    assert config.default_precision == jnp.dtype("float32")
    assert isinstance(tts_config.text_decoder_config, NeuTTSTextDecoderConfig)
    assert isinstance(tts_config.audio_decoder_config, NeuCodecAudioDecoderConfig)

    rope_config = tts_config.text_decoder_config.decoder_config.transformer_config.layer_configs[0].rope_config
    assert isinstance(rope_config, LinearScalingRoPEConfig)
    assert rope_config.scaling_factor == 32.0


def test_tts_model_import_weights_keeps_weightless_audio_decoder_when_weights_are_absent(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_neutts_config(config_path)
    config = HFNeuTTSConfig.from_json(config_path)
    tts_config = config.to_tts_config(
        context_length=None,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    model = tts_config.empty()

    loaded_model = model.import_weights({"text_decoder": model.text_decoder.export_weights()})

    assert loaded_model.audio_decoder is model.audio_decoder


def test_neutts_nano_is_registered_as_first_class_tts_model() -> None:
    spec = ModelRegistry.build(allow_third_party_plugins=False).repo_to_model["neuphonic/neutts-nano"]

    assert spec.model_type is ModelType.TTS_MODEL
    assert spec.config_type is HFNeuTTSConfig
    assert spec.family == "NeuTTS-Nano"
    assert spec.name == "NeuTTS-Nano"


def test_neutts_generator_config_initializes_neutts_generator(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_neutts_config(config_path)
    config = HFNeuTTSConfig.from_json(config_path)
    tts_config = config.to_tts_config(
        context_length=None,
        activation_precision=jnp.float32,
        accumulation_precision=jnp.float32,
    )
    message_processor_config = TTSMessageProcessorConfig(prompt_template="{{ messages[0].content }}")
    message_processor = TTSMessageProcessor(
        message_processor_config,
        Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")),
    )
    generator_config = TTSGeneratorConfig(
        tts_config=tts_config,
        message_processor_config=message_processor_config,
    )

    generator = generator_config.init_generator(tts_config.empty(), message_processor)

    assert isinstance(generator, NeuTTSGenerator)
