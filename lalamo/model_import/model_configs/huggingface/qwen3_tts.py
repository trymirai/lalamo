import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.common import load_parameters
from lalamo.model_import.loaders.qwen3_tts_loaders import (
    load_qwen3_tts_audio_decoder,
    load_qwen3_tts_text_decoder,
)
from lalamo.model_import.model_configs import ForeignTTSConfig
from lalamo.modules import LalamoModule, TTSConfig, TTSModel
from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder, StubTextDecoderConfig
from lalamo.modules.audio.qwen3_tts.qwen3_tts_audio_decoding import (
    Qwen3TTSAudioDecoder,
    default_qwen3_tts_audio_decoder_config,
)
from lalamo.modules.audio.qwen3_tts.qwen3_tts_text_decoding import (
    Qwen3TTSTextDecoder,
    default_qwen3_tts_text_decoder_config,
)
from lalamo.modules.audio.text_to_speech import VocoderConfig

__all__ = ["Qwen3TTSTokenizer12HzConfig"]


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
    sliding_window: int
    use_sliding_window: bool
    vocab_size: int
    layer_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Qwen3TTSTalkerConfig:
    attention_bias: bool
    attention_dropout: float
    codec_bos_id: int
    codec_eos_token_id: int
    codec_nothing_id: int
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
    sliding_window: int
    text_hidden_size: int
    text_vocab_size: int
    use_sliding_window: bool
    vocab_size: int
    code_predictor_config: Qwen3TTSTalkerCodePredictorConfig
    layer_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Qwen3TTSTokenizer12HzConfig(ForeignTTSConfig):
    decoder_config: Qwen3TTSTokenizer12HzDecoderConfig
    decode_upsample_rate: int
    encode_downsample_rate: int
    encoder_valid_num_quantizers: int
    input_sample_rate: int
    model_type: str
    output_sample_rate: int
    torch_dtype: str

    talker_config: Qwen3TTSTalkerConfig | None = None
    tts_pad_token_id: int | None = None
    tts_bos_token_id: int | None = None
    tts_eos_token_id: int | None = None
    assistant_token_id: int | None = None
    im_start_token_id: int | None = None
    im_end_token_id: int | None = None

    def to_tts_config(
        self,
        context_length: int | None,
        activation_precision: DTypeLike,
        accumulation_precision: DTypeLike,  # noqa: ARG002
    ) -> TTSConfig:
        decoder_config = self.decoder_config
        _validate_decoder_config_assumptions(decoder_config)
        _validate_top_level_config_assumptions(self, decoder_config)

        audio_decoder_config = default_qwen3_tts_audio_decoder_config(
            precision=activation_precision,
            samplerate=self.output_sample_rate,
            decode_upsample_rate=self.decode_upsample_rate,
            codebook_size=decoder_config.codebook_size,
            codebook_dim=decoder_config.codebook_dim,
            hidden_size=decoder_config.hidden_size,
            latent_dim=decoder_config.latent_dim,
            max_position_embeddings=decoder_config.max_position_embeddings,
            rope_theta=decoder_config.rope_theta,
            num_attention_heads=decoder_config.num_attention_heads,
            num_key_value_heads=decoder_config.num_key_value_heads,
            head_dim=decoder_config.head_dim,
            attention_bias=decoder_config.attention_bias,
            sliding_window=decoder_config.sliding_window,
            intermediate_size=decoder_config.intermediate_size,
            layer_scale_initial_scale=decoder_config.layer_scale_initial_scale,
            rms_norm_eps=decoder_config.rms_norm_eps,
            num_hidden_layers=decoder_config.num_hidden_layers,
            num_quantizers=decoder_config.num_quantizers,
            num_semantic_quantizers=decoder_config.num_semantic_quantizers,
            upsample_rates=decoder_config.upsample_rates,
            upsampling_ratios=decoder_config.upsampling_ratios,
            decoder_dim=decoder_config.decoder_dim,
            enable_debug=False,
        )

        if self.talker_config is None:
            text_decoder_config = StubTextDecoderConfig(
                num_codebooks=self.encoder_valid_num_quantizers,
                codebook_size=decoder_config.codebook_size,
                precision=activation_precision,
            )
        else:
            talker = self.talker_config
            predictor = talker.code_predictor_config

            if talker.hidden_act != "silu":
                raise ValueError(f"Only talker hidden_act=silu is supported, got {talker.hidden_act!r}.")
            if predictor.hidden_act != "silu":
                raise ValueError(f"Only predictor hidden_act=silu is supported, got {predictor.hidden_act!r}.")
            if talker.attention_dropout != 0.0:
                raise ValueError(
                    "Talker attention dropout is not implemented in Lalamo;"
                    f" expected 0.0, got {talker.attention_dropout}.",
                )
            if predictor.attention_dropout != 0.0:
                raise ValueError(
                    "Code predictor attention dropout is not implemented in Lalamo;"
                    f" expected 0.0, got {predictor.attention_dropout}.",
                )

            tts_pad_token_id = self.tts_pad_token_id
            tts_bos_token_id = self.tts_bos_token_id
            tts_eos_token_id = self.tts_eos_token_id
            if tts_pad_token_id is None or tts_bos_token_id is None or tts_eos_token_id is None:
                raise ValueError(
                    "Qwen3-TTS talker config requires top-level `tts_pad_token_id`,"
                    " `tts_bos_token_id`, and `tts_eos_token_id`.",
                )

            text_decoder_config = default_qwen3_tts_text_decoder_config(
                precision=activation_precision,
                talker_vocab_size=talker.vocab_size,
                text_vocab_size=talker.text_vocab_size,
                talker_hidden_size=talker.hidden_size,
                text_hidden_size=talker.text_hidden_size,
                talker_intermediate_size=talker.intermediate_size,
                talker_num_hidden_layers=talker.num_hidden_layers,
                talker_num_attention_heads=talker.num_attention_heads,
                talker_num_key_value_heads=talker.num_key_value_heads,
                talker_head_dim=talker.head_dim,
                talker_max_position_embeddings=(
                    min(context_length, talker.max_position_embeddings)
                    if context_length
                    else talker.max_position_embeddings
                ),
                talker_rope_theta=talker.rope_theta,
                talker_rms_norm_eps=talker.rms_norm_eps,
                talker_attention_bias=talker.attention_bias,
                talker_sliding_window_sizes=_build_sliding_window_sizes(
                    num_hidden_layers=talker.num_hidden_layers,
                    use_sliding_window=talker.use_sliding_window,
                    sliding_window=talker.sliding_window,
                    layer_types=talker.layer_types,
                ),
                predictor_hidden_size=predictor.hidden_size,
                predictor_intermediate_size=predictor.intermediate_size,
                predictor_num_hidden_layers=predictor.num_hidden_layers,
                predictor_num_attention_heads=predictor.num_attention_heads,
                predictor_num_key_value_heads=predictor.num_key_value_heads,
                predictor_head_dim=predictor.head_dim,
                predictor_max_position_embeddings=(
                    min(context_length, predictor.max_position_embeddings)
                    if context_length
                    else predictor.max_position_embeddings
                ),
                predictor_rope_theta=predictor.rope_theta,
                predictor_rms_norm_eps=predictor.rms_norm_eps,
                predictor_attention_bias=predictor.attention_bias,
                predictor_sliding_window_sizes=_build_sliding_window_sizes(
                    num_hidden_layers=predictor.num_hidden_layers,
                    use_sliding_window=predictor.use_sliding_window,
                    sliding_window=predictor.sliding_window,
                    max_window_layers=predictor.max_window_layers,
                    layer_types=predictor.layer_types,
                ),
                predictor_vocab_size=predictor.vocab_size,
                num_code_groups=talker.num_code_groups,
                max_new_tokens=(
                    min(context_length, talker.max_position_embeddings)
                    if context_length
                    else talker.max_position_embeddings
                ),
                codec_bos_id=talker.codec_bos_id,
                codec_eos_token_id=talker.codec_eos_token_id,
                codec_pad_id=talker.codec_pad_id,
                codec_think_id=talker.codec_think_id,
                codec_nothing_id=talker.codec_nothing_id,
                codec_think_bos_id=talker.codec_think_bos_id,
                codec_think_eos_id=talker.codec_think_eos_id,
                tts_bos_token_id=tts_bos_token_id,
                tts_eos_token_id=tts_eos_token_id,
                tts_pad_token_id=tts_pad_token_id,
                im_start_token_id=self.im_start_token_id,
                assistant_token_id=self.assistant_token_id,
                im_end_token_id=self.im_end_token_id,
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

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        json_path = Path(json_path)
        with open(json_path) as f:
            raw_config = json.load(f)
        assert isinstance(raw_config, Mapping)
        raw_config = _merge_with_speech_tokenizer_config(raw_config, json_path)

        decoder_raw = _extract_decoder_config(raw_config)
        decoder_config = _structure_decoder_config(decoder_raw)

        decode_upsample_rate = int(
            raw_config.get(
                "decode_upsample_rate",
                math.prod((*decoder_config.upsample_rates, *decoder_config.upsampling_ratios)),
            ),
        )
        encode_downsample_rate = int(raw_config.get("encode_downsample_rate", decode_upsample_rate))
        encoder_valid_num_quantizers = int(
            raw_config.get("encoder_valid_num_quantizers", decoder_config.num_quantizers),
        )
        input_sample_rate = int(raw_config.get("input_sample_rate", 24000))
        model_type = str(raw_config.get("model_type", "qwen3_tts"))

        if "output_sample_rate" in raw_config:
            output_sample_rate = int(raw_config["output_sample_rate"])
        else:
            speaker_encoder_config = raw_config.get("speaker_encoder_config")
            if isinstance(speaker_encoder_config, Mapping) and isinstance(
                speaker_encoder_config.get("sample_rate"), int
            ):
                output_sample_rate = int(speaker_encoder_config["sample_rate"])
            else:
                output_sample_rate = 24000

        torch_dtype = _resolve_torch_dtype(raw_config)
        talker_config = _extract_talker_config(raw_config)

        tts_pad_token_id_raw = raw_config.get("tts_pad_token_id")
        tts_bos_token_id_raw = raw_config.get("tts_bos_token_id")
        tts_eos_token_id_raw = raw_config.get("tts_eos_token_id")
        assistant_token_id_raw = raw_config.get("assistant_token_id")
        im_start_token_id_raw = raw_config.get("im_start_token_id")
        im_end_token_id_raw = raw_config.get("im_end_token_id")

        return cls(
            decoder_config=decoder_config,
            decode_upsample_rate=decode_upsample_rate,
            encode_downsample_rate=encode_downsample_rate,
            encoder_valid_num_quantizers=encoder_valid_num_quantizers,
            input_sample_rate=input_sample_rate,
            model_type=model_type,
            output_sample_rate=output_sample_rate,
            torch_dtype=torch_dtype,
            talker_config=talker_config,
            tts_pad_token_id=int(tts_pad_token_id_raw) if isinstance(tts_pad_token_id_raw, int) else None,
            tts_bos_token_id=int(tts_bos_token_id_raw) if isinstance(tts_bos_token_id_raw, int) else None,
            tts_eos_token_id=int(tts_eos_token_id_raw) if isinstance(tts_eos_token_id_raw, int) else None,
            assistant_token_id=int(assistant_token_id_raw) if isinstance(assistant_token_id_raw, int) else None,
            im_start_token_id=int(im_start_token_id_raw) if isinstance(im_start_token_id_raw, int) else None,
            im_end_token_id=int(im_end_token_id_raw) if isinstance(im_end_token_id_raw, int) else None,
        )

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(self.torch_dtype)


def _merge_with_speech_tokenizer_config(raw_config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    speech_tokenizer_config_path = config_path.parent / "speech_tokenizer" / "config.json"
    if not speech_tokenizer_config_path.exists():
        return dict(raw_config)

    with open(speech_tokenizer_config_path) as speech_tokenizer_file:
        speech_tokenizer_config = json.load(speech_tokenizer_file)

    if not isinstance(speech_tokenizer_config, Mapping):
        return dict(raw_config)

    # The main config may contain talker/text fields while decoder fields are in speech_tokenizer/config.json.
    merged_config: dict[str, Any] = dict(speech_tokenizer_config)
    merged_config.update(raw_config)
    return merged_config


def _extract_decoder_config(raw_config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    decoder_config = raw_config.get("decoder_config")
    if isinstance(decoder_config, Mapping):
        return decoder_config

    minimal_keys = {
        "codebook_size",
        "hidden_size",
        "latent_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "num_quantizers",
        "decoder_dim",
    }
    if minimal_keys.issubset(raw_config.keys()):
        return raw_config

    return None


def _structure_decoder_config(raw_decoder_config: Mapping[str, Any] | None) -> Qwen3TTSTokenizer12HzDecoderConfig:
    if raw_decoder_config is None:
        return _default_decoder_config()

    field_names = {field.name for field in fields(Qwen3TTSTokenizer12HzDecoderConfig)}
    filtered: dict[str, Any] = {key: value for key, value in raw_decoder_config.items() if key in field_names}

    if "upsample_rates" in filtered:
        filtered["upsample_rates"] = tuple(int(rate) for rate in filtered["upsample_rates"])
    if "upsampling_ratios" in filtered:
        filtered["upsampling_ratios"] = tuple(int(rate) for rate in filtered["upsampling_ratios"])

    defaults = _default_decoder_config()
    for field in fields(Qwen3TTSTokenizer12HzDecoderConfig):
        filtered.setdefault(field.name, getattr(defaults, field.name))

    return Qwen3TTSTokenizer12HzDecoderConfig(**filtered)


def _default_decoder_config() -> Qwen3TTSTokenizer12HzDecoderConfig:
    return Qwen3TTSTokenizer12HzDecoderConfig(
        attention_dropout=0.0,
        attention_bias=False,
        codebook_dim=2048,
        codebook_size=2048,
        decoder_dim=1536,
        head_dim=64,
        hidden_act="silu",
        hidden_size=1024,
        intermediate_size=3072,
        latent_dim=1024,
        layer_scale_initial_scale=0.01,
        max_position_embeddings=8000,
        num_attention_heads=16,
        num_hidden_layers=8,
        num_key_value_heads=16,
        num_quantizers=16,
        num_semantic_quantizers=1,
        rms_norm_eps=1e-5,
        semantic_codebook_size=1024,
        rope_theta=10000.0,
        sliding_window=72,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        vector_quantization_hidden_dimension=1024,
    )


def _extract_talker_config(raw_config: Mapping[str, Any]) -> Qwen3TTSTalkerConfig | None:
    talker_raw = raw_config.get("talker_config")
    if not isinstance(talker_raw, Mapping):
        return None
    return _structure_talker_config(talker_raw)


def _structure_talker_config(raw_talker_config: Mapping[str, Any]) -> Qwen3TTSTalkerConfig:
    predictor_raw = raw_talker_config.get("code_predictor_config")
    if isinstance(predictor_raw, Mapping):
        predictor_config = _structure_predictor_config(predictor_raw)
    else:
        predictor_config = _default_predictor_config()

    field_names = {field.name for field in fields(Qwen3TTSTalkerConfig)}
    filtered = {key: value for key, value in raw_talker_config.items() if key in field_names}
    filtered["code_predictor_config"] = predictor_config

    defaults = _default_talker_config(predictor_config)
    for field in fields(Qwen3TTSTalkerConfig):
        filtered.setdefault(field.name, getattr(defaults, field.name))

    layer_types = filtered.get("layer_types")
    if isinstance(layer_types, Sequence):
        filtered["layer_types"] = tuple(str(value) for value in layer_types)
    else:
        filtered["layer_types"] = None

    return Qwen3TTSTalkerConfig(**filtered)


def _structure_predictor_config(raw_predictor_config: Mapping[str, Any]) -> Qwen3TTSTalkerCodePredictorConfig:
    field_names = {field.name for field in fields(Qwen3TTSTalkerCodePredictorConfig)}
    filtered = {key: value for key, value in raw_predictor_config.items() if key in field_names}

    defaults = _default_predictor_config()
    for field in fields(Qwen3TTSTalkerCodePredictorConfig):
        filtered.setdefault(field.name, getattr(defaults, field.name))

    layer_types = filtered.get("layer_types")
    if isinstance(layer_types, Sequence):
        filtered["layer_types"] = tuple(str(value) for value in layer_types)
    else:
        filtered["layer_types"] = None

    return Qwen3TTSTalkerCodePredictorConfig(**filtered)


def _default_predictor_config() -> Qwen3TTSTalkerCodePredictorConfig:
    return Qwen3TTSTalkerCodePredictorConfig(
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=128,
        hidden_act="silu",
        hidden_size=1024,
        intermediate_size=3072,
        max_position_embeddings=32768,
        max_window_layers=28,
        num_attention_heads=16,
        num_code_groups=32,
        num_hidden_layers=5,
        num_key_value_heads=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=4096,
        use_sliding_window=False,
        vocab_size=2048,
        layer_types=None,
    )


def _default_talker_config(
    predictor_config: Qwen3TTSTalkerCodePredictorConfig,
) -> Qwen3TTSTalkerConfig:
    return Qwen3TTSTalkerConfig(
        attention_bias=False,
        attention_dropout=0.0,
        codec_bos_id=4197,
        codec_eos_token_id=4198,
        codec_nothing_id=4203,
        codec_pad_id=4196,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_think_id=4202,
        head_dim=64,
        hidden_act="silu",
        hidden_size=1024,
        intermediate_size=2048,
        max_position_embeddings=32768,
        num_attention_heads=16,
        num_code_groups=32,
        num_hidden_layers=20,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        sliding_window=4096,
        text_hidden_size=2048,
        text_vocab_size=151936,
        use_sliding_window=False,
        vocab_size=3072,
        code_predictor_config=predictor_config,
        layer_types=None,
    )


def _build_sliding_window_sizes(
    *,
    num_hidden_layers: int,
    use_sliding_window: bool,
    sliding_window: int,
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
    path_candidates = (
        ParameterPath("decoder"),
        ParameterPath("speech_tokenizer.decoder"),
        ParameterPath("tokenizer.decoder"),
        ParameterPath("model.decoder"),
        ParameterPath(""),
    )
    for candidate in path_candidates:
        required_key = candidate / "pre_transformer" / "input_proj" / "weight"
        if required_key in weights_dict:
            return candidate

    available_keys = sorted(weights_dict)[:8]
    raise ValueError(
        f"Could not infer Qwen3-TTS decoder path from checkpoint. First keys: {available_keys}",
    )


def _detect_talker_path(weights_dict: Mapping[str, Array]) -> ParameterPath:
    path_candidates = (
        ParameterPath(""),
        ParameterPath("model"),
        ParameterPath("tts_model"),
    )
    for candidate in path_candidates:
        required_key = candidate / "talker" / "model" / "codec_embedding" / "weight"
        if required_key in weights_dict:
            return candidate

    available_keys = sorted(weights_dict)[:8]
    raise ValueError(
        f"Could not infer Qwen3-TTS talker path from checkpoint. First keys: {available_keys}",
    )


def _resolve_torch_dtype(raw_config: Mapping[str, Any]) -> str:
    torch_dtype = raw_config.get("torch_dtype")
    if isinstance(torch_dtype, str):
        return torch_dtype

    encoder_config = raw_config.get("encoder_config")
    if isinstance(encoder_config, Mapping):
        encoder_dtype = encoder_config.get("dtype")
        if isinstance(encoder_dtype, str):
            return encoder_dtype

    raise ValueError(
        "Qwen3-TTS config is missing dtype metadata; expected `torch_dtype` or `encoder_config.dtype`.",
    )


def _validate_decoder_config_assumptions(decoder_config: Qwen3TTSTokenizer12HzDecoderConfig) -> None:
    if decoder_config.hidden_act != "silu":
        raise ValueError(
            f"Only `hidden_act=silu` is supported for Qwen3-TTS decoder, got {decoder_config.hidden_act!r}.",
        )

    if decoder_config.attention_dropout != 0.0:
        raise ValueError(
            "Attention dropout is not implemented in Lalamo Qwen3-TTS decoder;"
            f" expected 0.0, got {decoder_config.attention_dropout}.",
        )


def _validate_top_level_config_assumptions(
    config: Qwen3TTSTokenizer12HzConfig,
    decoder_config: Qwen3TTSTokenizer12HzDecoderConfig,
) -> None:
    if config.encoder_valid_num_quantizers != decoder_config.num_quantizers:
        raise ValueError(
            "Mismatch between top-level and decoder quantizer counts:"
            f" encoder_valid_num_quantizers={config.encoder_valid_num_quantizers},"
            f" decoder.num_quantizers={decoder_config.num_quantizers}.",
        )

    upsample_rate = math.prod((*decoder_config.upsample_rates, *decoder_config.upsampling_ratios))
    if config.decode_upsample_rate != upsample_rate:
        raise ValueError(
            "decode_upsample_rate does not match decoder upsampling factors:"
            f" decode_upsample_rate={config.decode_upsample_rate},"
            f" product(upsample_rates, upsampling_ratios)={upsample_rate}.",
        )
