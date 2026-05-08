import os
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Protocol, cast

from lalamo.audio.tts_message_processor import TTSMessage, VoicePrompt

SPEECH_TOKEN_PATTERN = re.compile(r"<\|speech_(\d+)\|>")


class _PhonemizerBackend(Protocol):
    def phonemize(self, text: list[str]) -> list[str]: ...


class _EspeakWrapper(Protocol):
    @staticmethod
    def set_library(path: str) -> None: ...


def speech_codes_to_text(speech_codes: Iterable[int]) -> str:
    return "".join(f"<|speech_{speech_code}|>" for speech_code in speech_codes)


def build_neutts_prompt_text(
    *,
    input_phones: str,
    reference_phones: str,
    reference_codes: Iterable[int],
) -> str:
    input_phones = input_phones.strip()
    reference_phones = reference_phones.strip()
    if not input_phones:
        raise ValueError("NeuTTS input text must not be empty after phonemization.")
    if not reference_phones:
        raise ValueError("NeuTTS reference text must not be empty after phonemization.")

    reference_codes_text = speech_codes_to_text(reference_codes)
    if not reference_codes_text:
        raise ValueError("NeuTTS reference audio did not produce any codec tokens.")

    return (
        "user: Convert the text to speech:"
        f"<|TEXT_PROMPT_START|>{reference_phones} {input_phones}<|TEXT_PROMPT_END|>\n"
        f"assistant:<|SPEECH_GENERATION_START|>{reference_codes_text}"
    )


def parse_neutts_speech_tokens(text: str) -> tuple[int, ...]:
    speech_tokens = tuple(int(match.group(1)) for match in SPEECH_TOKEN_PATTERN.finditer(text))
    if not speech_tokens:
        raise ValueError("No valid speech tokens found in the output.")
    return speech_tokens


def require_neutts_message(messages: Iterable[TTSMessage]) -> TTSMessage:
    message_tuple = tuple(messages)
    if len(message_tuple) != 1:
        raise ValueError("NeuTTS currently supports exactly one TTS message per request.")

    (message,) = message_tuple
    if not message.content.strip():
        raise ValueError("NeuTTS input text must not be empty.")

    voice_prompt = message.voice_prompt
    if voice_prompt is None:
        raise ValueError("NeuTTS requires a voice prompt with reference audio and reference text.")
    if not voice_prompt.reference_text.strip():
        raise ValueError("NeuTTS reference text must not be empty.")

    return message


def _configure_bundled_espeak() -> None:
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return

    wrapper = cast("_EspeakWrapper", EspeakWrapper)
    wrapper.set_library(str(espeakng_loader.get_library_path()))
    os.environ["ESPEAK_DATA_PATH"] = str(espeakng_loader.get_data_path())


@cache
def _get_espeak_backend(language_code: str) -> _PhonemizerBackend:
    try:
        from phonemizer.backend import EspeakBackend
    except ImportError as e:
        raise ImportError(
            "NeuTTS phonemization requires the optional phonemizer dependency. Install lalamo with the neutts extra.",
        ) from e

    try:
        _configure_bundled_espeak()
        return cast(
            "_PhonemizerBackend",
            EspeakBackend(
                language=language_code,
                preserve_punctuation=True,
                with_stress=True,
                words_mismatch="ignore",
                language_switch="remove-flags",
            ),
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize the eSpeak phonemizer backend required by NeuTTS. "
            "Install eSpeak NG or use the lalamo NeuTTS extra on a platform with eSpeak available.",
        ) from e


def phonemize_neutts_text(text: str, *, language_code: str) -> str:
    backend = _get_espeak_backend(language_code)
    phonemes = backend.phonemize([text])[0]
    return " ".join(phonemes.split())


def build_voice_prompt(reference_audio_path: Path | str | None, reference_text: str | None) -> VoicePrompt | None:
    if reference_audio_path is None and reference_text is None:
        return None
    if reference_audio_path is None or reference_text is None:
        raise ValueError("--ref-audio and --ref-text must be provided together.")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("--ref-text must not be empty.")
    return VoicePrompt(reference_audio_path=reference_audio_path, reference_text=reference_text)
