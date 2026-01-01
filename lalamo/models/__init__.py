from .language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from .router import Router, RouterConfig

# TTS support is optional; avoid importing heavy/optional dependencies (e.g. fish_speech)
# when users only need language-model conversion/inference.
try:
    from .tts_model import ForeignTTSModel, TTSConfig, TTSGenerator  # type: ignore
except ModuleNotFoundError as e:  # pragma: no cover
    # Provide lightweight stubs so the CLI can still be imported and used for
    # language-model workflows without installing optional speech dependencies.
    from dataclasses import dataclass
    from enum import Enum
    from pathlib import Path
    from typing import Optional

    _TTS_IMPORT_ERROR: ModuleNotFoundError = e

    class ForeignTTSModel(Enum):
        FISH_AUDIO = "fishaudio"

    @dataclass(frozen=True)
    class TTSConfig:  # type: ignore[no-redef]
        @classmethod
        def load_model_from_foreign_model_preset(cls, preset: ForeignTTSModel, path_to_checkpoints: Path) -> "TTSGenerator":
            raise ModuleNotFoundError(
                "Optional TTS dependencies are not installed. Install the speech extras/dev group to use `lalamo tts`."
            ) from _TTS_IMPORT_ERROR

        @classmethod
        def try_locate_audio_model_path(cls, preset: ForeignTTSModel) -> Optional[Path]:
            raise ModuleNotFoundError(
                "Optional TTS dependencies are not installed. Install the speech extras/dev group to use `lalamo tts`."
            ) from _TTS_IMPORT_ERROR

    class TTSGenerator:  # type: ignore[no-redef]
        pass

__all__ = [
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
    "Router",
    "RouterConfig",
    # Only present when optional TTS deps are installed.
    "ForeignTTSModel",
    "TTSConfig",
    "TTSGenerator",
]
