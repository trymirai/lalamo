from functools import lru_cache
from types import ModuleType

from lalamo.utils import setup_custom_logger

audio_logger = setup_custom_logger("lalamo-audio")


@lru_cache(maxsize=1)
def get_pyaudio() -> ModuleType | None:
    try:
        import pyaudio  # pyright: ignore[reportMissingModuleSource], pyaudio is optional to install
    except ImportError as e:
        msg = f"Failed to import 'pyaudio' package: {e}."
        audio_logger.warning(msg)
        return None

    return pyaudio
