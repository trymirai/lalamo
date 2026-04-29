from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .fishaudio_consts import (
    _default_audio_codec_config,
)


@dataclass(frozen=True)
class FishAudioSpecialInferenceTokens:
    semantic_begin_id: int
    semantic_end_id: int
    im_end_token_id: int


def get_default_fishaudio_dac_config() -> Mapping[Any, Any]:
    return deepcopy(_default_audio_codec_config)
