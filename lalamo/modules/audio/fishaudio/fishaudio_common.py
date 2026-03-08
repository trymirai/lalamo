from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from lalamo.sampling import SamplingPolicy, make_policy

from .fishaudio_consts import (
    DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE,
    DEFAULT_FISH_AUDIO_SAMPLING_TOP_P,
    _default_audio_codec_config,
)


def default_fishaudio_sampling_policy() -> SamplingPolicy:
    return make_policy(
        temperature=DEFAULT_FISH_AUDIO_SAMPLING_TEMPERATURE,
        top_p=DEFAULT_FISH_AUDIO_SAMPLING_TOP_P,
    )


@dataclass(frozen=True)
class FishAudioSpecialInferenceTokens:
    semantic_begin_id: int
    semantic_end_id: int
    im_end_token_id: int


def get_default_fishaudio_dac_config() -> Mapping[Any, Any]:
    return deepcopy(_default_audio_codec_config)
