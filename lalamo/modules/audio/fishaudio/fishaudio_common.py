from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from lalamo.sampling import SamplingPolicy

from .fishaudio_consts import _default_audio_codec_config


def default_fishaudio_sampling_policy() -> SamplingPolicy:
    return SamplingPolicy.init(
        temperature=0.8008,
        top_p=0.8008,
        repetition_penalty=1.1016,
    )


@dataclass(frozen=True)
class FishAudioSpecialInferenceTokens:
    semantic_begin_id: int
    semantic_end_id: int
    im_end_token_id: int


def get_default_fishaudio_dac_config() -> Mapping[Any, Any]:
    return deepcopy(_default_audio_codec_config)
