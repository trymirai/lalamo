from enum import StrEnum
from attr import dataclass

import numpy as np


class AudioEncoding(StrEnum):
    pcm = "PCM"
    ulaw = "uLAW"
    mlaw = "mLAW"


@dataclass
class AudioRenderingConfig:
    samplerate: int
    output_channels: int
    bitrate: int
    encoding: AudioEncoding

    def init(self) -> "AudioRenderer":
        return AudioRenderer(
            config=self,
        )


@dataclass
class AudioRenderer:
    config: AudioRenderingConfig

    def render_to_file(self, audio: np.ndarray) -> None:
        pass

    def play(self, audio: np.ndarray) -> None:
        pass
