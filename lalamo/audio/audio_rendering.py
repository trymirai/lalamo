from enum import StrEnum

import numpy as np
from attr import dataclass


class AudioEncoding(StrEnum):
    pcm = "PCM"
    ulaw = "uLAW"
    mlaw = "mLAW"


@dataclass
class AudioRenderingConfig:
    samplerate: int
    output_channels: int
    bitwidth: int
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

    def condition_signal(
        self,
        generated_audio: np.ndarray,
        generated_audio_properties: AudioRenderingConfig,  # noqa: ARG002
    ) -> np.ndarray:
        return generated_audio
