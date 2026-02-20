from dataclasses import dataclass
from enum import StrEnum


class AudioEncoding(StrEnum):
    PCM = "pcm"
    ULAW = "uLAW"
    ALAW = "aLAW"


@dataclass(frozen=True)
class AudioRenderingSettings:
    samplerate: int
    output_channels: int
    bitwidth: int
    encoding: AudioEncoding
