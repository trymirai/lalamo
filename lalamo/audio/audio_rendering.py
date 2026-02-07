from enum import StrEnum

from attr import dataclass


class AudioEncoding(StrEnum):
    PCM = "pcm"
    ULAW = "uLAW"
    ALAW = "aLAW"


@dataclass
class AudioRenderingSettings:
    samplerate: int
    output_channels: int
    bitwidth: int
    encoding: AudioEncoding
