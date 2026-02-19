from collections.abc import Mapping
from typing import TYPE_CHECKING

from jaxtyping import Array

from lalamo.common import ParameterPath

# from .executorch import load_executorch
from .huggingface import load_huggingface_classifier, load_huggingface_decoder

if TYPE_CHECKING:
    from lalamo.modules.audio.fishaudio import DescriptAudioCodec
    from lalamo.modules.audio.qwen3_tts.qwen3_tts_audio_decoding import Qwen3TTSAudioDecoder


def load_audio_decoder(
    module: "DescriptAudioCodec | Qwen3TTSAudioDecoder",
    weights_dict: Mapping[str, Array],
    base_path: ParameterPath,
) -> "DescriptAudioCodec | Qwen3TTSAudioDecoder":
    from lalamo.modules.audio.fishaudio import DescriptAudioCodec
    from lalamo.modules.audio.qwen3_tts.qwen3_tts_audio_decoding import Qwen3TTSAudioDecoder

    from .fishaudio_loaders import load_fishaudio_audio_decoder
    from .qwen3_tts_loaders import load_qwen3_tts_audio_decoder

    if isinstance(module, DescriptAudioCodec):
        return load_fishaudio_audio_decoder(module, weights_dict, base_path)
    if isinstance(module, Qwen3TTSAudioDecoder):
        return load_qwen3_tts_audio_decoder(module, weights_dict, base_path)
    raise TypeError(f"Unsupported audio decoder module type: {type(module)!r}")


__all__ = [
    "load_huggingface_classifier",
    # "load_executorch",
    "load_huggingface_decoder",
    "load_audio_decoder",
]
