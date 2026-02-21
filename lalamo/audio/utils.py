import string

import numpy as np
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel

__all__ = ["dummy_char_level_tokenizer_config", "play_mono_audio"]

DEFAULT_SAMPLERATE: int = 44100


def play_mono_audio(audio: np.ndarray, samplerate: int, audio_chunk_size: int = 1024) -> None:
    import pyaudio  # type: ignore[reportMissingImports], pyaudio is optional to install,

    if audio.dtype not in [np.float32, np.float16, np.float64]:
        raise ValueError("Input audio datatype is expected to be a floating point")
    (n_samples,) = audio.shape
    # better to clip then to overflow
    audio = np.clip(audio, -1.0, 1.0)
    # very dumb conversion to PCM16
    pcm_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

    audio_chunk_size = 1024
    num_chunks = int(np.ceil(n_samples / audio_chunk_size))

    # actual size of each chunk might not be exactly 'audio_chunk_size' but not critical here
    chunks = np.array_split(pcm_audio, num_chunks)

    audio_interface = pyaudio.PyAudio()
    output_stream = audio_interface.open(
        format=audio_interface.get_format_from_width(pcm_audio.dtype.itemsize),
        channels=1,
        rate=samplerate,
        output=True,
    )

    for chunk in chunks:
        output_stream.write(chunk.tobytes())


def dummy_char_level_tokenizer_config() -> str:
    chars = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " ")
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")  # type: ignore[assignment]

    return tokenizer.to_str()
