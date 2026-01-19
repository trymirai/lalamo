import numpy as np

from lalamo.audio import audio_logger, get_pyaudio

__all__ = ["play_mono_audio"]

DEFAULT_SAMPLERATE: int = 44100


def play_mono_audio(audio: np.ndarray, samplerate: int, audio_chunk_size: int = 1024) -> None:
    pyaudio = get_pyaudio()
    if pyaudio is None:
        audio_logger.warning("'pyaudio' package is not imported, unable to replay audio.")
        return

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
