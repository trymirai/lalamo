import jax.numpy as jnp

from lalamo.audio.tts_message_processor import TTSMessage
from lalamo.model_import.common import import_model
from lalamo.models import TTSGenerator
from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodec
from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder
from lalamo.modules.audio.text_to_speech import TTSModel


def test_nanocodec_model_spec_loading() -> None:
    """Test end-to-end model loading via NanoCodecForeignConfig (model spec path).

    Exercises the full pipeline: NanoCodecForeignConfig -> TTSConfig -> TTSModel
    with StubTextDecoder + NanoCodec audio decoder, then runs generate_speech.
    """

    message_to_generate = TTSMessage(content="Some noise will be generated here", speaker_id="0", style="unsupported")

    generator, _ = import_model(model_spec="nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps", precision=jnp.float32)

    assert isinstance(generator, TTSGenerator)
    assert isinstance(generator.tts_model, TTSModel)
    assert isinstance(generator.tts_model.text_decoder, StubTextDecoder)
    assert isinstance(generator.tts_model.audio_decoder, NanoCodec)

    generation_result = generator.generate_speech([message_to_generate])
    audio = generation_result.audio

    assert float(jnp.min(audio)) >= -1.0
    assert float(jnp.max(audio)) <= 1.0
