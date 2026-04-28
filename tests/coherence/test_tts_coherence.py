import numpy as np
import pytest
from transformers import pipeline

from lalamo.audio.tts_codec import TTSMessage
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from lalamo.models.tts_model import TTSGenerator
from lalamo.module import Keychain
from tests.conftest import ConvertModel, filter_specs
from tests.model_test_tiers import COHERENCE_TTS_REPOS

PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The capital of France is Paris", ("paris",)),
    ("Water is made of hydrogen and oxygen", ("water", "hydrogen")),
)

coherence_tts_specs = filter_specs(model_type=ModelType.TTS_MODEL, repos=frozenset(COHERENCE_TTS_REPOS))


@pytest.mark.parametrize("spec", coherence_tts_specs, ids=[s.repo for s in coherence_tts_specs])
def test_tts_coherence(spec: ModelSpec, convert_model: ConvertModel) -> None:
    converted_path = convert_model(spec.repo)
    model = TTSGenerator.load_model(converted_path)

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

    failures: list[str] = []
    keychain = Keychain.init(0)
    for text, expected_keywords in PHRASES:
        message = TTSMessage(content=text, speaker_id="speaker:0", style="interleave")
        keychain, phrase_keychain = keychain.split()
        result = model.generate_speech(
            [message],
            keychain=phrase_keychain,
        )

        audio = np.squeeze(result.audio).astype(np.float32)
        transcription: str = asr({"sampling_rate": result.audio_params.samplerate, "raw": audio})["text"]

        missing = [kw for kw in expected_keywords if kw not in transcription.lower()]
        if missing:
            failures.append(f"Phrase {text!r}: transcription {transcription!r} missing keywords {missing}")

    assert not failures, "\n".join(failures)
