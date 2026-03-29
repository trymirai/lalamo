from pathlib import Path

import numpy as np
import pytest
from transformers import pipeline

from lalamo.audio.tts_message_processor import TTSMessage
from lalamo.model_import.model_specs.common import ModelSpec, ModelType
from lalamo.models.tts_model import TTSGenerator
from tests.conftest import ConvertModel, filter_specs
from tests.model_test_tiers import COHERENCE_TTS_REPOS

PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The capital of France is Paris", ("paris",)),
    ("Water is made of hydrogen and oxygen", ("water", "hydrogen")),
)

coherence_tts_specs = filter_specs(model_type=ModelType.TTS_MODEL, repos=frozenset(COHERENCE_TTS_REPOS))


@pytest.mark.parametrize("spec", coherence_tts_specs, ids=[s.repo for s in coherence_tts_specs])
def test_tts_coherence(spec: ModelSpec, convert_model: ConvertModel, tmp_path: Path) -> None:
    converted_path = convert_model(spec.repo)
    model = TTSGenerator.load_model(converted_path)

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

    failures: list[str] = []
    for text, expected_keywords in PHRASES:
        message = TTSMessage(content=text, speaker_id="speaker:0", style="interleave")
        result = model.generate_speech([message])

        audio = np.squeeze(result.audio).astype(np.float32)
        transcription: str = asr({"sampling_rate": result.audio_params.samplerate, "raw": audio})["text"]

        missing = [kw for kw in expected_keywords if kw not in transcription.lower()]
        if missing:
            failures.append(f"Phrase {text!r}: transcription {transcription!r} missing keywords {missing}")

    assert not failures, "\n".join(failures)
