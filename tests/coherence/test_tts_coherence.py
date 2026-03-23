from pathlib import Path

import numpy as np
from transformers import pipeline

from lalamo.audio.tts_message_processor import TTSMessage
from lalamo.models.tts_model import TTSGenerator
from tests.conftest import ConvertModel

FISHAUDIO_REPO = "fishaudio/s1-mini"

PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("The capital of France is Paris", ("paris",)),
    ("Water is made of hydrogen and oxygen", ("water", "hydrogen")),
)


def test_fishaudio_coherence(convert_model: ConvertModel, tmp_path: Path) -> None:
    converted_path = convert_model(FISHAUDIO_REPO)
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
            failures.append(
                f"Phrase {text!r}: transcription {transcription!r} missing keywords {missing}"
            )

    assert not failures, "\n".join(failures)
