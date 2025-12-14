import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import logging

import huggingface_hub
import jax
import torch
from fish_speech.models.text2semantic.inference import generate_long, init_model
from fish_speech.tokenizer import FishTokenizer
from huggingface_hub import HfApi
from jax import numpy as jnp
from pytest import fixture

from lalamo.models import ForeignTTSModel, TTSConfig
from lalamo.modules.audio.tts_request_factory import TTSMessage

testlog = logging.getLogger("tts_test_logger")


@fixture
def fish_audio_local_model_path() -> Path:
    # TODO: (peter.glushkov) replace this one with actual ModelSpec
    fish_audiod_repo_id = "fishaudio/openaudio-s1-mini"

    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

    api = HfApi()
    cache_info = api.model_info(fish_audiod_repo_id)
    commit_hash = cache_info.sha

    return fish_audio_model_info.repo_path / "snapshots" / str(commit_hash)


def get_tts_message() -> TTSMessage:
    test_text = "le text pour le testeax"
    return TTSMessage(content=test_text, speaker_id="angry_trevor", style="catatonic")


def test_fishaudio_text_tokenization(fish_audio_local_model_path: Path) -> None:
    with jax.disable_jit():
        tts_generator = TTSConfig.load_model_from_foreign_model_preset(
            ForeignTTSModel.FISH_AUDIO, fish_audio_local_model_path
        )
        fish_tokenizer = FishTokenizer.from_pretrained(str(fish_audio_local_model_path))

        tts_message = get_tts_message()
        raw_message = tts_generator.message_processor.render_request([tts_message])

        tokens_fish = jnp.asarray(fish_tokenizer.encode(raw_message))
        tokens_hf = tts_generator.tokenize_text([tts_message])

        testlog.debug(f"raw message: {raw_message}")
        testlog.debug(f"Tokenized text HF= {tokens_hf}")
        testlog.debug(f"Tokenized text FISH = {tokens_fish}")

        assert jnp.all(tokens_fish == tokens_hf[0])


def test_fish_audio_text_decoding(fish_audio_local_model_path: Path) -> None:
    device = "cpu"
    with jax.disable_jit():
        tts_generator = TTSConfig.load_model_from_foreign_model_preset(
            ForeignTTSModel.FISH_AUDIO, fish_audio_local_model_path
        )
        tts_message = get_tts_message()

        # Lalamo inference
        tokenized_text = tts_generator.tokenize_text([tts_message])
        semantic_tokens = tts_generator.decode_text(tokenized_text)

        # Fish-audio inference
        raw_message = tts_generator.message_processor.render_request([tts_message])
        precision = torch.bfloat16
        fish_model, decode_one_token = init_model(fish_audio_local_model_path, device, precision)
        fish_generator = generate_long(
            model=fish_model, device=device, decode_one_token=decode_one_token, text=raw_message
        )
        fish_tokens = [token.codes for token in fish_generator if token.action == "sample"]

    testlog.debug(f"Fish tokens: {fish_tokens}")
    testlog.debug(f"Lalamo tokens: {semantic_tokens}")

    # if part.tokens is None:
    #     assert part.text is not None
    #     tokens = tokenizer.encode(part.text)
    # else:
    #     tokens = part.tokens

    # tokens = torch.tensor(tokens, dtype=torch.int)
