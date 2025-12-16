import os
from pathlib import Path
import sys

# TODO(peter.glushkov): dont forget to remove this after debuggin done
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

import logging

import huggingface_hub
import jax
import numpy as np
import pytest
import torch
from fish_speech.models.text2semantic import inference as fish_inference
from fish_speech.tokenizer import FishTokenizer
from huggingface_hub import HfApi
from jax import numpy as jnp
from pytest import fixture

from lalamo.models import ForeignTTSModel, TTSConfig
from lalamo.modules.audio.foreign.fish_audio import (
    FishAudioSamplingParams,
    FishAudioTextDecoder,
    FishAudioTextDecoderConfig,
    FishAudioTextDecoderResult,
    logits_to_probs,
    sample,
)
from lalamo.modules.audio.foreign.fish_audio_thin_wrapper import FishAudioTextDecoder_Foreign
from lalamo.modules.audio.tts_request_factory import TTSMessage
from lalamo.modules.torch_interop import jax_to_torch, torch_to_jax

_testlog = logging.getLogger("tts_test_logger")


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
    test_text = "text"
    return TTSMessage(content=test_text, speaker_id="speaker:0", style="interleave")


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

        _testlog.debug(f"raw message: {raw_message}")
        _testlog.debug(f"Tokenized text HF= {tokens_hf}")
        _testlog.debug(f"Tokenized text FISH = {tokens_fish}")

        assert jnp.all(tokens_fish == tokens_hf[0])


@torch.no_grad
def test_decode_one_token(fish_audio_local_model_path: Path) -> None:
    tts_message = get_tts_message()

    tts_generator = TTSConfig.load_model_from_foreign_model_preset(
        ForeignTTSModel.FISH_AUDIO, fish_audio_local_model_path
    )
    assert isinstance(tts_generator.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = tts_generator.text_decoder.fish_model

    sampling_params = FishAudioSamplingParams(
        argmax_decoding=True, top_p=0.808, temperature=0.808, repetition_penalty=1.1016
    )

    # -- preparing inputs for lalamo
    tokenized_text_lalamo = jnp.array(tts_generator.message_processor.tokenize_request([tts_message]))[None, :]
    n_tokens = tokenized_text_lalamo.shape[-1]
    input_pos = jnp.arange(n_tokens)[None, :]
    output_fish = tts_generator.text_decoder(tokenized_text_lalamo, sampling_params=sampling_params)

    # -- lalamo model setup and inference
    lalamo_model: FishAudioTextDecoder = FishAudioTextDecoderConfig.load_model(fish_model, jnp.bfloat16)
    decode_result: FishAudioTextDecoderResult = lalamo_model(
        text_tokens=tokenized_text_lalamo, input_pos=input_pos, sampling_params=sampling_params
    )
    output_lalamo = decode_result.token_codes

    _testlog.info(f"output_fish. : {output_fish}")
    _testlog.info(f"output_lalamo: {output_lalamo}")

    assert output_fish[:, 0].tolist() == output_lalamo[0].tolist()


def test_logits_to_probs_jax_basic() -> None:
    """Test that logits_to_probs_jax produces valid probability distributions."""
    logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    probs = logits_to_probs(logits, top_p=1.0, temperature=1.0)

    assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-5)
    assert jnp.all(probs >= 0)
    assert probs[4] > probs[3] > probs[2] > probs[1] > probs[0]


def test_logits_to_probs_jax_temperature() -> None:
    """Test that temperature affects the probability distribution correctly."""
    logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    probs_low_temp = logits_to_probs(logits, top_p=1.0, temperature=0.1)
    probs_high_temp = logits_to_probs(logits, top_p=1.0, temperature=10.0)

    assert probs_low_temp[4] > probs_high_temp[4]
    assert jnp.std(probs_high_temp) < jnp.std(probs_low_temp)


def test_sample_jax_respects_temperature() -> None:
    """Test that lower temperature makes sampling more deterministic."""
    logits = jnp.array([[[0.0, 0.0, 0.0, 0.0, 10.0]]])

    sampling_params = FishAudioSamplingParams(
        temperature=0.01, top_p=1.0, repetition_penalty=1.0, argmax_decoding=False
    )

    high_count = 0
    for i in range(50):
        key = jax.random.PRNGKey(i)
        token, _ = sample(logits, key=key, sampling_params=sampling_params)
        if int(token) == 4:
            high_count += 1

    assert high_count >= 45, f"Expected token 4 at least 45/50 times, got {high_count}"


def test_logits_to_probs_jax_vs_pytorch_basic() -> None:
    """Test that JAX and PyTorch logits_to_probs produce similar probability distributions."""
    logits_jax = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    logits_torch = jax_to_torch(logits_jax)

    temperature = 1.0
    top_p = 0.9

    probs_jax = logits_to_probs(logits_jax, top_p=top_p, temperature=temperature)

    probs_torch = fish_inference.logits_to_probs(
        logits_torch.clone(),
        temperature=torch.tensor(temperature),
        top_p=torch.tensor(top_p),
        repetition_penalty=torch.tensor(1.0),
        previous_tokens=None,
    )

    probs_torch_as_jax = torch_to_jax(probs_torch)

    # Both should produce valid probability distributions
    assert jnp.isclose(jnp.sum(probs_jax), 1.0, atol=1e-5)
    assert jnp.isclose(jnp.sum(probs_torch_as_jax), 1.0, atol=1e-5)

    # Check relative ordering is preserved (highest logit = highest prob)
    assert jnp.argmax(probs_jax) == jnp.argmax(probs_torch_as_jax)


def test_logits_to_probs_jax_vs_pytorch_temperature_scaling() -> None:
    """Test that both implementations scale with temperature similarly."""
    logits_jax = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    logits_torch = jax_to_torch(logits_jax)

    for temperature in [0.1, 0.5, 1.0, 2.0]:
        probs_jax = logits_to_probs(logits_jax, top_p=1.0, temperature=temperature)
        probs_torch = fish_inference.logits_to_probs(
            logits_torch.clone(),
            temperature=torch.tensor(temperature),
            top_p=torch.tensor(1.0),
            repetition_penalty=torch.tensor(1.0),
            previous_tokens=None,
        )
        probs_torch_as_jax = torch_to_jax(probs_torch)

        # Argmax should match for all temperatures
        assert jnp.argmax(probs_jax) == jnp.argmax(probs_torch_as_jax), f"Argmax mismatch at temperature={temperature}"


def test_logits_to_probs_jax_vs_pytorch_top_p() -> None:
    """Test that top-p filtering works similarly in both implementations."""
    logits_jax = jnp.array([0.1, 0.2, 0.3, 5.0, 10.0])  # Clear top-2 tokens
    logits_torch = jax_to_torch(logits_jax)

    # With low top_p, only the top tokens should have non-zero probability
    top_p = 0.5

    probs_jax = logits_to_probs(logits_jax, temperature=1.0, top_p=top_p)
    probs_torch = fish_inference.logits_to_probs(
        logits_torch.clone(),
        temperature=torch.tensor(1.0),
        top_p=torch.tensor(top_p),
        repetition_penalty=torch.tensor(1.0),
        previous_tokens=None,
    )
    probs_torch_as_jax = torch_to_jax(probs_torch)

    # Both should zero out low-probability tokens
    # The top token (index 4) should have the highest probability in both
    assert jnp.argmax(probs_jax) == 4
    assert jnp.argmax(probs_torch_as_jax) == 4

    # Lower logit tokens should have zero or near-zero probability
    assert probs_jax[0] < 0.01
    assert probs_torch_as_jax[0] < 0.01


def test_sample_jax_vs_pytorch_deterministic() -> None:
    """Test that both sampling implementations pick the same token with very low temperature."""
    # Create logits with a clear winner
    logits_jax = jnp.array([[[0.0, 0.0, 0.0, 0.0, 100.0]]])  # Token 4 is dominant
    logits_torch = jax_to_torch(logits_jax)

    temperature = 0.001  # Very low temperature = nearly deterministic
    sampling_params = FishAudioSamplingParams(
        argmax_decoding=False, top_p=1.0, temperature=temperature, repetition_penalty=1.0
    )

    key = jax.random.PRNGKey(42)
    token_jax, _ = sample(logits_jax, key=key, sampling_params=sampling_params)

    token_torch, _ = fish_inference.sample(
        logits_torch.clone(),
        temperature=torch.tensor(temperature),
        top_p=torch.tensor(1.0),
        repetition_penalty=torch.tensor(1.0),
        previous_tokens=None,
    )

    # Both should pick token 4 (the dominant one)
    assert int(token_jax) == 4
    assert int(token_torch.item()) == 4


def test_sample_jax_vs_pytorch_distribution_similarity() -> None:
    """Test that JAX and PyTorch sampling produce similar distributions over many samples."""
    logits_jax = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    logits_torch = jax_to_torch(logits_jax)

    temperature = 1.0
    top_p = 1.0
    num_samples = 200

    sampling_params = FishAudioSamplingParams(
        argmax_decoding=False, top_p=1.0, temperature=temperature, repetition_penalty=1.0
    )

    jax_counts = [0] * 5
    for i in range(num_samples):
        key = jax.random.PRNGKey(i)
        token, _ = sample(logits_jax, key=key, sampling_params=sampling_params)
        jax_counts[int(token)] += 1

    torch_counts = [0] * 5
    torch.manual_seed(42)
    for _ in range(num_samples):
        token, _ = fish_inference.sample(
            logits_torch.clone(),
            temperature=torch.tensor(temperature),
            top_p=torch.tensor(top_p),
            repetition_penalty=torch.tensor(1.0),
            previous_tokens=None,
        )
        torch_counts[int(token.item())] += 1

    # Both should favor higher-indexed tokens (higher logits)
    # Token 4 should be sampled most frequently in both
    assert jax_counts[4] == max(jax_counts), f"JAX didn't favor token 4: {jax_counts}"
    assert torch_counts[4] == max(torch_counts), f"PyTorch didn't favor token 4: {torch_counts}"

    # Token 0 should be sampled least frequently in both
    assert jax_counts[0] == min(jax_counts), f"JAX sampled token 0 too often: {jax_counts}"
    assert torch_counts[0] == min(torch_counts), f"PyTorch sampled token 0 too often: {torch_counts}"


@pytest.mark.skip(reason="skipped by default")
@torch.no_grad
def test_full_utterance_decoding(fish_audio_local_model_path: Path) -> None:
    tts_message = get_tts_message()

    tts_generator = TTSConfig.load_model_from_foreign_model_preset(
        ForeignTTSModel.FISH_AUDIO, fish_audio_local_model_path
    )
    tokenized_text = tts_generator.tokenize_text([tts_message])
    assert isinstance(tts_generator.text_decoder, FishAudioTextDecoder_Foreign)
    sampling_params = FishAudioSamplingParams(
        argmax_decoding=True, temperature=0.808, top_p=0.808, repetition_penalty=1.1016
    )

    fishaudio_wrapper_semantic_tokens = tts_generator.text_decoder.decode_utterance(
        tokenized_text, sampling_params=sampling_params
    )

    lalamo_text_decoder = FishAudioTextDecoderConfig.load_model(tts_generator.text_decoder.fish_model, jnp.bfloat16)
    lalamo_semantic_tokens = lalamo_text_decoder.decode_utterance(tokenized_text, sampling_params=sampling_params)

    _testlog.info(f"Fishaudio wrapper tokens: {fishaudio_wrapper_semantic_tokens}")
    _testlog.info(f"Lalamo tokens.          : {lalamo_semantic_tokens}")

    np.save("fish_codes.npy", np.array(fishaudio_wrapper_semantic_tokens))
    np.save("lalamo_codes.npy", np.array(lalamo_semantic_tokens))
