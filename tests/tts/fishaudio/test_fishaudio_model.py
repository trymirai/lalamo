import logging
from pathlib import Path

import jax
import torch
from fish_speech.models.dac import inference as fish_dac_inference
from fish_speech.models.dac.modded_dac import DAC
from jax import numpy as jnp

from lalamo.audio.tts_message_processor import TTSMessage
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_descript_audio_codec,
)
from lalamo.model_import.model_configs.huggingface.fishaudio import (
    instantiate_dac_config_from_fishaudio_config,
    load_fishaudio_text_decoder,
)
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.sampling import GreedyPolicy
from tests.tts.fishaudio.fishaudio_sampling import sampling_params_from_policy
from tests.tts.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from tests.tts.fishaudio.fishaudio_torch_stuff import FishAudioFromTorch

from .fishaudio_torch_stuff import from_fish_audio_config, prepare_state_dict_for_lalamo_loaders

_testlog = logging.getLogger("tts_test_logger")


@torch.no_grad
def test_decode_one_token(fish_audio_local_model_path: Path) -> None:
    test_text = "this is a test message with speaker 0"
    tts_message = TTSMessage(content=test_text, speaker_id="speaker:0", style="interleave")

    # Load PyTorch-wrapped model for reference output
    pytorch_tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(fish_audio_local_model_path)
    assert isinstance(pytorch_tts_generator.tts_model.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = pytorch_tts_generator.tts_model.text_decoder.fish_model

    # Create Lalamo text decoder config from PyTorch model config
    precision = jnp.bfloat16
    lalamo_config = from_fish_audio_config(fish_model.config, fish_model.tokenizer, precision)

    # Convert PyTorch weights to JAX and load into Lalamo text decoder
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_model.state_dict())
    lalamo_text_decoder = load_fishaudio_text_decoder(lalamo_config.empty(), weights_dict)

    sampling_policy = GreedyPolicy()
    key = jax.random.PRNGKey(123)

    # Prepare inputs
    tokenized_text = jnp.array(pytorch_tts_generator.message_processor.tokenize_request([tts_message]))[None, :]
    n_tokens = tokenized_text.shape[-1]
    input_pos = jnp.arange(n_tokens)[None, :]

    # Run PyTorch model
    output_pytorch = pytorch_tts_generator.tts_model.text_decoder(
        tokenized_text,
        sampling_params=sampling_params_from_policy(sampling_policy),
    )

    # Run Lalamo model
    decode_result = lalamo_text_decoder(
        text_tokens=tokenized_text,
        input_pos=input_pos,
        sampling_policy=sampling_policy,
        key=key,
    )
    output_lalamo = decode_result.token_codes

    _testlog.info(f"[generate token] pytorch: {output_pytorch}")
    _testlog.info(f"[generate token] lalamo : {output_lalamo}")

    assert output_pytorch[:, 0].tolist() == output_lalamo[0].tolist()


@torch.no_grad
def test_dac_matches_pytorch(fish_audio_local_model_path) -> None:
    """Test that Lalamo DAC matches PyTorch DAC from FishAudio.

    This test loads a real DAC model checkpoint, creates a Lalamo DAC module
    using load_descript_audio_codec(), and compares full inference (codes -> audio)
    between both implementations.
    """

    audio_chkpt_path = fish_audio_local_model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    fish_dac = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(fish_dac, DAC)
    fish_dac.eval()

    # Load Lalamo DAC using fishaudio_loaders directly
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_dac.state_dict())
    audio_decoder_cfg = instantiate_dac_config_from_fishaudio_config(get_default_fishaudio_dac_config())
    lalamo_dac = audio_decoder_cfg.empty()
    lalamo_dac = load_descript_audio_codec(lalamo_dac, weights_dict)

    fish_dac_omega_config = get_default_fishaudio_dac_config()

    # Create test input: random codes
    torch.manual_seed(42)
    fish_quantizer_config = fish_dac_omega_config["quantizer"]
    codebook_size = fish_quantizer_config["codebook_size"]
    batch_size = 1
    num_tokens = 10
    n_codebooks = fish_quantizer_config["n_codebooks"]
    test_codes_torch = torch.randint(0, codebook_size, (batch_size, n_codebooks, num_tokens))
    test_codes_jax = torch_to_jax(test_codes_torch).astype(jnp.int32)

    # Run FishAudio DAC inference (quantizer.decode + decoder)
    z_fish = fish_dac.quantizer.decode(test_codes_torch)  # (batch, latent_dim, tokens_upsampled)
    audio_fish = fish_dac.decoder(z_fish)  # (batch, 1, audio_samples)
    # Run Lalamo DAC inference
    audio_lalamo = lalamo_dac(test_codes_jax)  # (batch, audio_samples, 1) - NTC format

    # Convert for comparison (both to NTC format)
    audio_fish_ntc = torch_to_jax(audio_fish).transpose(0, 2, 1)  # NCT -> NTC

    audio_diff = audio_lalamo - audio_fish_ntc
    assert audio_fish_ntc.shape == audio_lalamo.shape, (
        f"Shape mismatch: FishAudio {audio_fish_ntc.shape} vs Lalamo {audio_lalamo.shape}"
    )
    assert jnp.allclose(audio_fish_ntc, audio_lalamo, atol=1e-3), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(audio_diff))}"
    )
