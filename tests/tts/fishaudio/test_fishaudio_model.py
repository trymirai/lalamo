import logging
from pathlib import Path
from typing import Protocol, cast

import jax
import torch
from fish_speech.models.dac import inference as fish_dac_inference
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.llama import DualARModelArgs
from jax import numpy as jnp

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.fishaudio_loaders import (
    load_descript_audio_codec,
)
from lalamo.model_import.model_configs.huggingface.fishaudio import (
    instantiate_dac_config_from_fishaudio_config,
    load_fishaudio_text_decoder,
)
from lalamo.models.tts_codec import TTSMessage
from lalamo.module import Keychain
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.sampling import SamplingPolicy
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.tts.fishaudio.fishaudio_sampling import sampling_params_from_policy
from tests.tts.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from tests.tts.fishaudio.fishaudio_torch_stuff import FishAudioFromTorch

from .fishaudio_torch_stuff import from_fish_audio_config, prepare_state_dict_for_lalamo_loaders

_testlog = logging.getLogger("tts_test_logger")


class _TorchCodeQuantizer(Protocol):
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad
def test_decode_one_token(fish_audio_local_model_path: Path) -> None:
    test_text = "this is a test message with speaker 0"
    tts_message = TTSMessage(content=test_text, speaker_id="speaker:0", style="interleave")

    # Load PyTorch-wrapped model for reference output
    pytorch_tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(
        fish_audio_local_model_path,
        precision=torch.float32,
    )
    assert isinstance(pytorch_tts_generator.tts_model.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = pytorch_tts_generator.tts_model.text_decoder.fish_model

    # Create Lalamo text decoder config from PyTorch model config
    assert isinstance(fish_model.config, DualARModelArgs)
    lalamo_config = from_fish_audio_config(fish_model.config, fish_model.tokenizer)

    # Convert PyTorch weights to JAX and load into Lalamo text decoder
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_model.state_dict())
    lalamo_text_decoder = load_fishaudio_text_decoder(
        lalamo_config.init(EmptyInitializer(dtype=jnp.bfloat16)), weights_dict
    )

    sampling_policy = SamplingPolicy.init(temperature=0.0)
    vmapped_keys = jax.random.key(123)

    # Prepare inputs
    tokenized_text = jnp.array(pytorch_tts_generator.token_codec.encode_request([tts_message]))[None, :]
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
        keychain=Keychain(vmapped_keys=vmapped_keys, batch_key=jax.random.key(456)),
    )
    output_lalamo = decode_result.token_codes

    _testlog.info(f"[generate token] pytorch: {output_pytorch}")
    _testlog.info(f"[generate token] lalamo : {output_lalamo}")

    assert output_pytorch[:, 0].tolist() == output_lalamo[0].tolist()


@torch.no_grad
def test_dac_matches_pytorch(fish_audio_local_model_path: Path) -> None:
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
    audio_decoder_cfg = instantiate_dac_config_from_fishaudio_config(
        get_default_fishaudio_dac_config(),
    )
    lalamo_dac = audio_decoder_cfg.init(EmptyInitializer(dtype=jnp.float32))
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
    fish_quantizer = cast("_TorchCodeQuantizer", fish_dac.quantizer)
    z_fish = fish_quantizer.decode(test_codes_torch)  # (batch, latent_dim, tokens_upsampled)
    audio_fish = fish_dac.decoder(z_fish)  # (batch, 1, audio_samples)
    # Run Lalamo DAC inference
    audio_lalamo = lalamo_dac(test_codes_jax, keychain=Keychain.init(0))  # (batch, audio_samples, 1) - NTC format

    # Convert for comparison (both to NTC format)
    audio_fish_ntc = torch_to_jax(audio_fish).transpose(0, 2, 1)  # NCT -> NTC

    assert audio_fish_ntc.shape == audio_lalamo.shape, (
        f"Shape mismatch: FishAudio {audio_fish_ntc.shape} vs Lalamo {audio_lalamo.shape}"
    )
    assert_close(
        result=audio_lalamo,
        reference=audio_fish_ntc,
        atol=3.5e-3,
        operation_name="test_dac_matches_pytorch",
    )
