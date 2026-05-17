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
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.sampling import SamplingPolicy
from lalamo.utils.torch_interop import torch_to_jax
from tests.common import assert_close
from tests.helpers import make_test_sharding_config
from tests.tts.fishaudio.fishaudio_sampling import sampling_params_from_policy
from tests.tts.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from tests.tts.fishaudio.fishaudio_torch_stuff import FishAudioFromTorch

from .fishaudio_torch_stuff import from_fish_audio_config, prepare_state_dict_for_lalamo_loaders


class _TorchCodeQuantizer(Protocol):
    def decode(self, codes: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad
def test_decode_one_token(fish_audio_local_model_path: Path) -> None:
    test_text = "this is a test message with speaker 0"
    tts_message = TTSMessage(content=test_text, speaker_id="speaker:0", style="interleave")

    pytorch_tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(
        fish_audio_local_model_path,
        precision=torch.float32,
    )
    assert isinstance(pytorch_tts_generator.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = pytorch_tts_generator.text_decoder.fish_model

    assert isinstance(fish_model.config, DualARModelArgs)
    lalamo_config = from_fish_audio_config(fish_model.config, fish_model.tokenizer)

    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_model.state_dict())
    lalamo_text_decoder = load_fishaudio_text_decoder(
        lalamo_config.init(
            EmptyInitializer(dtype=jnp.bfloat16, sharding_config=make_test_sharding_config()),
        ),
        weights_dict,
    )

    sampling_policy = SamplingPolicy.init(temperature=0.0)
    vmapped_keys = jax.random.key(123)

    tokenized_text = jnp.array(pytorch_tts_generator.token_codec.encode_request([tts_message]))[None, :]
    n_tokens = tokenized_text.shape[-1]
    input_pos = jnp.arange(n_tokens)[None, :]

    output_pytorch = pytorch_tts_generator.text_decoder(
        tokenized_text,
        sampling_params=sampling_params_from_policy(sampling_policy),
    )

    decode_result = lalamo_text_decoder(
        text_tokens=tokenized_text,
        input_pos=input_pos,
        sampling_policy=sampling_policy,
        keychain=Keychain(
            vmapped_keys=vmapped_keys,
            batch_key=jax.random.key(456),
            sharding_config=ShardingConfig.replicated(),
        ),
        forward_pass_config=DecoderForwardPassConfig.for_tracer_tests(),
    )
    output_lalamo = decode_result.token_codes

    assert output_pytorch[:, 0].tolist() == output_lalamo[0].tolist()


@torch.no_grad
def test_dac_matches_pytorch(fish_audio_local_model_path: Path) -> None:
    audio_chkpt_path = fish_audio_local_model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    fish_dac = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(fish_dac, DAC)
    fish_dac.eval()

    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_dac.state_dict())
    audio_decoder_cfg = instantiate_dac_config_from_fishaudio_config(
        get_default_fishaudio_dac_config(),
    )
    lalamo_dac = audio_decoder_cfg.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
    )
    lalamo_dac = load_descript_audio_codec(lalamo_dac, weights_dict)

    fish_dac_omega_config = get_default_fishaudio_dac_config()

    torch.manual_seed(42)
    fish_quantizer_config = fish_dac_omega_config["quantizer"]
    codebook_size = fish_quantizer_config["codebook_size"]
    batch_size = 1
    num_tokens = 10
    n_codebooks = fish_quantizer_config["n_codebooks"]
    test_codes_torch = torch.randint(0, codebook_size, (batch_size, n_codebooks, num_tokens))
    test_codes_jax = torch_to_jax(test_codes_torch).astype(jnp.int32)

    fish_quantizer = cast("_TorchCodeQuantizer", fish_dac.quantizer)
    z_fish = fish_quantizer.decode(test_codes_torch)  # (batch, latent_dim, tokens_upsampled)
    audio_fish = fish_dac.decoder(z_fish)  # (batch, 1, audio_samples)
    audio_lalamo = lalamo_dac(
        test_codes_jax, keychain=Keychain.init(0, sharding_config=make_test_sharding_config())
    )  # (batch, audio_samples, 1) - NTC format

    audio_fish_ntc = torch_to_jax(audio_fish).transpose(0, 2, 1)  # NCT -> NTC

    assert_close(
        result=audio_lalamo,
        reference=audio_fish_ntc,
        atol=3.5e-3,
        operation_name="test_dac_matches_pytorch",
    )
