from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import import_model
from lalamo.model_import.loaders.nanocodec_loaders import load_nanocodec
from lalamo.model_import.model_configs.nanocodec import NanoCodecForeignConfig
from lalamo.models import TTSModel
from lalamo.models.tts_codec import TTSMessage
from lalamo.module import Keychain
from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodec
from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import make_test_sharding_config
from tests.tts.nanocodec.nanocodec_torch_stuff import (
    AudioCodecModel,
    load_nemo_data,
    try_locate_fish_audio_model_path,
)
from tests.tts.utils import generate_harmonic_row, prepare_state_dict_for_lalamo_loaders


@pytest.fixture
def cached_nemo_model() -> tuple[Mapping, Mapping, Path]:
    model_path = try_locate_fish_audio_model_path()
    if model_path is None:
        pytest.skip("NeMo model not found in HuggingFace cache")
    if not model_path.exists():
        pytest.skip(f"NeMo model path is invalid: {model_path}")
    state_dict, config = load_nemo_data(model_path)
    if state_dict is None or config is None:
        pytest.skip(f"Failed to load NeMo model from provided path: {model_path}")
    return state_dict, config, model_path


def test_lalamo_nanocodec_matches_torch(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    state_dict, config, _ = cached_nemo_model

    cfg = DictConfig(config)
    torch_model = AudioCodecModel(cfg)
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    lalamo_config = NanoCodecForeignConfig.from_nemo_config(config).to_tts_config(None).audio_decoder_config
    lalamo_model = lalamo_config.init(EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()))
    assert isinstance(lalamo_model, NanoCodec)

    weights_dict = prepare_state_dict_for_lalamo_loaders(dict(state_dict))
    lalamo_model = load_nanocodec(lalamo_model, weights_dict)

    sample_rate = config["sample_rate"]
    harmonic_signal, _ = generate_harmonic_row(sample_rate, sample_rate, 200.0)
    harmonic_signal = harmonic_signal.astype(np.float32)

    audio_torch_input = torch.from_numpy(harmonic_signal).unsqueeze(0)
    audio_len = torch.tensor([len(harmonic_signal)])

    with torch.no_grad():
        tokens_torch, tokens_len = torch_model.encode(audio=audio_torch_input, audio_len=audio_len)
        audio_torch_decoded, _ = torch_model.decode(tokens=tokens_torch, tokens_len=tokens_len)

    tokens_np = tokens_torch[0].numpy()

    tokens_jax = jnp.array(tokens_np)
    audio_lalamo = lalamo_model.audio_from_codes(
        tokens_jax, keychain=Keychain.init(0, sharding_config=make_test_sharding_config())
    )

    np.testing.assert_allclose(
        np.array(audio_lalamo),
        audio_torch_decoded[0].numpy(),
        rtol=5e-2,
        atol=5e-2,
    )


def test_nanocodec_model_spec_loading() -> None:
    message_to_generate = TTSMessage(content="Some noise will be generated here", speaker_id="0", style="unsupported")

    generator, _ = import_model(
        model_spec="nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps",
        sharding_config=ShardingConfig.replicated(),
        dtype=jnp.float32,
    )

    assert isinstance(generator, TTSModel)
    assert isinstance(generator.text_decoder, StubTextDecoder)
    assert isinstance(generator.audio_decoder, NanoCodec)

    generation_result = generator.generate_speech(
        [message_to_generate],
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )
    audio = generation_result.audio

    assert float(jnp.min(audio)) >= -1.0
    assert float(jnp.max(audio)) <= 1.0
