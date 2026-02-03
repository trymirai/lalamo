import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from tests.tts.cambai.nanocodec_torch_stuff import (
    AudioCodecModel,
    CausalHiFiGANDecoder,
    GroupFiniteScalarQuantizer,
    load_nemo_data,
    try_locate_fish_audio_model_path,
)
from tests.tts.utils import generate_pseudo_voice_signal

_testlog = logging.getLogger("tts_test_logger")


@pytest.fixture
def cached_nemo_model() -> tuple[Mapping, Mapping, Path]:
    """Pytest fixture that provides cached NeMo model data."""
    model_path = try_locate_fish_audio_model_path()
    if model_path is None:
        pytest.skip("NeMo model not found in HuggingFace cache")
    if not model_path.exists():
        pytest.skip(f"NeMo model path is invalid: {model_path}")
    state_dict, config = load_nemo_data(model_path)
    if state_dict is None or config is None:
        pytest.skip(f"Failed to load NeMo model from provided path: {model_path}")
    return state_dict, config, model_path


def test_group_finite_scalar_quantizer_encode_decode(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test quantizer instantiation and encode/decode roundtrip."""
    _, cfg, _ = cached_nemo_model
    cfg = DictConfig(cfg)
    vq_cfg = cfg.vector_quantizer

    quantizer = GroupFiniteScalarQuantizer(
        num_groups=vq_cfg.num_groups,
        num_levels_per_group=list(vq_cfg.num_levels_per_group),
    )

    assert quantizer.num_groups == 13
    assert quantizer.codebook_dim_per_group == 4  # len([8, 7, 6, 6])
    assert quantizer.codebook_dim == 52  # 13 groups * 4 dims per group

    batch_size = 2
    seq_len = 10
    inputs = torch.randn(batch_size, quantizer.codebook_dim, seq_len)
    input_len = torch.tensor([seq_len, seq_len])

    # Encode
    indices = quantizer.encode(inputs=inputs, input_len=input_len)
    assert indices.shape == (13, batch_size, seq_len)  # [num_groups, B, T]

    # Decode
    dequantized = quantizer.decode(indices=indices, input_len=input_len)
    assert dequantized.shape == (batch_size, quantizer.codebook_dim, seq_len)


def test_causal_hifigan_decoder_forward(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test decoder instantiation and forward pass."""
    _, cfg, _ = cached_nemo_model
    cfg = DictConfig(cfg)
    dec_cfg = cfg.audio_decoder

    decoder = CausalHiFiGANDecoder(
        input_dim=dec_cfg.input_dim,
        up_sample_rates=list(dec_cfg.up_sample_rates),
        base_channels=dec_cfg.base_channels,
        activation=dec_cfg.activation,
        output_activation=dec_cfg.output_activation,
        pad_mode=dec_cfg.pad_mode,
        n_groups_equal_to_out_channels=dec_cfg.n_groups_equal_to_out_channels,
    )

    assert decoder.up_sample_rates == [7, 7, 6, 3, 2]

    batch_size = 2
    seq_len = 10
    input_dim = dec_cfg.input_dim  # 52

    inputs = torch.randn(batch_size, input_dim, seq_len)
    input_len = torch.tensor([seq_len, seq_len])

    audio, audio_len = decoder(inputs=inputs, input_len=input_len)

    expected_upsample = np.array(cfg.audio_decoder.up_sample_rates).prod()
    expected_audio_len = seq_len * expected_upsample

    assert audio.shape == (batch_size, expected_audio_len)
    assert audio_len.tolist() == [expected_audio_len, expected_audio_len]
    assert audio.min() >= -1.0
    assert audio.max() <= 1.0


def test_audio_codec_model_decode(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test AudioCodecModel instantiation and decode() method."""
    _, cfg, _ = cached_nemo_model
    cfg = DictConfig(cfg)
    model = AudioCodecModel(cfg)

    batch_size = 1
    num_codebooks = cfg.vector_quantizer.num_groups
    seq_len = 5

    # Simulate token indices (shape: [B, C, T] where C is num_codebooks)
    codebook_size = np.array(cfg.vector_quantizer.num_levels_per_group).prod()
    tokens = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_len), dtype=torch.int32)
    tokens_len = torch.tensor([seq_len])

    # Decode tokens to audio
    audio, _ = model.decode(tokens=tokens, tokens_len=tokens_len)

    expected_upsample = np.array(cfg.audio_decoder.up_sample_rates).prod()
    expected_audio_len = seq_len * expected_upsample

    assert audio.shape == (batch_size, expected_audio_len)
    assert audio.min() >= -1.0
    assert audio.max() <= 1.0


def test_try_locate_fish_audio_model_path() -> None:
    """Test that we can locate the cached NeMo model path."""
    model_path = try_locate_fish_audio_model_path()
    if model_path is None:
        pytest.skip("NeMo model not found in HuggingFace cache")

    assert model_path.exists(), f"Model path {model_path} does not exist"
    assert model_path.suffix == ".nemo", f"Expected .nemo file, got {model_path.suffix}"


def test_load_nemo_data() -> None:
    """Test loading state dict and config from .nemo file."""
    model_path = try_locate_fish_audio_model_path()
    if model_path is None:
        pytest.skip("NeMo model not found in HuggingFace cache")

    state_dict, config = load_nemo_data(model_path)

    assert isinstance(state_dict, dict), "state_dict should be a dict"
    assert isinstance(config, dict), "config should be a dict"
    assert len(state_dict) > 0, "state_dict should not be empty"
    assert "sample_rate" in config, "config should contain sample_rate"
    assert "vector_quantizer" in config, "config should contain vector_quantizer"
    assert "audio_decoder" in config, "config should contain audio_decoder"


def test_group_finite_scalar_quantizer_with_real_weights(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test GroupFiniteScalarQuantizer with real weights from cached model."""
    state_dict, config, _ = cached_nemo_model

    vq_cfg = config["vector_quantizer"]
    quantizer = GroupFiniteScalarQuantizer(
        num_groups=vq_cfg["num_groups"],
        num_levels_per_group=list(vq_cfg["num_levels_per_group"]),
    )

    # Extract quantizer weights from state_dict
    quantizer_state = {
        k.replace("vector_quantizer.", ""): v for k, v in state_dict.items() if k.startswith("vector_quantizer.")
    }

    if quantizer_state:
        quantizer.load_state_dict(quantizer_state)

    # Test encode/decode with real weights
    batch_size = 2
    seq_len = 10
    inputs = torch.randn(batch_size, quantizer.codebook_dim, seq_len)
    input_len = torch.tensor([seq_len, seq_len])

    indices = quantizer.encode(inputs=inputs, input_len=input_len)
    assert indices.shape == (quantizer.num_groups, batch_size, seq_len)

    dequantized = quantizer.decode(indices=indices, input_len=input_len)
    assert dequantized.shape == (batch_size, quantizer.codebook_dim, seq_len)


def test_causal_hifigan_decoder_with_real_weights(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test CausalHiFiGANDecoder with real weights from cached model."""
    state_dict, config, _ = cached_nemo_model

    dec_cfg = config["audio_decoder"]
    decoder = CausalHiFiGANDecoder(
        input_dim=dec_cfg["input_dim"],
        up_sample_rates=list(dec_cfg["up_sample_rates"]),
        base_channels=dec_cfg["base_channels"],
        activation=dec_cfg["activation"],
        output_activation=dec_cfg["output_activation"],
        pad_mode=dec_cfg.get("pad_mode", "zeros"),
        n_groups_equal_to_out_channels=dec_cfg.get("n_groups_equal_to_out_channels", True),
    )

    # Extract decoder weights from state_dict
    decoder_state = {
        k.replace("audio_decoder.", ""): v for k, v in state_dict.items() if k.startswith("audio_decoder.")
    }

    if decoder_state:
        decoder.load_state_dict(decoder_state)

    # Test forward pass with real weights
    batch_size = 2
    seq_len = 10
    input_dim = dec_cfg["input_dim"]

    inputs = torch.randn(batch_size, input_dim, seq_len)
    input_len = torch.tensor([seq_len, seq_len])

    audio, audio_len = decoder(inputs=inputs, input_len=input_len)

    expected_upsample = np.prod(dec_cfg["up_sample_rates"])
    expected_audio_len = seq_len * expected_upsample

    assert audio.shape == (batch_size, expected_audio_len)
    assert audio_len.tolist() == [expected_audio_len, expected_audio_len]
    assert audio.min() >= -1.0
    assert audio.max() <= 1.0


def test_audio_codec_model_with_real_weights(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test full AudioCodecModel with real weights from cached model."""
    state_dict, config, _ = cached_nemo_model

    # Create DictConfig from the loaded config
    cfg = DictConfig(config)
    model = AudioCodecModel(cfg)

    # Load state dict (model's load_state_dict handles filtering)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Test decode with real weights
    batch_size = 1
    num_codebooks = config["vector_quantizer"]["num_groups"]
    seq_len = 5

    codebook_size = int(np.prod(config["vector_quantizer"]["num_levels_per_group"]))
    tokens = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_len), dtype=torch.int32)
    tokens_len = torch.tensor([seq_len])

    with torch.no_grad():
        audio, _ = model.decode(tokens=tokens, tokens_len=tokens_len)

    expected_upsample = int(np.prod(config["audio_decoder"]["up_sample_rates"]))
    expected_audio_len = seq_len * expected_upsample

    assert audio.shape == (batch_size, expected_audio_len)
    assert audio.min() >= -1.0
    assert audio.max() <= 1.0


def test_audio_codec_model_decode_deterministic(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test that decode produces deterministic output with same input tokens."""
    state_dict, config, _ = cached_nemo_model

    cfg = DictConfig(config)
    model = AudioCodecModel(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    batch_size = 1
    num_codebooks = config["vector_quantizer"]["num_groups"]
    seq_len = 5

    codebook_size = int(np.prod(config["vector_quantizer"]["num_levels_per_group"]))
    tokens = torch.randint(0, codebook_size, (batch_size, num_codebooks, seq_len), dtype=torch.int32)
    tokens_len = torch.tensor([seq_len])

    with torch.no_grad():
        audio1, _ = model.decode(tokens=tokens, tokens_len=tokens_len)
        audio2, _ = model.decode(tokens=tokens, tokens_len=tokens_len)

    assert torch.allclose(audio1, audio2), "Decode should be deterministic"
