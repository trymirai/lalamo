from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from lalamo.model_import.loaders.nanocodec_loaders import load_nanocodec
from lalamo.models import TTSGenerator
from lalamo.modules.audio.common_modules import (
    CausalConv1dConfig,
    CausalTransposeConv1dConfig,
)
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    Snake1dConfig,
)
from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodecConfig
from lalamo.modules.audio.nanocodec.nanocodec_consts import (
    DEFAULT_AUDIO_DECODER_INPUT_CONV_SIZE,
    DEFAULT_AUDIO_DECODER_OUTPUT_CONV_SIZE,
    DEFAULT_AUDIO_DECODER_RESBLOCK_DILATIONS,
    DEFAULT_AUDIO_DECODER_RESBLOCK_KERNEL_SIZES,
    DEFAULT_FSQ_EPS,
    DEFAULT_NANOCODEC_PRECISION,
)
from lalamo.modules.audio.nanocodec.nanocodec_modules import (
    CausalHiFiGANDecoderConfig,
    FiniteScalarQuantizerConfig,
    GroupFiniteScalarQuantizerConfig,
    HalfSnakeConfig,
    HiFiGANResBlockConfig,
    HiFiGANResLayerConfig,
    ResidualBlockConfig,
)
from tests.tts.nanocodec.nanocodec_torch_stuff import (
    AudioCodecModel,
    CausalHiFiGANDecoder,
    GroupFiniteScalarQuantizer,
    load_nemo_data,
    try_locate_fish_audio_model_path,
)
from tests.tts.utils import generate_harmonic_row, prepare_state_dict_for_lalamo_loaders


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


# =============================================================================
# End-to-End Lalamo vs PyTorch Tests
# =============================================================================


def _create_lalamo_nanocodec_config(config: Mapping) -> NanoCodecConfig:
    """Create Lalamo NanoCodecConfig from NeMo config dict."""
    nemo_quantizer_config = config["vector_quantizer"]
    nemo_decoder_config = config["audio_decoder"]

    # FSQ config for each group
    fsq_config = FiniteScalarQuantizerConfig(
        num_levels=tuple(nemo_quantizer_config["num_levels_per_group"]),
        eps=DEFAULT_FSQ_EPS,
        precision=DEFAULT_NANOCODEC_PRECISION,
    )

    # Group FSQ config
    quantizer_config = GroupFiniteScalarQuantizerConfig(
        num_groups=nemo_quantizer_config["num_groups"],
        quantizer_config=fsq_config,
    )

    # Build decoder config hierarchy
    snake_config = Snake1dConfig(precision=jnp.float32)
    activation_config = HalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    transpose_conv_config = CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)

    residual_block_config = ResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )
    hifigan_res_block_config = HiFiGANResBlockConfig(residual_block_config=residual_block_config)
    res_layer_config = HiFiGANResLayerConfig(hifigan_res_block_config=hifigan_res_block_config)

    decoder_config = CausalHiFiGANDecoderConfig(
        activation_config=activation_config,
        pre_conv_config=conv_config,
        transpose_conv_config=transpose_conv_config,
        res_layer_config=res_layer_config,
        post_conv_config=conv_config,
    )

    in_kernel_size = nemo_decoder_config.get("in_kernel_size", DEFAULT_AUDIO_DECODER_INPUT_CONV_SIZE)
    out_kernel_size = nemo_decoder_config.get("out_kernel_size", DEFAULT_AUDIO_DECODER_OUTPUT_CONV_SIZE)
    resblock_kernel_sizes = nemo_decoder_config.get(
        "resblock_kernel_sizes",
        DEFAULT_AUDIO_DECODER_RESBLOCK_KERNEL_SIZES,
    )
    resblock_dilation_sizes = nemo_decoder_config.get(
        "resblock_dilation_sizes",
        DEFAULT_AUDIO_DECODER_RESBLOCK_DILATIONS,
    )

    return NanoCodecConfig(
        precision=jnp.float32,
        quantizer_config=quantizer_config,
        decoder_config=decoder_config,
        samplerate=config["sample_rate"],
        base_channels=nemo_decoder_config["base_channels"],
        up_sample_rates=tuple(nemo_decoder_config["up_sample_rates"]),
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilations=resblock_dilation_sizes,
    )


def test_lalamo_nanocodec_matches_torch(cached_nemo_model: tuple[Mapping, Mapping, Path]) -> None:
    """Test that Lalamo NanoCodec produces same output as PyTorch AudioCodecModel.

    Uses a harmonic signal encoded by PyTorch and decoded by both implementations.
    """
    state_dict, config, _ = cached_nemo_model

    # Create and load PyTorch model
    cfg = DictConfig(config)
    torch_model = AudioCodecModel(cfg)
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    # Create Lalamo config and model
    lalamo_config = _create_lalamo_nanocodec_config(config)
    lalamo_model = lalamo_config.empty()

    # Load weights into Lalamo model
    weights_dict = prepare_state_dict_for_lalamo_loaders(dict(state_dict))
    lalamo_model = load_nanocodec(lalamo_model, weights_dict)

    # Generate a harmonic signal for encoding (2 seconds)
    sample_rate = config["sample_rate"]
    duration_samples = sample_rate * 2  # 2 seconds of audio
    f0 = 200.0  # fundamental frequency
    harmonic_signal, _ = generate_harmonic_row(sample_rate, duration_samples, f0)
    harmonic_signal = harmonic_signal.astype(np.float32)

    # Encode with PyTorch model
    audio_torch_input = torch.from_numpy(harmonic_signal).unsqueeze(0)  # [1, T]
    audio_len = torch.tensor([len(harmonic_signal)])

    with torch.no_grad():
        tokens_torch, tokens_len = torch_model.encode(audio=audio_torch_input, audio_len=audio_len)
        # tokens_torch shape: [B, C, T] where C is num_codebooks
        audio_torch_decoded, _ = torch_model.decode(tokens=tokens_torch, tokens_len=tokens_len)

    # Get tokens for single batch item: [C, T]
    tokens_np = tokens_torch[0].numpy()

    # Lalamo forward using audio_from_codes (expects [C, T] format)
    tokens_jax = jnp.array(tokens_np)
    audio_lalamo = lalamo_model.audio_from_codes(tokens_jax)

    # Compare outputs
    # Use relaxed tolerances for end-to-end test as small numerical differences
    # accumulate through the many layers of the HiFiGAN decoder
    np.testing.assert_allclose(
        np.array(audio_lalamo),
        audio_torch_decoded[0].numpy(),
        rtol=5e-2,
        atol=5e-2,
    )


def test_nanocodec_model_spec_loading() -> None:
    """Test end-to-end model loading via NanoCodecForeignConfig (model spec path).

    Exercises the full pipeline: NanoCodecForeignConfig -> TTSConfig -> TTSModel
    with StubTextDecoder + NanoCodec audio decoder, then runs generate_speech.
    """
    from lalamo.model_import.common import import_model
    from lalamo.modules.audio.nanocodec.audio_decoding import NanoCodec
    from lalamo.modules.audio.nanocodec.stub_text_decoder import StubTextDecoder
    from lalamo.modules.audio.text_to_speech import TTSMessage, TTSModel

    message_to_generate = TTSMessage(content="Some noise will be generated here", speaker_id="0", style="unsupported")

    generator, _ = import_model(model_spec="nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps", precision=jnp.float32)

    assert isinstance(generator, TTSGenerator)
    assert isinstance(generator.tts_model, TTSModel)
    assert isinstance(generator.tts_model.text_decoder, StubTextDecoder)
    assert isinstance(generator.tts_model.audio_decoder, NanoCodec)

    generation_result = generator.generate_speech([message_to_generate])
    audio = generation_result.audio

    assert float(jnp.min(audio)) >= -1.0
    assert float(jnp.max(audio)) <= 1.0
