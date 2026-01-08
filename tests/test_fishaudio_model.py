import logging
from pathlib import Path

import huggingface_hub
import jax
import numpy as np
import pytest
import torch
from fish_speech.models.text2semantic import inference as fish_inference
from fish_speech.tokenizer import FishTokenizer
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import vmap
from pytest import fixture

from lalamo.models import ForeignTTSModel, TTSConfig
from lalamo.modules import GELU
from lalamo.modules.audio.foreign.fishaudio_audio_decoding import (
    DescriptAudioCodec,
    DescriptAudioCodecConfig,
)
from lalamo.modules.audio.foreign.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.audio.foreign.fishaudio_modules import (
    AudioDecoderBlockSpatialParams,
    CausalConv1dConfig,
    CausalTransposeConv1dConfig,
    ConvNeXtBlockConfig,
    ConvNeXtSpatialParams,
    DACDecoderBlockConfig,
    DACDecoderConfig,
    DACDecoderSpatialParams,
    ResidualUnitConfig,
    ResidualUnitSpatialParams,
    ResidualVectorQuantizeConfig,
    Snake1dConfig,
    TransposeConvSpatialParams,
    UpsamplerConfig,
    UpsamplingBlockConfig,
    VectorQuantizeConfig,
)
from lalamo.modules.audio.foreign.fishaudio_sampling import FishAudioSamplingParams, logits_to_probs, sample
from lalamo.modules.audio.foreign.fishaudio_text_decoding import (
    FishAudioTextDecoder,
    FishAudioTextDecoderConfig,
    FishAudioTextDecoderResult,
)
from lalamo.modules.audio.foreign.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from lalamo.modules.audio.tts_request_factory import TTSMessage
from lalamo.modules.audio.utils import DTypeConvert
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
    test_text = "this is a test message with speaker 0"
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
    lalamo_model: FishAudioTextDecoder = FishAudioTextDecoderConfig.from_foreign_model(fish_model, jnp.bfloat16)
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


@pytest.mark.skip(reason="Temporary test used for full utterance tokens generation")
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

    lalamo_text_decoder = FishAudioTextDecoderConfig.from_foreign_model(
        tts_generator.text_decoder.fish_model, jnp.bfloat16
    )
    lalamo_semantic_tokens = lalamo_text_decoder.decode_utterance(tokenized_text, sampling_params=sampling_params)

    _testlog.info(f"Fishaudio wrapper tokens: {fishaudio_wrapper_semantic_tokens}")
    _testlog.info(f"Lalamo tokens.          : {lalamo_semantic_tokens}")

    np.save("fish_codes.npy", np.array(fishaudio_wrapper_semantic_tokens))
    np.save("lalamo_codes.npy", np.array(lalamo_semantic_tokens))


@torch.no_grad
def test_vector_quantize_decode_code() -> None:
    from dac.nn.quantize import VectorQuantize as DACVectorQuantize

    # Test parameters
    input_dim = 512
    codebook_size = 1024
    codebook_dim = 8
    num_tokens = 10
    batch_size = 2

    # Create DAC VectorQuantize
    dac_vq = DACVectorQuantize(input_dim=input_dim, codebook_size=codebook_size, codebook_dim=codebook_dim)
    dac_vq.eval()

    # Create Lalamo VectorQuantize with same config
    lalamo_vq_config = VectorQuantizeConfig(precision=jnp.float32)
    lalamo_vq = lalamo_vq_config.empty(
        input_dim=input_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    # Extract weights from DAC and load into Lalamo
    # DAC codebook: nn.Embedding(codebook_size, codebook_dim)
    dac_codebook_weights = dac_vq.codebook.weight.detach()  # (codebook_size, codebook_dim)

    # DAC out_proj: WNConv1d(codebook_dim, input_dim, kernel_size=1)
    # WNConv1d with kernel_size=1 is essentially a linear layer
    # Weight shape for Conv1d: (out_channels, in_channels, kernel_size) = (input_dim, codebook_dim, 1)
    dac_out_proj_weight = dac_vq.out_proj.weight.detach().squeeze(-1)  # (input_dim, codebook_dim)
    assert dac_vq.out_proj.bias is not None
    dac_out_proj_bias = dac_vq.out_proj.bias.detach()  # (input_dim,)

    lalamo_weights = {
        "codebook": {"weights": torch_to_jax(dac_codebook_weights)},
        "out_proj": {
            "weights": torch_to_jax(dac_out_proj_weight),
            "biases": torch_to_jax(dac_out_proj_bias),
        },
    }
    lalamo_vq = lalamo_vq.import_weights(lalamo_weights)

    torch.manual_seed(42)
    test_indices_torch = torch.randint(0, codebook_size, (batch_size, num_tokens))
    test_indices_jax = torch_to_jax(test_indices_torch).astype(jnp.int32)

    # DAC decode_code:
    # returns (B, D, T) - just embedding + transpose, no out_proj Then out_proj is applied: (B, input_dim, T)
    dac_embedded = dac_vq.decode_code(test_indices_torch)  # (B, codebook_dim, T)
    dac_output = dac_vq.out_proj(dac_embedded)  # (B, input_dim, T)
    dac_output = dac_output.permute(0, 2, 1)  # (B, T, input_dim) for comparison

    # Lalamo decode_code:
    # applies out_proj and returns (tokens, input_dim)
    lalamo_output = vmap(lalamo_vq.decode_code)(test_indices_jax)  # (B, T, input_dim)

    dac_output_jax = torch_to_jax(dac_output)

    _testlog.info(f"DAC output shape: {dac_output.shape}")
    _testlog.info(f"Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Max difference: {jnp.max(jnp.abs(dac_output_jax - lalamo_output))}")

    assert jnp.allclose(dac_output_jax, lalamo_output, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(dac_output_jax - lalamo_output))}"
    )


@torch.no_grad
def test_residual_vector_quantize_from_codes() -> None:
    from dac.nn.quantize import ResidualVectorQuantize as DACResidualVectorQuantize

    input_dim = 512
    n_codebooks = 9
    codebook_size = 1024
    codebook_dim = 8
    num_tokens = 10
    batch_size = 2

    dac_rvq = DACResidualVectorQuantize(
        input_dim=input_dim,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )
    dac_rvq.eval()

    lalamo_rvq_config = ResidualVectorQuantizeConfig(precision=jnp.float32)
    lalamo_rvq = lalamo_rvq_config.empty(
        input_dim=input_dim,
        codebook_size=codebook_size,
        codebook_dim=[codebook_dim] * n_codebooks,
    )

    # Extract weights from DAC and load into Lalamo
    lalamo_quantizer_weights = []
    for dac_q in dac_rvq.quantizers:
        dac_codebook_weights = dac_q.codebook.weight.detach()
        dac_out_proj_weight = dac_q.out_proj.weight.detach().squeeze(-1)
        assert dac_q.out_proj.bias is not None
        dac_out_proj_bias = dac_q.out_proj.bias.detach()

        lalamo_quantizer_weights.append(
            {
                "codebook": {"weights": torch_to_jax(dac_codebook_weights)},
                "out_proj": {
                    "weights": torch_to_jax(dac_out_proj_weight),
                    "biases": torch_to_jax(dac_out_proj_bias),
                },
            }
        )

    lalamo_weights = {"quantizers": lalamo_quantizer_weights}
    lalamo_rvq = lalamo_rvq.import_weights(lalamo_weights)

    torch.manual_seed(42)
    test_codes_torch = torch.randint(0, codebook_size, (batch_size, n_codebooks, num_tokens))
    test_codes_jax = torch_to_jax(test_codes_torch).astype(jnp.int32)

    # DAC from_codes returns (z_q, z_p, codes) where z_q is (B, input_dim, T)
    dac_output, _, _ = dac_rvq.from_codes(test_codes_torch)
    dac_output = dac_output.permute(0, 2, 1)  # (B, T, input_dim) for comparison

    # Lalamo __call__ handles batching internally via vmap
    lalamo_output = lalamo_rvq(test_codes_jax)  # (B, T, input_dim)

    # Compare outputs
    dac_output_jax = torch_to_jax(dac_output)

    _testlog.info(f"DAC RVQ output shape: {dac_output.shape}")
    _testlog.info(f"Lalamo RVQ output shape: {lalamo_output.shape}")
    _testlog.info(f"Max difference: {jnp.max(jnp.abs(dac_output_jax - lalamo_output))}")

    assert jnp.allclose(dac_output_jax, lalamo_output, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(dac_output_jax - lalamo_output))}"
    )


@torch.no_grad
def test_causal_conv1d_matches_pytorch() -> None:
    """Test that Lalamo CausalConv1d matches PyTorch CausalConvNet."""
    from fish_speech.models.dac.rvq import CausalConvNet

    # Test parameters
    batch_size = 2
    in_channels = 64
    out_channels = 128
    kernel_size = 4
    stride = 2
    dilation = 1
    groups = 1
    seq_length = 100

    # Create PyTorch module
    torch_conv = CausalConvNet(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    torch_conv.eval()

    # Create Lalamo module with same config
    lalamo_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    lalamo_conv = lalamo_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    # Extract weights from PyTorch and load into Lalamo
    torch_weights = torch_conv.conv.weight.detach()  # (out_channels, in_channels/groups, kernel_size)
    torch_biases = torch_conv.conv.bias.detach()  # (out_channels,)

    lalamo_weights = {
        "weights": torch_to_jax(torch_weights),
        "biases": torch_to_jax(torch_biases),
    }
    lalamo_conv = lalamo_conv.import_weights(lalamo_weights)

    # Create test input: PyTorch uses (batch, channels, sequence)
    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, in_channels, seq_length)
    # JAX module uses (batch, sequence, channels) - transpose for JAX
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both implementations
    torch_output = torch_conv(test_input_torch)
    lalamo_output = lalamo_conv(test_input_jax)

    # Compare outputs - transpose JAX output back to (batch, channels, sequence) for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"PyTorch CausalConvNet output shape: {torch_output.shape}")
    _testlog.info(f"Lalamo CausalConv1d output shape: {lalamo_output.shape}")
    _testlog.info(f"Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_causal_conv1d_with_dilation() -> None:
    """Test CausalConv1d with dilation > 1."""
    from fish_speech.models.dac.rvq import CausalConvNet

    batch_size = 2
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    stride = 1
    dilation = 2
    groups = 1
    seq_length = 50

    torch_conv = CausalConvNet(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    torch_conv.eval()

    lalamo_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    lalamo_conv = lalamo_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    torch_weights = torch_conv.conv.weight.detach()
    torch_biases = torch_conv.conv.bias.detach()

    lalamo_weights = {
        "weights": torch_to_jax(torch_weights),
        "biases": torch_to_jax(torch_biases),
    }
    lalamo_conv = lalamo_conv.import_weights(lalamo_weights)

    torch.manual_seed(123)
    test_input_torch = torch.randn(batch_size, in_channels, seq_length)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    torch_output = torch_conv(test_input_torch)
    lalamo_output = lalamo_conv(test_input_jax)

    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"Dilated conv - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"Dilated conv - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Dilated conv - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5)


@torch.no_grad
def test_causal_conv1d_grouped() -> None:
    """Test CausalConv1d with grouped convolution (depthwise)."""
    from fish_speech.models.dac.rvq import CausalConvNet

    batch_size = 2
    channels = 64
    kernel_size = 7
    stride = 1
    dilation = 1
    groups = channels  # Depthwise convolution
    seq_length = 100

    torch_conv = CausalConvNet(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    torch_conv.eval()

    lalamo_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    lalamo_conv = lalamo_config.empty(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    torch_weights = torch_conv.conv.weight.detach()
    torch_biases = torch_conv.conv.bias.detach()

    lalamo_weights = {
        "weights": torch_to_jax(torch_weights),
        "biases": torch_to_jax(torch_biases),
    }
    lalamo_conv = lalamo_conv.import_weights(lalamo_weights)

    torch.manual_seed(456)
    test_input_torch = torch.randn(batch_size, channels, seq_length)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    torch_output = torch_conv(test_input_torch)
    lalamo_output = lalamo_conv(test_input_jax)

    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"Grouped conv - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"Grouped conv - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Grouped conv - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5)


@torch.no_grad
def test_causal_transpose_conv1d_matches_pytorch() -> None:
    """Test that Lalamo CausalTransposeConv1d matches PyTorch CausalTransConvNet."""
    from fish_speech.models.dac.rvq import CausalTransConvNet

    # Test parameters
    batch_size = 2
    in_channels = 128
    out_channels = 64
    kernel_size = 4
    stride = 2
    dilation = 1
    seq_length = 50

    # Create PyTorch module
    torch_conv = CausalTransConvNet(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )
    torch_conv.eval()

    # Create Lalamo module with same config
    lalamo_config = CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)
    lalamo_conv = lalamo_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    # Extract weights from PyTorch and load into Lalamo
    # PyTorch ConvTranspose1d weight shape: (in_channels, out_channels, kernel_size)
    torch_weights = torch_conv.conv.weight.detach()
    torch_biases = torch_conv.conv.bias.detach()

    lalamo_weights = {
        "weights": torch_to_jax(torch_weights),
        "biases": torch_to_jax(torch_biases),
    }
    lalamo_conv = lalamo_conv.import_weights(lalamo_weights)

    # Create test input: PyTorch uses (batch, channels, sequence)
    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, in_channels, seq_length)
    # JAX module uses (batch, sequence, channels) - transpose for JAX
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both implementations
    torch_output = torch_conv(test_input_torch)
    lalamo_output = lalamo_conv(test_input_jax)

    # Compare outputs - transpose JAX output back to (batch, channels, sequence) for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"PyTorch CausalTransConvNet output shape: {torch_output.shape}")
    _testlog.info(f"Lalamo CausalTransposeConv1d output shape: {lalamo_output.shape}")
    _testlog.info(f"Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_causal_transpose_conv1d_various_strides() -> None:
    """Test CausalTransposeConv1d with various stride values."""
    from fish_speech.models.dac.rvq import CausalTransConvNet

    batch_size = 2
    in_channels = 64
    out_channels = 32
    seq_length = 25

    for kernel_size, stride in [(2, 2), (4, 2), (4, 4), (8, 4)]:
        torch_conv = CausalTransConvNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
        )
        torch_conv.eval()

        lalamo_config = CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)
        lalamo_conv = lalamo_config.empty(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
        )

        torch_weights = torch_conv.conv.weight.detach()
        torch_biases = torch_conv.conv.bias.detach()

        lalamo_weights = {
            "weights": torch_to_jax(torch_weights),
            "biases": torch_to_jax(torch_biases),
        }
        lalamo_conv = lalamo_conv.import_weights(lalamo_weights)

        torch.manual_seed(42)
        test_input_torch = torch.randn(batch_size, in_channels, seq_length)
        test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

        torch_output = torch_conv(test_input_torch)
        lalamo_output = lalamo_conv(test_input_jax)

        torch_output_jax = torch_to_jax(torch_output)
        lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

        _testlog.info(
            f"TransConv kernel={kernel_size}, stride={stride} - shapes: {torch_output.shape} vs {lalamo_output.shape}"
        )
        _testlog.info(
            f"TransConv kernel={kernel_size}, stride={stride} - "
            f"max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
        )

        assert torch_output_jax.shape == lalamo_output_nct.shape, (
            f"Shape mismatch for kernel={kernel_size}, stride={stride}: "
            f"PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
        )
        assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
            f"Output mismatch for kernel={kernel_size}, stride={stride}. "
            f"Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
        )


@torch.no_grad
def test_causal_conv_transpose_roundtrip() -> None:
    """Test that conv followed by transpose conv produces expected output length."""
    from fish_speech.models.dac.rvq import CausalConvNet, CausalTransConvNet

    batch_size = 2
    channels = 64
    kernel_size = 4
    stride = 2
    seq_length = 100

    # PyTorch roundtrip
    torch_conv = CausalConvNet(channels, channels, kernel_size, stride=stride)
    torch_trans = CausalTransConvNet(channels, channels, kernel_size, stride=stride)
    torch_conv.eval()
    torch_trans.eval()

    # Lalamo roundtrip
    conv_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    trans_config = CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)

    lalamo_conv = conv_config.empty(channels, channels, kernel_size, stride=stride)
    lalamo_trans = trans_config.empty(channels, channels, kernel_size, stride=stride)

    # Load weights
    lalamo_conv = lalamo_conv.import_weights(
        {
            "weights": torch_to_jax(torch_conv.conv.weight.detach()),
            "biases": torch_to_jax(torch_conv.conv.bias.detach()),
        }
    )
    lalamo_trans = lalamo_trans.import_weights(
        {
            "weights": torch_to_jax(torch_trans.conv.weight.detach()),
            "biases": torch_to_jax(torch_trans.conv.bias.detach()),
        }
    )

    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, channels, seq_length)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Forward pass through both
    torch_down = torch_conv(test_input_torch)
    torch_up = torch_trans(torch_down)

    lalamo_down = lalamo_conv(test_input_jax)
    lalamo_up = lalamo_trans(lalamo_down)

    # Compare intermediate and final outputs - transpose JAX outputs for comparison
    torch_down_jax = torch_to_jax(torch_down)
    torch_up_jax = torch_to_jax(torch_up)
    lalamo_down_nct = lalamo_down.transpose(0, 2, 1)
    lalamo_up_nct = lalamo_up.transpose(0, 2, 1)

    _testlog.info(f"Roundtrip - Input shape: {test_input_torch.shape}")
    _testlog.info(f"Roundtrip - After conv: PyTorch {torch_down.shape}, Lalamo {lalamo_down.shape}")
    _testlog.info(f"Roundtrip - After trans: PyTorch {torch_up.shape}, Lalamo {lalamo_up.shape}")

    assert torch_down_jax.shape == lalamo_down_nct.shape
    assert torch_up_jax.shape == lalamo_up_nct.shape
    assert jnp.allclose(torch_down_jax, lalamo_down_nct, atol=1e-5)
    assert jnp.allclose(torch_up_jax, lalamo_up_nct, atol=1e-5)


@torch.no_grad
def test_convnext_block_matches_pytorch() -> None:
    """Test that Lalamo ConvNeXtBlock matches PyTorch ConvNeXtBlock."""
    from fish_speech.models.dac.rvq import ConvNeXtBlock as PyTorchConvNeXtBlock

    batch_size = 2
    dim = 64
    kernel_size = 7
    dilation = 1
    mlp_ratio = 4.0
    layer_scale_init_value = 1e-6
    seq_length = 50

    # Create PyTorch module
    torch_block = PyTorchConvNeXtBlock(
        dim=dim,
        layer_scale_init_value=layer_scale_init_value,
        mlp_ratio=mlp_ratio,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    torch_block.eval()

    # Create Lalamo module
    lalamo_config = ConvNeXtBlockConfig(
        precision=jnp.float32,
        activation=GELU(),
    )
    spatial_params = ConvNeXtSpatialParams(
        mlp_ratio=mlp_ratio,
        kernel_size=kernel_size,
        dilation=dilation,
        layer_scale_init_value=layer_scale_init_value,
    )
    lalamo_block = lalamo_config.empty(dim=dim, spatial_params=spatial_params)

    # Extract weights from PyTorch and map to Lalamo format
    # dwconv weights: (out_channels, in_channels/groups, kernel_size)
    dwconv_weights = torch_block.dwconv.conv.weight.detach()
    dwconv_biases = torch_block.dwconv.conv.bias.detach()

    # LayerNorm: weight (Lalamo LayerNorm only does element-wise mult, uses "scales")
    # Note: PyTorch LayerNorm bias is not used in Lalamo's simplified LayerNorm
    norm_scale = torch_block.norm.weight.detach()
    norm_bias = torch_block.norm.bias.detach()

    # pwconv1: Linear (dim -> hidden_dim)
    pwconv1_weights = torch_block.pwconv1.weight.detach()  # (hidden_dim, dim)
    pwconv1_biases = torch_block.pwconv1.bias.detach()

    # pwconv2: Linear (hidden_dim -> dim)
    pwconv2_weights = torch_block.pwconv2.weight.detach()  # (dim, hidden_dim)
    pwconv2_biases = torch_block.pwconv2.bias.detach()

    # gamma (layer scale) - Lalamo uses LayerNorm with "scales" key
    gamma = torch_block.gamma.detach() if torch_block.gamma is not None else None

    lalamo_weights = {
        "dwconv": {
            "weights": torch_to_jax(dwconv_weights),
            "biases": torch_to_jax(dwconv_biases),
        },
        "norm": {"scales": torch_to_jax(norm_scale), "bias": torch_to_jax(norm_bias)},
        "pwconv1": {
            "weights": torch_to_jax(pwconv1_weights),
            "biases": torch_to_jax(pwconv1_biases),
        },
        "pwconv2": {
            "weights": torch_to_jax(pwconv2_weights),
            "biases": torch_to_jax(pwconv2_biases),
        },
    }
    if gamma is not None:
        lalamo_weights["gamma"] = {"scales": torch_to_jax(gamma)}

    lalamo_block = lalamo_block.import_weights(lalamo_weights)

    # Create test input: PyTorch uses (batch, channels, sequence)
    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, dim, seq_length)
    # JAX uses (batch, sequence, channels)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both
    torch_output = torch_block(test_input_torch, apply_residual=True)
    lalamo_output = lalamo_block(test_input_jax, apply_residual=True)

    # Compare - transpose JAX output back for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"ConvNeXtBlock - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"ConvNeXtBlock - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"ConvNeXtBlock - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_upsampling_block_matches_pytorch() -> None:
    """Test that Lalamo UpsamplingBlock matches PyTorch upsampling block from DAC model.

    This test loads a real DAC model checkpoint, extracts the first upsampling block,
    transfers weights to the Lalamo implementation, and compares outputs.
    """
    from fish_speech.models.dac import inference as fish_dac_inference
    from fish_speech.models.dac.modded_dac import DAC

    # Load DAC model
    fish_audiod_repo_id = "fishaudio/openaudio-s1-mini"
    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

    api = HfApi()
    cache_info = api.model_info(fish_audiod_repo_id)
    commit_hash = cache_info.sha

    model_path = fish_audio_model_info.repo_path / "snapshots" / str(commit_hash)
    audio_chkpt_path = model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    model = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(model, DAC)

    config = get_default_fishaudio_dac_config()
    input_dim = config["quantizer"]["input_dim"]
    downsample_factor = config["quantizer"]["downsample_factor"]
    downsample_dims = config["quantizer"].get("downsample_dims") or [input_dim] * len(downsample_factor)

    # Build all_dims: (input_dim,) + tuple(downsample_dims)
    all_dims = (input_dim,) + tuple(downsample_dims)

    # Extract the first upsampling block from the quantizer.
    # Upsample blocks are in reversed order of downsample factors.
    fish_upsampler_block = model.quantizer.upsample[0]

    # Determine dimensions for the first upsample block
    reversed_indices = list(reversed(list(enumerate(downsample_factor))))
    idx, factor = reversed_indices[0]
    in_channels = all_dims[idx + 1]
    out_channels = all_dims[idx]
    upsample_kernel_size = factor
    upsample_stride = factor

    _testlog.info(f"UpsamplingBlock config: in={in_channels}, out={out_channels}, kernel={upsample_kernel_size}")

    # Create Lalamo UpsamplingBlock
    lalamo_config = UpsamplingBlockConfig(precision=jnp.float32)
    trans_conv_params = TransposeConvSpatialParams(
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_kernel_size=upsample_kernel_size,
        upsample_stride=upsample_stride,
    )
    convnext_spatial_params = ConvNeXtSpatialParams(
        mlp_ratio=4.0,
        kernel_size=7,
        dilation=1,
        layer_scale_init_value=1e-6,
    )
    lalamo_block = lalamo_config.empty(
        trans_conv_params=trans_conv_params,
        convnext_spatial_params=convnext_spatial_params,
    )

    # Extract weights from PyTorch upsampler block (Sequential of CausalTransConvNet, ConvNeXtBlock)
    torch_trans_conv = fish_upsampler_block[0]
    torch_convnext = fish_upsampler_block[1]

    # Trans conv weights
    trans_conv_weights = torch_trans_conv.conv.weight.detach()
    trans_conv_biases = torch_trans_conv.conv.bias.detach()

    # ConvNeXt weights
    dwconv_weights = torch_convnext.dwconv.conv.weight.detach()
    dwconv_biases = torch_convnext.dwconv.conv.bias.detach()
    norm_scale = torch_convnext.norm.weight.detach()
    norm_bias = torch_convnext.norm.bias.detach()
    pwconv1_weights = torch_convnext.pwconv1.weight.detach()
    pwconv1_biases = torch_convnext.pwconv1.bias.detach()
    pwconv2_weights = torch_convnext.pwconv2.weight.detach()
    pwconv2_biases = torch_convnext.pwconv2.bias.detach()
    gamma = torch_convnext.gamma.detach() if torch_convnext.gamma is not None else None

    lalamo_weights = {
        "trans_conv": {
            "weights": torch_to_jax(trans_conv_weights),
            "biases": torch_to_jax(trans_conv_biases),
        },
        "convnext": {
            "dwconv": {
                "weights": torch_to_jax(dwconv_weights),
                "biases": torch_to_jax(dwconv_biases),
            },
            "norm": {"scales": torch_to_jax(norm_scale), "bias": torch_to_jax(norm_bias)},
            "pwconv1": {
                "weights": torch_to_jax(pwconv1_weights),
                "biases": torch_to_jax(pwconv1_biases),
            },
            "pwconv2": {
                "weights": torch_to_jax(pwconv2_weights),
                "biases": torch_to_jax(pwconv2_biases),
            },
        },
    }
    if gamma is not None:
        lalamo_weights["convnext"]["gamma"] = {"scales": torch_to_jax(gamma)}

    # Create test input
    batch_size = 1
    seq_length = 10

    torch.manual_seed(42)
    # PyTorch uses (batch, channels, sequence)
    test_input_torch = torch.randn(batch_size, in_channels, seq_length)
    # JAX uses (batch, sequence, channels)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both
    torch_output = fish_upsampler_block(test_input_torch)
    with jax.disable_jit():
        lalamo_block = lalamo_block.import_weights(lalamo_weights)
        lalamo_output = lalamo_block(test_input_jax)

    # Compare - transpose JAX output back for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"UpsamplingBlock - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"UpsamplingBlock - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"UpsamplingBlock - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-4), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_upsampler_matches_pytorch() -> None:
    """Test that Lalamo Upsampler matches the full DAC quantizer upsampler.

    This test loads a real DAC model checkpoint, extracts all upsampling blocks,
    transfers weights to the Lalamo Upsampler, and compares outputs.
    """
    from fish_speech.models.dac import inference as fish_dac_inference
    from fish_speech.models.dac.modded_dac import DAC

    # Load DAC model
    fish_audiod_repo_id = "fishaudio/openaudio-s1-mini"
    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

    api = HfApi()
    cache_info = api.model_info(fish_audiod_repo_id)
    commit_hash = cache_info.sha

    model_path = fish_audio_model_info.repo_path / "snapshots" / str(commit_hash)
    audio_chkpt_path = model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    model = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(model, DAC)

    config = get_default_fishaudio_dac_config()
    input_dim = config["quantizer"]["input_dim"]
    downsample_factor = config["quantizer"]["downsample_factor"]
    downsample_dims = config["quantizer"].get("downsample_dims") or [input_dim] * len(downsample_factor)

    # Build all_dims: (input_dim,) + tuple(downsample_dims)
    all_dims = (input_dim,) + tuple(downsample_dims)

    # Get the full upsampler from DAC
    fish_upsampler = model.quantizer.upsample
    num_blocks = len(fish_upsampler)

    _testlog.info(f"Upsampler has {num_blocks} blocks")

    # Build block parameters in the same order as PyTorch
    # upsample is built with reversed(enumerate(downsample_factor))
    reversed_indices = list(reversed(list(enumerate(downsample_factor))))
    block_params: list[TransposeConvSpatialParams] = []
    for idx, factor in reversed_indices:
        in_channels = all_dims[idx + 1]
        out_channels = all_dims[idx]
        kernel_size = factor
        stride = factor
        block_params.append(
            TransposeConvSpatialParams(
                in_channels=in_channels,
                out_channels=out_channels,
                upsample_kernel_size=kernel_size,
                upsample_stride=stride,
            ),
        )
        _testlog.info(
            f"Block {len(block_params) - 1}: in={in_channels}, out={out_channels}, k={kernel_size}, s={stride}"
        )

    # Create Lalamo Upsampler config - one UpsamplingBlockConfig per block
    block_configs = tuple(UpsamplingBlockConfig(precision=jnp.float32) for _ in range(num_blocks))
    convnext_spatial_params = ConvNeXtSpatialParams(
        mlp_ratio=4.0,
        kernel_size=7,
        dilation=1,
        layer_scale_init_value=1e-6,
    )
    upsampler_config = UpsamplerConfig(block_configs=block_configs)
    lalamo_upsampler = upsampler_config.empty(
        trans_conv_params_per_block=tuple(block_params),
        convnext_spatial_params=convnext_spatial_params,
    )

    # Extract weights from all PyTorch upsampler blocks
    blocks_weights = []
    for fish_block in fish_upsampler:
        torch_trans_conv = fish_block[0]
        torch_convnext = fish_block[1]

        # Trans conv weights
        trans_conv_weights = torch_trans_conv.conv.weight.detach()
        trans_conv_biases = torch_trans_conv.conv.bias.detach()

        # ConvNeXt weights
        dwconv_weights = torch_convnext.dwconv.conv.weight.detach()
        dwconv_biases = torch_convnext.dwconv.conv.bias.detach()
        norm_scale = torch_convnext.norm.weight.detach()
        norm_bias = torch_convnext.norm.bias.detach()
        pwconv1_weights = torch_convnext.pwconv1.weight.detach()
        pwconv1_biases = torch_convnext.pwconv1.bias.detach()
        pwconv2_weights = torch_convnext.pwconv2.weight.detach()
        pwconv2_biases = torch_convnext.pwconv2.bias.detach()
        gamma = torch_convnext.gamma.detach() if torch_convnext.gamma is not None else None

        block_weight = {
            "trans_conv": {
                "weights": torch_to_jax(trans_conv_weights),
                "biases": torch_to_jax(trans_conv_biases),
            },
            "convnext": {
                "dwconv": {
                    "weights": torch_to_jax(dwconv_weights),
                    "biases": torch_to_jax(dwconv_biases),
                },
                "norm": {"scales": torch_to_jax(norm_scale), "bias": torch_to_jax(norm_bias)},
                "pwconv1": {
                    "weights": torch_to_jax(pwconv1_weights),
                    "biases": torch_to_jax(pwconv1_biases),
                },
                "pwconv2": {
                    "weights": torch_to_jax(pwconv2_weights),
                    "biases": torch_to_jax(pwconv2_biases),
                },
            },
        }
        if gamma is not None:
            block_weight["convnext"]["gamma"] = {"scales": torch_to_jax(gamma)}

        blocks_weights.append(block_weight)

    lalamo_weights = {"blocks": blocks_weights}

    # Create test input - use the input dimension of the first block (deepest level)
    batch_size = 1
    seq_length = 10
    first_in_channels = block_params[0].in_channels

    torch.manual_seed(42)
    # PyTorch uses (batch, channels, sequence)
    test_input_torch = torch.randn(batch_size, first_in_channels, seq_length)
    # JAX uses (batch, sequence, channels)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both
    torch_output = fish_upsampler(test_input_torch)

    with jax.disable_jit():
        lalamo_upsampler = lalamo_upsampler.import_weights(lalamo_weights)
        lalamo_output = lalamo_upsampler(test_input_jax)

    # Compare - transpose JAX output back for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"Upsampler - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"Upsampler - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Upsampler - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-3), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_snake1d_matches_pytorch() -> None:
    """Test that Lalamo Snake1d matches PyTorch Snake1d."""
    from dac.nn.layers import Snake1d as PyTorchSnake1d

    batch_size = 2
    channels = 64
    seq_length = 100

    # Create PyTorch module
    torch_snake = PyTorchSnake1d(channels)
    torch_snake.eval()

    # Create Lalamo module
    lalamo_config = Snake1dConfig(precision=jnp.float32)
    lalamo_snake = lalamo_config.empty(channels)

    # Extract weights from PyTorch: alpha is (1, channels, 1) in PyTorch
    torch_alpha = torch_snake.alpha.detach().squeeze()  # (channels,)

    lalamo_weights = {"alpha": torch_to_jax(torch_alpha)}
    lalamo_snake = lalamo_snake.import_weights(lalamo_weights)

    # Create test input: PyTorch uses (batch, channels, sequence)
    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, channels, seq_length)
    # JAX uses (batch, sequence, channels)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both
    torch_output = torch_snake(test_input_torch)
    lalamo_output = lalamo_snake(test_input_jax)

    # Compare - transpose JAX output back for comparison
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"Snake1d - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"Snake1d - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Snake1d - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_residual_unit_matches_pytorch() -> None:
    """Test that Lalamo ResidualUnit matches PyTorch ResidualUnit."""
    from fish_speech.models.dac.modded_dac import ResidualUnit as PyTorchResidualUnit

    batch_size = 2
    dim = 64
    dilation = 3
    seq_length = 100

    # Create PyTorch module (causal=True)
    torch_res_unit = PyTorchResidualUnit(dim=dim, dilation=dilation, causal=True)
    torch_res_unit.eval()

    # Create Lalamo module
    lalamo_config = ResidualUnitConfig(precision=jnp.float32, causal=True)
    spatial_params = ResidualUnitSpatialParams(dilation=dilation, kernel_size=7)
    lalamo_res_unit = lalamo_config.empty(dim=dim, spatial_params=spatial_params)

    # Extract weights from PyTorch
    # block is Sequential(Snake1d, Conv, Snake1d, Conv)
    torch_block = torch_res_unit.block

    # Snake1d weights
    snake1_alpha = torch_block[0].alpha.detach().squeeze()  # (channels,)
    snake2_alpha = torch_block[2].alpha.detach().squeeze()  # (channels,)

    # Conv weights (CausalWNConv1d wraps nn.Conv1d)
    conv1_weights = torch_block[1].conv.weight.detach()
    conv1_biases = torch_block[1].conv.bias.detach()
    conv2_weights = torch_block[3].conv.weight.detach()
    conv2_biases = torch_block[3].conv.bias.detach()

    lalamo_weights = {
        "snake1": {"alpha": torch_to_jax(snake1_alpha)},
        "conv1": {
            "weights": torch_to_jax(conv1_weights),
            "biases": torch_to_jax(conv1_biases),
        },
        "snake2": {"alpha": torch_to_jax(snake2_alpha)},
        "conv2": {
            "weights": torch_to_jax(conv2_weights),
            "biases": torch_to_jax(conv2_biases),
        },
    }
    lalamo_res_unit = lalamo_res_unit.import_weights(lalamo_weights)

    # Create test input
    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, dim, seq_length)
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

    # Run both
    torch_output = torch_res_unit(test_input_torch)
    lalamo_output = lalamo_res_unit(test_input_jax)

    # Compare
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"ResidualUnit - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"ResidualUnit - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"ResidualUnit - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-5), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_decoder_block_matches_pytorch() -> None:
    """Test that Lalamo DecoderBlock matches PyTorch DecoderBlock."""
    from fish_speech.models.dac.modded_dac import DecoderBlock as PyTorchDecoderBlock

    batch_size = 2
    input_dim = 128
    output_dim = 64
    stride = 4
    seq_length = 25

    # Create PyTorch module (causal=True)
    torch_decoder_block = PyTorchDecoderBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        stride=stride,
        causal=True,
        n_t_layer=0,  # No transformer
    )
    torch_decoder_block.eval()

    # Create Lalamo module
    lalamo_config = DACDecoderBlockConfig(precision=jnp.float32, causal=True)
    spatial_params = AudioDecoderBlockSpatialParams(
        input_dim=input_dim,
        output_dim=output_dim,
        stride=stride,
    )
    lalamo_decoder_block = lalamo_config.empty(spatial_params=spatial_params)

    # Extract weights from PyTorch
    # block is Sequential(Snake1d, TransConv, ResUnit, ResUnit, ResUnit)
    torch_block = torch_decoder_block.block

    # Snake weights
    snake_alpha = torch_block[0].alpha.detach().squeeze()

    # TransConv weights
    trans_conv_weights = torch_block[1].conv.weight.detach()
    trans_conv_biases = torch_block[1].conv.bias.detach()

    def extract_res_unit_weights(res_unit):
        block = res_unit.block
        return {
            "snake1": {"alpha": torch_to_jax(block[0].alpha.detach().squeeze())},
            "conv1": {
                "weights": torch_to_jax(block[1].conv.weight.detach()),
                "biases": torch_to_jax(block[1].conv.bias.detach()),
            },
            "snake2": {"alpha": torch_to_jax(block[2].alpha.detach().squeeze())},
            "conv2": {
                "weights": torch_to_jax(block[3].conv.weight.detach()),
                "biases": torch_to_jax(block[3].conv.bias.detach()),
            },
        }

    lalamo_weights = {
        "snake": {"alpha": torch_to_jax(snake_alpha)},
        "trans_conv": {
            "weights": torch_to_jax(trans_conv_weights),
            "biases": torch_to_jax(trans_conv_biases),
        },
        "res_unit1": extract_res_unit_weights(torch_block[2]),
        "res_unit2": extract_res_unit_weights(torch_block[3]),
        "res_unit3": extract_res_unit_weights(torch_block[4]),
    }

    with jax.disable_jit():
        lalamo_decoder_block = lalamo_decoder_block.import_weights(lalamo_weights)

        # Create test input
        torch.manual_seed(42)
        test_input_torch = torch.randn(batch_size, input_dim, seq_length)
        test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

        # Run both
        torch_output = torch_decoder_block(test_input_torch)
        lalamo_output = lalamo_decoder_block(test_input_jax)

    # Compare
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"DecoderBlock - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"DecoderBlock - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"DecoderBlock - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-4), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_audio_decoder_matches_pytorch() -> None:
    """Test that Lalamo AudioDecoder matches PyTorch Decoder."""
    from fish_speech.models.dac.modded_dac import Decoder as PyTorchDecoder

    batch_size = 1
    input_channel = 512  # latent dim from quantizer
    channels = 1536  # decoder_dim
    rates = (8, 8, 4, 2)  # upsampling rates
    d_out = 1
    seq_length = 10  # Short sequence for testing

    # Create PyTorch module (causal=True, no transformers)
    torch_decoder = PyTorchDecoder(
        input_channel=input_channel,
        channels=channels,
        rates=list(rates),
        d_out=d_out,
        causal=True,
        n_transformer_layers=[0, 0, 0, 0],
    )
    torch_decoder.eval()

    # Create Lalamo module
    lalamo_config = DACDecoderConfig(precision=jnp.float32, causal=True)
    spatial_params = DACDecoderSpatialParams(
        input_channel=input_channel,
        channels=channels,
        rates=rates,
        d_out=d_out,
    )
    lalamo_decoder = lalamo_config.empty(spatial_params=spatial_params)

    # Extract weights from PyTorch Decoder
    # The model is a Sequential: [first_conv, DecoderBlock..., final_snake, final_conv, Tanh]
    torch_model = torch_decoder.model

    # First conv (index 0)
    first_conv_weights = torch_model[0].conv.weight.detach()
    first_conv_biases = torch_model[0].conv.bias.detach()

    # DecoderBlocks (indices 1 to len(rates))
    def extract_decoder_block_weights(decoder_block):
        block = decoder_block.block
        # block is Sequential(Snake1d, TransConv, ResUnit, ResUnit, ResUnit)
        snake_alpha = block[0].alpha.detach().squeeze()
        trans_conv_weights = block[1].conv.weight.detach()
        trans_conv_biases = block[1].conv.bias.detach()

        def extract_res_unit_weights(res_unit):
            res_block = res_unit.block
            return {
                "snake1": {"alpha": torch_to_jax(res_block[0].alpha.detach().squeeze())},
                "conv1": {
                    "weights": torch_to_jax(res_block[1].conv.weight.detach()),
                    "biases": torch_to_jax(res_block[1].conv.bias.detach()),
                },
                "snake2": {"alpha": torch_to_jax(res_block[2].alpha.detach().squeeze())},
                "conv2": {
                    "weights": torch_to_jax(res_block[3].conv.weight.detach()),
                    "biases": torch_to_jax(res_block[3].conv.bias.detach()),
                },
            }

        return {
            "snake": {"alpha": torch_to_jax(snake_alpha)},
            "trans_conv": {
                "weights": torch_to_jax(trans_conv_weights),
                "biases": torch_to_jax(trans_conv_biases),
            },
            "res_unit1": extract_res_unit_weights(block[2]),
            "res_unit2": extract_res_unit_weights(block[3]),
            "res_unit3": extract_res_unit_weights(block[4]),
        }

    decoder_blocks_weights = []
    for i in range(len(rates)):
        decoder_blocks_weights.append(extract_decoder_block_weights(torch_model[1 + i]))

    # Final snake (index 1 + len(rates))
    final_snake_alpha = torch_model[1 + len(rates)].alpha.detach().squeeze()

    # Final conv (index 2 + len(rates))
    final_conv_weights = torch_model[2 + len(rates)].conv.weight.detach()
    final_conv_biases = torch_model[2 + len(rates)].conv.bias.detach()

    lalamo_weights = {
        "first_conv": {
            "weights": torch_to_jax(first_conv_weights),
            "biases": torch_to_jax(first_conv_biases),
        },
        "decoder_blocks": decoder_blocks_weights,
        "final_snake": {"alpha": torch_to_jax(final_snake_alpha)},
        "final_conv": {
            "weights": torch_to_jax(final_conv_weights),
            "biases": torch_to_jax(final_conv_biases),
        },
    }

    with jax.disable_jit():
        lalamo_decoder = lalamo_decoder.import_weights(lalamo_weights)

        # Create test input
        torch.manual_seed(42)
        test_input_torch = torch.randn(batch_size, input_channel, seq_length)
        test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)

        # Run both
        torch_output = torch_decoder(test_input_torch)
        lalamo_output = lalamo_decoder(test_input_jax)

    # Compare
    torch_output_jax = torch_to_jax(torch_output)
    lalamo_output_nct = lalamo_output.transpose(0, 2, 1)

    _testlog.info(f"AudioDecoder - PyTorch output shape: {torch_output.shape}")
    _testlog.info(f"AudioDecoder - Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"AudioDecoder - Max difference: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}")

    assert torch_output_jax.shape == lalamo_output_nct.shape, (
        f"Shape mismatch: PyTorch {torch_output_jax.shape} vs Lalamo {lalamo_output_nct.shape}"
    )
    assert jnp.allclose(torch_output_jax, lalamo_output_nct, atol=1e-4), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}"
    )


@torch.no_grad
def test_dac_matches_pytorch() -> None:
    """Test that Lalamo DAC matches PyTorch DAC from FishAudio.

    This test loads a real DAC model checkpoint, creates a Lalamo DAC module
    using from_foreign_model(), and compares full inference (codes -> audio)
    between both implementations.
    """
    from fish_speech.models.dac import inference as fish_dac_inference
    from fish_speech.models.dac.modded_dac import DAC as FishDAC

    # Load FishAudio DAC model
    fish_audiod_repo_id = "fishaudio/openaudio-s1-mini"
    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_model_info = next(filter(lambda repo: repo.repo_id == fish_audiod_repo_id, repos))

    api = HfApi()
    cache_info = api.model_info(fish_audiod_repo_id)
    commit_hash = cache_info.sha

    model_path = fish_audio_model_info.repo_path / "snapshots" / str(commit_hash)
    audio_chkpt_path = model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    fish_dac = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(fish_dac, FishDAC)
    fish_dac.eval()

    # Load Lalamo DAC using from_foreign_model
    precision = jnp.float32
    lalamo_dac: DescriptAudioCodec = DescriptAudioCodecConfig.from_foreign_model(audio_chkpt_path, precision)

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

    _testlog.info(f"Test codes shape: {test_codes_torch.shape}")

    # Run FishAudio DAC inference (quantizer.decode + decoder)
    z_fish = fish_dac.quantizer.decode(test_codes_torch)  # (batch, latent_dim, tokens_upsampled)
    audio_fish = fish_dac.decoder(z_fish)  # (batch, 1, audio_samples)

    _testlog.info(f"FishAudio z shape: {z_fish.shape}")
    _testlog.info(f"FishAudio audio shape: {audio_fish.shape}")

    # Run Lalamo DAC inference
    audio_lalamo = lalamo_dac(test_codes_jax)  # (batch, audio_samples, 1) - NTC format

    _testlog.info(f"Lalamo audio shape: {audio_lalamo.shape}")

    # Convert for comparison (both to NTC format)
    audio_fish_ntc = torch_to_jax(audio_fish).transpose(0, 2, 1)  # NCT -> NTC

    # Compare final audio outputs
    audio_diff = audio_lalamo - audio_fish_ntc

    _testlog.info(
        f"DAC - Fish audio: shape={audio_fish.shape}, max={audio_fish.max():.6f}, "
        f"min={audio_fish.min():.6f}, avg={audio_fish.mean():.6f}"
    )
    _testlog.info(
        f"DAC - Lalamo audio: shape={audio_lalamo.shape}, max={audio_lalamo.max():.6f}, "
        f"min={audio_lalamo.min():.6f}, avg={audio_lalamo.mean():.6f}"
    )
    _testlog.info(
        f"DAC - audio diff: max={jnp.abs(audio_diff).max():.6f}, "
        f"avg={audio_diff.mean():.6f}, std={audio_diff.std():.6f}"
    )

    assert audio_fish_ntc.shape == audio_lalamo.shape, (
        f"Shape mismatch: FishAudio {audio_fish_ntc.shape} vs Lalamo {audio_lalamo.shape}"
    )
    assert jnp.allclose(audio_fish_ntc, audio_lalamo, atol=1e-3), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(audio_diff))}"
    )


def test_dtype_convert_roundtrip() -> None:
    """Test that DTypeConvert correctly converts dtypes between JAX and PyTorch."""
    # Test all supported dtypes: JAX -> PyTorch and back
    test_cases = [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("bfloat16", torch.bfloat16),
        ("int8", torch.int8),
        ("int16", torch.int16),
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("uint8", torch.uint8),
        ("bool", torch.bool),
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]

    for dtype_str, torch_dtype in test_cases:
        jax_dtype = jnp.dtype(dtype_str)

        # Test JAX dtype -> PyTorch
        assert DTypeConvert.to_torch(jax_dtype) == torch_dtype, f"Failed JAX->Torch for {dtype_str}"

        # Test PyTorch -> JAX
        assert DTypeConvert.to_jax(torch_dtype) == jax_dtype, f"Failed Torch->JAX for {dtype_str}"

        # Test string -> PyTorch
        assert DTypeConvert.to_torch(dtype_str) == torch_dtype, f"Failed str->Torch for {dtype_str}"

        # Test string -> JAX
        assert DTypeConvert.to_jax(dtype_str) == jax_dtype, f"Failed str->JAX for {dtype_str}"


@pytest.mark.skip(reason="Temporary test used for e2e TTS with FishAudio")
def test_fishaudio_lalamo_tts_generation(fish_audio_local_model_path: Path) -> None:
    """Test TTS generation using FISH_AUDIO_LALAMO preset."""
    model = TTSConfig.load_model_from_foreign_model_preset(
        ForeignTTSModel.FISH_AUDIO_LALAMO,
        fish_audio_local_model_path,
    )

    tts_message = TTSMessage.simple_message("Some text!")
    tts_result = model.generate_speech([tts_message])

    _testlog.info(f"Generated audio shape: {tts_result.audio.shape}")
    _testlog.info(f"Audio sample rate: {tts_result.audio_params.samplerate}")

    assert tts_result.audio is not None
    assert len(tts_result.audio) > 0
    assert tts_result.audio_params.samplerate > 0
