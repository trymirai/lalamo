import logging
from pathlib import Path

import huggingface_hub
import jax
import pytest
import torch
from fish_speech.tokenizer import FishTokenizer
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import vmap
from pytest import fixture

from lalamo.model_import.model_configs.huggingface.fishaudio import instantiate_dac_config_from_fishaudio_config
from lalamo.modules import GELU
from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
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
from lalamo.modules.audio.text_to_speech import TTSMessage
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.sampling import GreedyPolicy
from tests.tts.fishaudio.fishaudio_sampling import sampling_params_from_policy
from tests.tts.fishaudio.fishaudio_thin_wrapper import (
    FishAudioTextDecoder_Foreign,
)
from tests.tts.fishaudio.fishaudio_torch_stuff import FishAudioFromTorch

_testlog = logging.getLogger("tts_test_logger")


from fish_speech.models.dac import inference as fish_dac_inference
from fish_speech.models.dac.modded_dac import DAC as FishDAC


@fixture
def fish_audio_local_model_path() -> Path:
    fish_audio_repo_id = "fishaudio/s1-mini"

    repos = huggingface_hub.scan_cache_dir().repos
    fish_audio_candidate_repos = list(filter(lambda repo: repo.repo_id == fish_audio_repo_id, repos))
    if not fish_audio_candidate_repos:
        return Path(huggingface_hub.snapshot_download(fish_audio_repo_id))

    api = HfApi()
    cache_info = api.model_info(fish_audio_repo_id)

    for repo in fish_audio_candidate_repos:
        path = repo.repo_path / "snapshots" / str(cache_info.sha)

        if path.exists():
            return path

    pytest.skip("The cached fishaudio repository is in an invalid state")


def get_tts_message() -> TTSMessage:
    test_text = "this is a test message with speaker 0"
    return TTSMessage(content=test_text, speaker_id="speaker:0", style="interleave")


def test_fishaudio_text_tokenization(fish_audio_local_model_path: Path) -> None:
    with jax.disable_jit():
        tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(fish_audio_local_model_path)
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
    from lalamo.model_import.model_configs.huggingface.fishaudio import load_fishaudio_text_decoder

    from .fishaudio_torch_stuff import from_fish_audio_config, prepare_state_dict_for_lalamo_loaders

    tts_message = get_tts_message()

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
        tokenized_text, sampling_params=sampling_params_from_policy(sampling_policy),
    )

    # Run Lalamo model
    decode_result = lalamo_text_decoder(
        text_tokens=tokenized_text, input_pos=input_pos, sampling_policy=sampling_policy, key=key,
    )
    output_lalamo = decode_result.token_codes

    _testlog.info(f"output_pytorch: {output_pytorch}")
    _testlog.info(f"output_lalamo : {output_lalamo}")

    assert output_pytorch[:, 0].tolist() == output_lalamo[0].tolist()


def test_permute_for_rope_rotate_half() -> None:
    """Test that _permute_for_rope_rotate_half correctly reorders weights from interleaved to rotate-half format.

    Interleaved RoPE format: pairs are adjacent [x0, x1, x2, x3, ...]
        RoPE operates on pairs (x0, x1), (x2, x3), etc.

    Rotate-half format: evens first, then odds [x0, x2, ..., x1, x3, ...]
        RoPE operates by swapping halves.

    The function permutes rows of the weight matrix. For a single head with head_dim=8:
        Input rows:  [0, 1, 2, 3, 4, 5, 6, 7]  (interleaved pairs: 0-1, 2-3, 4-5, 6-7)
        Output rows: [0, 2, 4, 6, 1, 3, 5, 7]  (first half: 0,2,4,6; second half: 1,3,5,7)
    """
    import numpy as np

    from lalamo.model_import.loaders.fishaudio_loaders import _permute_for_rope_rotate_half

    # Single head, head_dim=8, in_features=3 (different from head_dim for clarity)
    num_heads = 1
    head_dim = 8
    in_features = 3
    out_features = num_heads * head_dim

    # Create weight matrix where each row is clearly identifiable
    # Using row_idx * 10 + col_idx pattern for traceability
    # Result looks like:
    #   [[ 0,  1,  2],   <- row 0
    #    [10, 11, 12],   <- row 1
    #    ...
    #    [70, 71, 72]]   <- row 7
    row_indices = jnp.arange(out_features)[:, None]  # (out_features, 1)
    col_indices = jnp.arange(in_features)[None, :]  # (1, in_features)
    weight = (row_indices * 10 + col_indices).astype(jnp.float32)

    result = _permute_for_rope_rotate_half(weight, num_heads, head_dim)

    # Manually trace through the transformation:
    # 1. Reshape (8, 3) -> (1, 4, 2, 3): groups pairs along axis 1, interleave index on axis 2
    #    [h=0, pair=0, interleave=0] = row 0
    #    [h=0, pair=0, interleave=1] = row 1
    #    [h=0, pair=1, interleave=0] = row 2
    #    [h=0, pair=1, interleave=1] = row 3
    #    [h=0, pair=2, interleave=0] = row 4
    #    [h=0, pair=2, interleave=1] = row 5
    #    [h=0, pair=3, interleave=0] = row 6
    #    [h=0, pair=3, interleave=1] = row 7
    # 2. Transpose (0,2,1,3) -> (1, 2, 4, 3): swap pair and interleave axes
    #    [h=0, interleave=0, pair=0..3] = rows 0,2,4,6  -> new rows 0,1,2,3
    #    [h=0, interleave=1, pair=0..3] = rows 1,3,5,7  -> new rows 4,5,6,7
    # 3. Reshape back (8, 3)
    # Expected row order: [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]

    print("\n=== Single head test ===")
    print(f"num_heads={num_heads}, head_dim={head_dim}, in_features={in_features}")
    print(f"\nInput weight matrix (interleaved format):\n{np.array(weight)}")
    print(f"\nOutput weight matrix (rotate-half format):\n{np.array(result)}")

    # Verify by checking which input row ended up in each output row
    for out_row in range(out_features):
        # Each row has unique first element = row_idx * 10
        first_elem = int(result[out_row, 0])
        source_row = first_elem // 10
        print(f"  Output row {out_row} <- Input row {source_row}")

    # Verify the transformation is correct:
    # First half (rows 0-3 in output) <- even input rows (0,2,4,6)
    # Second half (rows 4-7 in output) <- odd input rows (1,3,5,7)
    expected_mapping = [0, 2, 4, 6, 1, 3, 5, 7]
    for out_idx, in_idx in enumerate(expected_mapping):
        assert jnp.allclose(result[out_idx], weight[in_idx]), f"Output row {out_idx} should be input row {in_idx}"

    print("\n2D matrix permutation test passed!")

    # === Test 1D vector case ===
    # This is used for bias vectors where we have a single head_dim worth of values
    # The permutation is the same: interleaved pairs -> grouped halves
    print("\n=== 1D vector test ===")

    head_dim_1d = 8
    # Create 1D vector with traceable values: [0, 10, 20, 30, 40, 50, 60, 70]
    vector = (jnp.arange(head_dim_1d) * 10).astype(jnp.float32)

    result_1d = _permute_for_rope_rotate_half(vector, num_heads=1, head_dim=head_dim_1d)

    # Trace through transformation:
    # 1. Reshape (8,) -> (4, 2):
    #    [[0, 10], [20, 30], [40, 50], [60, 70]]
    # 2. Transpose (2, 4):
    #    [[0, 20, 40, 60], [10, 30, 50, 70]]
    # 3. Reshape (8,):
    #    [0, 20, 40, 60, 10, 30, 50, 70]
    # Expected index mapping: [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]

    print(f"head_dim={head_dim_1d}")
    print(f"\nInput vector (interleaved format):\n{np.array(vector)}")
    print(f"\nOutput vector (rotate-half format):\n{np.array(result_1d)}")

    # Verify by checking which input index ended up at each output index
    for out_idx in range(head_dim_1d):
        in_value = int(result_1d[out_idx])
        source_idx = in_value // 10
        print(f"  Output[{out_idx}] = {in_value} <- Input[{source_idx}]")

    # Same mapping as 2D case: [0,2,4,6,1,3,5,7]
    expected_mapping_1d = [0, 2, 4, 6, 1, 3, 5, 7]
    expected_1d = vector[jnp.array(expected_mapping_1d)]

    assert jnp.allclose(result_1d, expected_1d), (
        f"1D permutation incorrect.\nGot: {result_1d}\nExpected: {expected_1d}"
    )

    print("\n1D vector permutation test passed!")


@torch.no_grad
def test_vector_quantize_decode_code() -> None:
    from dac.nn.quantize import VectorQuantize as DACVectorQuantize

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_vector_quantize

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

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
    lalamo_vq_config = VectorQuantizeConfig(
        precision=jnp.float32,
        codebook_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=jnp.float32,
        ),
        out_proj_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    lalamo_vq = lalamo_vq_config.empty(
        input_dim=input_dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(dac_vq.state_dict(), prefix="vq")
    lalamo_vq = load_vector_quantize(lalamo_vq, weights_dict, ParameterPath("vq"))

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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_residual_vector_quantize

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

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

    vq_config = VectorQuantizeConfig(
        precision=jnp.float32,
        codebook_config=TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=jnp.float32,
        ),
        out_proj_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    lalamo_rvq_config = ResidualVectorQuantizeConfig(
        precision=jnp.float32,
        vq_config=vq_config,
    )
    lalamo_rvq = lalamo_rvq_config.empty(
        input_dim=input_dim,
        codebook_size=codebook_size,
        codebook_dim=[codebook_dim] * n_codebooks,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(dac_rvq.state_dict(), prefix="rvq")
    lalamo_rvq = load_residual_vector_quantize(lalamo_rvq, weights_dict, ParameterPath("rvq"))

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
            f"TransConv kernel={kernel_size}, stride={stride} - shapes: {torch_output.shape} vs {lalamo_output.shape}",
        )
        _testlog.info(
            f"TransConv kernel={kernel_size}, stride={stride} - "
            f"max diff: {jnp.max(jnp.abs(torch_output_jax - lalamo_output_nct))}",
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
        },
    )
    lalamo_trans = lalamo_trans.import_weights(
        {
            "weights": torch_to_jax(torch_trans.conv.weight.detach()),
            "biases": torch_to_jax(torch_trans.conv.bias.detach()),
        },
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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_convnext_block

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    batch_size = 2
    dim = 64
    kernel_size = 7
    dilation = 1
    mlp_ratio = 4.0
    layer_scale_init_value = 1e-6
    seq_length = 50

    torch_block = PyTorchConvNeXtBlock(
        dim=dim,
        layer_scale_init_value=layer_scale_init_value,
        mlp_ratio=mlp_ratio,
        kernel_size=kernel_size,
        dilation=dilation,
    )
    torch_block.eval()

    lalamo_config = ConvNeXtBlockConfig(
        precision=jnp.float32,
        activation=GELU(),
        dwconv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        norm_config=NormalizationConfig(
            scale_precision=jnp.float32,
            accumulation_precision=jnp.float32,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        ),
        pwconv_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    spatial_params = ConvNeXtSpatialParams(
        mlp_ratio=mlp_ratio,
        kernel_size=kernel_size,
        dilation=dilation,
        layer_scale_init_value=layer_scale_init_value,
    )
    lalamo_block = lalamo_config.empty(dim=dim, spatial_params=spatial_params)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_block.state_dict(), prefix="block")
    lalamo_block = load_convnext_block(lalamo_block, weights_dict, ParameterPath("block"))

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
def test_upsampling_block_matches_pytorch(fish_audio_local_model_path) -> None:
    """Test that Lalamo UpsamplingBlock matches PyTorch upsampling block from DAC model.

    This test loads a real DAC model checkpoint, extracts the first upsampling block,
    transfers weights to the Lalamo implementation, and compares outputs.
    """
    audio_chkpt_path = fish_audio_local_model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    model = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(model, FishDAC)

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
    convnext_config = ConvNeXtBlockConfig(
        precision=jnp.float32,
        activation=GELU(approximate=False),
        dwconv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        norm_config=NormalizationConfig(
            scale_precision=jnp.float32,
            accumulation_precision=jnp.float32,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        ),
        pwconv_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    lalamo_config = UpsamplingBlockConfig(
        precision=jnp.float32,
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        convnext_config=convnext_config,
    )
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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_upsampling_block

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    path = ParameterPath("block")
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_upsampler_block.state_dict(), prefix="block")

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
        lalamo_block = load_upsampling_block(lalamo_block, weights_dict, path)
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
def test_upsampler_matches_pytorch(fish_audio_local_model_path) -> None:
    """Test that Lalamo Upsampler matches the full DAC quantizer upsampler.

    This test loads a real DAC model checkpoint, extracts all upsampling blocks,
    transfers weights to the Lalamo Upsampler, and compares outputs.
    """
    audio_chkpt_path = fish_audio_local_model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    model = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(model, FishDAC)

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
            f"Block {len(block_params) - 1}: in={in_channels}, out={out_channels}, k={kernel_size}, s={stride}",
        )

    # Create Lalamo Upsampler config - one UpsamplingBlockConfig per block
    convnext_config = ConvNeXtBlockConfig(
        precision=jnp.float32,
        activation=GELU(approximate=False),
        dwconv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        norm_config=NormalizationConfig(
            scale_precision=jnp.float32,
            accumulation_precision=jnp.float32,
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            use_bias=True,
        ),
        pwconv_config=FullPrecisionLinearConfig(precision=jnp.float32),
    )
    upsampling_block_config = UpsamplingBlockConfig(
        precision=jnp.float32,
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        convnext_config=convnext_config,
    )
    block_configs = tuple(upsampling_block_config for _ in range(num_blocks))
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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_upsampler

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    path = ParameterPath("upsample")
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_upsampler.state_dict(), prefix="upsample")

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
        lalamo_upsampler = load_upsampler(lalamo_upsampler, weights_dict, path)
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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_snake1d

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    batch_size = 2
    channels = 64
    seq_length = 100

    # Create PyTorch module
    torch_snake = PyTorchSnake1d(channels)
    torch_snake.eval()

    # Create Lalamo module
    lalamo_config = Snake1dConfig(precision=jnp.float32)
    lalamo_snake = lalamo_config.empty(channels)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_snake.state_dict(), prefix="snake")
    lalamo_snake = load_snake1d(lalamo_snake, weights_dict, ParameterPath("snake"))

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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_residual_unit

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    batch_size = 2
    dim = 64
    dilation = 3
    seq_length = 100

    # Create PyTorch module (causal=True)
    torch_res_unit = PyTorchResidualUnit(dim=dim, dilation=dilation, causal=True)
    torch_res_unit.eval()

    # Create Lalamo module
    lalamo_config = ResidualUnitConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        causal=True,
    )
    spatial_params = ResidualUnitSpatialParams(dilation=dilation, kernel_size=7)
    lalamo_res_unit = lalamo_config.empty(dim=dim, spatial_params=spatial_params)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_res_unit.state_dict(), prefix="res")
    lalamo_res_unit = load_residual_unit(lalamo_res_unit, weights_dict, ParameterPath("res"))

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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_audio_decoder_block

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

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
    res_unit_config = ResidualUnitConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        causal=True,
    )
    lalamo_config = DACDecoderBlockConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        res_unit_config=res_unit_config,
        causal=True,
    )
    spatial_params = AudioDecoderBlockSpatialParams(
        input_dim=input_dim,
        output_dim=output_dim,
        stride=stride,
    )
    lalamo_decoder_block = lalamo_config.empty(spatial_params=spatial_params)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder_block.state_dict(), prefix="dec_block")

    with jax.disable_jit():
        lalamo_decoder_block = load_audio_decoder_block(lalamo_decoder_block, weights_dict, ParameterPath("dec_block"))

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

    from lalamo.common import ParameterPath
    from lalamo.model_import.loaders.fishaudio_loaders import load_audio_decoder

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

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
    res_unit_config = ResidualUnitConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        causal=True,
    )
    decoder_block_config = DACDecoderBlockConfig(
        precision=jnp.float32,
        snake_config=Snake1dConfig(precision=jnp.float32),
        trans_conv_config=CausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True),
        res_unit_config=res_unit_config,
        causal=True,
    )
    lalamo_config = DACDecoderConfig(
        precision=jnp.float32,
        conv_config=CausalConv1dConfig(precision=jnp.float32, has_biases=True),
        snake_config=Snake1dConfig(precision=jnp.float32),
        decoder_block_config=decoder_block_config,
        causal=True,
    )
    spatial_params = DACDecoderSpatialParams(
        input_channel=input_channel,
        channels=channels,
        rates=rates,
        d_out=d_out,
    )
    lalamo_decoder = lalamo_config.empty(spatial_params=spatial_params)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder.state_dict(), prefix="decoder")

    with jax.disable_jit():
        lalamo_decoder = load_audio_decoder(lalamo_decoder, weights_dict, ParameterPath("decoder"))

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
def test_single_text_transformer_layer(fish_audio_local_model_path: Path) -> None:
    """Test that a single transformer layer matches between FishAudio and Lalamo.

    Based on sandbox/fish_audio_sandbox_text.py::compare_fish_transformer_layer_to_lalamo().
    Compares the first layer output between FishAudio and Lalamo implementations.
    """
    from lalamo.model_import.loaders.fishaudio_loaders import load_transformer_block

    from .fishaudio_torch_stuff import ConfigMapping, prepare_state_dict_for_lalamo_loaders

    # Load the FishAudio model
    pytorch_tts_generator = FishAudioFromTorch.build_foreign_fish_audio_tts_generator(fish_audio_local_model_path)
    assert isinstance(pytorch_tts_generator.tts_model.text_decoder, FishAudioTextDecoder_Foreign)
    fish_model = pytorch_tts_generator.tts_model.text_decoder.fish_model
    config = fish_model.config

    # Create Lalamo transformer using config mapping
    precision = jnp.bfloat16
    transformer_cfg, _ = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(config, precision)
    lalamo_transformer = transformer_cfg.empty()

    # Load weights
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_model.state_dict())
    lalamo_transformer = load_transformer_block(lalamo_transformer, weights_dict)

    # Create random embedded input (test layer computation, not embedding)
    batch_size = 1
    seq_length = 16
    model_dim = config.dim

    torch.manual_seed(42)
    embedded_input_torch = torch.randn(batch_size, seq_length, model_dim, dtype=torch.bfloat16)
    embedded_input_lalamo = torch_to_jax(embedded_input_torch)

    max_seq_len = seq_length
    input_pos = torch.arange(max_seq_len, device=embedded_input_torch.device)
    input_pos_lalamo = torch_to_jax(input_pos)[None, :]

    # Get mask and RoPE embeddings from fish model
    mask = fish_model.causal_mask[None, None, input_pos, :max_seq_len]
    freqs_cis = fish_model.freqs_cis[input_pos]

    # Run first layer from each model
    fish_layer = fish_model.layers[0]
    fish_layer_result = fish_layer(embedded_input_torch, freqs_cis, mask, input_pos=input_pos)

    assert lalamo_transformer.global_rope is not None
    pos_emb_lalamo = vmap(lalamo_transformer.global_rope)(input_pos_lalamo)
    lalamo_layer = lalamo_transformer.layers[0]
    lalamo_layer_result = lalamo_layer(embedded_input_lalamo, pos_emb_lalamo)

    # Compare outputs per token position
    fish_output_jax = torch_to_jax(fish_layer_result)
    lalamo_output = lalamo_layer_result.outputs

    for k in range(max_seq_len):
        diff = fish_output_jax[0][k] - lalamo_output[0][k]
        _testlog.info(f"(token_idx={k}) Layer output diff: min={diff.min()}, max={diff.max()}, std={diff.std()}")

    # Overall comparison
    total_diff = jnp.abs(fish_output_jax - lalamo_output)
    max_diff = float(jnp.max(total_diff))
    mean_diff = float(jnp.mean(total_diff))
    _testlog.info(f"Single layer - Max difference: {max_diff}")
    _testlog.info(f"Single layer - Mean difference: {mean_diff}")

    # Position 0 should match well (no RoPE applied) - validates weight loading
    pos0_diff = jnp.abs(fish_output_jax[0, 0] - lalamo_output[0, 0])
    pos0_max_diff = float(jnp.max(pos0_diff))
    _testlog.info(f"Position 0 max difference: {pos0_max_diff}")
    assert pos0_max_diff < 0.02, f"Position 0 mismatch indicates weight loading issue: {pos0_max_diff}"

    assert fish_output_jax.shape == lalamo_output.shape, (
        f"Shape mismatch: FishAudio {fish_output_jax.shape} vs Lalamo {lalamo_output.shape}"
    )
    assert jnp.allclose(fish_output_jax, lalamo_output, atol=1e-1), (
        f"Outputs don't match. Max diff: {max_diff}, mean diff: {mean_diff}"
    )


@torch.no_grad
def test_audio_transformer_inference() -> None:
    """Test that the post-module Transformer from DAC config can be instantiated and run.

    This test creates both a FishAudio WindowLimitedTransformer and a Lalamo Transformer
    using the same config from get_default_fishaudio_dac_config, loads weights from
    FishAudio into Lalamo, and verifies both can be instantiated correctly.
    """
    from fish_speech.models.dac.modded_dac import ModelArgs, WindowLimitedTransformer

    from lalamo.model_import.loaders.fishaudio_loaders import load_transformer_block
    from lalamo.model_import.model_configs.huggingface.fishaudio import (
        lalamo_transformer_cfg_from_fish_audio_codec_cfg,
    )
    from lalamo.modules.audio.fishaudio.fishaudio_common import get_default_fishaudio_dac_config

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    fish_dac_config = get_default_fishaudio_dac_config()
    post_module_config_dict = fish_dac_config["quantizer"]["post_module"]
    post_module_transformer_foreign = post_module_config_dict["config"]

    # === Create FishAudio WindowLimitedTransformer ===
    torch.manual_seed(42)
    fish_model_args = ModelArgs(
        block_size=post_module_transformer_foreign["block_size"],
        n_layer=post_module_transformer_foreign["n_layer"],
        n_head=post_module_transformer_foreign["n_head"],
        dim=post_module_transformer_foreign["dim"],
        intermediate_size=post_module_transformer_foreign["intermediate_size"],
        n_local_heads=post_module_transformer_foreign["n_local_heads"],
        head_dim=post_module_transformer_foreign["head_dim"],
        rope_base=post_module_transformer_foreign["rope_base"],
        norm_eps=post_module_transformer_foreign["norm_eps"],
        dropout_rate=post_module_transformer_foreign["dropout_rate"],
        attn_dropout_rate=post_module_transformer_foreign["attn_dropout_rate"],
        channels_first=post_module_transformer_foreign["channels_first"],
    )
    fish_transformer = WindowLimitedTransformer(
        config=fish_model_args,
        input_dim=post_module_config_dict["input_dim"],
        window_size=post_module_config_dict["window_size"],
        causal=post_module_config_dict["causal"],
    )
    fish_transformer.eval()

    _testlog.info(f"FishAudio WindowLimitedTransformer instantiated with {len(fish_transformer.layers)} layers")

    # === Create Lalamo Transformer ===
    precision = jnp.float32
    transformer_cfg = lalamo_transformer_cfg_from_fish_audio_codec_cfg(
        post_module_transformer_foreign,
        precision,
        window_size=post_module_config_dict["window_size"],
        input_dim=post_module_config_dict["input_dim"],
    )

    _testlog.info(
        f"Lalamo Transformer config: model_dim={transformer_cfg.model_dim}, "
        f"hidden_dim={transformer_cfg.hidden_dim}, "
        f"context_length={transformer_cfg.context_length}, "
        f"num_layers={len(transformer_cfg.layer_configs)}",
    )

    # Instantiate an empty Lalamo transformer
    lalamo_transformer = transformer_cfg.empty()
    _testlog.info(f"Lalamo Transformer instantiated with {len(lalamo_transformer.layers)} layers")

    # === Load weights from FishAudio into Lalamo ===
    weights_dict = prepare_state_dict_for_lalamo_loaders(fish_transformer.state_dict())
    lalamo_transformer = load_transformer_block(lalamo_transformer, weights_dict)

    _testlog.info("Weights loaded from FishAudio WindowLimitedTransformer into Lalamo Transformer")

    # === Run inference and compare outputs ===
    from lalamo.modules import ForwardPassMode

    batch_size = 1
    seq_length = 32
    input_dim = post_module_config_dict["input_dim"]

    torch.manual_seed(42)
    test_input_torch = torch.randn(batch_size, input_dim, seq_length, dtype=torch.float32)

    fish_output = fish_transformer(test_input_torch)

    # For Lalamo, we need (B, T, C) format and position indices
    test_input_jax = torch_to_jax(test_input_torch).transpose(0, 2, 1)  # (B, C, T) -> (B, T, C)
    input_pos = jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))
    lalamo_result = lalamo_transformer(
        inner_features=test_input_jax,
        token_positions=input_pos,
        state=None,
        return_updated_state=False,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        forward_pass_config=None,
    )
    lalamo_output = lalamo_result.outputs

    # Convert FishAudio output for comparison (B, C, T) -> (B, T, C)
    fish_output_jax = torch_to_jax(fish_output).transpose(0, 2, 1)

    _testlog.info(f"FishAudio output shape: {fish_output.shape}")
    _testlog.info(f"Lalamo output shape: {lalamo_output.shape}")
    _testlog.info(f"Max difference: {jnp.max(jnp.abs(fish_output_jax - lalamo_output))}")
    _testlog.info(f"Mean difference: {jnp.mean(jnp.abs(fish_output_jax - lalamo_output))}")

    assert fish_output_jax.shape == lalamo_output.shape, (
        f"Shape mismatch: FishAudio {fish_output_jax.shape} vs Lalamo {lalamo_output.shape}"
    )
    assert jnp.allclose(fish_output_jax, lalamo_output, atol=1e-4), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(fish_output_jax - lalamo_output))}"
    )


@torch.no_grad
def test_dac_matches_pytorch(fish_audio_local_model_path) -> None:
    """Test that Lalamo DAC matches PyTorch DAC from FishAudio.

    This test loads a real DAC model checkpoint, creates a Lalamo DAC module
    using load_descript_audio_codec(), and compares full inference (codes -> audio)
    between both implementations.
    """
    from lalamo.model_import.loaders.fishaudio_loaders import load_descript_audio_codec

    from .fishaudio_torch_stuff import prepare_state_dict_for_lalamo_loaders

    audio_chkpt_path = fish_audio_local_model_path / "codec.pth"
    config_name = "modded_dac_vq"
    device = "cpu"

    fish_dac = fish_dac_inference.load_model(config_name, audio_chkpt_path, device=device)
    assert isinstance(fish_dac, FishDAC)
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
        f"min={audio_fish.min():.6f}, avg={audio_fish.mean():.6f}",
    )
    _testlog.info(
        f"DAC - Lalamo audio: shape={audio_lalamo.shape}, max={audio_lalamo.max():.6f}, "
        f"min={audio_lalamo.min():.6f}, avg={audio_lalamo.mean():.6f}",
    )
    _testlog.info(
        f"DAC - audio diff: max={jnp.abs(audio_diff).max():.6f}, "
        f"avg={audio_diff.mean():.6f}, std={audio_diff.std():.6f}",
    )

    assert audio_fish_ntc.shape == audio_lalamo.shape, (
        f"Shape mismatch: FishAudio {audio_fish_ntc.shape} vs Lalamo {audio_lalamo.shape}"
    )
    assert jnp.allclose(audio_fish_ntc, audio_lalamo, atol=1e-3), (
        f"Outputs don't match. Max diff: {jnp.max(jnp.abs(audio_diff))}"
    )
