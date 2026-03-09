"""Tests comparing Lalamo and PyTorch implementations of Qwen3 TTS modules."""

import jax.numpy as jnp
import numpy as np
import torch

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.qwen3_tts_loaders import (
    load_qwen3_tts_causal_transpose_conv1d,
    load_qwen3_tts_euclidean_codebook,
    load_qwen3_tts_residual_unit,
    load_qwen3_tts_snake_beta,
)
from lalamo.modules.audio.common_modules import CausalConv1dConfig
from lalamo.modules.audio.qwen3_tts.qwen3_tts_modules import (
    Qwen3TTSCausalTransposeConv1dConfig,
    Qwen3TTSEuclideanCodebookConfig,
    Qwen3TTSResidualUnitConfig,
    Qwen3TTSSnakeBetaConfig,
)
from tests.tts.qwen3.reference.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    EuclideanCodebook as TorchEuclideanCodebook,
)
from tests.tts.qwen3.reference.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalTransConvNet as TorchCausalTransConvNet,
)
from tests.tts.qwen3.reference.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit as TorchResidualUnit,
)
from tests.tts.qwen3.reference.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    SnakeBeta as TorchSnakeBeta,
)
from tests.tts.utils import prepare_state_dict_for_lalamo_loaders


def random_normal(*shape: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_snake_beta_matches_torch() -> None:
    channels = 64

    torch_snake = TorchSnakeBeta(channels)
    torch_snake.alpha.data = torch.from_numpy(random_normal(channels, seed=10))
    torch_snake.beta.data = torch.from_numpy(random_normal(channels, seed=11))

    lalamo_snake = Qwen3TTSSnakeBetaConfig(precision=jnp.float32).empty(channels)
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_snake.state_dict())
    lalamo_snake = load_qwen3_tts_snake_beta(lalamo_snake, weights_dict, ParameterPath())

    inputs_nct = random_normal(2, channels, 50)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    with torch.no_grad():
        output_torch_nct = torch_snake(torch.from_numpy(inputs_nct)).numpy()

    output_lalamo_nct = np.transpose(np.array(lalamo_snake(jnp.array(inputs_nsc))), (0, 2, 1))

    np.testing.assert_allclose(output_lalamo_nct, output_torch_nct, rtol=1e-5, atol=1e-5)


def test_causal_transpose_conv1d_matches_torch() -> None:
    in_channels, out_channels = 64, 32
    kernel_size, stride = 16, 8

    torch_conv = TorchCausalTransConvNet(in_channels, out_channels, kernel_size, stride)

    lalamo_conv = Qwen3TTSCausalTransposeConv1dConfig(
        precision=jnp.float32, has_biases=True,
    ).empty(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_conv.state_dict())
    lalamo_conv = load_qwen3_tts_causal_transpose_conv1d(lalamo_conv, weights_dict, ParameterPath("conv"))

    inputs_nct = random_normal(2, in_channels, 10)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    with torch.no_grad():
        output_torch_nct = torch_conv(torch.from_numpy(inputs_nct)).numpy()

    output_lalamo_nct = np.transpose(np.array(lalamo_conv(jnp.array(inputs_nsc))), (0, 2, 1))

    np.testing.assert_allclose(output_lalamo_nct, output_torch_nct, rtol=1e-5, atol=1e-5)


def test_residual_unit_matches_torch() -> None:
    dim, dilation = 32, 3

    torch_unit = TorchResidualUnit(dim, dilation)

    snake_config = Qwen3TTSSnakeBetaConfig(precision=jnp.float32)
    conv_config = CausalConv1dConfig(precision=jnp.float32, has_biases=True)
    lalamo_unit = Qwen3TTSResidualUnitConfig(
        precision=jnp.float32, snake_config=snake_config, conv_config=conv_config,
    ).empty(dim=dim, dilation=dilation)

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_unit.state_dict())
    lalamo_unit = load_qwen3_tts_residual_unit(lalamo_unit, weights_dict, ParameterPath())

    inputs_nct = random_normal(2, dim, 50)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    with torch.no_grad():
        output_torch_nct = torch_unit(torch.from_numpy(inputs_nct)).numpy()

    output_lalamo_nct = np.transpose(np.array(lalamo_unit(jnp.array(inputs_nsc))), (0, 2, 1))

    np.testing.assert_allclose(output_lalamo_nct, output_torch_nct, rtol=1e-5, atol=1e-5)


def test_euclidean_codebook_decode_matches_torch() -> None:
    dim, codebook_size = 64, 128

    torch_codebook = TorchEuclideanCodebook(dim, codebook_size)
    torch_codebook.cluster_usage.data = torch.from_numpy(random_normal(codebook_size, seed=20).clip(0.1))
    torch_codebook.embedding_sum.data = torch.from_numpy(random_normal(codebook_size, dim, seed=21))

    lalamo_codebook = Qwen3TTSEuclideanCodebookConfig(precision=jnp.float32).empty(dim, codebook_size)
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_codebook.state_dict())
    lalamo_codebook = load_qwen3_tts_euclidean_codebook(lalamo_codebook, weights_dict, ParameterPath())

    rng = np.random.default_rng(42)
    codes_np = rng.integers(0, codebook_size, (2, 10), dtype=np.int32)

    with torch.no_grad():
        output_torch = torch_codebook.decode(torch.from_numpy(codes_np)).numpy()

    output_lalamo = np.array(lalamo_codebook.decode(jnp.array(codes_np)))

    np.testing.assert_allclose(output_lalamo, output_torch, rtol=1e-5, atol=1e-5)
