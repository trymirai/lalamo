"""Tests comparing JAX and PyTorch implementations of NanoCodec modules.

These tests verify that the Lalamo JAX implementation produces outputs
that match the PyTorch reference implementation from NeMo.
"""

import jax.numpy as jnp
import numpy as np
import torch

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.nanocodec_loaders import (
    load_causal_conv1d,
    load_causal_hifigan_decoder,
    load_causal_transpose_conv1d,
    load_half_snake,
    load_hifigan_res_block,
    load_hifigan_res_layer,
    load_residual_block,
)
from lalamo.modules.audio.cambai.cambai_modules import (
    CausalHiFiGANDecoderConfig as JaxCausalHiFiGANDecoderConfig,
    CausalTransposeConv1dConfig as JaxCausalTransposeConv1dConfig,
    FiniteScalarQuantizerConfig as JaxFSQConfig,
    GroupFiniteScalarQuantizerConfig as JaxGroupFSQConfig,
    HalfSnakeConfig as JaxHalfSnakeConfig,
    HiFiGANResBlockConfig as JaxHiFiGANResBlockConfig,
    HiFiGANResLayerConfig as JaxHiFiGANResLayerConfig,
    ResidualBlockConfig as JaxResidualBlockConfig,
)
from lalamo.modules.audio.fishaudio.fishaudio_modules import (
    CausalConv1dConfig as JaxCausalConv1dConfig,
    Snake1dConfig as JaxSnake1dConfig,
)
from tests.tts.cambai.nanocodec_torch_stuff import (
    CausalConv1dNorm as TorchCausalConv1dNorm,
    CausalConvTranspose1dNorm as TorchCausalConvTranspose1dNorm,
    CausalHiFiGANDecoder as TorchCausalHiFiGANDecoder,
    FiniteScalarQuantizer as TorchFSQ,
    GroupFiniteScalarQuantizer as TorchGroupFSQ,
    HalfSnake as TorchHalfSnake,
    HiFiGANResBlock as TorchHiFiGANResBlock,
    HiFiGANResLayer as TorchHiFiGANResLayer,
    ResidualBlock as TorchResidualBlock,
)
from tests.tts.utils import prepare_state_dict_for_lalamo_loaders

# =============================================================================
# Quantizer Tests
# =============================================================================


def test_fsq_forward_matches_torch() -> None:
    """Test FSQ forward pass produces same codes and indices as PyTorch."""
    num_levels = [8, 7, 6, 6]

    jax_config = JaxFSQConfig(num_levels=tuple(num_levels), eps=1e-3)
    jax_quantizer = jax_config.empty()
    torch_quantizer = TorchFSQ(num_levels=num_levels, eps=1e-3)

    np.random.seed(42)
    batch_size, seq_len = 2, 10
    inputs_np = np.random.randn(batch_size, 4, seq_len).astype(np.float32)

    # JAX
    codes_jax, indices_jax = jax_quantizer(jnp.array(inputs_np))

    # PyTorch
    input_len = torch.tensor([seq_len, seq_len])
    codes_torch, indices_torch = torch_quantizer(torch.from_numpy(inputs_np), input_len)

    np.testing.assert_allclose(np.array(codes_jax), codes_torch.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.array(indices_jax), indices_torch.numpy())


def test_fsq_decode_matches_torch() -> None:
    """Test FSQ decode produces same outputs as PyTorch for all codebook entries."""
    num_levels = [8, 7, 6, 6]

    jax_config = JaxFSQConfig(num_levels=tuple(num_levels), eps=1e-3)
    jax_quantizer = jax_config.empty()
    torch_quantizer = TorchFSQ(num_levels=num_levels, eps=1e-3)

    # Test all codebook entries
    codebook_size = jax_quantizer.codebook_size
    indices_np = np.arange(codebook_size).reshape(1, 1, -1).astype(np.int32)

    # JAX
    decoded_jax = jax_quantizer.decode(jnp.array(indices_np))

    # PyTorch
    input_len = torch.tensor([codebook_size])
    decoded_torch = torch_quantizer.decode(torch.from_numpy(indices_np), input_len)

    np.testing.assert_allclose(np.array(decoded_jax), decoded_torch.numpy(), rtol=1e-5, atol=1e-5)


def test_group_fsq_forward_matches_torch() -> None:
    """Test GroupFSQ forward pass produces same codes and indices as PyTorch."""
    num_groups = 13
    num_levels_per_group = [8, 7, 6, 6]

    fsq_config = JaxFSQConfig(num_levels=tuple(num_levels_per_group), eps=1e-3)
    jax_config = JaxGroupFSQConfig(num_groups=num_groups, quantizer_config=fsq_config)
    jax_quantizer = jax_config.empty()
    torch_quantizer = TorchGroupFSQ(num_groups=num_groups, num_levels_per_group=num_levels_per_group, eps=1e-3)

    np.random.seed(42)
    batch_size, seq_len = 2, 10
    channels = jax_quantizer.codebook_dim  # 52
    inputs_np = np.random.randn(batch_size, channels, seq_len).astype(np.float32)

    # JAX
    codes_jax, indices_jax = jax_quantizer(jnp.array(inputs_np))

    # PyTorch
    input_len = torch.tensor([seq_len, seq_len])
    codes_torch, indices_torch = torch_quantizer(torch.from_numpy(inputs_np), input_len)

    np.testing.assert_allclose(np.array(codes_jax), codes_torch.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.array(indices_jax), indices_torch.numpy())


def test_group_fsq_decode_matches_torch() -> None:
    """Test GroupFSQ decode produces same outputs as PyTorch."""
    num_groups = 13
    num_levels_per_group = [8, 7, 6, 6]

    fsq_config = JaxFSQConfig(num_levels=tuple(num_levels_per_group), eps=1e-3)
    jax_config = JaxGroupFSQConfig(num_groups=num_groups, quantizer_config=fsq_config)
    jax_quantizer = jax_config.empty()
    torch_quantizer = TorchGroupFSQ(num_groups=num_groups, num_levels_per_group=num_levels_per_group, eps=1e-3)

    np.random.seed(42)
    batch_size, seq_len = 2, 10
    codebook_size_per_group = 8 * 7 * 6 * 6  # 2016
    indices_np = np.random.randint(0, codebook_size_per_group, (num_groups, batch_size, seq_len)).astype(np.int32)

    # JAX
    decoded_jax = jax_quantizer.decode(jnp.array(indices_np))

    # PyTorch
    input_len = torch.tensor([seq_len, seq_len])
    decoded_torch = torch_quantizer.decode(torch.from_numpy(indices_np), input_len)

    np.testing.assert_allclose(np.array(decoded_jax), decoded_torch.numpy(), rtol=1e-5, atol=1e-5)


# =============================================================================
# Activation Tests
# =============================================================================


def test_half_snake_forward_matches_torch() -> None:
    """Test HalfSnake forward pass produces same output as PyTorch.

    Tests both default alpha (ones) and custom alpha values.
    """
    channels = 128

    # Create JAX HalfSnake
    snake_config = JaxSnake1dConfig(precision=jnp.float32)
    jax_config = JaxHalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    jax_half_snake = jax_config.random_init(channels, key=None)

    # Create PyTorch HalfSnake
    torch_half_snake = TorchHalfSnake(channels)

    # Set custom alpha values and load using loader
    np.random.seed(123)
    snake_channels = channels // 2
    custom_alpha = np.random.randn(snake_channels).astype(np.float32) * 0.5 + 1.0

    # Update PyTorch alpha: shape is (1, snake_channels, 1)
    torch_half_snake.snake_act.alpha.data = torch.from_numpy(custom_alpha).reshape(1, -1, 1)

    # Get state_dict and load using loader
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_half_snake.state_dict())
    jax_half_snake = load_half_snake(jax_half_snake, weights_dict, ParameterPath())

    np.random.seed(42)
    batch_size, seq_len = 2, 100
    # PyTorch uses NCT format: (batch, channels, time)
    inputs_nct = np.random.randn(batch_size, channels, seq_len).astype(np.float32)
    # JAX uses NSC format: (batch, sequence, channels)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_half_snake(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward
    output_torch_nct = torch_half_snake(torch.from_numpy(inputs_nct))

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


# =============================================================================
# Convolution Tests
# =============================================================================


def test_causal_conv1d_forward_matches_torch() -> None:
    """Test CausalConv1d forward pass matches PyTorch CausalConv1dNorm (after weight norm removal)."""
    in_channels = 64
    out_channels = 128
    kernel_size = 7
    stride = 1
    dilation = 1

    # Create PyTorch conv and remove weight norm to get fused weights
    torch_conv = TorchCausalConv1dNorm(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=True,
    )

    # Create JAX conv
    jax_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)
    jax_conv = jax_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    # Load weights using loader
    # state_dict already has 'conv.' prefix from TorchCausalConv1dNorm's self.conv
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_conv.state_dict())
    jax_conv = load_causal_conv1d(jax_conv, weights_dict, ParameterPath("conv"))

    np.random.seed(42)
    batch_size, seq_len = 2, 50
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, in_channels, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_conv(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward
    input_len = torch.tensor([seq_len, seq_len])
    output_torch_nct = torch_conv(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_causal_conv1d_with_dilation_matches_torch() -> None:
    """Test CausalConv1d with dilation matches PyTorch."""
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    stride = 1
    dilation = 3

    torch_conv = TorchCausalConv1dNorm(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=True,
    )

    jax_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)
    jax_conv = jax_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    # Load weights using loader
    # state_dict already has 'conv.' prefix from TorchCausalConv1dNorm's self.conv
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_conv.state_dict())
    jax_conv = load_causal_conv1d(jax_conv, weights_dict, ParameterPath("conv"))

    np.random.seed(42)
    batch_size, seq_len = 2, 50
    inputs_nct = np.random.randn(batch_size, in_channels, seq_len).astype(np.float32)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    output_jax_nsc = jax_conv(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    input_len = torch.tensor([seq_len, seq_len])
    output_torch_nct = torch_conv(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_causal_transpose_conv1d_forward_matches_torch() -> None:
    """Test CausalTransposeConv1d forward pass matches PyTorch CausalConvTranspose1dNorm."""
    in_channels = 128
    out_channels = 64
    kernel_size = 14
    stride = 7

    # Create PyTorch transposed conv (groups=out_channels by default in NanoCodec)
    torch_conv = TorchCausalConvTranspose1dNorm(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=out_channels,
        bias=True,
    )

    # Create JAX transposed conv
    jax_config = JaxCausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)
    jax_conv = jax_config.empty(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        groups=out_channels,
    )

    # Load weights using loader (handles weight transformation automatically)
    # state_dict already has 'conv.' prefix from TorchCausalConvTranspose1dNorm's self.conv
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_conv.state_dict())
    jax_conv = load_causal_transpose_conv1d(jax_conv, weights_dict, ParameterPath("conv"))

    np.random.seed(42)
    batch_size, seq_len = 2, 10
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, in_channels, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_conv(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward - manually apply conv and trimming without mask
    # (PyTorch's forward() applies mask_sequence_tensor which zeros out positions > input_len)
    with torch.no_grad():
        hidden_states = torch_conv.conv(torch.from_numpy(inputs_nct))
        # unpad (same logic as in CausalConvTranspose1dNorm.forward)
        end = hidden_states.shape[-1] - torch_conv.padding_right
        output_torch_nct = hidden_states[..., torch_conv.padding_left : end]

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


# =============================================================================
# Residual Block Tests
# =============================================================================


def test_residual_block_forward_matches_torch() -> None:
    """Test ResidualBlock forward pass matches PyTorch."""
    channels = 128
    kernel_size = 3
    dilation = 3

    # Create PyTorch residual block (causal with half_snake activation)
    torch_block = TorchResidualBlock(
        channels=channels,
        filters=channels,
        kernel_size=kernel_size,
        dilation=dilation,
        activation="half_snake",
        is_causal=True,
    )
    torch_block.remove_weight_norm()

    # Create JAX residual block
    snake_config = JaxSnake1dConfig(precision=jnp.float32)
    activation_config = JaxHalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)

    jax_block_config = JaxResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )
    jax_block = jax_block_config.empty(
        channels=channels,
        kernel_size=kernel_size,
        dilation=dilation,
    )

    # Load weights using loader
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_block.state_dict())
    jax_block = load_residual_block(jax_block, weights_dict, ParameterPath())

    np.random.seed(42)
    batch_size, seq_len = 2, 50
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, channels, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_block(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward (without masking)
    with torch.no_grad():
        input_len = torch.tensor([seq_len, seq_len])
        output_torch_nct = torch_block(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_hifigan_res_block_forward_matches_torch() -> None:
    """Test HiFiGANResBlock forward pass matches PyTorch."""
    channels = 128
    kernel_size = 3
    dilations = (1, 3, 5)

    # Create PyTorch HiFiGANResBlock (causal with half_snake activation)
    torch_block = TorchHiFiGANResBlock(
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
        activation="half_snake",
        is_causal=True,
    )
    for res_block in torch_block.res_blocks:
        res_block.remove_weight_norm()

    # Create JAX HiFiGANResBlock
    snake_config = JaxSnake1dConfig(precision=jnp.float32)
    activation_config = JaxHalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)
    residual_block_config = JaxResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )

    jax_block_config = JaxHiFiGANResBlockConfig(residual_block_config=residual_block_config)
    jax_block = jax_block_config.empty(
        channels=channels,
        kernel_size=kernel_size,
        dilations=dilations,
    )

    # Load weights using loader
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_block.state_dict())
    jax_block = load_hifigan_res_block(jax_block, weights_dict, ParameterPath())

    np.random.seed(42)
    batch_size, seq_len = 2, 50
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, channels, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_block(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward
    with torch.no_grad():
        input_len = torch.tensor([seq_len, seq_len])
        output_torch_nct = torch_block(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_hifigan_res_layer_forward_matches_torch() -> None:
    """Test HiFiGANResLayer forward pass matches PyTorch."""
    channels = 64
    kernel_sizes = (3, 7, 11)
    dilations = (1, 3, 5)

    # Create PyTorch HiFiGANResLayer (causal with half_snake activation)
    torch_layer = TorchHiFiGANResLayer(
        channels=channels,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
        activation="half_snake",
        is_causal=True,
    )
    for hifigan_res_block in torch_layer.res_blocks:
        for res_block in hifigan_res_block.res_blocks:
            res_block.remove_weight_norm()

    # Create JAX HiFiGANResLayer
    snake_config = JaxSnake1dConfig(precision=jnp.float32)
    activation_config = JaxHalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)
    residual_block_config = JaxResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )
    hifigan_res_block_config = JaxHiFiGANResBlockConfig(residual_block_config=residual_block_config)

    jax_layer_config = JaxHiFiGANResLayerConfig(hifigan_res_block_config=hifigan_res_block_config)
    jax_layer = jax_layer_config.empty(
        channels=channels,
        kernel_sizes=kernel_sizes,
        dilations=dilations,
    )

    # Load weights using loader
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_layer.state_dict())
    jax_layer = load_hifigan_res_layer(jax_layer, weights_dict, ParameterPath())

    np.random.seed(42)
    batch_size, seq_len = 2, 50
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, channels, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward
    output_jax_nsc = jax_layer(jnp.array(inputs_nsc))
    output_jax_nct = np.transpose(np.array(output_jax_nsc), (0, 2, 1))

    # PyTorch forward
    with torch.no_grad():
        input_len = torch.tensor([seq_len, seq_len])
        output_torch_nct = torch_layer(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(output_jax_nct, output_torch_nct.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_causal_hifigan_decoder_forward_matches_torch() -> None:
    """Test CausalHiFiGANDecoder forward pass matches PyTorch."""
    # Use smaller config for faster testing
    input_dim = 52  # NanoCodec uses 52 (13 groups * 4 dims)
    base_channels = 128  # Reduced from 512
    up_sample_rates = (4, 4)  # Reduced from (8, 8, 2, 2)
    in_kernel_size = 7
    out_kernel_size = 3
    resblock_kernel_sizes = (3, 7)  # Reduced from (3, 7, 11)
    resblock_dilations = (1, 3)  # Reduced from (1, 3, 5)

    # Create PyTorch decoder
    torch_decoder = TorchCausalHiFiGANDecoder(
        input_dim=input_dim,
        up_sample_rates=up_sample_rates,
        base_channels=base_channels,
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilations,
        activation="half_snake",
        output_activation="tanh",
    )
    torch_decoder.remove_weight_norm()

    # Create JAX decoder
    snake_config = JaxSnake1dConfig(precision=jnp.float32)
    activation_config = JaxHalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = JaxCausalConv1dConfig(precision=jnp.float32, has_biases=True)
    transpose_conv_config = JaxCausalTransposeConv1dConfig(precision=jnp.float32, has_biases=True)

    residual_block_config = JaxResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )
    hifigan_res_block_config = JaxHiFiGANResBlockConfig(residual_block_config=residual_block_config)
    res_layer_config = JaxHiFiGANResLayerConfig(hifigan_res_block_config=hifigan_res_block_config)

    jax_decoder_config = JaxCausalHiFiGANDecoderConfig(
        activation_config=activation_config,
        pre_conv_config=conv_config,
        transpose_conv_config=transpose_conv_config,
        res_layer_config=res_layer_config,
        post_conv_config=conv_config,
    )
    jax_decoder = jax_decoder_config.empty(
        input_dim=input_dim,
        base_channels=base_channels,
        up_sample_rates=up_sample_rates,
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilations=resblock_dilations,
    )

    # Load weights using loader
    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder.state_dict())
    jax_decoder = load_causal_hifigan_decoder(jax_decoder, weights_dict)

    np.random.seed(42)
    batch_size, seq_len = 2, 10
    # PyTorch uses NCT format
    inputs_nct = np.random.randn(batch_size, input_dim, seq_len).astype(np.float32)
    # JAX uses NSC format
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    # JAX forward - returns (batch, audio_length)
    output_jax = jax_decoder(jnp.array(inputs_nsc))

    # PyTorch forward - returns (audio, audio_len)
    with torch.no_grad():
        input_len = torch.tensor([seq_len, seq_len])
        output_torch, _ = torch_decoder(torch.from_numpy(inputs_nct), input_len)

    np.testing.assert_allclose(np.array(output_jax), output_torch.detach().numpy(), rtol=1e-5, atol=1e-5)
