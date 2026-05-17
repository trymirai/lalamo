import jax.numpy as jnp
import numpy as np
import torch

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.nanocodec_loaders import (
    load_causal_hifigan_decoder,
)
from lalamo.modules.audio.common_modules import CausalConv1dConfig
from lalamo.modules.audio.fishaudio.fishaudio_modules import Snake1dConfig
from lalamo.modules.audio.nanocodec.nanocodec_consts import DEFAULT_FSQ_EPS
from lalamo.modules.audio.nanocodec.nanocodec_modules import (
    CausalHiFiGANDecoderConfig,
    CausalTransposeConv1dConfig,
    FiniteScalarQuantizerConfig,
    GroupFiniteScalarQuantizerConfig,
    HalfSnakeConfig,
    HiFiGANResBlockConfig,
    HiFiGANResLayerConfig,
    ResidualBlockConfig,
)
from tests.common import assert_close
from tests.helpers import make_test_sharding_config
from tests.tts.nanocodec import nanocodec_torch_stuff as nanocodec_torch
from tests.tts.utils import prepare_state_dict_for_lalamo_loaders


def random_normal(*shape: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_fsq_matches_torch() -> None:
    num_levels = [8, 7, 6, 6]
    lalamo_config = FiniteScalarQuantizerConfig(
        num_levels=tuple(num_levels),
        eps=DEFAULT_FSQ_EPS,
    )
    lalamo_quantizer = lalamo_config.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
    )
    torch_quantizer = nanocodec_torch.FiniteScalarQuantizer(num_levels=num_levels, eps=1e-3)

    batch_size, seq_len = 2, 10
    inputs_np = random_normal(batch_size, 4, seq_len)

    codes_lalamo = lalamo_quantizer.encode(jnp.array(inputs_np))

    input_len = torch.tensor([seq_len, seq_len])
    _, indices_torch = torch_quantizer(torch.from_numpy(inputs_np), input_len)
    np.testing.assert_allclose(np.array(codes_lalamo), indices_torch[0].numpy(), rtol=1e-5, atol=1e-5)

    codebook_size = lalamo_quantizer.config.codebook_size
    indices_np = np.arange(codebook_size).reshape(1, -1).astype(np.int32)

    decoded_lalamo = lalamo_quantizer(jnp.array(indices_np))

    input_len = torch.tensor([codebook_size])
    decoded_torch = torch_quantizer.decode(torch.from_numpy(indices_np).reshape((1, -1, 1)), input_len)

    np.testing.assert_allclose(
        np.array(decoded_lalamo),
        decoded_torch.numpy().transpose((2, 0, 1)),
        rtol=1e-5,
    )


def test_group_fsq_matches_torch() -> None:
    num_groups = 13
    num_levels_per_group = [8, 7, 6, 6]
    fsq_config = FiniteScalarQuantizerConfig(
        num_levels=tuple(num_levels_per_group),
        eps=DEFAULT_FSQ_EPS,
    )
    lalamo_config = GroupFiniteScalarQuantizerConfig(num_groups=num_groups, quantizer_config=fsq_config)
    lalamo_quantizer = lalamo_config.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
    )
    torch_quantizer = nanocodec_torch.GroupFiniteScalarQuantizer(
        num_groups=num_groups,
        num_levels_per_group=num_levels_per_group,
        eps=1e-3,
    )

    batch_size, seq_len = 2, 10
    channels = lalamo_quantizer.config.codebook_dim
    inputs_np = random_normal(batch_size, channels, seq_len)

    codes_lalamo = lalamo_quantizer.encode(jnp.array(inputs_np))

    input_len = torch.tensor([seq_len, seq_len])
    _, indices_torch = torch_quantizer(torch.from_numpy(inputs_np), input_len)

    np.testing.assert_allclose(np.array(codes_lalamo), indices_torch.numpy(), rtol=1e-5, atol=1e-5)

    rng = np.random.default_rng(42)
    codebook_size_per_group = 8 * 7 * 6 * 6
    indices_np = rng.integers(0, codebook_size_per_group, (batch_size, seq_len, num_groups), dtype=np.int32)

    decoded_lalamo = lalamo_quantizer.decode(jnp.array(indices_np))
    input_len = torch.tensor([seq_len, seq_len])
    indices_torch = torch.from_numpy(indices_np).permute((2, 0, 1))
    decoded_torch = torch_quantizer.decode(indices_torch, input_len)

    np.testing.assert_allclose(
        np.array(decoded_lalamo),
        decoded_torch.numpy().transpose((0, 2, 1)),
        rtol=1e-5,
    )


def test_causal_hifigan_decoder_forward_matches_torch() -> None:
    input_dim = 52
    base_channels = 128
    up_sample_rates = (4, 4)
    in_kernel_size = 7
    out_kernel_size = 3
    resblock_kernel_sizes = (3, 7)
    resblock_dilations = (1, 3)

    torch_decoder = nanocodec_torch.CausalHiFiGANDecoder(
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

    snake_config = Snake1dConfig()
    activation_config = HalfSnakeConfig(snake_config=snake_config, leaky_relu_negative_slope=0.01)
    conv_config = CausalConv1dConfig(has_biases=True)
    transpose_conv_config = CausalTransposeConv1dConfig(has_biases=True)

    residual_block_config = ResidualBlockConfig(
        activation_config=activation_config,
        conv_config=conv_config,
    )
    hifigan_res_block_config = HiFiGANResBlockConfig(residual_block_config=residual_block_config)
    res_layer_config = HiFiGANResLayerConfig(hifigan_res_block_config=hifigan_res_block_config)

    lalamo_decoder_config = CausalHiFiGANDecoderConfig(
        activation_config=activation_config,
        pre_conv_config=conv_config,
        transpose_conv_config=transpose_conv_config,
        res_layer_config=res_layer_config,
        post_conv_config=conv_config,
    )
    lalamo_decoder = lalamo_decoder_config.init(
        EmptyInitializer(dtype=jnp.float32, sharding_config=make_test_sharding_config()),
        input_dim=input_dim,
        base_channels=base_channels,
        up_sample_rates=up_sample_rates,
        in_kernel_size=in_kernel_size,
        out_kernel_size=out_kernel_size,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilations=resblock_dilations,
    )

    weights_dict = prepare_state_dict_for_lalamo_loaders(torch_decoder.state_dict())
    lalamo_decoder = load_causal_hifigan_decoder(lalamo_decoder, weights_dict)

    batch_size, seq_len = 2, 10
    inputs_nct = random_normal(batch_size, input_dim, seq_len)
    inputs_nsc = np.transpose(inputs_nct, (0, 2, 1))

    output_lalamo = lalamo_decoder(jnp.array(inputs_nsc))

    with torch.no_grad():
        input_len = torch.tensor([seq_len, seq_len])
        output_torch, _ = torch_decoder(torch.from_numpy(inputs_nct), input_len)

    assert_close(
        result=jnp.asarray(output_lalamo),
        reference=jnp.asarray(output_torch.detach().numpy()),
        operation_name="test_causal_hifigan_decoder_forward_matches_torch",
    )
