import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.audio.neutts.codec_modules import (
    NeuCodecFSQConfig,
    NeuCodecResidualFSQConfig,
    NeuCodecResnetBlockConfig,
    NeuCodecTransformerBlockConfig,
)


def test_neucodec_fsq_indices_to_codes_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    finite_scalar_quantization = pytest.importorskip("vector_quantize_pytorch.finite_scalar_quantization")
    indices = [[0, 1, 255, 4**7, 16383, 65535]]
    torch_fsq = finite_scalar_quantization.FSQ(levels=[4] * 8)
    lalamo_fsq = NeuCodecFSQConfig(levels=(4,) * 8, precision=jnp.float32).empty()

    expected_codes = torch_fsq.indices_to_codes(torch.tensor(indices, dtype=torch.long)).detach().cpu().numpy()
    actual_codes = np.asarray(lalamo_fsq.indices_to_codes(jnp.asarray(indices, dtype=jnp.int32)))

    np.testing.assert_allclose(actual_codes, expected_codes, rtol=0, atol=0)


def test_neucodec_residual_fsq_get_output_from_indices_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    vector_quantize_pytorch = pytest.importorskip("vector_quantize_pytorch")
    indices = np.asarray([[[12], [34], [56], [65535]]], dtype=np.int64)
    torch_residual_fsq = vector_quantize_pytorch.ResidualFSQ(dim=2048, levels=[4] * 8, num_quantizers=1)
    lalamo_residual_fsq = NeuCodecResidualFSQConfig(
        levels=(4,) * 8,
        num_quantizers=1,
        output_dim=2048,
        precision=jnp.float32,
    ).empty()
    project_out_weights = {
        "weights": jnp.asarray(torch_residual_fsq.project_out.weight.detach().cpu().numpy()),
        "biases": jnp.asarray(torch_residual_fsq.project_out.bias.detach().cpu().numpy()),
    }
    lalamo_residual_fsq = lalamo_residual_fsq.import_weights(
        {
            "fsq": lalamo_residual_fsq.fsq.export_weights(),
            "project_out": project_out_weights,
        },
    )

    with torch.no_grad():
        expected_output = torch_residual_fsq.get_output_from_indices(torch.as_tensor(indices, dtype=torch.long))
    actual_output = np.asarray(lalamo_residual_fsq.get_output_from_indices(jnp.asarray(indices, dtype=jnp.int32)))

    np.testing.assert_allclose(actual_output, expected_output.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)


def test_neucodec_resnet_block_matches_torch_vocos_reference() -> None:
    torch = pytest.importorskip("torch")
    codec_decoder_vocos = pytest.importorskip("neucodec.codec_decoder_vocos")
    torch.manual_seed(0)
    torch_block = codec_decoder_vocos.ResnetBlock(
        in_channels=32,
        out_channels=32,
        temb_channels=0,
        dropout=0.1,
    ).eval()
    lalamo_block = NeuCodecResnetBlockConfig(channels=32, precision=jnp.float32).empty()
    torch_input = torch.linspace(-1.0, 1.0, steps=1 * 32 * 7, dtype=torch.float32).reshape(1, 32, 7)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy()).transpose(0, 2, 1)

    state_dict = torch_block.state_dict()
    lalamo_block = lalamo_block.import_weights(
        {
            "norm1": {
                "weights": jnp.asarray(state_dict["norm1.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["norm1.bias"].detach().cpu().numpy()),
            },
            "conv1": {
                "weights": jnp.asarray(state_dict["conv1.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["conv1.bias"].detach().cpu().numpy()),
            },
            "norm2": {
                "weights": jnp.asarray(state_dict["norm2.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["norm2.bias"].detach().cpu().numpy()),
            },
            "conv2": {
                "weights": jnp.asarray(state_dict["conv2.weight"].detach().cpu().numpy()),
                "biases": jnp.asarray(state_dict["conv2.bias"].detach().cpu().numpy()),
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_block(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_block(lalamo_input)).transpose(0, 2, 1)

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)


def test_neucodec_transformer_block_matches_torch_vocos_reference() -> None:
    torch = pytest.importorskip("torch")
    bs_roformer5 = pytest.importorskip("neucodec.bs_roformer5")
    torchtune_modules = pytest.importorskip("torchtune.modules")
    torch.manual_seed(0)
    rotary = torchtune_modules.RotaryPositionalEmbeddings(dim=16)
    torch_block = bs_roformer5.TransformerBlock(dim=64, n_heads=4, rotary_embed=rotary).eval()
    lalamo_block = NeuCodecTransformerBlockConfig(dim=64, num_heads=4, rotary_dim=16, precision=jnp.float32).empty()
    torch_input = torch.linspace(-0.5, 0.5, steps=1 * 5 * 64, dtype=torch.float32).reshape(1, 5, 64)
    lalamo_input = jnp.asarray(torch_input.detach().cpu().numpy())

    state_dict = torch_block.state_dict()
    lalamo_block = lalamo_block.import_weights(
        {
            "att_norm": {
                "weights": jnp.asarray(state_dict["att_norm.weight"].detach().cpu().numpy()),
            },
            "ffn_norm": {
                "weights": jnp.asarray(state_dict["ffn_norm.weight"].detach().cpu().numpy()),
            },
            "att": {
                "c_attn": {
                    "weights": jnp.asarray(state_dict["att.c_attn.weight"].detach().cpu().numpy()),
                },
                "c_proj": {
                    "weights": jnp.asarray(state_dict["att.c_proj.weight"].detach().cpu().numpy()),
                },
            },
            "mlp": {
                "fc1": {
                    "weights": jnp.asarray(state_dict["mlp.fc1.weight"].detach().cpu().numpy()),
                },
                "fc2": {
                    "weights": jnp.asarray(state_dict["mlp.fc2.weight"].detach().cpu().numpy()),
                },
            },
        },
    )

    with torch.no_grad():
        expected_output = torch_block(torch_input).detach().cpu().numpy()
    actual_output = np.asarray(lalamo_block(lalamo_input))

    np.testing.assert_allclose(actual_output, expected_output, rtol=2e-4, atol=2e-4)
