import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.audio.neutts.codec_modules import NeuCodecFSQConfig, NeuCodecResidualFSQConfig


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
