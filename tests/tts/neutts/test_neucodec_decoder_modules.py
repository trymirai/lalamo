import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules.audio.neutts.codec_modules import NeuCodecFSQConfig


def test_neucodec_fsq_indices_to_codes_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    finite_scalar_quantization = pytest.importorskip("vector_quantize_pytorch.finite_scalar_quantization")
    indices = [[0, 1, 255, 4**7, 16383, 65535]]
    torch_fsq = finite_scalar_quantization.FSQ(levels=[4] * 8)
    lalamo_fsq = NeuCodecFSQConfig(levels=(4,) * 8, precision=jnp.float32).empty()

    expected_codes = torch_fsq.indices_to_codes(torch.tensor(indices, dtype=torch.long)).detach().cpu().numpy()
    actual_codes = np.asarray(lalamo_fsq.indices_to_codes(jnp.asarray(indices, dtype=jnp.int32)))

    np.testing.assert_allclose(actual_codes, expected_codes, rtol=0, atol=0)
