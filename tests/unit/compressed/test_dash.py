import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.dash import DashCodes, dash_matrix, dash_mlx_matrix
from tests.common import assert_close
from tests.helpers import make_test_sharding_config


def random_codes(key: jax.Array, shape: tuple[int, int], code_range: DashCodes) -> jax.Array:
    top = code_range.value
    return 2 * jax.random.randint(key, shape, 0, (top + 1) // 2) - top


@pytest.mark.parametrize(
    ("code_range", "groups"), [(DashCodes.TWO_BIT, 1), (DashCodes.TWO_BIT, 3), (DashCodes.THREE_BIT, 3)]
)
def test_dash_mlx_matrix_decodes_codes_times_scales_exactly(code_range: DashCodes, groups: int) -> None:
    rows, columns = 16, 384
    codes = random_codes(jax.random.key(0), (rows, columns), code_range)
    scales = jax.random.uniform(jax.random.key(1), (rows, groups), minval=0.01, maxval=0.1).astype(jnp.float16)
    matrix = dash_mlx_matrix(
        codes, scales.astype(jnp.float32), code_range=code_range, sharding_config=make_test_sharding_config()
    )

    assert (matrix.spec.bits, matrix.spec.group_size) == (code_range.mlx_bits, 128)
    reference = codes.astype(jnp.float32) * jnp.repeat(scales.astype(jnp.float32), columns // groups, axis=1)
    np.testing.assert_array_equal(np.asarray(matrix.decompress()), np.asarray(reference))


def test_dash_matrix_matches_the_dense_fold_of_the_exporter() -> None:
    # The Dash exporter folds W = ((codes * scale) @ blockdiag(H128)) * signs with the orthonormal Sylvester H128.
    rows, columns = 8, 256
    codes = random_codes(jax.random.key(2), (rows, columns), DashCodes.TWO_BIT)
    scales = jax.random.uniform(jax.random.key(3), (rows, 1), minval=0.01, maxval=0.1)
    signs = jnp.where(jax.random.bernoulli(jax.random.key(4), shape=(columns,)), 1, -1)
    sylvester = np.ones((1, 1))
    while sylvester.shape[0] < 128:
        sylvester = np.block([[sylvester, sylvester], [sylvester, -sylvester]])
    block_hadamard = np.kron(np.eye(columns // 128), sylvester / np.sqrt(128))
    reference = (np.asarray(codes.astype(jnp.float32) * scales) @ block_hadamard) * np.asarray(signs)

    matrix = dash_matrix(codes, scales, signs, sharding_config=make_test_sharding_config())

    assert_close(result=matrix.decompress(), reference=jnp.asarray(reference, dtype=jnp.float32), atol=1e-6, rtol=1e-6)
