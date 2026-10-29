import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from lalamo.compressed import SymRow2Spec
from lalamo.compressed.symrow2 import pack_layout16, unpack_layout16
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def test_pack_round_trip() -> None:
    codes = np.random.default_rng(0).integers(0, 4, (48, 384), dtype=np.uint8)
    packed = pack_layout16(jnp.asarray(codes))
    assert packed.shape == (48, 96)
    assert (np.asarray(unpack_layout16(packed)) == codes).all()


@pytest.mark.parametrize(
    ("row", "col", "byte_index", "byte_value"),
    [(0, 5, 17, 3), (17, 130, 1570, 3), (9, 47, 183, 12), (31, 255, 2047, 192)],
)
def test_pack_places_code_in_layout16_position(row: int, col: int, byte_index: int, byte_value: int) -> None:
    codes = np.zeros((32, 256), dtype=np.uint8)
    codes[row, col] = 3
    packed = np.asarray(pack_layout16(jnp.asarray(codes))).reshape(-1)
    assert np.flatnonzero(packed).tolist() == [byte_index]
    assert packed[byte_index] == byte_value


def test_fit_matches_reference_and_scales_are_bf16() -> None:
    weights = jax.random.normal(jax.random.key(0), (32, 256)).at[3].set(0.0)
    matrix = SymRow2Spec().compress(weights, sharding_config=make_test_sharding_config())

    w = np.asarray(weights)
    best_error, best_scales = np.full(32, np.inf), np.ones(32, np.float32)
    for fraction in np.array([0.6, 0.7, 0.8, 0.9, 1.0], np.float32):
        s = np.maximum(fraction * np.abs(w).max(1) / 1.5, 1e-6).astype(ml_dtypes.bfloat16).astype(np.float32)
        error = ((s[:, None] * (np.clip(np.rint(w / s[:, None] + 1.5), 0, 3) - 1.5) - w) ** 2).sum(1)
        best_scales, best_error = np.where(error < best_error, s, best_scales), np.minimum(error, best_error)
    codes = np.clip(np.rint(w / best_scales[:, None] + 1.5), 0, 3)

    assert matrix.scales.dtype == jnp.bfloat16
    assert (np.asarray(matrix.scales.astype(jnp.float32)) == best_scales).all()
    assert (np.asarray(unpack_layout16(matrix.weights)) == codes).all()


@pytest.mark.parametrize("shape", [(24, 128), (16, 192), (2, 16, 128)])
def test_rejects_unsupported_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="SymRow2 needs"):
        SymRow2Spec().compress(jnp.zeros(shape), sharding_config=make_test_sharding_config())


def test_spec_json_tag() -> None:
    assert SymRow2Spec().to_json() == {"type": "SymRow2Spec", "layout": "output_input"}
