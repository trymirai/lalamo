from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.s_surface import SSurfaceKind, SSurfaceMatrix, SSurfaceSpec
from lalamo.compressed.utils.s_gains import SScaleAxis
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of, with_sharding
from lalamo.weight_matrix import Layout, ShapeDtypeSpec
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.fixture(params=list(SSurfaceKind))
def matrix(request: pytest.FixtureRequest) -> SSurfaceMatrix:
    kind = request.param
    layout = Layout.INPUT_OUTPUT if kind == SSurfaceKind.D4 else Layout.OUTPUT_INPUT
    filename = "s_surface_i4.npz" if kind == SSurfaceKind.I4 else "s_surfaces_hyb036.npz"
    prefix = "" if kind == SSurfaceKind.I4 else f"{kind}_"
    states = 1 << SSurfaceSpec(kind, layout).code_bits
    with np.load(Path(__file__).parent / "data" / filename) as data:
        table = (
            jnp.asarray(data["table"])
            if kind == SSurfaceKind.D4
            else jnp.arange(1 - states, states, 2, dtype=jnp.int8)[:, None]
        )
        return (
            SSurfaceMatrix(
                spec=SSurfaceSpec(kind, layout),
                sharding_config=make_test_sharding_config(),
                is_sharded=True,
                codes=jnp.asarray(data[f"{prefix}codes"]),
                row_scales=jax.lax.bitcast_convert_type(jnp.asarray(data[f"{prefix}row_scale_bits"]), jnp.bfloat16),
                ladder_indices=jnp.asarray(data[f"{prefix}ladder_indices"]),
                ladder=jnp.asarray(data["ladder"]),
                table=table,
                signs=jnp.asarray(data["signs"]),
            )
            .astype(jnp.float32)
            .switch_sharding_config(make_test_sharding_config())
        )


def reference_weights(matrix: SSurfaceMatrix) -> np.ndarray:
    rows, columns = matrix.shape
    codes = np.asarray(matrix.codes)
    if matrix.spec.kind == SSurfaceKind.D4:
        levels = np.asarray(matrix.table)[codes].reshape(rows, columns)
    elif matrix.spec.kind == SSurfaceKind.I3:
        levels = np.array(
            [
                [2 * ((int.from_bytes(row.tobytes(), "little") >> (3 * col)) & 7) - 7 for col in range(columns)]
                for row in codes
            ],
            dtype=np.float32,
        )
    else:
        levels = np.array(
            [
                [2 * ((int(row[col // 2]) >> (4 if col % 2 == 0 else 0)) & 15) - 15 for col in range(columns)]
                for row in codes
            ],
            dtype=np.float32,
        )
    packed = np.asarray(matrix.ladder_indices)
    indices = np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, -1)
    scales = (
        np.asarray(matrix.row_scales, dtype=np.float32)[:, None] * np.asarray(matrix.ladder, dtype=np.float32)[indices]
    )
    rotated = levels * np.repeat(scales, 64, axis=1)
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(5):
        h = np.block([[h, h], [h, -h]])
    return (rotated.reshape(rows, -1, 32) @ (h / np.sqrt(np.float32(32)))).reshape(rows, columns) * np.asarray(
        matrix.signs
    )


def test_s_surface_matches_torch_fitted_rows(matrix: SSurfaceMatrix) -> None:
    kind = matrix.spec.kind
    filename = "s_surface_i4.npz" if kind == SSurfaceKind.I4 else "s_surfaces_hyb036.npz"
    prefix = "" if kind == SSurfaceKind.I4 else f"{kind}_"
    with np.load(Path(__file__).parent / "data" / filename) as data:
        expected = data[f"{prefix}expected"]
    if matrix.spec.layout == Layout.INPUT_OUTPUT:
        expected = expected.T
    np.testing.assert_allclose(matrix.decompress(), expected, atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize("matrix", [SSurfaceKind.I4], indirect=True)
def test_i4_qat_stages_match_torch_bf16_weights(matrix: SSurfaceMatrix) -> None:
    axes = (SScaleAxis.ROW, SScaleAxis.ROW, SScaleAxis.ROW, SScaleAxis.COLUMN)
    with np.load(Path(__file__).parent / "data/s_qat_rows.npz") as data:
        matrix = replace(
            matrix,
            spec=replace(matrix.spec, post_gain_axes=axes),
            post_gains=tuple(jnp.asarray(data[name]) for name in ("i4_t0", "i4_t5", "i4_row_gain", "i4_column_gain")),
        ).switch_sharding_config(matrix.sharding_config)
        expected = jax.lax.bitcast_convert_type(jnp.asarray(data["i4_expected_bits"]), jnp.bfloat16)
    # The producer's dense H32 matmul leaves tiny residuals at exact cancellation zeros.
    actual = np.asarray(matrix.decompress())
    nonzero = actual != 0
    np.testing.assert_array_equal(actual[nonzero], np.asarray(expected)[nonzero])
    np.testing.assert_allclose(actual[~nonzero], np.asarray(expected)[~nonzero], atol=1e-8, rtol=0)
    template = ShapeDtypeSpec().compress(
        dummy_array(matrix.shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    np.testing.assert_array_equal(restored.decompress(), matrix.decompress())
    inputs = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    np.testing.assert_allclose(matrix.dot(inputs, keychain=keychain), expected.astype(jnp.float32) @ inputs, atol=2e-5)
    embedding = replace(matrix, spec=replace(matrix.spec, layout=Layout.INPUT_OUTPUT))
    np.testing.assert_array_equal(
        embedding.lookup_embedding(0, keychain=keychain, dtype=jnp.float32), np.asarray(matrix.decompress())[0]
    )


def test_s_surface_native_reload_preserves_payload(matrix: SSurfaceMatrix) -> None:
    shape = matrix.decompress().shape
    template = matrix.spec.compress(
        dummy_array(shape, matrix.dtype, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
        is_sharded=False,
    )
    restored = template.load_exported(matrix.export())
    for old, new in zip(jax.tree.leaves(matrix), jax.tree.leaves(restored), strict=True):
        assert old.dtype == new.dtype
        np.testing.assert_array_equal(old, new)


def test_s_surface_native_load_keeps_checkpoint_dtype(matrix: SSurfaceMatrix) -> None:
    matrix = matrix.astype(jnp.bfloat16)
    template = ShapeDtypeSpec(matrix.spec.layout).compress(
        dummy_array(matrix.decompress().shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    assert restored.dtype == jnp.bfloat16
    for old, new in zip(jax.tree.leaves(matrix), jax.tree.leaves(restored), strict=True):
        assert old.dtype == new.dtype
        np.testing.assert_array_equal(old, new)


def test_s_surface_lookup_preserves_requested_rows(matrix: SSurfaceMatrix) -> None:
    if matrix.spec.layout != Layout.INPUT_OUTPUT:
        with pytest.raises(ValueError, match="input-output"):
            matrix.lookup_embedding(0, keychain=Keychain.init(0, sharding_config=matrix.sharding_config))
        return
    expected = reference_weights(matrix)
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    for index in (2, jnp.array([3, 0, 1], dtype=jnp.int32)):
        actual = matrix.lookup_embedding(index, keychain=keychain)
        np.testing.assert_allclose(actual, expected[np.asarray(index)], atol=2e-7, rtol=1e-6)


def test_s_readout_crosses_decode_batch_boundary(matrix: SSurfaceMatrix) -> None:
    matrix = replace(matrix, spec=SSurfaceSpec(matrix.spec.kind, Layout.OUTPUT_INPUT))
    repeated = replace(
        matrix,
        codes=jnp.asarray(np.tile(np.asarray(matrix.codes), (80, 1))),
        row_scales=jnp.asarray(np.tile(np.asarray(matrix.row_scales), 80)),
        ladder_indices=jnp.asarray(np.tile(np.asarray(matrix.ladder_indices), (80, 1))),
    ).switch_sharding_config(matrix.sharding_config)
    x = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    expected = np.tile(np.asarray(matrix.decompress() @ x), 80)
    actual = repeated.dot(x, keychain=Keychain.init(0, sharding_config=matrix.sharding_config))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["fully_sharded_data_parallel", "tensor_parallel", "data_parallel"])
def test_s_surface_batched_dot_sharding(matrix: SSurfaceMatrix, mode: str) -> None:
    config = getattr(ShardingConfig, mode)(jax.devices("cpu")[:4])
    with jax.set_mesh(config.mesh):
        matrix = replace(matrix, spec=SSurfaceSpec(matrix.spec.kind, Layout.OUTPUT_INPUT))
        matrix = matrix.switch_sharding_config(config)
        inputs = with_sharding(
            jnp.linspace(-1, 1, 4 * matrix.shape[1]).reshape(4, -1),
            config.resolve_sharding((LogicalAxis.BATCH, None)),
        )
        keychain = Keychain.init(0, sharding_config=config)
        actual = jax.jit(jax.vmap(lambda x: matrix.dot(x, keychain=keychain)))(inputs)
        expected = np.asarray(inputs) @ np.asarray(matrix.decompress()).T
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert sharding_of(actual).spec == sharding_of(inputs).spec
