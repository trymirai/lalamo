from dataclasses import replace
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.compressed.s_trellis import STrellisMatrix, STrellisSpec, full_rotation
from lalamo.compressed.utils.s_gains import SScaleAxis
from lalamo.module import Keychain
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.sharding import LogicalAxis, ShardingConfig, sharding_of, with_sharding
from lalamo.weight_matrix import MatmulConfig, ShapeDtypeSpec
from tests.helpers import make_sharding, make_test_sharding_config
from tests.unit.compressed.s_trellis_fixture import load_saved_trellis

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.fixture(
    params=[
        ("v2_k2", 2, 4, 0),
        ("v2_k3", 2, 6, 0),
        ("v4_k2", 4, 8, 64),
        ("v4_k2_connected", 4, 8, 0),
        ("v2_k4_connected", 2, 8, 0),
    ]
)
def matrix(request: pytest.FixtureRequest) -> STrellisMatrix:
    name, width, bits, restart = request.param
    return load_saved_trellis(name, STrellisSpec(width, bits, restart))


def reference_states(matrix: STrellisMatrix) -> np.ndarray:
    spec = matrix.spec
    blocks, steps, block_bytes = spec.tape_shape(matrix.shape[1])
    tapes = np.asarray(matrix.codes).reshape(matrix.shape[0], blocks, block_bytes)
    states = np.empty((matrix.shape[0], blocks, steps), dtype=np.uint32)
    for row in range(matrix.shape[0]):
        for block in range(blocks):
            tape = tapes[row, block].tobytes()
            state = int.from_bytes(tape[:2], "little")
            symbols = int.from_bytes(tape[2:], "little")
            states[row, block, 0] = state
            for step in range(1, steps):
                symbol = (symbols >> ((step - 1) * spec.transition_bits)) & ((1 << spec.transition_bits) - 1)
                state = ((state << spec.transition_bits) | symbol) & 65535
                states[row, block, step] = state
    return states.reshape(matrix.shape[0], -1)


def test_saved_s_tapes_and_two_stage_scales_decode_exactly(matrix: STrellisMatrix) -> None:
    states = reference_states(matrix)
    expected = np.asarray(matrix.table)[states].reshape(matrix.shape)
    expected = expected * np.asarray(matrix.scales, dtype=np.float32)[:, None]
    expected = expected * np.asarray(matrix.gains, dtype=np.float32)[:, None]

    np.testing.assert_array_equal(matrix.spec.states(matrix.codes, matrix.shape[1]), states)
    np.testing.assert_array_equal(matrix.rotated_weights(), expected)
    np.testing.assert_array_equal(eqx.filter_jit(lambda m: m.rotated_weights())(matrix), expected)


@pytest.mark.parametrize(
    "name,width,bits,restart",
    [
        ("v2_k2", 2, 4, 0),
        ("v2_k3", 2, 6, 0),
        ("v4_k2", 4, 8, 64),
        ("v4_k2_connected", 4, 8, 0),
        ("v2_k4_connected", 2, 8, 0),
    ],
)
def test_s_tapes_reproduce_torch_hessian_fit(
    name: str, width: Literal[2, 4], bits: Literal[4, 6, 8], restart: Literal[0, 64]
) -> None:
    matrix = load_saved_trellis(name, STrellisSpec(width, bits, restart))
    filename = "s_trellis_muse.npz" if name.endswith("_connected") else "s_trellis_hyb036.npz"
    with np.load(Path(__file__).parent / "data" / filename) as data:
        np.testing.assert_array_equal(matrix.rotated_weights(), data[f"{name}_rotated"])


def test_qat_rounds_after_rotation_and_preserves_saved_gains() -> None:
    original = load_saved_trellis("v4_k2_connected", STrellisSpec(4, 8, 0))
    with np.load(Path(__file__).parent / "data/s_qat_rows.npz") as data:
        gain = jnp.asarray(data["muse_up_gain"])
    matrix = replace(
        original,
        spec=replace(original.spec, post_gain_axes=(SScaleAxis.ROW,)),
        post_gains=(gain,),
    ).switch_sharding_config(original.sharding_config)
    expected = (np.asarray(original.decompress(), dtype=np.float32) * np.asarray(gain)[:, None]).astype(jnp.bfloat16)
    np.testing.assert_array_equal(matrix.decompress(), expected)
    template = ShapeDtypeSpec().compress(
        dummy_array(matrix.shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    np.testing.assert_array_equal(restored.decompress(), expected)
    assert restored.dtype == jnp.bfloat16


def test_reround_fp32_scales_and_kept_old_gain_stages_survive_native_load() -> None:
    original = load_saved_trellis("v4_k2", STrellisSpec(4, 8, 64))
    extra = jnp.array([0.995, 1.003, 1.004, 1.011], dtype=jnp.float32)
    matrix = replace(
        original,
        spec=replace(original.spec, scale_dtype="float32", pre_gain_count=1),
        scales=original.scales.astype(jnp.float32) * jnp.float32(1.0003),
        pre_gains=(extra,),
    ).switch_sharding_config(original.sharding_config)
    states = reference_states(matrix)
    expected = np.asarray(matrix.table)[states].reshape(matrix.shape)
    expected = expected * np.asarray(matrix.scales)[:, None]
    expected = expected * np.asarray(matrix.gains, dtype=np.float32)[:, None]
    expected = expected * np.asarray(extra)[:, None]
    np.testing.assert_array_equal(matrix.rotated_weights(), expected)
    np.testing.assert_array_equal(eqx.filter_jit(lambda m: m.rotated_weights())(matrix), expected)
    template = ShapeDtypeSpec().compress(
        dummy_array(matrix.shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    assert isinstance(restored, STrellisMatrix)
    np.testing.assert_array_equal(restored.scales, matrix.scales)
    np.testing.assert_array_equal(restored.rotated_weights(), expected)


def test_s_checkpoint_export_load_and_resharding_preserve_payload(matrix: STrellisMatrix) -> None:
    template = matrix.spec.compress(
        dummy_array(matrix.shape, matrix.dtype, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
        is_sharded=False,
    )
    restored = template.load_exported(matrix.export())
    assert isinstance(restored, STrellisMatrix)
    for expected, actual in zip(jax.tree.leaves(matrix), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(restored.rotated_weights(), matrix.rotated_weights())


def test_full_rotation_matches_explicit_kronecker_product() -> None:
    values = np.arange(96, dtype=np.float32).reshape(4, 24) / 17
    q = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))[0].astype(np.float32)
    h = np.ones((1, 1), dtype=np.float32)
    for _ in range(3):
        h = np.block([[h, h], [h, -h]])
    rotation = np.kron(h / np.sqrt(np.float32(8)), q)
    np.testing.assert_allclose(full_rotation(jnp.asarray(values), jnp.asarray(q)), values @ rotation, atol=3e-6)


def test_s_matrix_forward_and_transpose_use_saved_rotation(matrix: STrellisMatrix) -> None:
    config = MatmulConfig.for_tracer_tests()
    keychain = Keychain.init(0, sharding_config=matrix.sharding_config)
    weights = matrix.decompress().astype(jnp.float32)
    x = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    y = jnp.arange(matrix.shape[0], dtype=jnp.float32)
    np.testing.assert_allclose(
        matrix.dot(x, keychain=keychain, forward_pass_config=config), weights @ x, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        matrix.dot(y, keychain=keychain, forward_pass_config=config, transposed=True),
        weights.T @ y,
        atol=1e-5,
        rtol=1e-5,
    )


def test_s_matrix_refuses_to_refit_dense_weights() -> None:
    with pytest.raises(ValueError, match="saved parameters"):
        STrellisSpec(2, 4, 0).compress(jnp.ones((4, 16)), sharding_config=make_test_sharding_config())


def test_s_trellis_native_load_keeps_checkpoint_dtype(matrix: STrellisMatrix) -> None:
    template = ShapeDtypeSpec().compress(
        dummy_array(matrix.shape, None, make_sharding((None, None))),
        sharding_config=matrix.sharding_config,
    )
    restored = template.load_exported(matrix.export())
    assert restored.dtype == jnp.bfloat16
    for old, new in zip(jax.tree.leaves(matrix), jax.tree.leaves(restored), strict=True):
        assert old.dtype == new.dtype
        np.testing.assert_array_equal(old, new)


def test_s_trellis_dot_crosses_decode_batch_boundary(matrix: STrellisMatrix) -> None:
    # Keep this boundary check independent of BF16 rounding at rotation ties.
    matrix = matrix.astype(jnp.float32)
    repeated = replace(
        matrix,
        codes=jnp.asarray(np.tile(np.asarray(matrix.codes), (80, 1))),
        scales=jnp.asarray(np.tile(np.asarray(matrix.scales), 80)),
        gains=jnp.asarray(np.tile(np.asarray(matrix.gains), 80)),
    ).switch_sharding_config(matrix.sharding_config)
    x = jnp.linspace(-1, 1, matrix.shape[1], dtype=jnp.float32)
    expected = np.tile(np.asarray(matrix.decompress().astype(x.dtype) @ x), 80)
    actual = repeated.dot(x, keychain=Keychain.init(0, sharding_config=matrix.sharding_config))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["fully_sharded_data_parallel", "tensor_parallel", "data_parallel"])
def test_s_trellis_batched_dot_sharding(matrix: STrellisMatrix, mode: str) -> None:
    config = getattr(ShardingConfig, mode)(jax.devices("cpu")[:4])
    with jax.set_mesh(config.mesh):
        matrix = matrix.switch_sharding_config(config).astype(jnp.float32)
        inputs = with_sharding(
            jnp.linspace(-1, 1, 4 * matrix.shape[1]).reshape(4, -1),
            config.resolve_sharding((LogicalAxis.BATCH, None)),
        )
        keychain = Keychain.init(0, sharding_config=config)
        actual = jax.jit(jax.vmap(lambda x: matrix.dot(x, keychain=keychain)))(inputs)
        expected = np.asarray(inputs) @ np.asarray(matrix.decompress()).T
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert sharding_of(actual).spec == sharding_of(inputs).spec
