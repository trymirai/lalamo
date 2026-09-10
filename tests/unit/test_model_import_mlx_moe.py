"""Tests for loading MoE experts stored in the MLX affine format (stacked, one tensor per projection).

Expectations come from the MLX format definition, not from running the loader: an affine-quantized
weight is `value = code * scale + bias`, with one scale/bias pair per group of `group_size` inputs,
and codes packed little-end-first into uint32 words. Every assertion below recomputes that in numpy
from the tensors handed to the loader, so a transposed axis, a wrong bit width or a swapped up/gate
half cannot be masked -- those are exactly the failures that load cleanly and only show up as quality
loss much later.
"""

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange
from jaxtyping import Array

from lalamo.compressed.mlx import MLXMatrixForInference
from lalamo.initializer import EmptyInitializer
from lalamo.model_import.loaders.huggingface import load_moe
from lalamo.module import ForwardPassMode, Keychain, LogicalAxis
from lalamo.modules.activations import SiLU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import (
    DenseMLPConfig,
    MixtureOfExperts,
    MixtureOfExpertsConfig,
    MLPForwardPassConfig,
    SoftmaxRouting,
    _take_moe_expert_leaf,
)
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrix
from tests.helpers import make_sharding, make_test_sharding_config

BITS = 4
GROUP_SIZE = 4
MODEL_DIM = 8
EXPERT_HIDDEN_DIM = 8
NUM_EXPERTS = 4
NUM_ACTIVE = 2
NUM_GROUPS = MODEL_DIM // GROUP_SIZE
PACKED_COLS = MODEL_DIM * BITS // 32
MAX_CODE = 2**BITS - 1


def _moe_template() -> MixtureOfExperts:
    linear_config = LinearConfig()
    expert_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )
    config = MixtureOfExpertsConfig(
        expert_config=expert_config,
        router_config=linear_config,
        routing_function=SoftmaxRouting(),
        num_routed_experts=NUM_EXPERTS,
        num_active_routed_experts=NUM_ACTIVE,
        router_has_biases=False,
        num_shared_experts=1,
        expert_hidden_dim=EXPERT_HIDDEN_DIM,
        gate_config=linear_config,
    )
    initializer = EmptyInitializer(default_dtype=jnp.bfloat16, sharding_config=make_test_sharding_config())
    return config.init(initializer, model_dim=MODEL_DIM, hidden_dim=EXPERT_HIDDEN_DIM)


def _pack(codes: np.ndarray) -> Array:
    # MLX packs `32 // bits` codes per uint32 word, lowest code in the least significant bits.
    values_per_word = 32 // BITS
    grouped = rearrange(
        jnp.asarray(codes, dtype=jnp.uint32),
        "... (words values) -> ... words values",
        values=values_per_word,
    )
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(BITS)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint32)


def _dequantize(codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    # The MLX affine contract: value = code * scale + bias, one (scale, bias) per group of inputs.
    return codes * np.repeat(scales, GROUP_SIZE, axis=-1) + np.repeat(biases, GROUP_SIZE, axis=-1)


def _materialize(matrix: WeightMatrix, expert: int) -> np.ndarray:
    """Rows of one expert's matrix, read out through the public `dot` on the standard basis.

    The expert axis is sharded, so a bare `leaf[expert]` has no unambiguous output sharding. Reuse
    the very helper the decode path uses, which pins the result to a replicated layout -- that also
    means this reads the matrix exactly the way the model will.
    """
    per_expert = jax.tree_util.tree_map(partial(_take_moe_expert_leaf, index=jnp.asarray(expert)), matrix)
    basis = jnp.eye(per_expert.shape[-1], dtype=jnp.bfloat16)
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())
    columns = jax.vmap(lambda row: per_expert.dot(row, keychain=keychain))(basis)
    return np.asarray(jax.device_get(columns), dtype=np.float32).T


class _Triplet:
    """One MLX tensor group as the checkpoint stores it, plus the values it is supposed to decode to."""

    def __init__(self, codes: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> None:
        self.codes = codes
        self.scales = scales
        self.biases = biases
        self.values = _dequantize(codes, scales, biases)

    def as_weights(self, path: ParameterPath) -> dict[str, Array]:
        return {
            path / "weight": _pack(self.codes),
            path / "scales": jnp.asarray(self.scales, dtype=jnp.bfloat16),
            path / "biases": jnp.asarray(self.biases, dtype=jnp.bfloat16),
        }


def _triplet(rows: int, *, leading: tuple[int, ...], seed: int, cols: int = MODEL_DIM) -> _Triplet:
    rng = np.random.default_rng(seed)
    shape = (*leading, rows, cols)
    group_shape = (*leading, rows, cols // GROUP_SIZE)
    codes = rng.integers(0, MAX_CODE + 1, size=shape).astype(np.float32)
    # Powers of two keep the affine reconstruction exact in bfloat16, so the test measures the
    # loader's layout rather than rounding noise.
    scales = 2.0 ** rng.integers(-3, 2, size=group_shape).astype(np.float32)
    biases = rng.integers(-2, 3, size=group_shape).astype(np.float32)
    return _Triplet(codes, scales, biases)


def _constant_triplet(rows: int, value: float, *, leading: tuple[int, ...], cols: int = MODEL_DIM) -> _Triplet:
    shape = (*leading, rows, cols)
    group_shape = (*leading, rows, cols // GROUP_SIZE)
    return _Triplet(
        np.full(shape, value, dtype=np.float32),
        np.ones(group_shape, dtype=np.float32),
        np.zeros(group_shape, dtype=np.float32),
    )


def _checkpoint(
    experts_key: str,
    up: _Triplet,
    gate: _Triplet,
    down: _Triplet,
    router_weights: Array,
) -> tuple[Mapping[str, Array], ParameterPath]:
    path = ParameterPath()
    experts_path = path / experts_key
    shared_path = path / "shared_expert"
    shared_up = _constant_triplet(EXPERT_HIDDEN_DIM, 3.0, leading=())
    shared_gate = _constant_triplet(EXPERT_HIDDEN_DIM, 5.0, leading=())
    shared_down = _constant_triplet(MODEL_DIM, 7.0, leading=(), cols=EXPERT_HIDDEN_DIM)
    weights: dict[str, Array] = {
        path / "gate" / "weight": router_weights,
        path / "shared_expert_gate" / "weight": jnp.zeros((1, MODEL_DIM), dtype=jnp.bfloat16),
        **up.as_weights(experts_path / "up_proj"),
        **gate.as_weights(experts_path / "gate_proj"),
        **down.as_weights(experts_path / "down_proj"),
        **shared_up.as_weights(shared_path / "up_proj"),
        **shared_gate.as_weights(shared_path / "gate_proj"),
        **shared_down.as_weights(shared_path / "down_proj"),
    }
    return weights, path


def _load(experts_key: str = "experts", *, seed: int = 0) -> tuple[MixtureOfExperts, _Triplet, _Triplet, _Triplet]:
    up = _triplet(EXPERT_HIDDEN_DIM, leading=(NUM_EXPERTS,), seed=seed)
    gate = _triplet(EXPERT_HIDDEN_DIM, leading=(NUM_EXPERTS,), seed=seed + 100)
    down = _triplet(MODEL_DIM, leading=(NUM_EXPERTS,), seed=seed + 200, cols=EXPERT_HIDDEN_DIM)
    router_weights = jnp.asarray(
        np.random.default_rng(seed + 300).standard_normal((NUM_EXPERTS, MODEL_DIM)),
        dtype=jnp.bfloat16,
    )
    weights, path = _checkpoint(experts_key, up, gate, down, router_weights)
    return load_moe(_moe_template(), weights, path), up, gate, down


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("experts_key", ["experts", "switch_mlp"], ids=["hf-name", "mlx-name"])
def test_stacked_mlx_experts_load_as_mlx_matrices_under_either_name(experts_key: str) -> None:
    # MLX conversions rename the stack to `switch_mlp`; the payload is identical, so both names must
    # produce the same kind of matrix with the shapes the module template declares.
    loaded, *_ = _load(experts_key)

    up_weights = loaded.routed_experts.up_projection.weights
    down_weights = loaded.routed_experts.down_projection.weights
    assert isinstance(up_weights, MLXMatrixForInference)
    assert isinstance(down_weights, MLXMatrixForInference)
    assert up_weights.shape == (NUM_EXPERTS, 2 * EXPERT_HIDDEN_DIM, MODEL_DIM)
    assert down_weights.shape == (NUM_EXPERTS, MODEL_DIM, EXPERT_HIDDEN_DIM)
    # Both are inferred from the tensor shapes alone, never read from config.json.
    assert up_weights.spec.bits == BITS
    assert up_weights.spec.group_size == GROUP_SIZE
    assert down_weights.spec.bits == BITS
    assert down_weights.spec.group_size == GROUP_SIZE


@pytest.mark.usefixtures("fake_mesh")
def test_up_occupies_the_first_half_of_the_fused_rows_and_gate_the_second() -> None:
    # The fused matrix is [up; gate] everywhere else in this loader, and the two halves are only
    # distinguishable by value -- a swap keeps every shape valid and silently ruins the MLP.
    up = _constant_triplet(EXPERT_HIDDEN_DIM, 1.0, leading=(NUM_EXPERTS,))
    gate = _constant_triplet(EXPERT_HIDDEN_DIM, 2.0, leading=(NUM_EXPERTS,))
    down = _constant_triplet(MODEL_DIM, 3.0, leading=(NUM_EXPERTS,), cols=EXPERT_HIDDEN_DIM)
    weights, path = _checkpoint("switch_mlp", up, gate, down, jnp.zeros((NUM_EXPERTS, MODEL_DIM), jnp.bfloat16))

    loaded = load_moe(_moe_template(), weights, path)

    fused = _materialize(loaded.routed_experts.up_projection.weights, expert=0)
    np.testing.assert_array_equal(fused[:EXPERT_HIDDEN_DIM], np.ones((EXPERT_HIDDEN_DIM, MODEL_DIM), np.float32))
    np.testing.assert_array_equal(fused[EXPERT_HIDDEN_DIM:], 2.0 * np.ones((EXPERT_HIDDEN_DIM, MODEL_DIM), np.float32))


@pytest.mark.usefixtures("fake_mesh")
def test_every_expert_decodes_to_the_affine_reconstruction_of_its_own_tensors() -> None:
    # value = code * scale + bias, per group, per expert (MLX affine definition). Recomputed in numpy
    # from the tensors handed to the loader; catches a shared scale, an off-by-one group boundary and
    # a leading axis collapsed across experts.
    loaded, up, gate, down = _load("switch_mlp", seed=7)

    for expert in range(NUM_EXPERTS):
        expected_fused = np.concatenate([up.values[expert], gate.values[expert]], axis=0)
        np.testing.assert_allclose(
            _materialize(loaded.routed_experts.up_projection.weights, expert),
            expected_fused,
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            _materialize(loaded.routed_experts.down_projection.weights, expert),
            down.values[expert],
            rtol=0,
            atol=0,
        )


@pytest.mark.usefixtures("fake_mesh")
def test_the_single_shared_expert_gets_the_leading_axis_the_template_expects() -> None:
    # The shared expert is stored without an expert axis, but the module keeps one of size 1; adding
    # it in the wrong place would either crash or silently transpose the matrix.
    loaded, *_ = _load("switch_mlp")

    assert loaded.shared_experts is not None
    shared_up = loaded.shared_experts.up_projection.weights
    assert isinstance(shared_up, MLXMatrixForInference)
    assert shared_up.shape == (1, 2 * EXPERT_HIDDEN_DIM, MODEL_DIM)
    materialized = _materialize(shared_up, expert=0)
    np.testing.assert_array_equal(materialized[:EXPERT_HIDDEN_DIM], 3.0)
    np.testing.assert_array_equal(materialized[EXPERT_HIDDEN_DIM:], 5.0)


@pytest.mark.usefixtures("fake_mesh")
def test_the_router_stays_full_precision_when_the_experts_are_quantized() -> None:
    # This is the machine-checkable form of "routing interventions are isolated": they act on router
    # logits only, so compressing the experts must leave the router bit-for-bit untouched.
    loaded, *_ = _load("switch_mlp", seed=3)

    router_weights = loaded.router.weights
    assert isinstance(router_weights, FullPrecisionMatrix)
    weights, path = _checkpoint(
        "switch_mlp",
        _constant_triplet(EXPERT_HIDDEN_DIM, 1.0, leading=(NUM_EXPERTS,)),
        _constant_triplet(EXPERT_HIDDEN_DIM, 2.0, leading=(NUM_EXPERTS,)),
        _constant_triplet(MODEL_DIM, 3.0, leading=(NUM_EXPERTS,), cols=EXPERT_HIDDEN_DIM),
        router_weights.weights,
    )
    other = load_moe(_moe_template(), weights, path)
    assert isinstance(other.router.weights, FullPrecisionMatrix)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(other.router.weights.weights)),
        np.asarray(jax.device_get(router_weights.weights)),
    )


@pytest.mark.usefixtures("fake_mesh")
def test_a_loaded_mlx_moe_runs_and_agrees_between_the_decode_and_prefill_paths() -> None:
    # Compressed experts disable the ragged kernel, so prefill falls back to the chunked scatter path
    # (mlp.py `use_ragged`). Both paths must still compute the same thing on the same single token --
    # this is the end-to-end check that stacked MLX weights survive the gather in either branch.
    loaded, *_ = _load("switch_mlp", seed=11)
    inputs = jax.device_put(
        jnp.arange(2 * 1 * MODEL_DIM, dtype=jnp.float32).reshape(2, 1, MODEL_DIM).astype(jnp.bfloat16) / 10,
        make_sharding((LogicalAxis.BATCH, None, None)),
    )
    lengths = jax.device_put(jnp.array([1, 1], dtype=jnp.int32), make_sharding((LogicalAxis.BATCH,)))

    decoded = loaded(
        inputs,
        forward_pass_config=MLPForwardPassConfig(mode=ForwardPassMode.SINGLE_TOKEN),
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    ).outputs
    prefilled = loaded(
        inputs,
        lengths_without_padding=lengths,
        forward_pass_config=MLPForwardPassConfig(moe_chunk_size_ratio=0.5),
        keychain=Keychain.init(1, sharding_config=make_test_sharding_config()),
    ).outputs

    assert jnp.all(jnp.isfinite(jax.device_get(decoded)))
    np.testing.assert_allclose(
        np.asarray(jax.device_get(prefilled), dtype=np.float32),
        np.asarray(jax.device_get(decoded), dtype=np.float32),
        # bfloat16 accumulation differs between the two kernels; one mantissa step is 2**-8.
        rtol=2.0**-8,
        atol=2.0**-8,
    )
