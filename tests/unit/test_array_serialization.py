import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct

from lalamo.compressed import AWQMatrix, AWQSpec, LoRAMatrix, LoRASpec, MLXMatrix, MLXSpec
from lalamo.compressed.utils import (
    expand_last_axis_groups,
    grouped_last_axis_shape,
    round_to_unsigned_grid,
    stochastic_round_to_unsigned_grid,
)
from lalamo.exportable import Exportable
from lalamo.initializer import EmptyInitializer
from lalamo.weight_matrix import (
    FullPrecisionSpec,
    GradientEstimator,
    Layout,
    MatmulConfig,
    WeightMatrix,
    WeightMatrixSpec,
)
from tests.common import assert_close


class MatrixContainer(Exportable, eqx.Module):
    layer: WeightMatrix


def manual_mlx_dequantize(
    weights: jax.Array,
    scales: jax.Array,
    biases: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> jax.Array:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_biases = expand_last_axis_groups(biases, group_size=group_size)
    quantized = jnp.round(jnp.clip((weights - expanded_biases) / expanded_scales, 0, (2**bits) - 1))
    return quantized * expanded_scales + expanded_biases


def manual_awq_dequantize(
    weights: jax.Array,
    scales: jax.Array,
    zero_points: jax.Array,
    *,
    bits: int,
    group_size: int,
) -> jax.Array:
    expanded_scales = expand_last_axis_groups(scales, group_size=group_size)
    expanded_zero_points = expand_last_axis_groups(zero_points, group_size=group_size)
    quantized_weights = jnp.round(jnp.clip(weights, 0, (2**bits) - 1))
    quantized_zero_points = jnp.round(jnp.clip(expanded_zero_points, 0, (2**bits) - 1))
    return (quantized_weights - quantized_zero_points) * expanded_scales


def test_awq_spec_json_round_trip() -> None:
    spec = AWQSpec(bits=4, group_size=4, layout=Layout.INPUT_OUTPUT)
    restored = WeightMatrixSpec.from_json(spec.to_json())
    assert restored == spec


def test_mlx_spec_json_round_trip() -> None:
    spec = MLXSpec(bits=8, group_size=4, layout=Layout.OUTPUT_INPUT)
    restored = WeightMatrixSpec.from_json(spec.to_json())
    assert restored == spec


def test_awq_export_round_trip() -> None:
    spec = AWQSpec(bits=4, group_size=4)
    original = MatrixContainer(layer=spec.compress(jax.random.normal(jax.random.key(0), (4, 8))))
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, AWQMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_mlx_export_round_trip() -> None:
    spec = MLXSpec(bits=4, group_size=4)
    original = MatrixContainer(layer=spec.compress(jax.random.normal(jax.random.key(1), (4, 8))))
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, MLXMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_lora_export_round_trip() -> None:
    spec = LoRASpec(rank=2)
    original = MatrixContainer(
        layer=LoRAMatrix(
            spec=spec,
            down=jnp.ones((4, 2)),
            up=jnp.arange(16, dtype=jnp.float32).reshape(2, 8),
        ),
    )
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, LoRAMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_awq_input_output_layout() -> None:
    weights = jax.random.normal(jax.random.key(2), (5, 8))
    matrix = AWQSpec(bits=4, group_size=4, layout=Layout.INPUT_OUTPUT).compress(weights)
    token_index = 3
    vector = jax.random.normal(jax.random.key(3), (8,))

    assert_close(
        result=matrix.lookup_embedding(token_index, dequant_key=jax.random.key(4)),
        reference=matrix.decompress()[token_index, :],
    )
    assert_close(
        result=matrix.dot(vector, dequant_key=jax.random.key(5)),
        reference=vector @ matrix.decompress(),
    )


def test_awq_compresses_with_deterministic_min_max_reference() -> None:
    weights = jnp.array([[-1.0, 0.0, 0.5, 1.0]], dtype=jnp.float32)
    matrix = AWQSpec(bits=4, group_size=4).compress(weights)

    expected_scales = jnp.array([[2 / 15]], dtype=jnp.float32)
    expected_zero_points = jnp.array([[7.5]], dtype=jnp.float32)
    expected_master_weights = weights / expected_scales + expected_zero_points

    assert_close(result=matrix.scales, reference=expected_scales)
    assert_close(result=matrix.zero_points, reference=expected_zero_points)
    assert_close(result=matrix.weights, reference=expected_master_weights)
    assert_close(
        result=matrix.decompress(),
        reference=manual_awq_dequantize(
            expected_master_weights,
            expected_scales,
            expected_zero_points,
            bits=4,
            group_size=4,
        ),
    )


def test_awq_decompress_backpropagates_to_weights_and_zero_points_not_scales() -> None:
    matrix = AWQMatrix(
        spec=AWQSpec(bits=4, group_size=1),
        weights=jnp.array([[1.25, 2.75]], dtype=jnp.float32),
        scales=jnp.ones((1, 2), dtype=jnp.float32),
        zero_points=jnp.array([[0.25, 1.25]], dtype=jnp.float32),
    )

    def loss(weights: jax.Array, scales: jax.Array, zero_points: jax.Array) -> jax.Array:
        test_matrix = AWQMatrix(spec=matrix.spec, weights=weights, scales=scales, zero_points=zero_points)
        return jnp.sum(test_matrix.decompress())

    weights_grad, scales_grad, zero_points_grad = jax.grad(loss, argnums=(0, 1, 2))(
        matrix.weights,
        matrix.scales,
        matrix.zero_points,
    )

    assert_close(result=weights_grad, reference=jnp.ones_like(matrix.weights))
    assert_close(result=scales_grad, reference=jnp.zeros_like(matrix.scales))
    assert_close(result=zero_points_grad, reference=-jnp.ones_like(matrix.zero_points))


def test_awq_stochastic_rounding_quantizes_zero_points() -> None:
    num_samples = 4096
    matrix = AWQMatrix(
        spec=AWQSpec(bits=4, group_size=1, layout=Layout.INPUT_OUTPUT),
        weights=jnp.full((1, num_samples), 1.25, dtype=jnp.float32),
        scales=jnp.ones((1, num_samples), dtype=jnp.float32),
        zero_points=jnp.full((1, num_samples), 0.25, dtype=jnp.float32),
    )

    result = matrix.lookup_embedding(
        0,
        dequant_key=jax.random.key(22),
        forward_pass_config=MatmulConfig(gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING),
    )

    assert abs(jnp.mean(result).item() - 1.0) < 0.04


def test_mlx_input_output_layout() -> None:
    weights = jax.random.normal(jax.random.key(6), (8, 6))
    matrix = MLXSpec(bits=4, group_size=4, layout=Layout.INPUT_OUTPUT).compress(weights)
    token_index = 1
    vector = jax.random.normal(jax.random.key(7), (6,))

    assert_close(
        result=matrix.lookup_embedding(token_index, dequant_key=jax.random.key(8)),
        reference=matrix.decompress()[token_index, :],
    )
    assert_close(
        result=matrix.dot(vector, dequant_key=jax.random.key(9)),
        reference=vector @ matrix.decompress(),
    )


def test_full_precision_input_output_lookup_uses_stored_row() -> None:
    weights = jax.random.normal(jax.random.key(19), (8, 6))
    matrix = FullPrecisionSpec(layout=Layout.INPUT_OUTPUT).compress(weights)
    token_index = 2

    assert_close(
        result=matrix.lookup_embedding(token_index, dequant_key=jax.random.key(20)),
        reference=matrix.decompress()[token_index, :],
    )


def test_mlx_compresses_with_deterministic_min_max_reference() -> None:
    weights = jnp.array(
        [
            [0.0, 0.2, 0.55, 1.0],
            [4.0, 4.2, 4.55, 5.0],
        ],
        dtype=jnp.float32,
    )
    matrix = MLXSpec(bits=4, group_size=4).compress(weights)

    expected_scales = jnp.array([[1 / 15], [1 / 15]], dtype=jnp.float32)
    expected_biases = jnp.array([[0.0], [4.0]], dtype=jnp.float32)

    assert_close(result=matrix.scales, reference=expected_scales)
    assert_close(result=matrix.biases, reference=expected_biases)
    assert_close(
        result=matrix.decompress(),
        reference=manual_mlx_dequantize(weights, expected_scales, expected_biases, bits=4, group_size=4),
    )


def test_mlx_affine_scale_avoids_half_precision_overflow() -> None:
    max_float16 = jnp.finfo(jnp.float16).max
    weights = jnp.array([[-max_float16, 0.0, max_float16 / 2, max_float16]], dtype=jnp.float16)

    matrix = MLXSpec(bits=4, group_size=4).compress(weights)

    assert jnp.all(jnp.isfinite(matrix.scales))


def test_mlx_rejects_group_size_that_does_not_divide_stored_last_dimension() -> None:
    weights = jnp.ones((2, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="divisible"):
        MLXSpec(bits=4, group_size=4).compress(weights)


@pytest.mark.parametrize(
    ("value", "incoming_gradient", "expected_gradient"),
    [
        (-1.0, 2.0, 2.0),
        (-1.0, -2.0, 0.0),
        (4.0, -2.0, -2.0),
        (16.0, 2.0, 0.0),
        (16.0, -2.0, -2.0),
    ],
)
def test_unsigned_grid_round_backward_respects_clipped_gradient_direction(
    value: float,
    incoming_gradient: float,
    expected_gradient: float,
) -> None:
    result = jax.grad(lambda scalar: round_to_unsigned_grid(scalar, bits=4) * incoming_gradient)(
        jnp.array(value, dtype=jnp.float32),
    )

    assert_close(result=result, reference=jnp.array(expected_gradient, dtype=jnp.float32))


def test_stochastic_unsigned_grid_round_backward_uses_same_clipped_surrogate() -> None:
    result = jax.grad(
        lambda scalar: (
            stochastic_round_to_unsigned_grid(
                scalar,
                bits=4,
                dequant_key=jax.random.key(18),
            )
            * -2.0
        ),
    )(jnp.array(16.0, dtype=jnp.float32))

    assert_close(result=result, reference=jnp.array(-2.0, dtype=jnp.float32))


def test_mlx_decompress_does_not_backpropagate_to_affine_parameters() -> None:
    matrix = MLXMatrix(
        spec=MLXSpec(bits=4, group_size=1),
        weights=jnp.array([[0.25, 1.5, 2.75]], dtype=jnp.float32),
        scales=jnp.ones((1, 3), dtype=jnp.float32),
        biases=jnp.zeros((1, 3), dtype=jnp.float32),
    )

    def loss(scales: jax.Array, biases: jax.Array) -> jax.Array:
        test_matrix = MLXMatrix(spec=matrix.spec, weights=matrix.weights, scales=scales, biases=biases)
        return jnp.sum(test_matrix.decompress())

    scale_grad, bias_grad = jax.grad(loss, argnums=(0, 1))(matrix.scales, matrix.biases)

    assert_close(result=scale_grad, reference=jnp.zeros_like(matrix.scales))
    assert_close(result=bias_grad, reference=jnp.zeros_like(matrix.biases))


@pytest.mark.parametrize(
    ("layout", "output_dim", "input_dim"),
    [
        (Layout.OUTPUT_INPUT, 6, 8),
        (Layout.INPUT_OUTPUT, 12, 6),
    ],
)
def test_mlx_init_shapes_and_shardings_match_full_precision_partition(
    layout: Layout,
    output_dim: int,
    input_dim: int,
) -> None:
    initializer = EmptyInitializer(dtype=jnp.float32)
    full_precision = FullPrecisionSpec(layout=layout).init(initializer, (2,), output_dim, input_dim)
    matrix = MLXSpec(bits=4, group_size=4, layout=layout).init(initializer, (2,), output_dim, input_dim)

    assert isinstance(matrix.weights, ShapeDtypeStruct)
    assert matrix.weights.shape == full_precision.weights.shape
    assert matrix.scales.shape == grouped_last_axis_shape(
        full_precision.weights.shape, group_size=matrix.spec.group_size
    )
    assert matrix.biases.shape == grouped_last_axis_shape(
        full_precision.weights.shape, group_size=matrix.spec.group_size
    )
    assert matrix.weights.sharding == full_precision.weights.sharding
    assert matrix.scales.sharding == full_precision.weights.sharding
    assert matrix.biases.sharding == full_precision.weights.sharding


def test_mlx_lookup_embedding_matches_selected_row_dequantization() -> None:
    weights = jax.random.normal(jax.random.key(10), (8, 6))
    matrix = MLXSpec(bits=4, group_size=4, layout=Layout.INPUT_OUTPUT).compress(weights)
    index = 5

    result = matrix.lookup_embedding(index, dequant_key=jax.random.key(11))
    reference = manual_mlx_dequantize(
        matrix.weights[index, :],
        matrix.scales[index, :],
        matrix.biases[index, :],
        bits=matrix.spec.bits,
        group_size=matrix.spec.group_size,
    )

    assert_close(result=result, reference=reference)


@pytest.mark.parametrize(
    ("layout", "weights", "vector"),
    [
        (
            Layout.OUTPUT_INPUT,
            jax.random.normal(jax.random.key(12), (12, 8)),
            jax.random.normal(jax.random.key(13), (8,)),
        ),
        (
            Layout.INPUT_OUTPUT,
            jax.random.normal(jax.random.key(14), (12, 6)),
            jax.random.normal(jax.random.key(15), (6,)),
        ),
    ],
)
def test_mlx_dot_matches_manual_dequantized_matmul(
    layout: Layout,
    weights: jax.Array,
    vector: jax.Array,
) -> None:
    matrix = MLXSpec(bits=4, group_size=4, layout=layout).compress(weights)
    dequantized_weights = manual_mlx_dequantize(
        matrix.weights,
        matrix.scales,
        matrix.biases,
        bits=matrix.spec.bits,
        group_size=matrix.spec.group_size,
    )
    reference = vector @ dequantized_weights if layout == Layout.INPUT_OUTPUT else dequantized_weights @ vector

    result = matrix.dot(vector, dequant_key=jax.random.key(16))

    assert_close(result=result, reference=reference)


def test_mlx_stochastic_rounding_is_unbiased_on_fractional_bins() -> None:
    num_samples = 4096
    matrix = MLXMatrix(
        spec=MLXSpec(bits=4, group_size=1),
        weights=jnp.full((1, num_samples), 1.25, dtype=jnp.float32),
        scales=jnp.ones((1, num_samples), dtype=jnp.float32),
        biases=jnp.zeros((1, num_samples), dtype=jnp.float32),
    )
    vector = jnp.full((num_samples,), 1 / num_samples, dtype=jnp.float32)

    result = matrix.dot(
        vector,
        dequant_key=jax.random.key(17),
        forward_pass_config=MatmulConfig(gradient_estimator=GradientEstimator.STOCHASTIC_ROUNDING),
    )

    assert abs(result[0].item() - 1.25) < 0.03
