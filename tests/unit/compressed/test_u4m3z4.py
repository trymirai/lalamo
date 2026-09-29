import jax.numpy as jnp
import pytest

from lalamo.compressed.u4m3z4 import U4M3Z4QuantizedWeights


@pytest.mark.parametrize(("rows", "group_size"), [(4, 32), (5, 64)])
def test_export_byte_layout(rows: int, group_size: int) -> None:
    cols = 4 * group_size
    codes = jnp.tile(jnp.array([0, 15, 2, 3], dtype=jnp.uint8), (rows, cols // 4)).at[1, 0].set(1)
    multipliers = jnp.ones((rows, 4), dtype=jnp.uint8).at[0, :2].set(jnp.array([8, 2], dtype=jnp.uint8))
    multipliers = multipliers.at[1, 0].set(3)
    zero_points = jnp.zeros((rows, 4), dtype=jnp.uint8).at[0, :2].set(jnp.array([15, 1], dtype=jnp.uint8))
    zero_points = zero_points.at[1, 0].set(2)
    output_scales = jnp.arange(1, rows + 1, dtype=jnp.float32)

    exported = U4M3Z4QuantizedWeights(codes, output_scales, multipliers, zero_points, group_size).export()

    expected_weights = jnp.tile(jnp.array([0xF0, 0x32], dtype=jnp.uint8), (rows, cols // 4))
    expected_weights = expected_weights.at[1, 0].set(0xF1)
    expected_multipliers_and_zero_points = jnp.zeros((4, rows + (-rows % 4)), dtype=jnp.uint8)
    expected_multipliers_and_zero_points = expected_multipliers_and_zero_points.at[0, 0].set(0x7F)
    expected_multipliers_and_zero_points = expected_multipliers_and_zero_points.at[1, 0].set(0x09)
    expected_multipliers_and_zero_points = expected_multipliers_and_zero_points.at[0, 1].set(0x12)
    assert set(exported.arrays) == {"weights", "scales", "multipliers_and_zero_points"}
    assert jnp.array_equal(exported.arrays["weights"], expected_weights)
    assert jnp.array_equal(exported.arrays["multipliers_and_zero_points"], expected_multipliers_and_zero_points)
    assert jnp.array_equal(exported.arrays["scales"], output_scales)
    assert exported.metadata == {
        "spec": {
            "type": "U4M3Z4Spec",
            "bits": 4,
            "group_size": group_size,
            "multiplier_bits": 3,
            "zero_point_bits": 4,
            "layout": "output_input",
        },
    }


@pytest.mark.parametrize("cols", [127, 160])
def test_export_rejects_unaligned_k(cols: int) -> None:
    with pytest.raises(ValueError, match="columns divisible"):
        U4M3Z4QuantizedWeights(
            jnp.zeros((4, cols), dtype=jnp.uint8),
            jnp.ones((4,), dtype=jnp.float32),
            jnp.ones((4, 4), dtype=jnp.uint8),
            jnp.zeros((4, 4), dtype=jnp.uint8),
        )


@pytest.mark.parametrize(
    ("code", "multiplier", "zero_point", "message"),
    [(16, 1, 0, "codes"), (0, 0, 0, "multipliers"), (0, 1, 16, "zero_points")],
)
def test_export_rejects_unrepresentable_values(code: int, multiplier: int, zero_point: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        U4M3Z4QuantizedWeights(
            jnp.full((4, 128), code, dtype=jnp.uint8),
            jnp.ones((4,), dtype=jnp.float32),
            jnp.full((4, 4), multiplier, dtype=jnp.uint8),
            jnp.full((4, 4), zero_point, dtype=jnp.uint8),
        )
