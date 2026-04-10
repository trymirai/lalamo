import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.arrays.awq import AWQArray, AWQSpec
from lalamo.arrays.mixture import MixtureArray
from lalamo.arrays.full_precision import FullPrecisionArray, FullPrecisionSpec
from lalamo.arrays.lora import LoRAArray
from lalamo.arrays.mlx import MLXArray, MLXSpec
from lalamo.serialization import UzuSerializable
from tests.common import assert_close


def test_full_precision_round_trip() -> None:
    arr = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.ones((4, 8)))
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


def test_awq_round_trip() -> None:
    key = jax.random.key(0)
    arr = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


def test_lora_round_trip() -> None:
    arr = LoRAArray(
        down=jnp.ones((4, 2)),
        up=jnp.ones((2, 8)),
    )
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


def test_mixture_full_precision_and_lora_round_trip() -> None:
    fp = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.ones((4, 8)))
    lora = LoRAArray(down=jnp.ones((4, 2)), up=jnp.ones((2, 8)))
    mixture = MixtureArray(parts=(fp, lora), coefficients=jnp.array([1.0, 0.5]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())


def test_cross_type_awq_from_full_precision_skeleton() -> None:
    key = jax.random.key(42)
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    data = awq.to_uzu()
    skeleton = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8)))
    restored = skeleton.from_uzu(data)
    assert isinstance(restored, type(awq))
    assert_close(result=restored.materialize(), reference=awq.materialize())


def test_cross_type_mlx_from_full_precision_skeleton() -> None:
    key = jax.random.key(99)
    mlx = MLXSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    data = mlx.to_uzu()
    skeleton = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8)))
    restored = skeleton.from_uzu(data)
    assert isinstance(restored, MLXArray)
    assert_close(result=restored.materialize(), reference=mlx.materialize())


def test_cross_type_module_with_multiple_quant_types() -> None:
    class TwoLayerModule(UzuSerializable, eqx.Module):
        layer_a: FullPrecisionArray | AWQArray
        layer_b: FullPrecisionArray | MLXArray

    key = jax.random.key(0)
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    mlx = MLXSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (8, 4)))

    original = TwoLayerModule(layer_a=awq, layer_b=mlx)
    data = original.to_uzu()

    fp_skeleton = TwoLayerModule(
        layer_a=FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8))),
        layer_b=FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((8, 4))),
    )
    restored = fp_skeleton.from_uzu(data)

    assert isinstance(restored.layer_a, AWQArray)
    assert isinstance(restored.layer_b, MLXArray)
    assert_close(result=restored.layer_a.materialize(), reference=original.layer_a.materialize())
    assert_close(result=restored.layer_b.materialize(), reference=original.layer_b.materialize())


def test_cross_type_module_partial_quantization() -> None:
    class ThreeLayerModule(UzuSerializable, eqx.Module):
        first: FullPrecisionArray | MLXArray
        second: FullPrecisionArray
        third: FullPrecisionArray | MLXArray

    key = jax.random.key(1)
    mlx = MLXSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    fp = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jax.random.normal(key, (4, 4)))

    original = ThreeLayerModule(first=mlx, second=fp, third=mlx)
    data = original.to_uzu()

    skeleton = ThreeLayerModule(
        first=FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8))),
        second=FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 4))),
        third=FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8))),
    )
    restored = skeleton.from_uzu(data)

    assert isinstance(restored.first, MLXArray)
    assert isinstance(restored.second, FullPrecisionArray)
    assert isinstance(restored.third, MLXArray)
    assert_close(result=restored.second.materialize(), reference=fp.materialize())


def test_mixture_full_precision_and_awq_round_trip() -> None:
    fp = FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.ones((4, 8)))
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(jax.random.key(0), (4, 8)))
    mixture = MixtureArray(parts=(fp, awq), coefficients=jnp.array([1.0, 1.0]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())


def test_mixture_cross_type_from_full_precision_skeleton() -> None:
    key = jax.random.key(7)
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    mlx = MLXSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    mixture = MixtureArray(parts=(awq, mlx), coefficients=jnp.array([0.6, 0.4]))

    data = mixture.to_uzu()

    fp_skeleton = MixtureArray(
        parts=(
            FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8))),
            FullPrecisionArray(spec=FullPrecisionSpec(), weights=jnp.zeros((4, 8))),
        ),
        coefficients=jnp.zeros(2),
    )
    restored = fp_skeleton.from_uzu(data)

    assert isinstance(restored.parts[0], AWQArray)
    assert isinstance(restored.parts[1], MLXArray)
    assert_close(result=restored.materialize(), reference=mixture.materialize())
