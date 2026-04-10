import jax
import jax.numpy as jnp

from lalamo.arrays.awq import AWQSpec
from lalamo.arrays.composite import MixtureArray
from lalamo.arrays.full_precision import FullPrecisionArray
from lalamo.arrays.lora import LoRAArray
from tests.common import assert_close


def test_full_precision_round_trip() -> None:
    arr = FullPrecisionArray(weights=jnp.ones((4, 8)))
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
    fp = FullPrecisionArray(weights=jnp.ones((4, 8)))
    lora = LoRAArray(down=jnp.ones((4, 2)), up=jnp.ones((2, 8)))
    mixture = MixtureArray(parts=(fp, lora), coefficients=jnp.array([1.0, 0.5]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())


def test_cross_type_awq_from_full_precision_skeleton() -> None:
    key = jax.random.key(42)
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(key, (4, 8)))
    data = awq.to_uzu()
    skeleton = FullPrecisionArray(weights=jnp.zeros((4, 8)))
    restored = skeleton.from_uzu(data)
    assert isinstance(restored, type(awq))
    assert_close(result=restored.materialize(), reference=awq.materialize())


def test_mixture_full_precision_and_awq_round_trip() -> None:
    fp = FullPrecisionArray(weights=jnp.ones((4, 8)))
    awq = AWQSpec(bits=4, group_size=4, dtype=jnp.float32).compress(jax.random.normal(jax.random.key(0), (4, 8)))
    mixture = MixtureArray(parts=(fp, awq), coefficients=jnp.array([1.0, 1.0]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())
