import jax
import jax.numpy as jnp
import pytest

from lalamo.arrays.awq import AWQQuantArray
from lalamo.arrays.composite import MixtureArray
from lalamo.arrays.full_precision import FullPrecisionArray
from lalamo.arrays.lora import LoRAArray
from tests.common import assert_close


@pytest.mark.fast
def test_full_precision_round_trip() -> None:
    arr = FullPrecisionArray(weights=jnp.ones((4, 8)))
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


@pytest.mark.fast
def test_awq_round_trip() -> None:
    key = jax.random.key(0)
    arr = AWQQuantArray.compress(jax.random.normal(key, (4, 8)), bits=4, group_size=4)
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


@pytest.mark.fast
def test_lora_round_trip() -> None:
    arr = LoRAArray(
        down=jnp.ones((4, 2)),
        up=jnp.ones((2, 8)),
    )
    data = arr.to_uzu()
    restored = arr.from_uzu(data)
    assert_close(result=restored.materialize(), reference=arr.materialize())


@pytest.mark.fast
def test_mixture_full_precision_and_lora_round_trip() -> None:
    fp = FullPrecisionArray(weights=jnp.ones((4, 8)))
    lora = LoRAArray(down=jnp.ones((4, 2)), up=jnp.ones((2, 8)))
    mixture = MixtureArray(parts=(fp, lora), coefficients=jnp.array([1.0, 0.5]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())


@pytest.mark.fast
def test_mixture_full_precision_and_awq_round_trip() -> None:
    fp = FullPrecisionArray(weights=jnp.ones((4, 8)))
    awq = AWQQuantArray.compress(jax.random.normal(jax.random.key(0), (4, 8)), bits=4, group_size=4)
    mixture = MixtureArray(parts=(fp, awq), coefficients=jnp.array([1.0, 1.0]))

    data = mixture.to_uzu()
    restored = mixture.from_uzu(data)
    assert_close(result=restored.materialize(), reference=mixture.materialize())
