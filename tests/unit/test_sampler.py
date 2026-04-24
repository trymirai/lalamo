import jax.numpy as jnp
import numpy as np

from lalamo.speculator.sampler import GumbelMaxSampler, GumbelSeed


def test_gumbel_seed_derive_and_advance_are_bit_reproducible() -> None:
    base = GumbelSeed(42)
    assert base.derive(1).value == 2485467453280205223
    assert base.derive(2).value == 1937481673661862836
    assert base.derive(10).value == 15375987271337636094
    assert base.advance(0).value == 111486301962
    assert base.advance(17).value == 111486301979

    assert GumbelSeed(0).derive(1).value == 14424038852845472812
    assert GumbelSeed(0xDEADBEEF).derive(7).value == 2803409933834008760


def test_gumbel_max_sampler_matches_single_seed_sample() -> None:
    logits = jnp.asarray(np.random.RandomState(0).randn(32).astype(np.float32))
    seed = GumbelSeed(123)
    single = seed.sample(logits)

    sampler = GumbelMaxSampler()
    batch = sampler.sample(logits[None, :], jnp.asarray([seed.value & 0xFFFFFFFF], dtype=jnp.uint32))
    assert int(batch[0]) == single
