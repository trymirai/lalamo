import jax
import pytest
from jaxtyping import PRNGKeyArray

RANDOM_SEED = 42


@pytest.fixture
def rng_key() -> PRNGKeyArray:
    return jax.random.PRNGKey(RANDOM_SEED)
