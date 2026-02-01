import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int

from .bitshift_codebook_config import BitShiftCodebookConfig

__all__ = [
    "BitShiftCodebookStates",
]


class BitShiftCodebookStates(eqx.Module):
    states: Int[Array, " number_of_states"]
    state_deltas: Int[Array, " number_of_deltas"]
    candidate_states: Int[Array, "number_of_reduced_states number_of_deltas"]

    @classmethod
    def create(cls, config: BitShiftCodebookConfig) -> "BitShiftCodebookStates":
        states = jnp.arange(config.number_of_states, dtype=jnp.int32)

        state_deltas = jnp.arange(config.number_of_deltas, dtype=jnp.int32)
        state_deltas = state_deltas << (config.state_bits - config.bits_per_step)

        candidate_states = (states >> config.bits_per_step)[:: config.number_of_deltas]
        candidate_states = candidate_states[:, None] + state_deltas[None, :]

        return cls(
            states=states,
            state_deltas=state_deltas,
            candidate_states=candidate_states,
        )
