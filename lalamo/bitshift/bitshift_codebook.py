import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .bitshift_codebook_config import BitShiftCodebookConfig
from .lut_provider import LUTProvider

__all__ = [
    "BitShiftCodebook",
]


class BitShiftCodebook(eqx.Module):
    config: BitShiftCodebookConfig = eqx.field(static=True)
    lut_provider: LUTProvider
    states: Int[Array, " number_of_states"]
    transitions: Int[Array, "number_of_reduced_states transitions_per_state"]

    @property
    def lut(self) -> Float[Array, "chunk_size number_of_states"]:
        return self.lut_provider.lut

    @classmethod
    def create(cls, config: BitShiftCodebookConfig, lut_provider: LUTProvider) -> "BitShiftCodebook":
        states = jnp.arange(config.number_of_states, dtype=jnp.int32)

        transition_offsets = jnp.arange(config.transitions_per_state, dtype=jnp.int32)
        transition_offsets = transition_offsets << (config.state_bits - config.bits_per_step)

        transitions = (states >> config.bits_per_step)[:: config.transitions_per_state]
        transitions = transitions[:, None] + transition_offsets[None, :]
        assert transitions.shape == (config.number_of_reduced_states, config.transitions_per_state)

        return cls(
            config=config,
            lut_provider=lut_provider,
            states=states,
            transitions=transitions,
        )

    def reconstruct(self, states: Int[Array, "..."]) -> Float[Array, "chunk_size ..."]:
        return self.lut[:, states]
