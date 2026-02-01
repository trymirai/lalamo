import equinox as eqx
from jaxtyping import Array, Float, Int

from .bitshift_codebook_config import BitShiftCodebookConfig
from .bitshift_codebook_states import BitShiftCodebookStates
from .lut_provider import LUTProvider

__all__ = [
    "BitShiftCodebook",
]


class BitShiftCodebook(eqx.Module):
    config: BitShiftCodebookConfig = eqx.field(static=True)
    lut_provider: LUTProvider
    codebook_states: BitShiftCodebookStates
    reconstructed_states: Float[Array, "values_per_step number_of_states"]

    @property
    def lut(self) -> Float[Array, "values_per_step number_of_states"]:
        return self.lut_provider.lut

    @classmethod
    def create(cls, config: BitShiftCodebookConfig, lut_provider: LUTProvider) -> "BitShiftCodebook":
        codebook_states = BitShiftCodebookStates.create(config)
        reconstructed_states = cls._reconstruct(lut_provider.lut, codebook_states.states)

        return cls(
            config=config,
            lut_provider=lut_provider,
            codebook_states=codebook_states,
            reconstructed_states=reconstructed_states,
        )

    @staticmethod
    def _reconstruct(
        lut: Float[Array, "values_per_step number_of_states"], states: Int[Array, "..."]
    ) -> Float[Array, "values_per_step ..."]:
        return lut[:, states]
