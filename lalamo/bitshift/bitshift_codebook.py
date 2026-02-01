import equinox as eqx

from .bitshift_codebook_config import BitshiftCodebookConfig

__all__ = [
    "BitShiftCodebook",
]


class BitShiftCodebook(eqx.Module):
    config: BitshiftCodebookConfig = eqx.field(static=True)
