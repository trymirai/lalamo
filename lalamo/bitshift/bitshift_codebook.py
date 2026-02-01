import equinox as eqx

from .bitshift_codebook_config import BitshiftCodebookConfig
from .lut_provider import LUTProvider

__all__ = [
    "BitShiftCodebook",
]


class BitShiftCodebook(eqx.Module):
    config: BitshiftCodebookConfig = eqx.field(static=True)
    lut_provider: LUTProvider = eqx.field(static=True)
