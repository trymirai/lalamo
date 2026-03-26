from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import QuantArray


class QuantFormat(Enum):
    FULL_PRECISION = "full_precision"
    AWQ = "awq"
    MLX = "mlx"

    @property
    def array_class(self) -> type[QuantArray]:
        from .awq import AWQQuantArray
        from .full_precision import FullPrecisionArray
        from .mlx import MLXQuantArray

        return {
            QuantFormat.FULL_PRECISION: FullPrecisionArray,
            QuantFormat.AWQ: AWQQuantArray,
            QuantFormat.MLX: MLXQuantArray,
        }[self]
