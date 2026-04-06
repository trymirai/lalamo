from enum import Enum


class QuantFormat(Enum):
    FULL_PRECISION = "full_precision"
    AWQ = "awq"
    MLX = "mlx"

    @property
    def array_class(self) -> "type[CompressedArray]":
        from .awq import AWQQuantArray
        from .full_precision import FullPrecisionArray
        from .mlx import MLXQuantArray

        return {
            QuantFormat.FULL_PRECISION: FullPrecisionArray,
            QuantFormat.AWQ: AWQQuantArray,
            QuantFormat.MLX: MLXQuantArray,
        }[self]
