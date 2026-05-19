from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, Float

from lalamo.weight_matrix import CompressionImplementation, WeightMatrixSpec

__all__ = [
    "QuantizedSpec",
]


@dataclass(frozen=True)
class QuantizedSpec(WeightMatrixSpec):
    @property
    @abstractmethod
    def input_block_size(self) -> int: ...

    @property
    @abstractmethod
    def output_block_size(self) -> int: ...

    def quantize_block(
        self,
        weights: Float[Array, "out_block_channels in_block_channels"],
    ) -> Float[Array, "out_block_channels in_block_channels"]:
        expected_shape = (self.output_block_size, self.input_block_size)
        if weights.shape != expected_shape:
            raise ValueError(f"Expected quantization block shape {expected_shape}, got {weights.shape}")
        compressed = self.compress(weights, implementation=CompressionImplementation.TRAINING, is_sharded=False)
        return compressed.decompress()

    @property
    @abstractmethod
    def rate(self) -> float: ...

    @property
    @abstractmethod
    def distortion(self) -> float: ...
