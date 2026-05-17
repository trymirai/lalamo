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
        weights: Float[Array, "*blocks out_block_channels in_block_channels"],
    ) -> Float[Array, "*blocks out_block_channels in_block_channels"]:
        expected_shape = (self.output_block_size, self.input_block_size)
        *_, output_block_size, input_block_size = weights.shape
        actual_shape = (output_block_size, input_block_size)
        if actual_shape != expected_shape:
            raise ValueError(f"Expected quantization block shape {expected_shape}, got {actual_shape}")
        compressed = self.compress(weights, implementation=CompressionImplementation.INFERENCE, is_sharded=False)
        return compressed.decompress()

    @property
    @abstractmethod
    def rate(self) -> float: ...

    @property
    @abstractmethod
    def distortion(self) -> float: ...
