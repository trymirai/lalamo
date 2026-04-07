from abc import abstractmethod
from dataclasses import dataclass

from jaxtyping import Array, Float

from lalamo.modules.common import Initializer

from .base import CompressedArray


@dataclass(frozen=True)
class CompressedArraySpec:
    @abstractmethod
    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray: ...

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> CompressedArray: ...


@dataclass(frozen=True)
class FullPrecisionSpec(CompressedArraySpec):
    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray:
        from .full_precision import FullPrecisionArray

        return FullPrecisionArray.compress(weights)

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> CompressedArray:
        from .full_precision import FullPrecisionArray

        return FullPrecisionArray(
            weights=initializer.normal(1.0, (*leading_dims, out_channels, in_channels), initializer.precision),
        )


@dataclass(frozen=True)
class AWQSpec(CompressedArraySpec):
    bits: int
    group_size: int

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray:
        from .awq import AWQQuantArray

        return AWQQuantArray.compress(weights, bits=self.bits, group_size=self.group_size)

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> CompressedArray:
        from .awq import AWQQuantArray

        num_groups = in_channels // self.group_size
        return AWQQuantArray(
            weights=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            zero_points=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
            bits=self.bits,
            group_size=self.group_size,
        )


@dataclass(frozen=True)
class MLXSpec(CompressedArraySpec):
    bits: int
    group_size: int

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray:
        from .mlx import MLXQuantArray

        return MLXQuantArray.compress(weights, bits=self.bits, group_size=self.group_size)

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> CompressedArray:
        from .mlx import MLXQuantArray

        num_groups = in_channels // self.group_size
        return MLXQuantArray(
            weights=initializer.zeros((*leading_dims, out_channels, in_channels), initializer.precision),
            scales=initializer.ones((*leading_dims, out_channels, num_groups), initializer.precision),
            biases=initializer.zeros((*leading_dims, out_channels, num_groups), initializer.precision),
            bits=self.bits,
            group_size=self.group_size,
        )


@dataclass(frozen=True)
class LoRASpec(CompressedArraySpec):
    rank: int

    def compress(self, weights: Float[Array, "... out_channels in_channels"]) -> CompressedArray:
        raise NotImplementedError("LoRA does not support compression from full-precision weights")

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        out_channels: int,
        in_channels: int,
    ) -> CompressedArray:
        from .lora import LoRAArray

        return LoRAArray(
            down=initializer.zeros((*leading_dims, out_channels, self.rank), initializer.precision),
            up=initializer.zeros((*leading_dims, self.rank, in_channels), initializer.precision),
        )
