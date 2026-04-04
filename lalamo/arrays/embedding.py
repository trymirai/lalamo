from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
from einops import rearrange

from lalamo.quantization import QuantizationMode, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Float, Int


class CompressedEmbedding(eqx.Module):
    @property
    @abc.abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abc.abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def activation_precision(self) -> DTypeLike: ...

    @abc.abstractmethod
    def dequantize(self) -> Float[Array, "vocabulary channels"]: ...


class FullPrecisionEmbedding(CompressedEmbedding):
    weights: Float[Array, "vocabulary channels"]

    @property
    def vocab_size(self) -> int:
        return self.weights.shape[0]

    @property
    def model_dim(self) -> int:
        return self.weights.shape[1]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.weights.dtype

    def dequantize(self) -> Float[Array, "vocabulary channels"]:
        return self.weights


class MLXQuantizedEmbedding(CompressedEmbedding):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, "vocabulary groups"]
    biases: Float[Array, "vocabulary groups"]
    group_size: int = eqx.field(static=True)
    quantization_mode: QuantizationMode = eqx.field(static=True)

    @property
    def vocab_size(self) -> int:
        return self.weights.shape[0]

    @property
    def model_dim(self) -> int:
        return self.weights.shape[1]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.scales.dtype

    def dequantize(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.quantization_mode)
        grouped = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )
        scales = rearrange(self.scales, "vocab groups -> vocab groups 1")
        biases = rearrange(self.biases, "vocab groups -> vocab groups 1")
        return rearrange(
            grouped * scales + biases,
            "vocab groups elements -> vocab (groups elements)",
        )

    def pack(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.weights, self.quantization_mode)
        casted = quantized.astype(self.quantization_mode.dtype)
        if self.quantization_mode == QuantizationMode.UINT4:
            return jax_uint4_to_packed_uint8(casted)
        return casted
