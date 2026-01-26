from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import RegistryABC
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8

from .common import (
    Initializer,
    LalamoConfig,
    LalamoModule,
)
from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfigBase",
    "MLXQuantizedTiedEmbedding",
    "MLXQuantizedTiedEmbeddingConfig",
    "MLXQuantizedUntiedEmbedding",
    "MLXQuantizedUntiedEmbeddingConfig",
    "MLXSemiQuantizedUntiedEmbedding",
    "MLXSemiQuantizedUntiedEmbeddingConfig",
    "QuantizedTiedEmbedding",
    "QuantizedTiedEmbeddingConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class EmbeddingConfigBase(LalamoConfig, RegistryABC):
    input_scale: float | None
    logit_soft_cap: float | None

    @abstractmethod
    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


class EmbeddingBase(LalamoModule):
    activation_precision: DTypeLike = eqx.field(static=True)
    input_scale: float | None = eqx.field(static=True)
    logit_soft_cap: float | None = eqx.field(static=True)

    @abstractmethod
    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]: ...

    @abstractmethod
    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self._prepare_input_weights()[x]
        if self.input_scale is not None:
            result = result * jnp.array(self.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self._prepare_output_weights() @ x
        if self.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.logit_soft_cap)
        return logits


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "TiedEmbedding":
        weights = initializer.normal(1.0, (vocab_size, model_dim), self.precision)
        return TiedEmbedding(
            weights=weights,
            activation_precision=self.precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
        )


class TiedEmbedding(EmbeddingBase):
    weights: Float[Array, "vocabulary channels"]

    @property
    def model_dim(self) -> int:
        _, model_dim = self.weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.weights.shape
        return vocab_size

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.weights

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.weights


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "UntiedEmbedding":
        input_weights = initializer.normal(1.0, (vocab_size, model_dim), self.precision)
        output_weights = initializer.normal(1.0, (vocab_size, model_dim), self.precision)
        return UntiedEmbedding(
            input_weights=input_weights,
            output_weights=output_weights,
            activation_precision=self.precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
        )


class UntiedEmbedding(EmbeddingBase):
    input_weights: Float[Array, "vocabulary channels"]
    output_weights: Float[Array, "channels vocabulary"]

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.input_weights

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.output_weights


@dataclass(frozen=True)
class QuantizedTiedEmbeddingConfig(EmbeddingConfigBase):
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "QuantizedTiedEmbedding":
        scales = initializer.ones((vocab_size,), self.activation_precision)
        weights = initializer.zeros((vocab_size, model_dim), self.activation_precision)
        return QuantizedTiedEmbedding(
            weights=weights,
            scales=scales,
            activation_precision=self.activation_precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
        )


class QuantizedTiedEmbedding(EmbeddingBase):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, " vocabulary"]

    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @property
    def model_dim(self) -> int:
        _, model_dim = self.weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.weights.shape
        return vocab_size

    @property
    def int_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.weights, self.embedding_quantization_mode)
        casted = quantized.astype(self.embedding_quantization_mode.dtype)

        if self.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.embedding_quantization_mode)
        quantized_weights = quantized_weights * self.scales.reshape(-1, 1)
        return quantized_weights

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return super().readout(x)


@dataclass(frozen=True)
class MLXQuantizedTiedEmbeddingConfig(EmbeddingConfigBase):
    group_size: int
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXQuantizedTiedEmbedding":
        model_groups = model_dim // self.group_size
        weights = initializer.zeros((vocab_size, model_dim), self.activation_precision)
        scales = initializer.ones((vocab_size, model_groups), self.activation_precision)
        biases = initializer.zeros((vocab_size, model_groups), self.activation_precision)
        return MLXQuantizedTiedEmbedding(
            weights=weights,
            scales=scales,
            biases=biases,
            activation_precision=self.activation_precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
            group_size=self.group_size,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
        )


class MLXQuantizedTiedEmbedding(EmbeddingBase):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, "vocabulary groups"]
    biases: Float[Array, "vocabulary groups"]

    group_size: int = eqx.field(static=True)
    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @property
    def model_dim(self) -> int:
        _, model_dim = self.weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.weights.shape
        return vocab_size

    @property
    def int_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.weights, self.embedding_quantization_mode)
        casted = quantized.astype(self.embedding_quantization_mode.dtype)

        if self.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.embedding_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )

        scales = rearrange(self.scales, "vocab groups -> vocab groups 1")

        biases = rearrange(self.biases, "vocab groups -> vocab groups 1")

        scaled_grouped_weights = grouped_weights * scales + biases

        result = rearrange(
            scaled_grouped_weights,
            "vocab groups elements -> vocab (groups elements)",
        )
        return result

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return super().readout(x)


@dataclass(frozen=True)
class MLXQuantizedUntiedEmbeddingConfig(EmbeddingConfigBase):
    group_size: int
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXQuantizedUntiedEmbedding":
        model_groups = model_dim // self.group_size
        return MLXQuantizedUntiedEmbedding(
            input_weights=initializer.zeros((vocab_size, model_dim), self.activation_precision),
            input_scales=initializer.ones((vocab_size, model_groups), self.activation_precision),
            input_biases=initializer.zeros((vocab_size, model_groups), self.activation_precision),
            output_weights=initializer.zeros((vocab_size, model_dim), self.activation_precision),
            output_scales=initializer.ones((vocab_size, model_groups), self.activation_precision),
            output_biases=initializer.zeros((vocab_size, model_groups), self.activation_precision),
            activation_precision=self.activation_precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
            group_size=self.group_size,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
        )


class MLXQuantizedUntiedEmbedding(EmbeddingBase):
    input_weights: Float[Array, "vocabulary channels"]
    input_scales: Float[Array, "vocabulary groups"]
    input_biases: Float[Array, "vocabulary groups"]
    output_weights: Float[Array, "vocabulary channels"]
    output_scales: Float[Array, "vocabulary groups"]
    output_biases: Float[Array, "vocabulary groups"]

    group_size: int = eqx.field(static=True)
    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    @property
    def int_input_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.input_weights, self.embedding_quantization_mode)
        casted = quantized.astype(self.embedding_quantization_mode.dtype)

        if self.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    @property
    def int_output_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.output_weights, self.embedding_quantization_mode)
        casted = quantized.astype(self.embedding_quantization_mode.dtype)

        if self.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.input_weights, self.embedding_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )

        scales = rearrange(self.input_scales, "vocab groups -> vocab groups 1")

        biases = rearrange(self.input_biases, "vocab groups -> vocab groups 1")

        scaled_grouped_weights = grouped_weights * scales + biases

        result = rearrange(
            scaled_grouped_weights,
            "vocab groups elements -> vocab (groups elements)",
        )
        return result

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.output_weights, self.embedding_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )

        scales = rearrange(self.output_scales, "vocab groups -> vocab groups 1")

        biases = rearrange(self.output_biases, "vocab groups -> vocab groups 1")

        scaled_grouped_weights = grouped_weights * scales + biases

        result = rearrange(
            scaled_grouped_weights,
            "vocab groups elements -> vocab (groups elements)",
        )
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return super().readout(x)


@dataclass(frozen=True)
class MLXSemiQuantizedUntiedEmbeddingConfig(EmbeddingConfigBase):
    group_size: int
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def init(
        self,
        initializer: Initializer,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXSemiQuantizedUntiedEmbedding":
        model_groups = model_dim // self.group_size
        input_weights = initializer.normal(1.0, (vocab_size, model_dim), self.activation_precision)
        output_weights = initializer.zeros((vocab_size, model_dim), self.activation_precision)
        output_scales = initializer.ones((vocab_size, model_groups), self.activation_precision)
        output_biases = initializer.zeros((vocab_size, model_groups), self.activation_precision)
        return MLXSemiQuantizedUntiedEmbedding(
            input_weights=input_weights,
            output_weights=output_weights,
            output_scales=output_scales,
            output_biases=output_biases,
            activation_precision=self.activation_precision,
            input_scale=self.input_scale,
            logit_soft_cap=self.logit_soft_cap,
            group_size=self.group_size,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
        )


class MLXSemiQuantizedUntiedEmbedding(EmbeddingBase):
    input_weights: Float[Array, "vocabulary channels"]
    output_weights: Float[Array, "vocabulary channels"]
    output_scales: Float[Array, "vocabulary groups"]
    output_biases: Float[Array, "vocabulary groups"]

    group_size: int = eqx.field(static=True)
    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    @property
    def int_output_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.output_weights, self.embedding_quantization_mode)
        casted = quantized.astype(self.embedding_quantization_mode.dtype)

        if self.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.input_weights

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.output_weights, self.embedding_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )

        scales = rearrange(self.output_scales, "vocab groups -> vocab groups 1")

        biases = rearrange(self.output_biases, "vocab groups -> vocab groups 1")

        scaled_grouped_weights = grouped_weights * scales + biases

        result = rearrange(
            scaled_grouped_weights,
            "vocab groups elements -> vocab (groups elements)",
        )
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return super().readout(x)
