from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, dummy_array
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8, jax_uint8_to_unpacked_uint4

from .common import (
    LalamoModule,
    register_config_union,
)
from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "MLXQuantizedTiedEmbedding",
    "MLXQuantizedTiedEmbeddingConfig",
    "QuantizedTiedEmbedding",
    "QuantizedTiedEmbeddingConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
]


@dataclass(frozen=True)
class EmbeddingConfigBase:
    input_scale: float | None
    logit_soft_cap: float | None

    @abstractmethod
    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "EmbeddingBase": ...

    @abstractmethod
    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


class EmbeddingBase[ConfigT: EmbeddingConfigBase](LalamoModule[ConfigT]):
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
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self._prepare_output_weights() @ x
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "TiedEmbedding":
        weights = jax.random.normal(key, (vocab_size, model_dim), dtype=self.precision)
        return TiedEmbedding(config=self, weights=weights)

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "TiedEmbedding":
        weights = dummy_array((vocab_size, model_dim), dtype=self.precision)
        return TiedEmbedding(config=self, weights=weights)


class TiedEmbedding(EmbeddingBase[TiedEmbeddingConfig]):
    weights: Float[Array, "vocabulary channels"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def __post_init__(self) -> None:
        if self.config.precision != self.weights.dtype:
            raise ValueError(
                f"Embedding dtype {self.weights.dtype} does not match the specified precision {self.config.precision}",
            )

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

    def export_weights(self) -> ParameterTree:
        return {"weights": self.weights}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(self, weights=weights["weights"])


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "UntiedEmbedding":
        input_key, output_key = jax.random.split(key)
        input_weights = jax.random.normal(input_key, (vocab_size, model_dim), dtype=self.precision)
        output_weights = jax.random.normal(output_key, (vocab_size, model_dim), dtype=self.precision)
        return UntiedEmbedding(
            config=self,
            input_weights=input_weights,
            output_weights=output_weights,
        )

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "UntiedEmbedding":
        input_weights = dummy_array((vocab_size, model_dim), dtype=self.precision)
        output_weights = dummy_array((vocab_size, model_dim), dtype=self.precision)
        return UntiedEmbedding(
            config=self,
            input_weights=input_weights,
            output_weights=output_weights,
        )


class UntiedEmbedding(EmbeddingBase[UntiedEmbeddingConfig]):
    input_weights: Float[Array, "vocabulary channels"]
    output_weights: Float[Array, "channels vocabulary"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    def __post_init__(self) -> None:
        if self.config.precision != self.input_weights.dtype:
            raise ValueError(
                f"Embedding dtype {self.input_weights.dtype} does not match",
                f" the specified precision {self.config.precision}",
            )
        if self.config.precision != self.output_weights.dtype:
            raise ValueError(
                f"Embedding dtype {self.output_weights.dtype} does not match"
                f" the specified precision {self.config.precision}",
            )
        input_vocab_size, input_model_dim = self.input_weights.shape
        output_vocab_size, output_model_dim = self.output_weights.shape
        if input_vocab_size != output_vocab_size:
            raise ValueError(
                f"Input vocab size {input_vocab_size} does not match the output vocab size {output_vocab_size}",
            )
        if input_model_dim != output_model_dim:
            raise ValueError(
                f"Input model dim {input_model_dim} does not match the output model dim {output_model_dim}",
            )

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.input_weights

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self.output_weights

    def export_weights(self) -> ParameterTree:
        return {
            "input_weights": self.input_weights,
            "output_weights": self.output_weights,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            input_weights=weights["input_weights"],
            output_weights=weights["output_weights"],
        )


@dataclass(frozen=True)
class QuantizedTiedEmbeddingConfig(EmbeddingConfigBase):
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "QuantizedTiedEmbedding":
        min_val, max_val = self.embedding_quantization_mode.range
        min_abs_val = min(abs(min_val), abs(max_val))
        scale = 1 / min_abs_val
        scales = scale * jnp.ones(vocab_size, dtype=self.activation_precision)
        weights = jax.random.normal(key, (vocab_size, model_dim), dtype=self.activation_precision)
        weights = quantize_weights(weights * min_abs_val, self.embedding_quantization_mode)
        return QuantizedTiedEmbedding(config=self, weights=weights, scales=scales)

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "QuantizedTiedEmbedding":
        scales = dummy_array(vocab_size, dtype=self.activation_precision)
        weights = dummy_array((vocab_size, model_dim), dtype=self.activation_precision)
        return QuantizedTiedEmbedding(config=self, weights=weights, scales=scales)


class QuantizedTiedEmbedding(EmbeddingBase[QuantizedTiedEmbeddingConfig]):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, " vocabulary"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    @property
    def model_dim(self) -> int:
        _, model_dim = self.weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.weights.shape
        return vocab_size

    def __post_init__(self) -> None:
        if self.weights.dtype != self.config.activation_precision:
            raise ValueError(
                f"Embedding dtype ({self.scales.dtype}) is not equal to specified activation precision"
                f" ({self.config.activation_precision})."
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        if self.scales.dtype != self.config.activation_precision:
            raise ValueError(
                f"Scales dtype {self.scales.dtype} does not match the specified activation precision"
                f" {self.config.activation_precision}"
                " Quantized layers require parameter dtypes to be equal to the activation precision.",
            )
        weights_vocab_size, _ = self.weights.shape
        (scales_vocab_size,) = self.scales.shape
        if weights_vocab_size != scales_vocab_size:
            raise ValueError(
                f"Embedding vocab size {weights_vocab_size} does not match"
                f" the scales dimension size {scales_vocab_size}",
            )

    @property
    def int_weights(self) -> Int[Array, "vocabulary channels"]:
        quantized = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        casted = quantized.astype(self.config.embedding_quantization_mode.dtype)

        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        quantized_weights = quantized_weights * self.scales.reshape(-1, 1)
        return quantized_weights

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        return super().readout(x)

    def export_weights(self) -> ParameterTree:
        return {
            "weights": self.int_weights,
            "scales": self.scales,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        stored_weights = weights["weights"]

        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            stored_weights = jax_uint8_to_unpacked_uint4(stored_weights)

        return replace(
            self,
            weights=stored_weights.astype(self.weights.dtype),
            scales=weights["scales"],
        )


@dataclass(frozen=True)
class MLXQuantizedTiedEmbeddingConfig(EmbeddingConfigBase):
    group_size: int
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "QuantizedTiedEmbedding":
        raise NotImplementedError

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXQuantizedTiedEmbedding":
        assert model_dim % self.group_size == 0
        model_groups = model_dim // self.group_size
        weights = dummy_array((vocab_size, model_dim), dtype=self.activation_precision)
        scales = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        biases = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        return MLXQuantizedTiedEmbedding(config=self, weights=weights, scales=scales, biases=biases)


class MLXQuantizedTiedEmbedding(EmbeddingBase[MLXQuantizedTiedEmbeddingConfig]):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, "vocabulary groups"]
    biases: Float[Array, "vocabulary groups"]

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

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
        quantized = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        casted = quantized.astype(self.config.embedding_quantization_mode.dtype)

        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            packed = jax_uint4_to_packed_uint8(casted)
        else:
            packed = casted

        return packed

    def _prepare_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        grouped_weights = rearrange(
            quantized_weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.config.group_size,
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
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        return super().readout(x)

    def export_weights(self) -> ParameterTree:
        return {
            "weights": self.int_weights,
            "scales": self.scales,
            "biases": self.biases,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["weights"], Array)
        assert isinstance(weights["scales"], Array)
        assert isinstance(weights["biases"], Array)

        unpacked_weights = weights["weights"]

        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(weights["weights"])

        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=weights["scales"],
            biases=weights["biases"],
        )


EmbeddingConfig = (
    TiedEmbeddingConfig | UntiedEmbeddingConfig | QuantizedTiedEmbeddingConfig | MLXQuantizedTiedEmbeddingConfig
)


register_config_union(EmbeddingConfig)  # type: ignore (pyright bug)
