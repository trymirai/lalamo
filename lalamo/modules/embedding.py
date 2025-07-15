from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterDict
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

from .common import LalamoModule, WeightLayout, register_config_union
from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
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
    logits_soft_cap: float | None

    @abstractmethod
    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: PRNGKeyArray,
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

    @classmethod
    def _default_weight_layout(cls) -> WeightLayout:
        return WeightLayout.INPUT_OUTPUT

    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self._prepare_input_weights()[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self._prepare_output_weights() @ x
        if self.config.logits_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logits_soft_cap)
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

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(weights=self.weights)


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


class UntiedEmbedding(EmbeddingBase[UntiedEmbeddingConfig]):
    input_weights: Float[Array, "vocabulary channels"]
    output_weights: Float[Array, "vocabulary channels"]

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

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:
        if weight_layout == WeightLayout.AUTO:
            weight_layout = self._default_weight_layout()

        match weight_layout:
            case WeightLayout.OUTPUT_INPUT:
                output_weights = self.output_weights
            case WeightLayout.INPUT_OUTPUT:
                output_weights = rearrange(self.output_weights, "token_ids channels -> channels token_ids")
            case _:
                raise ValueError(f"Unsupported weight layout: {weight_layout}")

        return ParameterDict(
            input_weights=self.input_weights,
            output_weights=output_weights,
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
        weights_vocab_size, weights_model_dim = self.weights.shape
        (scales_vocab_size,) = self.scales.shape
        if weights_vocab_size != scales_vocab_size:
            raise ValueError(
                f"Embedding vocab size {weights_vocab_size} does not match"
                f" the scales dimension size {scales_vocab_size}",
            )

    @property
    def int_weights(self) -> Int[Array, "vocabulary channels"]:
        result = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        return result.astype(self.config.embedding_quantization_mode.dtype)

    def _prepare_weights(self) -> Float[Array, "vocabulary channels"]:
        quantized_weights = quantize_weights(self.weights, self.config.embedding_quantization_mode)
        quantized_weights = quantized_weights * self.scales.reshape(-1, 1)
        return quantized_weights

    def _prepare_input_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    def _prepare_output_weights(self) -> Float[Array, "vocabulary channels"]:
        return self._prepare_weights()

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        return super().readout(x)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(
            weights=self.int_weights,
            scales=self.scales,
        )


EmbeddingConfig = TiedEmbeddingConfig | UntiedEmbeddingConfig | QuantizedTiedEmbeddingConfig


register_config_union(EmbeddingConfig)
