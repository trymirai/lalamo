from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import DEFAULT_PRECISION, DType
from fartsovka.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights

from .common import FartsovkaModule, ParameterDict, register_config_union
from .utils import apply_soft_capping

__all__ = [
    "AbstractEmbedding",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "QuantizedTiedEmbedding",
    "QuantizedTiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
    "EmbeddingConfigType",
]


class AbstractEmbedding(FartsovkaModule):
    vocab_dim: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)
    input_scale: float | None = eqx.field(static=True)
    logits_soft_cap: float | None = eqx.field(static=True)

    def _prepare_input_weights(self) -> Float[Array, "token_ids channels"]:
        raise NotImplementedError

    def _prepare_output_weights(self) -> Float[Array, "channels token_ids"]:
        raise NotImplementedError

    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens model_dim"]:
        result = self._prepare_input_weights()[x]
        if self.input_scale is not None:
            result = result * jnp.array(self.input_scale, dtype=result.dtype)
        return result

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " token_ids"]:
        logits = self._prepare_output_weights() @ x
        if self.logits_soft_cap is not None:
            logits = apply_soft_capping(logits, self.logits_soft_cap)
        return logits


@dataclass
class AbstractEmbeddingConfig[EmbeddingType: AbstractEmbedding]:
    def __call__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        *,
        key: PRNGKeyArray,
    ) -> EmbeddingType:
        raise NotImplementedError


class TiedEmbedding(AbstractEmbedding):
    weights: Float[Array, "token_ids channels"]

    precision: DType = eqx.field(static=True)

    def __init__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        precision: DType,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
        )
        self.precision = precision
        self.weights = jax.random.normal(key, (vocab_dim, model_dim), dtype=precision)

    def _prepare_input_weights(self) -> Float[Array, "token_ids channels"]:
        return self.weights

    def _prepare_output_weights(self) -> Float[Array, "channels token_ids"]:
        return self.weights

    def export_weights(self) -> ParameterDict:
        return ParameterDict(token_embeddings=self.weights)


@dataclass
class TiedEmbeddingConfig(AbstractEmbeddingConfig[TiedEmbedding]):
    precision: DType = DEFAULT_PRECISION

    def __call__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        *,
        key: PRNGKeyArray,
    ) -> TiedEmbedding:
        return TiedEmbedding(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
            precision=self.precision,
            key=key,
        )


class UntiedEmbedding(AbstractEmbedding):
    input_weights: Float[Array, "token_ids channels"]
    output_weights: Float[Array, "channels token_ids"]

    def __init__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        precision: DType,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
        )
        self.precision = precision

        input_key, output_key = jax.random.split(key)
        self.input_weights = jax.random.normal(input_key, (vocab_dim, model_dim), dtype=precision)
        self.output_weights = jax.random.normal(output_key, (model_dim, vocab_dim), dtype=precision)

    def _prepare_input_weights(self) -> Float[Array, "token_ids channels"]:
        return self.input_weights

    def _prepare_output_weights(self) -> Float[Array, "channels token_ids"]:
        return self.output_weights

    def export_weights(self) -> ParameterDict:
        return ParameterDict(
            input_weights=self.input_weights,
            output_weights=self.output_weights,
        )


class UntiedEmbeddingConfig(AbstractEmbeddingConfig[UntiedEmbedding]):
    precision: DType = DEFAULT_PRECISION

    def __call__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        *,
        key: PRNGKeyArray,
    ) -> UntiedEmbedding:
        return UntiedEmbedding(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
            precision=self.precision,
            key=key,
        )


class QuantizedTiedEmbedding(AbstractEmbedding):
    weights: Float[Array, "token_ids channels"]

    @property
    def int_weights(self) -> Int[Array, "token_ids channels"]:
        result = quantize_weights(self.weights, self.embedding_quantization_mode)
        return result.astype(self.embedding_quantization_mode.dtype)

    scales: Float[Array, " token_ids"]

    embedding_quantization_mode: QuantizationMode = eqx.field(static=True)
    activation_quantization_mode: QuantizationMode | None = eqx.field(static=True)
    activation_precision: DType = eqx.field(static=True)

    def __init__(
        self,
        *,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        embedding_quantization_mode: QuantizationMode,
        activation_quantization_mode: QuantizationMode | None,
        activation_precision: DType,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
        )

        self.embedding_quantization_mode = embedding_quantization_mode
        self.activation_quantization_mode = activation_quantization_mode
        self.activation_precision = activation_precision

        min_val, max_val = embedding_quantization_mode.range
        self.weights = jax.random.uniform(
            key,
            (vocab_dim, model_dim),
            minval=min_val,
            maxval=max_val,
            dtype=activation_precision,
        )
        self.scales = jnp.ones((vocab_dim,), dtype=activation_precision)

    def _prepare_weights(self) -> Float[Array, "out_channels in_channels"]:
        quantized_weights = quantize_weights(self.weights, self.embedding_quantization_mode)
        quantized_weights = quantized_weights * self.scales.reshape(-1, 1)
        return quantized_weights

    def _prepare_input_weights(self) -> Float[Array, "token_ids channels"]:
        return self._prepare_weights()

    def _prepare_output_weights(self) -> Float[Array, "channels token_ids"]:
        return self._prepare_weights()

    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " token_ids"]:
        if self.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.activation_quantization_mode)
        return super().readout(x)

    def export_weights(self) -> ParameterDict:
        exported_weights = quantize_weights(self.weights, self.embedding_quantization_mode)
        return ParameterDict(
            token_embeddings=exported_weights,
            scales=self.scales,
        )


@dataclass
class QuantizedTiedEmbeddingConfig(AbstractEmbeddingConfig[QuantizedTiedEmbedding]):
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DType = DEFAULT_PRECISION

    def __call__(
        self,
        vocab_dim: int,
        model_dim: int,
        input_scale: float | None,
        logits_soft_cap: float | None,
        *,
        key: PRNGKeyArray,
    ) -> QuantizedTiedEmbedding:
        return QuantizedTiedEmbedding(
            vocab_dim=vocab_dim,
            model_dim=model_dim,
            input_scale=input_scale,
            logits_soft_cap=logits_soft_cap,
            embedding_quantization_mode=self.embedding_quantization_mode,
            activation_quantization_mode=self.activation_quantization_mode,
            activation_precision=self.activation_precision,
            key=key,
        )


EmbeddingConfigType = TiedEmbeddingConfig | QuantizedTiedEmbeddingConfig | UntiedEmbeddingConfig

register_config_union(EmbeddingConfigType)
