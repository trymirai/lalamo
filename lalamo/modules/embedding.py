from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.common import ParameterTree, dummy_array, require_array, require_mapping
from lalamo.quantization import QuantizationMode, dynamically_quantize_activations, quantize_weights
from lalamo.utils import jax_uint4_to_packed_uint8, jax_uint8_to_unpacked_uint4

from .common import (
    LalamoModule,
    ParameterNorm,
    field,
    register_config_union,
)
from .utils import apply_soft_capping

__all__ = [
    "EmbeddingBase",
    "EmbeddingConfig",
    "MLXQuantizedTiedEmbedding",
    "MLXQuantizedTiedEmbeddingConfig",
    "MLXQuantizedUntiedEmbedding",
    "MLXQuantizedUntiedEmbeddingConfig",
    "MLXSemiQuantizedUntiedEmbedding",
    "MLXSemiQuantizedUntiedEmbeddingConfig",
    "TiedEmbedding",
    "TiedEmbeddingConfig",
    "UntiedEmbedding",
    "UntiedEmbeddingConfig",
    "quantize_tied_embedding_to_mlx",
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
        key: Key[Array, ""],
    ) -> "EmbeddingBase": ...

    @abstractmethod
    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "EmbeddingBase": ...


class EmbeddingBase[ConfigT: EmbeddingConfigBase](LalamoModule[ConfigT]):
    @abstractmethod
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]: ...

    @abstractmethod
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...


def _pack_embedding_weights(
    weights: Float[Array, "vocabulary channels"],
    quantization_mode: QuantizationMode,
) -> Int[Array, "vocabulary channels"]:
    quantized = quantize_weights(weights, quantization_mode).astype(quantization_mode.dtype)
    if quantization_mode == QuantizationMode.UINT4:
        return jax_uint4_to_packed_uint8(quantized)
    return quantized


def _dequantize_mlx_embedding_weights(
    weights: Float[Array, "vocabulary channels"],
    scales: Float[Array, "vocabulary groups"],
    biases: Float[Array, "vocabulary groups"],
    *,
    group_size: int,
    quantization_mode: QuantizationMode,
) -> Float[Array, "vocabulary channels"]:
    grouped_weights = rearrange(
        quantize_weights(weights, quantization_mode),
        "vocab (groups elements) -> vocab groups elements",
        elements=group_size,
    )
    return rearrange(
        grouped_weights * rearrange(scales, "vocab groups -> vocab groups 1")
        + rearrange(biases, "vocab groups -> vocab groups 1"),
        "vocab groups elements -> vocab (groups elements)",
    )


@dataclass(frozen=True)
class QuantizedEmbeddingConfigBase(EmbeddingConfigBase):
    group_size: int
    embedding_quantization_mode: QuantizationMode
    activation_quantization_mode: QuantizationMode | None
    activation_precision: DTypeLike

    @property
    def quantization(self) -> QuantizationMode:
        return self.embedding_quantization_mode

    def group_count(self, model_dim: int) -> int:
        assert model_dim % self.group_size == 0
        return model_dim // self.group_size


@dataclass(frozen=True)
class TiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: Key[Array, ""],
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
    weights: Float[Array, "vocabulary channels"] = field(norm=ParameterNorm.L_INF)

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

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self.weights[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self.weights @ x
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits

    def export_weights(self) -> ParameterTree:
        return {"weights": self.weights}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        weights = require_mapping(weights)
        return replace(self, weights=weights["weights"])


def quantize_tied_embedding_to_mlx(
    embedding: TiedEmbedding,
    *,
    group_size: int,
    embedding_quantization_mode: QuantizationMode,
    activation_quantization_mode: QuantizationMode | None = None,
    activation_precision: DTypeLike | None = None,
    num_iterations: int = 1,
    epsilon: float = 1e-5,
) -> "MLXQuantizedTiedEmbedding":
    if num_iterations <= 0:
        raise ValueError(f"Expected num_iterations to be positive, got {num_iterations}.")
    if epsilon <= 0:
        raise ValueError(f"Expected epsilon to be positive, got {epsilon}.")
    if embedding.model_dim % group_size != 0:
        raise ValueError(
            f"Embedding dimension {embedding.model_dim} is not divisible by group_size {group_size}.",
        )

    target_activation_precision = jnp.dtype(
        embedding.activation_precision if activation_precision is None else activation_precision,
    )
    quantization_min, quantization_max = [float(value) for value in embedding_quantization_mode.range]
    grouped_weights = rearrange(
        embedding.weights.astype(jnp.float32),
        "vocabulary (groups group_channels) -> vocabulary groups group_channels",
        group_channels=group_size,
    )

    group_min = grouped_weights.min(axis=-1)
    group_max = grouped_weights.max(axis=-1)
    group_span = group_max - group_min

    scales = group_span / (quantization_max - quantization_min)
    biases = group_min - quantization_min * scales

    is_constant_group = group_span < 1e-20
    group_mean = grouped_weights.mean(axis=-1)
    scales = jnp.where(is_constant_group, 1.0, scales)
    biases = jnp.where(is_constant_group, group_mean, biases)

    scales = jnp.maximum(scales, epsilon)

    def quantize(w: Array, s: Array, b: Array) -> Array:
        return jnp.clip(jnp.round((w - b[..., None]) / s[..., None]), quantization_min, quantization_max)

    for _ in range(num_iterations):
        quantized_grouped_weights = quantize(grouped_weights, scales, biases)

        quantized_mean = jnp.mean(quantized_grouped_weights, axis=-1)
        weights_mean = jnp.mean(grouped_weights, axis=-1)
        centered_quantized_weights = quantized_grouped_weights - quantized_mean[..., None]
        centered_weights = grouped_weights - weights_mean[..., None]
        denominator = jnp.sum(centered_quantized_weights * centered_quantized_weights, axis=-1)
        numerator = jnp.sum(centered_weights * centered_quantized_weights, axis=-1)

        refined_scales = jnp.where(denominator > 0.0, numerator / denominator, scales)
        refined_scales = jnp.maximum(refined_scales, epsilon)
        refined_biases = weights_mean - refined_scales * quantized_mean

        scales = jnp.where(is_constant_group, scales, refined_scales)
        biases = jnp.where(is_constant_group, biases, refined_biases)

    quantized_grouped_weights = quantize(grouped_weights, scales, biases)

    quantized_weights = rearrange(
        quantized_grouped_weights,
        "vocabulary groups group_channels -> vocabulary (groups group_channels)",
    )
    quantized_config = MLXQuantizedTiedEmbeddingConfig(
        input_scale=embedding.config.input_scale,
        logit_soft_cap=embedding.config.logit_soft_cap,
        group_size=group_size,
        embedding_quantization_mode=embedding_quantization_mode,
        activation_quantization_mode=activation_quantization_mode,
        activation_precision=target_activation_precision,
    )

    return MLXQuantizedTiedEmbedding(
        config=quantized_config,
        weights=quantized_weights.astype(target_activation_precision),
        scales=scales.astype(target_activation_precision),
        biases=biases.astype(target_activation_precision),
    )


@dataclass(frozen=True)
class UntiedEmbeddingConfig(EmbeddingConfigBase):
    precision: DTypeLike

    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: Key[Array, ""],
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
    input_weights: Float[Array, "vocabulary channels"] = field(norm=ParameterNorm.L_INF)
    output_weights: Float[Array, "vocabulary channels"] = field(norm=ParameterNorm.L_INF)

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
                f"Embedding dtype {self.input_weights.dtype} does not match"
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

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self.input_weights[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        logits = self.output_weights @ x
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits

    def export_weights(self) -> ParameterTree:
        return {
            "input_weights": self.input_weights,
            "output_weights": self.output_weights,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            input_weights=require_array(weights["input_weights"]),
            output_weights=require_array(weights["output_weights"]),
        )


@dataclass(frozen=True)
class MLXQuantizedTiedEmbeddingConfig(QuantizedEmbeddingConfigBase):
    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: Key[Array, ""],
    ) -> "MLXQuantizedTiedEmbedding":
        raise NotImplementedError

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXQuantizedTiedEmbedding":
        model_groups = self.group_count(model_dim)
        weights = dummy_array((vocab_size, model_dim), dtype=self.activation_precision)
        scales = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        biases = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        return MLXQuantizedTiedEmbedding(config=self, weights=weights, scales=scales, biases=biases)


class MLXQuantizedTiedEmbedding(EmbeddingBase[MLXQuantizedTiedEmbeddingConfig]):
    weights: Float[Array, "vocabulary channels"] = field(quantized=True, norm=ParameterNorm.L_INF)
    scales: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)
    biases: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)

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
        for name, value in (("Embedding", self.weights), ("Scale", self.scales), ("Bias", self.biases)):
            if value.dtype != self.config.activation_precision:
                raise ValueError(
                    f"{name} dtype {value.dtype} does not match"
                    f" the specified precision {self.config.activation_precision}",
                )
        vocab_size, model_dim = self.weights.shape
        expected_shape = (vocab_size, self.config.group_count(model_dim))
        if self.scales.shape != expected_shape:
            raise ValueError(f"Scale shape {self.scales.shape} does not match {expected_shape}")
        if self.biases.shape != expected_shape:
            raise ValueError(f"Bias shape {self.biases.shape} does not match {expected_shape}")

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = _dequantize_mlx_embedding_weights(
            self.weights,
            self.scales,
            self.biases,
            group_size=self.config.group_size,
            quantization_mode=self.config.embedding_quantization_mode,
        )[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        logits = (
            _dequantize_mlx_embedding_weights(
                self.weights,
                self.scales,
                self.biases,
                group_size=self.config.group_size,
                quantization_mode=self.config.embedding_quantization_mode,
            )
            @ x
        )
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits

    def export_weights(self) -> ParameterTree:
        return {
            "weights": _pack_embedding_weights(self.weights, self.config.embedding_quantization_mode),
            "scales": self.scales,
            "biases": self.biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_weights = require_array(weights["weights"])
        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            unpacked_weights = jax_uint8_to_unpacked_uint4(unpacked_weights)
        elif unpacked_weights.dtype != self.config.embedding_quantization_mode.dtype:
            raise ValueError(
                f"Expected packed embedding weights to have dtype {self.config.embedding_quantization_mode.dtype},"
                f" got {unpacked_weights.dtype}",
            )
        return replace(
            self,
            weights=unpacked_weights.astype(self.weights.dtype),
            scales=require_array(weights["scales"]),
            biases=require_array(weights["biases"]),
        )


@dataclass(frozen=True)
class MLXQuantizedUntiedEmbeddingConfig(QuantizedEmbeddingConfigBase):
    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: Key[Array, ""],
    ) -> "MLXQuantizedUntiedEmbedding":
        raise NotImplementedError

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXQuantizedUntiedEmbedding":
        model_groups = self.group_count(model_dim)
        return MLXQuantizedUntiedEmbedding(
            config=self,
            input_weights=dummy_array((vocab_size, model_dim), dtype=self.activation_precision),
            input_scales=dummy_array((vocab_size, model_groups), dtype=self.activation_precision),
            input_biases=dummy_array((vocab_size, model_groups), dtype=self.activation_precision),
            output_weights=dummy_array((vocab_size, model_dim), dtype=self.activation_precision),
            output_scales=dummy_array((vocab_size, model_groups), dtype=self.activation_precision),
            output_biases=dummy_array((vocab_size, model_groups), dtype=self.activation_precision),
        )


class MLXQuantizedUntiedEmbedding(EmbeddingBase[MLXQuantizedUntiedEmbeddingConfig]):
    input_weights: Float[Array, "vocabulary channels"] = field(quantized=True, norm=ParameterNorm.L_INF)
    input_scales: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)
    input_biases: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)
    output_weights: Float[Array, "vocabulary channels"] = field(quantized=True, norm=ParameterNorm.L_INF)
    output_scales: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)
    output_biases: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    def __post_init__(self) -> None:
        for name, value in (
            ("Input embedding", self.input_weights),
            ("Output embedding", self.output_weights),
            ("Input scale", self.input_scales),
            ("Input bias", self.input_biases),
            ("Output scale", self.output_scales),
            ("Output bias", self.output_biases),
        ):
            if value.dtype != self.config.activation_precision:
                raise ValueError(
                    f"{name} dtype {value.dtype} does not match"
                    f" the specified precision {self.config.activation_precision}",
                )
        input_vocab_size, model_dim = self.input_weights.shape
        output_vocab_size, output_model_dim = self.output_weights.shape
        if input_vocab_size != output_vocab_size:
            raise ValueError(
                f"Input vocab size {input_vocab_size} does not match output vocab size {output_vocab_size}",
            )
        if model_dim != output_model_dim:
            raise ValueError(f"Input model dim {model_dim} does not match output model dim {output_model_dim}")
        expected_shape = (input_vocab_size, self.config.group_count(model_dim))
        if self.input_scales.shape != expected_shape:
            raise ValueError(f"Input scale shape {self.input_scales.shape} does not match {expected_shape}")
        if self.input_biases.shape != expected_shape:
            raise ValueError(f"Input bias shape {self.input_biases.shape} does not match {expected_shape}")
        if self.output_scales.shape != expected_shape:
            raise ValueError(f"Output scale shape {self.output_scales.shape} does not match {expected_shape}")
        if self.output_biases.shape != expected_shape:
            raise ValueError(f"Output bias shape {self.output_biases.shape} does not match {expected_shape}")

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = _dequantize_mlx_embedding_weights(
            self.input_weights,
            self.input_scales,
            self.input_biases,
            group_size=self.config.group_size,
            quantization_mode=self.config.embedding_quantization_mode,
        )[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        logits = (
            _dequantize_mlx_embedding_weights(
                self.output_weights,
                self.output_scales,
                self.output_biases,
                group_size=self.config.group_size,
                quantization_mode=self.config.embedding_quantization_mode,
            )
            @ x
        )
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits

    def export_weights(self) -> ParameterTree:
        return {
            "input_weights": _pack_embedding_weights(self.input_weights, self.config.embedding_quantization_mode),
            "input_scales": self.input_scales,
            "input_biases": self.input_biases,
            "output_weights": _pack_embedding_weights(self.output_weights, self.config.embedding_quantization_mode),
            "output_scales": self.output_scales,
            "output_biases": self.output_biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_input_weights = require_array(weights["input_weights"])
        unpacked_output_weights = require_array(weights["output_weights"])
        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            unpacked_input_weights = jax_uint8_to_unpacked_uint4(unpacked_input_weights)
            unpacked_output_weights = jax_uint8_to_unpacked_uint4(unpacked_output_weights)
        else:
            expected_weight_dtype = self.config.embedding_quantization_mode.dtype
            if unpacked_input_weights.dtype != expected_weight_dtype:
                raise ValueError(
                    f"Expected packed input embedding weights to have dtype {expected_weight_dtype},"
                    f" got {unpacked_input_weights.dtype}",
                )
            if unpacked_output_weights.dtype != expected_weight_dtype:
                raise ValueError(
                    f"Expected packed output embedding weights to have dtype {expected_weight_dtype},"
                    f" got {unpacked_output_weights.dtype}",
                )
        return replace(
            self,
            input_weights=unpacked_input_weights.astype(self.input_weights.dtype),
            input_scales=require_array(weights["input_scales"]),
            input_biases=require_array(weights["input_biases"]),
            output_weights=unpacked_output_weights.astype(self.output_weights.dtype),
            output_scales=require_array(weights["output_scales"]),
            output_biases=require_array(weights["output_biases"]),
        )


@dataclass(frozen=True)
class MLXSemiQuantizedUntiedEmbeddingConfig(QuantizedEmbeddingConfigBase):
    def random_init(
        self,
        vocab_size: int,
        model_dim: int,
        *,
        key: Key[Array, ""],
    ) -> "MLXSemiQuantizedUntiedEmbedding":
        raise NotImplementedError

    def empty(
        self,
        vocab_size: int,
        model_dim: int,
    ) -> "MLXSemiQuantizedUntiedEmbedding":
        model_groups = self.group_count(model_dim)
        input_weights = dummy_array((vocab_size, model_dim), dtype=self.activation_precision)
        output_weights = dummy_array((vocab_size, model_dim), dtype=self.activation_precision)
        output_scales = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        output_biases = dummy_array((vocab_size, model_groups), dtype=self.activation_precision)
        return MLXSemiQuantizedUntiedEmbedding(
            config=self,
            input_weights=input_weights,
            output_weights=output_weights,
            output_scales=output_scales,
            output_biases=output_biases,
        )


class MLXSemiQuantizedUntiedEmbedding(EmbeddingBase[MLXSemiQuantizedUntiedEmbeddingConfig]):
    input_weights: Float[Array, "vocabulary channels"] = field(norm=ParameterNorm.L_INF)
    output_weights: Float[Array, "vocabulary channels"] = field(quantized=True, norm=ParameterNorm.L_INF)
    output_scales: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)
    output_biases: Float[Array, "vocabulary groups"] = field(norm=ParameterNorm.L_INF)

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.activation_precision

    @property
    def model_dim(self) -> int:
        _, model_dim = self.input_weights.shape
        return model_dim

    @property
    def vocab_size(self) -> int:
        vocab_size, _ = self.input_weights.shape
        return vocab_size

    def __post_init__(self) -> None:
        for name, value in (
            ("Input embedding", self.input_weights),
            ("Output embedding", self.output_weights),
            ("Output scale", self.output_scales),
            ("Output bias", self.output_biases),
        ):
            if value.dtype != self.config.activation_precision:
                raise ValueError(
                    f"{name} dtype {value.dtype} does not match"
                    f" the specified precision {self.config.activation_precision}",
                )
        input_vocab_size, model_dim = self.input_weights.shape
        output_vocab_size, output_model_dim = self.output_weights.shape
        if input_vocab_size != output_vocab_size:
            raise ValueError(
                f"Input vocab size {input_vocab_size} does not match output vocab size {output_vocab_size}",
            )
        if model_dim != output_model_dim:
            raise ValueError(f"Input model dim {model_dim} does not match output model dim {output_model_dim}")
        expected_shape = (input_vocab_size, self.config.group_count(model_dim))
        if self.output_scales.shape != expected_shape:
            raise ValueError(f"Output scale shape {self.output_scales.shape} does not match {expected_shape}")
        if self.output_biases.shape != expected_shape:
            raise ValueError(f"Output bias shape {self.output_biases.shape} does not match {expected_shape}")

    @eqx.filter_jit
    def embed(self, x: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        result = self.input_weights[x]
        if self.config.input_scale is not None:
            result = result * jnp.array(self.config.input_scale, dtype=result.dtype)
        return result

    @eqx.filter_jit
    def readout(self, x: Float[Array, " channels"]) -> Float[Array, " vocabulary"]:
        if self.config.activation_quantization_mode is not None:
            x = dynamically_quantize_activations(x, self.config.activation_quantization_mode)
        logits = (
            _dequantize_mlx_embedding_weights(
                self.output_weights,
                self.output_scales,
                self.output_biases,
                group_size=self.config.group_size,
                quantization_mode=self.config.embedding_quantization_mode,
            )
            @ x
        )
        if self.config.logit_soft_cap is not None:
            logits = apply_soft_capping(logits, self.config.logit_soft_cap)
        return logits

    def export_weights(self) -> ParameterTree:
        return {
            "input_weights": self.input_weights,
            "output_weights": _pack_embedding_weights(self.output_weights, self.config.embedding_quantization_mode),
            "output_scales": self.output_scales,
            "output_biases": self.output_biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        input_weights = require_array(weights["input_weights"])
        unpacked_output_weights = require_array(weights["output_weights"])
        if self.config.embedding_quantization_mode == QuantizationMode.UINT4:
            unpacked_output_weights = jax_uint8_to_unpacked_uint4(unpacked_output_weights)
        elif unpacked_output_weights.dtype != self.config.embedding_quantization_mode.dtype:
            raise ValueError(
                "Expected packed output embedding weights to have dtype"
                f" {self.config.embedding_quantization_mode.dtype},"
                f" got {unpacked_output_weights.dtype}",
            )
        return replace(
            self,
            input_weights=input_weights,
            output_weights=unpacked_output_weights.astype(self.output_weights.dtype),
            output_scales=require_array(weights["output_scales"]),
            output_biases=require_array(weights["output_biases"]),
        )


EmbeddingConfig = (
    TiedEmbeddingConfig
    | UntiedEmbeddingConfig
    | MLXQuantizedTiedEmbeddingConfig
    | MLXQuantizedUntiedEmbeddingConfig
    | MLXSemiQuantizedUntiedEmbeddingConfig
)


register_config_union(EmbeddingConfig)
