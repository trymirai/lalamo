from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.common import ParameterTree, dummy_array, require_array, require_mapping
from lalamo.quantization import (
    QuantizationMode,
    dynamically_quantize_activations,
    pack_quantized_values,
    quantize_weights,
    unpack_quantized_values,
)

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
        if model_dim % self.group_size != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by group_size ({self.group_size})")
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
            "weights": pack_quantized_values(self.weights, self.config.embedding_quantization_mode),
            "scales": self.scales,
            "biases": self.biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_weights = unpack_quantized_values(
            require_array(weights["weights"]),
            self.config.embedding_quantization_mode,
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
            "input_weights": pack_quantized_values(self.input_weights, self.config.embedding_quantization_mode),
            "input_scales": self.input_scales,
            "input_biases": self.input_biases,
            "output_weights": pack_quantized_values(self.output_weights, self.config.embedding_quantization_mode),
            "output_scales": self.output_scales,
            "output_biases": self.output_biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        unpacked_input_weights = unpack_quantized_values(
            require_array(weights["input_weights"]),
            self.config.embedding_quantization_mode,
        )
        unpacked_output_weights = unpack_quantized_values(
            require_array(weights["output_weights"]),
            self.config.embedding_quantization_mode,
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
            "output_weights": pack_quantized_values(self.output_weights, self.config.embedding_quantization_mode),
            "output_scales": self.output_scales,
            "output_biases": self.output_biases,
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        input_weights = require_array(weights["input_weights"])
        unpacked_output_weights = unpack_quantized_values(
            require_array(weights["output_weights"]),
            self.config.embedding_quantization_mode,
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
