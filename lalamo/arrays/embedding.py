from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar, cast

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.common import ParameterPath
from lalamo.registry_abc import RegistryABC
from lalamo.serialization import UzuSerializable

from .base import ArraySpec
from .quantization_helpers import pack_quant_weights, unpack_quant_weights


@dataclass(frozen=True)
class CompressedEmbeddingSpec(ArraySpec):
    @abstractmethod
    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "CompressedEmbedding": ...


CompressedEmbeddingSpecT_co = TypeVar(
    "CompressedEmbeddingSpecT_co", bound=CompressedEmbeddingSpec | None, covariant=True
)


class CompressedEmbedding(
    UzuSerializable,
    RegistryABC,
    eqx.Module,
    Generic[CompressedEmbeddingSpecT_co],  # noqa: UP046
):
    spec: CompressedEmbeddingSpecT_co = eqx.field(static=True, default=None, kw_only=True)

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike: ...

    @abstractmethod
    def materialize(self) -> Float[Array, "vocabulary channels"]: ...

    @abstractmethod
    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]: ...

    def to_uzu(self) -> dict[str, Any]:
        result: dict[str, Any] = {"__class__": type(self).__name__}
        if self.spec is not None:
            result["__spec__"] = self.spec.to_json()
        result.update(super().to_uzu())
        return result

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: ParameterPath = ParameterPath(),  # noqa: B008
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        spec_key = prefix / "__spec__"
        if spec_key not in data:
            return super().from_uzu(data, prefix=prefix, sharding_config=sharding_config)
        return cast("Self", CompressedEmbeddingSpec.from_json(data[spec_key]).from_uzu(data, prefix))


@dataclass(frozen=True)
class FullPrecisionEmbeddingSpec(CompressedEmbeddingSpec):
    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "FullPrecisionEmbedding":
        return FullPrecisionEmbedding(spec=self, weights=data[prefix / "weights"])


class FullPrecisionEmbedding(CompressedEmbedding[FullPrecisionEmbeddingSpec]):
    weights: Float[Array, "vocabulary channels"]

    @property
    def vocab_size(self) -> int:
        vocab, _channels = self.weights.shape
        return vocab

    @property
    def model_dim(self) -> int:
        _vocab, channels = self.weights.shape
        return channels

    @property
    def dtype(self) -> DTypeLike:
        return self.weights.dtype

    def materialize(self) -> Float[Array, "vocabulary channels"]:
        return self.weights

    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        return self.weights[token_ids]


@dataclass(frozen=True)
class MLXEmbeddingSpec(CompressedEmbeddingSpec):
    bits: int
    group_size: int
    float_dtype: DTypeLike = jnp.float32

    def from_uzu(self, data: Mapping[str, Any], prefix: ParameterPath) -> "MLXQuantizedEmbedding":
        return MLXQuantizedEmbedding(
            spec=self,
            weights=unpack_quant_weights(data[prefix / "weights"], self.bits, self.float_dtype),
            scales=data[prefix / "scales"],
            biases=data[prefix / "biases"],
        )


class MLXQuantizedEmbedding(CompressedEmbedding[MLXEmbeddingSpec]):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, "vocabulary groups"]
    biases: Float[Array, "vocabulary groups"]

    @property
    def vocab_size(self) -> int:
        vocab, _channels = self.weights.shape
        return vocab

    @property
    def model_dim(self) -> int:
        _vocab, channels = self.weights.shape
        return channels

    @property
    def dtype(self) -> DTypeLike:
        return self.scales.dtype

    def to_uzu(self) -> dict[str, Any]:
        result = super().to_uzu()
        result["weights"] = pack_quant_weights(self.weights, self.spec.bits)
        return result

    def materialize(self) -> Float[Array, "vocabulary channels"]:
        grouped = rearrange(
            self.weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.spec.group_size,
        )
        scales = rearrange(self.scales, "vocab groups -> vocab groups 1")
        biases = rearrange(self.biases, "vocab groups -> vocab groups 1")
        return rearrange(
            grouped * scales + biases,
            "vocab groups elements -> vocab (groups elements)",
        )

    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        return self.materialize()[token_ids]
