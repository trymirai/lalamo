import abc
from collections.abc import Mapping
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.serialization import Serializable

from .base import pack_uint_to_uint8, unpack_uint8_to_uint


class CompressedEmbedding(Serializable, eqx.Module):
    _registry: ClassVar[dict[str, type["CompressedEmbedding"]]] = {}
    kind: ClassVar[str]

    def __init_subclass__(cls, kind: str, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        CompressedEmbedding._registry[kind] = cls
        cls.kind = kind

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
    def materialize(self) -> Float[Array, "vocabulary channels"]: ...

    @abc.abstractmethod
    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]: ...

    def to_uzu(self) -> dict[str, Any]:
        return {"__kind__": self.kind, **super().to_uzu()}

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> "CompressedEmbedding":
        kind = data["__kind__"]
        if not isinstance(kind, str):
            raise TypeError(f"Expected string kind, got {type(kind)}")
        return cls._registry[kind].from_uzu(data)


class FullPrecisionEmbedding(CompressedEmbedding, kind="full_precision_embedding"):
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
    def activation_precision(self) -> DTypeLike:
        return self.weights.dtype

    def materialize(self) -> Float[Array, "vocabulary channels"]:
        return self.weights

    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        return self.weights[token_ids]

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> CompressedEmbedding:
        if str(data.get("__kind__")) != cls.kind:
            return CompressedEmbedding.from_uzu(data)
        return cls(weights=data["weights"])


class MLXQuantizedEmbedding(CompressedEmbedding, kind="mlx_embedding"):
    weights: Float[Array, "vocabulary channels"]
    scales: Float[Array, "vocabulary groups"]
    biases: Float[Array, "vocabulary groups"]
    group_size: int = eqx.field(static=True)
    bits: int = eqx.field(static=True)

    @property
    def vocab_size(self) -> int:
        vocab, _channels = self.weights.shape
        return vocab

    @property
    def model_dim(self) -> int:
        _vocab, channels = self.weights.shape
        return channels

    @property
    def activation_precision(self) -> DTypeLike:
        return self.scales.dtype

    def materialize(self) -> Float[Array, "vocabulary channels"]:
        grouped = rearrange(
            self.weights,
            "vocab (groups elements) -> vocab groups elements",
            elements=self.group_size,
        )
        scales = rearrange(self.scales, "vocab groups -> vocab groups 1")
        biases = rearrange(self.biases, "vocab groups -> vocab groups 1")
        return rearrange(
            grouped * scales + biases,
            "vocab groups elements -> vocab (groups elements)",
        )

    def lookup(self, token_ids: Int[Array, " tokens"]) -> Float[Array, "tokens channels"]:
        return self.materialize()[token_ids]

    def to_uzu(self) -> dict[str, Any]:
        return {
            "__kind__": self.kind,
            "qweight": pack_uint_to_uint8(self.weights.astype(jnp.uint8), self.bits),
            "scales": self.scales,
            "biases": self.biases,
            "bits": self.bits,
            "group_size": self.group_size,
        }

    @classmethod
    def from_uzu(cls, data: Mapping[str, Any]) -> CompressedEmbedding:
        if str(data.get("__kind__")) != cls.kind:
            return CompressedEmbedding.from_uzu(data)
        bits = int(data["bits"])
        group_size = int(data["group_size"])
        weights = unpack_uint8_to_uint(data["qweight"], bits)
        return cls(
            weights=weights.astype(data["scales"].dtype),
            scales=data["scales"],
            biases=data["biases"],
            group_size=group_size,
            bits=bits,
        )
