import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    from lalamo.modules.common import ShardingConfig

import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.serialization import UzuSerializable

from .quantization_helpers import pack_uint_to_uint8, quantize_to_grid, unpack_uint8_to_uint


class CompressedEmbedding(UzuSerializable, eqx.Module):
    _registry: ClassVar[dict[str, type["CompressedEmbedding"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        CompressedEmbedding._registry[cls.__name__] = cls

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
        return {"__class__": type(self).__name__, **super().to_uzu()}

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,
    ) -> Self:
        class_key = f"{prefix}.__class__" if prefix else "__class__"
        stored_class_name = data.get(class_key)
        if isinstance(stored_class_name, str) and stored_class_name != type(self).__name__:
            target_cls = CompressedEmbedding._registry[stored_class_name]
            placeholder = target_cls.__new__(target_cls)
            return placeholder.from_uzu(data, prefix=prefix, sharding_config=sharding_config)  # type: ignore[return-value]
        return super().from_uzu(data, prefix=prefix, sharding_config=sharding_config)  # type: ignore[invalid-return-type]


class FullPrecisionEmbedding(CompressedEmbedding):
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


class MLXQuantizedEmbedding(CompressedEmbedding):
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
        int_weights = quantize_to_grid(self.weights, self.bits).astype(jnp.uint8)
        return {
            "__class__": type(self).__name__,
            "qweight": pack_uint_to_uint8(int_weights, self.bits),
            "scales": self.scales,
            "biases": self.biases,
            "bits": self.bits,
            "group_size": self.group_size,
        }

    def from_uzu(
        self,
        data: Mapping[str, Any],
        prefix: str = "",
        sharding_config: "ShardingConfig | None" = None,  # noqa: ARG002
    ) -> Self:
        def _key(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        bits = int(data[_key("bits")])
        group_size = int(data[_key("group_size")])
        weights = unpack_uint8_to_uint(data[_key("qweight")], bits)
        return type(self)(
            weights=weights.astype(data[_key("scales")].dtype),
            scales=data[_key("scales")],
            biases=data[_key("biases")],
            group_size=group_size,
            bits=bits,
        )
