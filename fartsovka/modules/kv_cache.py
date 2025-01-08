from dataclasses import dataclass

from jaxtyping import Array, Float

__all__ = ["KVCacheLayerSlice"]


@dataclass
class KVCacheLayerSlice:
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]
