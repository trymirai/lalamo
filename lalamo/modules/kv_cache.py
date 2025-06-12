import equinox as eqx
from jaxtyping import Array, Float

__all__ = ["KVCacheLayerSlice"]


class KVCacheLayerSlice(eqx.Module):
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]
