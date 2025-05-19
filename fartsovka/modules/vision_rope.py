from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import equinox as eqx

from fartsovka.common import DType, ParameterDict
from .common import FartsovkaModule
from .rope import RoPEConfigBase, RoPE, PositionalEmbeddings

__all__ = [
    "VisionRoPE",
    "VisionPositionalEmbeddings",
]

class VisionPositionalEmbeddings:
    cosines: Float[Array, "tokens head_dim"]
    sines:   Float[Array, "tokens head_dim"]

    def __init__(self, *, cosines: Float[Array, "tokens head_dim"], sines: Float[Array, "tokens head_dim"]):
            self.cosines = cosines
            self.sines = sines

class VisionRoPE(FartsovkaModule[RoPEConfigBase]):
    config: RoPEConfigBase = eqx.static_field()
    base_rope: RoPE = eqx.static_field() # RoPE for H and W dimensions
    head_dim: int = eqx.static_field()
    spatial_merge_size: int = eqx.static_field()

    def __init__(self, *, config: RoPEConfigBase, head_dim: int, spatial_merge_size: int):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for 2‑D RoPE.")
        
        self.config = config
        self.head_dim = head_dim
        self.spatial_merge_size = spatial_merge_size
        
        half_dim = head_dim // 2
        self.base_rope = config.init(
            head_dim=half_dim,
            num_timesteps=config.max_sequence_length
        )

    def __call__(self, position_thw_grid_coords: Int[Array, "... 3"]) -> VisionPositionalEmbeddings:
        if not (
            isinstance(position_thw_grid_coords, jnp.ndarray)
            and position_thw_grid_coords.ndim == 2
            and position_thw_grid_coords.shape[-1] == 3
        ):
            raise TypeError(
                "VisionRoPE expected [N,3] grid_thw tensor; "
                f"got {type(position_thw_grid_coords)} with shape {getattr(position_thw_grid_coords, 'shape', None)}"
            )

        s = int(self.spatial_merge_size)
        pos_chunks = []

        for t_i, h_i, w_i in jnp.asarray(position_thw_grid_coords).astype(jnp.int32):
            t_i, h_i, w_i = int(t_i), int(h_i), int(w_i)

            hpos = jnp.arange(h_i)[:, None].repeat(w_i, axis=1)
            hpos = hpos.reshape(h_i // s, s, w_i // s, s)
            hpos = jnp.transpose(hpos, (0, 2, 1, 3)).reshape(-1)

            wpos = jnp.arange(w_i)[None, :].repeat(h_i, axis=0)
            wpos = wpos.reshape(h_i // s, s, w_i // s, s)
            wpos = jnp.transpose(wpos, (0, 2, 1, 3)).reshape(-1)

            ids_hw = jnp.stack([hpos, wpos], axis=-1)
            ids_thw = jnp.repeat(ids_hw, t_i, axis=0)
            pos_chunks.append(ids_thw)

        pos_ids = jnp.concatenate(pos_chunks, axis=0)

        half_dim = self.head_dim // 2

        rope = self.config.init_with_positions(
            head_dim=half_dim,
            positions=pos_ids
        )

        return VisionPositionalEmbeddings(cosines=rope.cosines, sines=rope.sines)

    def export_weights(self) -> ParameterDict:
        return ParameterDict()
