from dataclasses import dataclass
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int
from einops import rearrange

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .rope import RoPE, RoPEConfigBase, PositionalEmbeddings

__all__ = [
    "VisionRoPE",
]

class VisionRoPE(FartsovkaModule[RoPEConfigBase]):
    config: RoPEConfigBase = eqx.static_field()
    base_rope: RoPE = eqx.static_field()
    head_dim: int = eqx.static_field()
    spatial_merge_size: int = eqx.static_field()

    def __call__(self, position_thw_grid_coords: Int[Array, "... 3"]) -> PositionalEmbeddings:
        if not (
            isinstance(position_thw_grid_coords, jnp.ndarray)
            and position_thw_grid_coords.ndim == 2
            and position_thw_grid_coords.shape[-1] == 3
        ):
            raise TypeError(
                "VisionRoPE expected [N,3] grid_thw tensor; "
                f"got {type(position_thw_grid_coords)} with shape {getattr(position_thw_grid_coords, 'shape', None)}",
            )

        s = int(self.spatial_merge_size)
        pos_chunks = []

        for t_i, h_i, w_i in jnp.asarray(position_thw_grid_coords).astype(jnp.int32):
            t_i, h_i, w_i = int(t_i), int(h_i), int(w_i)

            h_coords_matrix = jnp.arange(h_i)[:, None].repeat(w_i, axis=1)
            hpos = rearrange(h_coords_matrix, '(NH SH) (NW SW) -> (NH NW SH SW)', SH=s, SW=s, NH=h_i//s, NW=w_i//s)

            w_coords_matrix = jnp.arange(w_i)[None, :].repeat(h_i, axis=0)
            wpos = rearrange(w_coords_matrix, '(NH SH) (NW SW) -> (NH NW SH SW)', SH=s, SW=s, NH=h_i//s, NW=w_i//s)

            ids_hw = jnp.stack([hpos, wpos], axis=-1)
            ids_thw = jnp.repeat(ids_hw, t_i, axis=0)
            pos_chunks.append(ids_thw)

        pos_ids = jnp.concatenate(pos_chunks, axis=0)

        half_dim = self.head_dim // 2

        rope = self.config.init_with_positions(
            head_dim=half_dim,
            positions=pos_ids,
        )

        return PositionalEmbeddings(cosines=rope.cosines, sines=rope.sines)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(cosines=self.base_rope.cosines, sines=self.base_rope.sines)
