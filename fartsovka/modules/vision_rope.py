import equinox as eqx
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Int

from fartsovka.common import ParameterDict

from .common import FartsovkaModule
from .rope import PositionalEmbeddings, RoPE, RoPEConfigBase

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

        spatial_merge_size = int(self.spatial_merge_size)
        position_chunks = []

        for time_dim, height_dim, width_dim in jnp.asarray(position_thw_grid_coords).astype(jnp.int32):
            time_dim, height_dim, width_dim = int(time_dim), int(height_dim), int(width_dim)

            # Create height coordinate matrix - each row contains the same height coordinate
            height_coords_matrix = jnp.arange(height_dim)[:, None].repeat(width_dim, axis=1)
            height_positions = rearrange(height_coords_matrix,
                                       "(num_height_blocks spatial_height) (num_width_blocks spatial_width) -> (num_height_blocks num_width_blocks spatial_height spatial_width)",
                                       spatial_height=spatial_merge_size, spatial_width=spatial_merge_size,
                                       num_height_blocks=height_dim//spatial_merge_size, num_width_blocks=width_dim//spatial_merge_size)

            # Create width coordinate matrix - each column contains the same width coordinate
            width_coords_matrix = jnp.arange(width_dim)[None, :].repeat(height_dim, axis=0)
            width_positions = rearrange(width_coords_matrix,
                                      "(num_height_blocks spatial_height) (num_width_blocks spatial_width) -> (num_height_blocks num_width_blocks spatial_height spatial_width)",
                                      spatial_height=spatial_merge_size, spatial_width=spatial_merge_size,
                                      num_height_blocks=height_dim//spatial_merge_size, num_width_blocks=width_dim//spatial_merge_size)

            spatial_position_ids = jnp.stack([height_positions, width_positions], axis=-1)
            temporal_spatial_position_ids = jnp.repeat(spatial_position_ids, time_dim, axis=0)
            position_chunks.append(temporal_spatial_position_ids)

        position_ids = jnp.concatenate(position_chunks, axis=0)

        half_head_dim = self.head_dim // 2

        rope = self.config.init_with_positions(
            head_dim=half_head_dim,
            positions=position_ids,
        )

        return PositionalEmbeddings(cosines=rope.cosines, sines=rope.sines)

    def export_weights(self) -> ParameterDict:
        return ParameterDict(cosines=self.base_rope.cosines, sines=self.base_rope.sines)
