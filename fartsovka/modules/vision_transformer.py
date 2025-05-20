from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .common import FartsovkaModule
from .normalization import RMSNorm, RMSNormConfig
from .patch_embedding import PatchEmbedding, PatchEmbeddingConfig
from .patch_merger import PatchMerger, PatchMergerConfig
from .rope import PositionalEmbeddings, RoPEConfigBase
from .vision_layer import VisionLayer, VisionLayerConfig
from .vision_rope import VisionRoPE

__all__ = [
    "VisionOutput",
    "VisionTransformer",
    "VisionTransformerConfig",
]

class VisionOutput(NamedTuple):
    output: Float[Array, "batch_size out_hidden_size"]

@dataclass
class VisionTransformerConfig:
    patch_embedding_config: PatchEmbeddingConfig
    rope_config: RoPEConfigBase
    layer_config: VisionLayerConfig
    patch_merger_config: PatchMergerConfig
    output_norm_config: RMSNormConfig

    image_size: int
    patch_size: int
    window_size: int = 112

    stage_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (160, 320, 640, 1280))
    stage_depths: tuple[int, ...] = field(default_factory=lambda: (2, 2, 18, 2))
    stage_num_heads: tuple[int, ...] = field(default_factory=lambda: (8, 16, 32, 64))
    stage_mlp_intermediate_dims: tuple[int, ...] = field(default_factory=lambda: (2560, 2560, 2560, 3420))

    attention_scale: float | None = None

    in_channels: int = 3
    temporal_patch_size: int = 2
    temporal_pos_scale_factor: int = 1
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048

    fullatt_block_indexes: tuple[int, ...] = field(default_factory=lambda: (7, 15, 23, 31))

    def __post_init__(self) -> None:
        num_stages = len(self.stage_hidden_dims)
        if not (len(self.stage_depths) == num_stages and
                len(self.stage_num_heads) == num_stages and
                len(self.stage_mlp_intermediate_dims) == num_stages):
            raise ValueError("Stage configuration arrays (dims, depths, heads, mlp_dims) must have the same length.")

        for i in range(num_stages):
            if self.stage_hidden_dims[i] % self.stage_num_heads[i] != 0:
                raise ValueError(
                    f"Stage {i}: Hidden dimension {self.stage_hidden_dims[i]} is not divisible by "
                    f"number of heads {self.stage_num_heads[i]}",
                )

        total_layers = sum(self.stage_depths)
        if any(idx >= total_layers for idx in self.fullatt_block_indexes):
             raise ValueError(
                 f"Some full attention block indexes {self.fullatt_block_indexes} are greater than or equal to "
                 f"the total number of layers {total_layers}",
             )

        if self.patch_size != self.patch_embedding_config.patch_size:
            raise ValueError(
                f"Patch size in VisionTransformerConfig ({self.patch_size}) does not match "
                f"patch size in PatchEmbeddingConfig ({self.patch_embedding_config.patch_size})",
            )

        if self.in_channels != self.patch_embedding_config.in_channels:
            raise ValueError(
                f"Input channels in VisionTransformerConfig ({self.in_channels}) does not match "
                f"input channels in PatchEmbeddingConfig ({self.patch_embedding_config.in_channels})",
            )

        if self.temporal_patch_size != self.patch_embedding_config.temporal_patch_size:
            raise ValueError(
                f"Temporal patch size in VisionTransformerConfig ({self.temporal_patch_size}) does not match "
                f"temporal patch size in PatchEmbeddingConfig ({self.patch_embedding_config.temporal_patch_size})",
            )

        if self.spatial_merge_size != self.patch_merger_config.spatial_merge_size:
            raise ValueError(
                f"Spatial merge size in VisionTransformerConfig ({self.spatial_merge_size}) does not match "
                f"spatial merge size in PatchMergerConfig ({self.patch_merger_config.spatial_merge_size})",
            )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "VisionTransformer":
        embedding_key, rope_key, stages_key, final_merger_key, norm_key = jax.random.split(key, 5)

        patch_embed_initial_weights = None
        patch_embed_initial_biases = None


        patch_embed = self.patch_embedding_config.random_init(
            hidden_size=self.stage_hidden_dims[0],
            key=embedding_key,
        )

        first_stage_head_dim = self.stage_hidden_dims[0] // self.stage_num_heads[0]

        if first_stage_head_dim % 2 != 0:
            raise ValueError(f"head_dim ({first_stage_head_dim}) must be even for 2‑D RoPE.")

        half_dim_for_rope = first_stage_head_dim // 2
        base_rope_for_vision = self.rope_config.init(
            head_dim=half_dim_for_rope,
            num_timesteps=self.rope_config.max_sequence_length,
        )

        rope = VisionRoPE(
            config=self.rope_config,
            head_dim=first_stage_head_dim,
            spatial_merge_size=self.spatial_merge_size,
            base_rope=base_rope_for_vision,
        )

        all_layers = []
        inter_stage_mergers = []
        current_dim = self.stage_hidden_dims[0]
        stages_keys = jax.random.split(stages_key, len(self.stage_hidden_dims))

        global_layer_index = 0
        for i in range(len(self.stage_hidden_dims)):
            stage_dim = self.stage_hidden_dims[i]
            stage_depth = self.stage_depths[i]
            stage_heads = self.stage_num_heads[i]
            stage_mlp_intermediate_dim = self.stage_mlp_intermediate_dims[i] # New way

            stage_key, merger_key = jax.random.split(stages_keys[i])

            stage_layers = []
            layers_keys = jax.random.split(stage_key, stage_depth)
            for j in range(stage_depth):
                layer_key = layers_keys[j]

                layer = self.layer_config.random_init(
                    model_dim=stage_dim,
                    hidden_dim=stage_mlp_intermediate_dim,
                    num_heads=stage_heads,
                    key=layer_key,
                )
                stage_layers.append(layer)
                global_layer_index += 1
            all_layers.append(tuple(stage_layers))

            if i < len(self.stage_hidden_dims) - 1:
                next_stage_dim = self.stage_hidden_dims[i+1]
                merger = self.patch_merger_config.random_init(
                    context_dim=stage_dim,
                    out_dim=next_stage_dim,
                    key=merger_key,
                )
                inter_stage_mergers.append(merger)


        last_stage_dim = self.stage_hidden_dims[-1]
        output_norm = self.output_norm_config.init(last_stage_dim)

        final_merger_cfg = self.patch_merger_config

        final_merger = final_merger_cfg.random_init(
            context_dim=last_stage_dim,
            out_dim=self.out_hidden_size,
            expansion_factor=1,
            key=final_merger_key,
        )

        return VisionTransformer(
            config=self,
            patch_embed=patch_embed,
            rope=rope,
            stages=tuple(all_layers),
            inter_stage_mergers=tuple(inter_stage_mergers),
            output_norm=output_norm,
            final_merger=final_merger,
        )


class VisionTransformer(FartsovkaModule[VisionTransformerConfig]):
    patch_embed: PatchEmbedding
    rope: VisionRoPE
    stages: tuple[tuple[VisionLayer, ...], ...]
    inter_stage_mergers: tuple[PatchMerger, ...]
    output_norm: RMSNorm
    final_merger: PatchMerger

    def get_window_index(self, grid_thw: Int[Array, "batch_size 3"]) -> tuple[jnp.ndarray, list[int]]:
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.config.window_size //
            self.config.spatial_merge_size //
            self.config.patch_size
        )
        spatial_merge_unit = self.config.spatial_merge_size * self.config.spatial_merge_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.config.spatial_merge_size
            llm_grid_w = grid_w // self.config.spatial_merge_size
            index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            padding_value = -100
            index_padded = jnp.pad(
                index,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=padding_value,
            )
            index_padded = rearrange(index_padded,
                                   "t (NWH VMH) (NWW VMW) -> t (NWH NWW) VMH VMW",
                                   NWH=num_windows_h, VMH=vit_merger_window_size,
                                   NWW=num_windows_w, VMW=vit_merger_window_size)

            seqlens = jnp.sum(index_padded != padding_value, axis=(2, 3)).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != padding_value]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = jnp.cumsum(seqlens, axis=0) * spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w)
        window_index = jnp.concatenate(window_index, axis=0)
        assert window_index.shape[0] == sum(
            grid_t * (grid_h // self.config.spatial_merge_size) * (grid_w // self.config.spatial_merge_size)
            for grid_t, grid_h, grid_w in grid_thw
        ), "window_index length mismatch – check reshape logic"

        return window_index, cu_window_seqlens

    def __call__(
        self,
        images: Float[Array, "channels time height width"],
        grid_thw: Int[Array, "batch 3"] | None = None,
    ) -> VisionOutput:
        if grid_thw is None:
            _, num_timesteps, height, width = images.shape
            batch_size = 1
            grid_thw = jnp.tile(
                jnp.array(
                    [[max(1, num_timesteps // self.config.temporal_patch_size),
                    height // self.config.patch_size,
                    width // self.config.patch_size]],
                    dtype=jnp.int32,
                ),
                (batch_size, 1),
            )

        hidden_states = self.patch_embed(images)
        position_embeddings = self.rope(grid_thw)
        window_index, cu_window_seqlens_list = self.get_window_index(grid_thw)
        window_index = jnp.asarray(window_index, dtype=jnp.int32)
        cu_window_seqlens = jnp.asarray(cu_window_seqlens_list, dtype=jnp.int32)
        cu_window_seqlens = jnp.unique(cu_window_seqlens)

        seq_len_pre_permute, _ = hidden_states.shape
        spatial_merge_unit = self.config.spatial_merge_size * self.config.spatial_merge_size
        num_channels = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(seq_len_pre_permute // spatial_merge_unit, spatial_merge_unit, num_channels)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len_pre_permute, num_channels)

        head_dim_rope = position_embeddings.cosines.shape[-1]
        permuted_cosines = position_embeddings.cosines.reshape(seq_len_pre_permute // spatial_merge_unit, spatial_merge_unit, head_dim_rope)
        permuted_cosines = permuted_cosines[window_index, :, :]
        permuted_cosines = permuted_cosines.reshape(seq_len_pre_permute, head_dim_rope)
        permuted_sines = position_embeddings.sines.reshape(seq_len_pre_permute // spatial_merge_unit, spatial_merge_unit, head_dim_rope)
        permuted_sines = permuted_sines[window_index, :, :]
        permuted_sines = permuted_sines.reshape(seq_len_pre_permute, head_dim_rope)

        layer_position_embeddings = PositionalEmbeddings(cosines=permuted_cosines, sines=permuted_sines)

        total_patches_per_item_in_grid = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        _cu_seqlens_full_intermediate = jnp.cumsum(total_patches_per_item_in_grid.astype(jnp.int32))
        cu_seqlens_full_attention = jnp.concatenate([jnp.array([0], dtype=jnp.int32), _cu_seqlens_full_intermediate])

        global_layer_idx = 0
        for stage_idx, stage_layers_in_current_stage in enumerate(self.stages):
            for layer_idx, layer_instance in enumerate(stage_layers_in_current_stage):
                is_full_attention_block = global_layer_idx in self.config.fullatt_block_indexes
                cu_seqlens_for_this_layer = cu_seqlens_full_attention if is_full_attention_block else cu_window_seqlens
                hidden_states = layer_instance(
                    hidden_states,
                    position_embeddings=layer_position_embeddings,
                    cumulative_seqlens=cu_seqlens_for_this_layer,
                )
                global_layer_idx += 1

            if stage_idx < len(self.inter_stage_mergers):
                merger_instance = self.inter_stage_mergers[stage_idx]
                hidden_states = merger_instance(hidden_states)

        hidden_states = self.final_merger(hidden_states)

        inv_window_index = jnp.argsort(window_index)
        hidden_states_after_unshuffle = hidden_states[inv_window_index, :]

        return VisionOutput(output=hidden_states_after_unshuffle)
