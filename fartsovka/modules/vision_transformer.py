from typing import Any, Dict
import jax


from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float, Int, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule
from .normalization import RMSNorm, RMSNormConfig
from .vision_rope import VisionRoPE, VisionRoPEConfig
from .vision_layer import VisionLayer, VisionLayerConfig
from .patch_embedding import PatchEmbedding, PatchEmbeddingConfig
from .patch_merger import PatchMerger, PatchMergerConfig

hf_activations: Dict[str, Any] = {}

__all__ = [
    "VisionConfig",
    "VisionOutput",
    "VisionTransformer",
]


class VisionOutput(NamedTuple):
    output: Float[Array, "batch_size out_hidden_size"]

@dataclass
class VisionConfig:
    patch_embedding_config: PatchEmbeddingConfig
    rope_config: VisionRoPEConfig 
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
        """Validate configuration values."""
        num_stages = len(self.stage_hidden_dims)
        if not (len(self.stage_depths) == num_stages and
                len(self.stage_num_heads) == num_stages and
                len(self.stage_mlp_intermediate_dims) == num_stages):
            raise ValueError("Stage configuration arrays (dims, depths, heads, mlp_dims) must have the same length.")

        # Validate individual stage configs
        for i in range(num_stages):
            if self.stage_hidden_dims[i] % self.stage_num_heads[i] != 0:
                raise ValueError(
                    f"Stage {i}: Hidden dimension {self.stage_hidden_dims[i]} is not divisible by "
                    f"number of heads {self.stage_num_heads[i]}"
                )

        # Revisit fullatt_block_indexes validation based on total layers across stages
        total_layers = sum(self.stage_depths)
        if any(idx >= total_layers for idx in self.fullatt_block_indexes):
             raise ValueError(
                 f"Some full attention block indexes {self.fullatt_block_indexes} are greater than or equal to "
                 f"the total number of layers {total_layers}"
             )

        # Verify consistency between patch configs (Keep relevant checks)
        if self.patch_size != self.patch_embedding_config.patch_size:
            raise ValueError(
                f"Patch size in VisionConfig ({self.patch_size}) does not match "
                f"patch size in PatchEmbeddingConfig ({self.patch_embedding_config.patch_size})"
            )
        
        if self.in_channels != self.patch_embedding_config.in_channels:
            raise ValueError(
                f"Input channels in VisionConfig ({self.in_channels}) does not match "
                f"input channels in PatchEmbeddingConfig ({self.patch_embedding_config.in_channels})"
            )
        
        if self.temporal_patch_size != self.patch_embedding_config.temporal_patch_size:
            raise ValueError(
                f"Temporal patch size in VisionConfig ({self.temporal_patch_size}) does not match "
                f"temporal patch size in PatchEmbeddingConfig ({self.patch_embedding_config.temporal_patch_size})"
            )
        
        # Verify consistency between patch merger configs
        if self.spatial_merge_size != self.patch_merger_config.spatial_merge_size:
            raise ValueError(
                f"Spatial merge size in VisionConfig ({self.spatial_merge_size}) does not match "
                f"spatial merge size in PatchMergerConfig ({self.patch_merger_config.spatial_merge_size})"
            )

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
        loaded_weights: ParameterDict | None = None,
    ) -> "VisionTransformer":
        """Initialize a VisionTransformer with random weights, adapting for stages, or from loaded_weights."""
        embedding_key, rope_key, stages_key, final_merger_key, norm_key = jax.random.split(key, 5)

        # --- Patch Embedding (Handles initial projection to first stage dim) ---
        patch_embed_initial_weights = None
        patch_embed_initial_biases = None
        if loaded_weights is not None:
            patch_embed_initial_weights = loaded_weights.get("patch_embed.weights")
            if self.patch_embedding_config.has_bias:
                 patch_embed_initial_biases = loaded_weights.get("patch_embed.biases")
            
            if patch_embed_initial_weights is not None:
                print(f"DEBUG VisionConfig.random_init: Found patch_embed.weights in loaded_weights.")
            if patch_embed_initial_biases is not None:
                print(f"DEBUG VisionConfig.random_init: Found patch_embed.biases in loaded_weights.")

        patch_embed = self.patch_embedding_config.random_init(
            hidden_size=self.stage_hidden_dims[0],
            key=embedding_key,
            initial_weights=patch_embed_initial_weights,
            initial_biases=patch_embed_initial_biases,
        )

        first_stage_head_dim = self.stage_hidden_dims[0] // self.stage_num_heads[0]
        num_patches_per_dim  = self.image_size // self.patch_size     # e.g. 224//14 → 16
        num_timesteps_for_rope = num_patches_per_dim * num_patches_per_dim  # 16×16 = 256

        rope = self.rope_config.init(
            head_dim=first_stage_head_dim,
            num_timesteps=num_timesteps_for_rope,   # ← updated value
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
                print(f"DEBUG: Created layer: {layer}")
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
            key=final_merger_key,
            expansion_factor=1,
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


class VisionTransformer(FartsovkaModule[VisionConfig]):
    """Vision Transformer model for processing images, with multi-stage architecture."""

    patch_embed: PatchEmbedding
    rope: VisionRoPE
    stages: tuple[tuple[VisionLayer, ...], ...]
    inter_stage_mergers: tuple[PatchMerger, ...]
    output_norm: RMSNorm
    final_merger: PatchMerger

    def _apply_norm(self, norm_layer: RMSNorm, hidden_states: Float[Array, "seq_len hidden_size"]) -> Float[Array, "seq_len hidden_size"]:
        """Apply RMSNorm, handling potential shape mismatches."""
        if hidden_states.shape[-1] != norm_layer.input_dim:
            print(f"WARN: Adjusting hidden_states shape ({hidden_states.shape}) for norm layer ({norm_layer.input_dim})")
            target_dim = norm_layer.input_dim
            current_dim = hidden_states.shape[-1]
            if current_dim < target_dim:
                # Pad hidden_states
                padding_width = ((0, 0),) * (hidden_states.ndim - 1) + ((0, target_dim - current_dim),)
                hidden_states = jnp.pad(hidden_states, padding_width, mode='constant')
            else:
                # Truncate hidden_states
                hidden_states = hidden_states[..., :target_dim]
            print(f"Adjusted hidden_states shape to: {hidden_states.shape}")
        return vmap(norm_layer, in_axes=0)(hidden_states)

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
                mode='constant',
                constant_values=padding_value
            )
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = jnp.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
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
        images: Float[Array, "batch channels time height width"],
        grid_thw: Int[Array, "batch 3"] | None = None,
    ) -> VisionOutput:
        if grid_thw is None:
            B, _, T, H, W = images.shape
            grid_thw = jnp.tile(
                jnp.array(
                    [[max(1, T // self.config.temporal_patch_size),
                    H // self.config.patch_size,
                    W // self.config.patch_size]],
                    dtype=jnp.int32,
                ),
                (B, 1),
            )

        hidden_states = self.patch_embed(images)
        rotary_pos_emb_raw = self.rope(grid_thw)
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

        head_dim_rope = rotary_pos_emb_raw.cosines.shape[-1]
        permuted_cosines = rotary_pos_emb_raw.cosines.reshape(seq_len_pre_permute // spatial_merge_unit, spatial_merge_unit, head_dim_rope)
        permuted_cosines = permuted_cosines[window_index, :, :]
        permuted_cosines = permuted_cosines.reshape(seq_len_pre_permute, head_dim_rope)
        permuted_sines = rotary_pos_emb_raw.sines.reshape(seq_len_pre_permute // spatial_merge_unit, spatial_merge_unit, head_dim_rope)
        permuted_sines = permuted_sines[window_index, :, :]
        permuted_sines = permuted_sines.reshape(seq_len_pre_permute, head_dim_rope)
        position_embeddings_tuple = (permuted_cosines, permuted_sines)

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
                    position_embeddings_tuple=position_embeddings_tuple, 
                    cu_seqlens=cu_seqlens_for_this_layer,
                )
                global_layer_idx += 1

            if stage_idx < len(self.inter_stage_mergers):
                merger_instance = self.inter_stage_mergers[stage_idx]
                hidden_states = merger_instance(hidden_states)
                

        hidden_states = self.final_merger(hidden_states)
        
        inv_window_index = jnp.argsort(window_index)
        hidden_states_after_unshuffle = hidden_states[inv_window_index, :] 

        return VisionOutput(output=hidden_states_after_unshuffle)
