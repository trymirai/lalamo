import numpy as np
from typing import Any, Union, Optional, Dict, TYPE_CHECKING
import jax

if TYPE_CHECKING:
    import torch

# Full _print_hf_tensor_stats function, adapted for Fartsovka context
# Global dictionary to store activations (potentially unused in this direct context but part of original func)
hf_activations: Dict[str, Any] = {}

def _print_hf_tensor_stats(tensor: Union["torch.Tensor", np.ndarray, jax.Array, None], name: str, store_key: Optional[str] = None, num_elements_to_show: int = 5) -> None:
    """Helper function to print tensor statistics for PyTorch, NumPy, or JAX tensors."""
    # Local import of torch for runtime use if available
    torch_module = None
    try:
        import torch as torch_runtime
        torch_module = torch_runtime
    except ImportError:
        pass # torch is not available

    if tensor is None:
        print(f"DEBUG FS STATS: --- {name} (Input was None) ---")
        if store_key: hf_activations[store_key] = None
        return

    data_np: np.ndarray
    original_dtype_str = str(tensor.dtype) if hasattr(tensor, 'dtype') else 'N/A'
    tensor_name_suffix = ""

    if torch_module and isinstance(tensor, torch_module.Tensor):
        tensor_name_suffix = "(Torch)"
        try:
            data_np = tensor.detach().cpu().float().numpy()
        except Exception as e:
            print(f"DEBUG FS STATS: Could not convert Torch tensor {name} to NumPy. Error: {e}")
            if store_key: hf_activations[store_key] = "Conversion Error"
            return
    elif isinstance(tensor, np.ndarray):
        tensor_name_suffix = "(NumPy)"
        if tensor.dtype != np.float32:
            data_np = tensor.astype(np.float32)
        else:
            data_np = tensor
    elif isinstance(tensor, jax.Array):
        tensor_name_suffix = "(JAX)"
        # Convert JAX array to NumPy array for stats
        _np_array = np.asarray(tensor)
        if _np_array.dtype != np.float32:
            data_np = _np_array.astype(np.float32)
        else:
            data_np = _np_array
    else:
        print(f"DEBUG FS STATS: Unsupported tensor type for {name}: {type(tensor)}")
        if store_key: hf_activations[store_key] = "Unsupported Type"
        return

    flat_tensor = data_np.flatten()
    nan_count = np.sum(np.isnan(flat_tensor))
    inf_count = np.sum(np.isinf(flat_tensor))

    print(f"DEBUG FS STATS: --- {name} {tensor_name_suffix} ---")
    print(f"  Shape: {data_np.shape}, Dtype (original): {original_dtype_str}, Dtype (for stats): {data_np.dtype}")
    if flat_tensor.size > 0:
        rms_val = np.sqrt(np.mean(data_np**2))
        print(f"  Min: {np.min(data_np):.6f}, Max: {np.max(data_np):.6f}, Mean: {np.mean(data_np):.6f}, Sum: {np.sum(data_np):.6f}, RMS: {rms_val:.6f}")
        print(f"  NaNs: {nan_count}, Infs: {inf_count}")
        if flat_tensor.size > 2 * num_elements_to_show:
            print(f"  First {num_elements_to_show} elements: {flat_tensor[:num_elements_to_show]}")
            print(f"  Last  {num_elements_to_show} elements: {flat_tensor[-num_elements_to_show:]}")
        else:
            print(f"  Elements: {flat_tensor}")
    else:
        print(f"  Tensor is empty.")
    print(f"DEBUG FS STATS: --- END {name} ---")
    if store_key:
        hf_activations[store_key] = data_np

# ---------------------------------------------------------------------
# Local helper – always converts to NumPy so stats are comparable to HF
# ---------------------------------------------------------------------
def _debug_stats_fs(tensor, name: str) -> None:
    """Wrapper around _print_hf_tensor_stats used only for FS logs."""
    if tensor is None:
        return
    import numpy as _np
    _print_hf_tensor_stats(_np.asarray(tensor), name, name)

from dataclasses import dataclass, field
from typing import NamedTuple, Callable, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule
from .mlp import MLP, MLPConfig
from .attention import VisionSdpaAttention, VisionSdpaAttentionConfig
from .normalization import RMSNorm, RMSNormConfig
from .vision_rope import VisionRoPE, VisionRoPEConfig
from .linear import FullPrecisionLinearConfig, LinearBase

__all__ = [
    "VisionConfig",
    "VisionOutput",
    "VisionTransformer",
    "PatchEmbeddingConfig",
    "VisionLayerConfig",
    "PatchEmbedding",
    "VisionLayer",
    "PatchMergerConfig",
    "PatchMerger",
]


class VisionOutput(NamedTuple):
    output: Float[Array, "batch_size out_hidden_size"]


@dataclass
class PatchEmbeddingConfig:
    """Configuration for the patch embedding layer of vision transformer."""
    
    precision: DType
    patch_size: int
    temporal_patch_size: int = 2
    in_channels: int = 3
    has_bias: bool = False
    
    def random_init(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        initial_weights: Float[Array, "hidden_size temporal_patch_size patch_size patch_size in_channels"] | None = None,
        initial_biases: Float[Array, "hidden_size"] | None = None,
    ) -> "PatchEmbedding":
        """Initialize a PatchEmbedding with random weights or from provided initial weights/biases."""
        
        final_weights: Float[Array, "hidden_size temporal_patch_size patch_size patch_size in_channels"]
        if initial_weights is not None:
            # TODO: Add shape/dtype validation for initial_weights if desired
            final_weights = initial_weights
            print(f"DEBUG PatchEmbeddingConfig.random_init: using provided initial_weights with shape: {final_weights.shape}")
        else:
            print(f"DEBUG PatchEmbeddingConfig.random_init: received hidden_size for random init: {hidden_size}")
            # Calculate the size of the convolution kernel
            kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
            
            # Initialize the weights for the convolution
            kernel_shape = (hidden_size, *kernel_size, self.in_channels)
            
            # Use fan_in initialization for weights (Kaiming initialization)
            fan_in = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
            stddev = (1 / fan_in) ** 0.5
            
            final_weights = jax.random.normal(key, kernel_shape, dtype=self.precision) * stddev
            print(f"DEBUG PatchEmbeddingConfig.random_init: created random weights with shape: {final_weights.shape}")

        biases_to_use: Float[Array, "hidden_size"] | None = None
        if self.has_bias:
            if initial_biases is not None:
                biases_to_use = initial_biases
                print(f"DEBUG PatchEmbeddingConfig.random_init: using provided initial_biases with shape: {biases_to_use.shape}")
            else:
                biases_to_use = jnp.zeros(hidden_size, dtype=self.precision)
                print(f"DEBUG PatchEmbeddingConfig.random_init: created zero biases with shape: {biases_to_use.shape}")
        
        return PatchEmbedding(config=self, weights=final_weights, biases=biases_to_use)


class PatchEmbedding(FartsovkaModule[PatchEmbeddingConfig]):
    # (E, Pt, Ph, Pw, C) after parameter swap during HF weight import
    weights: Float[Array, "hidden_size temporal_patch_size patch_size patch_size in_channels"]
    biases:  Float[Array, "hidden_size"] | None = None


    def __call__(
        self,
        images: Float[Array, "batch_size channels time height width"],
    ) -> Float[Array, "num_patches hidden_size"]:
        B, C, T_img, H_img, W_img = images.shape
        Pt, Ph, Pw = (
            self.config.temporal_patch_size,
            self.config.patch_size,
            self.config.patch_size,
        )

        assert C == self.config.in_channels, (
            f"expected {self.config.in_channels} channels, got {C}"
        )
        assert (
            T_img % Pt == 0 and H_img % Ph == 0 and W_img % Pw == 0
        ), "image dims must be divisible by patch sizes"

        E = self.weights.shape[0]       
        nT, nH, nW = T_img // Pt, H_img // Ph, W_img // Pw

        patches = (
            images.reshape(
                B, C,
                nT, Pt,
                nH, Ph,
                nW, Pw,
            )
            .reshape(-1, C, Pt, Ph, Pw)                     # (N, C, Pt, Ph, Pw)
        )

        input_is_f32 = images.dtype == jnp.float32

        kernel = (
            self.weights.astype(jnp.float32) if input_is_f32 else self.weights
        )
        bias = (
            None
            if self.biases is None
            else (self.biases.astype(jnp.float32) if input_is_f32 else self.biases)
        )

        # flatten kernel to (E, C*Pt*Ph*Pw)
        kernel_flat = jnp.transpose(kernel, (0, 4, 1, 2, 3)).reshape(E, -1)
        patches_flat = patches.reshape(patches.shape[0], -1).astype(jnp.float32)

        out = patches_flat @ kernel_flat.T
        if bias is not None:
            out = out + bias

        return out.astype(images.dtype)

    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        params = ParameterDict(weights=self.weights)
        if self.biases is not None:
            params["biases"] = self.biases
        return params


@dataclass
class VisionLayerConfig:
    norm_config: RMSNormConfig        
    attention_config: VisionSdpaAttentionConfig
    mlp_config: MLPConfig
    
    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
    ) -> "VisionLayer":
        norm1_key, attn_key, norm2_key, mlp_key = jax.random.split(key, 4)
        
        # Initialize the normalization layers
        norm1 = self.norm_config.init(model_dim)
        norm2 = self.norm_config.init(model_dim)
        
        attention = self.attention_config.random_init(
            model_dim=model_dim,
            num_heads=num_heads,
            key=attn_key,
        )

        # Initialize the MLP module
        mlp = self.mlp_config.random_init(
            model_dim=model_dim, 
            hidden_dim=hidden_dim,
            key=mlp_key,
        )
        
        return VisionLayer(
            config=self,
            norm1=norm1,
            attention=attention,
            norm2=norm2,
            mlp=mlp,
        )


class VisionLayer(FartsovkaModule[VisionLayerConfig]):
    """Transformer layer for the vision model."""
    
    norm1: RMSNorm
    attention: VisionSdpaAttention
    norm2: RMSNorm
    mlp: MLP
    
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

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        position_embeddings_tuple: tuple[Float[Array, "seq_len head_dim"], Float[Array, "seq_len head_dim"]], # New: (cos, sin) tables
        cu_seqlens: Int[Array, "n_plus_1"] | None = None, # For VisionSdpaAttention's mask
        *,
        debug_prefix: str | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        """Process hidden states through transformer layer."""

        prefix = debug_prefix or "FS_Layer"
        debug_enabled = debug_prefix is not None
        if debug_enabled:
            _debug_stats_fs(hidden_states, f"{prefix}_Input")

        residual = hidden_states


        normed_hidden_states = self._apply_norm(self.norm1, hidden_states)
        if debug_enabled:
            _debug_stats_fs(normed_hidden_states, f"{prefix}_Norm1")

        attention_output = self.attention(
            hidden_states=normed_hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings_tuple,
        )

        if debug_enabled:
            _debug_stats_fs(attention_output, f"{prefix}_AttnOutput")

        hidden_states_after_attn = residual + attention_output
        if debug_enabled:
            _debug_stats_fs(hidden_states_after_attn, f"{prefix}_AfterAttnResidual")

        # MLP branch
        residual_mlp = hidden_states_after_attn
        normed_hidden_states_mlp = self._apply_norm(self.norm2, hidden_states_after_attn)
        if debug_enabled:
            _debug_stats_fs(normed_hidden_states_mlp, f"{prefix}_Norm2")
        mlp_output = vmap(self.mlp, in_axes=0)(normed_hidden_states_mlp)
        if debug_enabled:
            _debug_stats_fs(mlp_output, f"{prefix}_MLPOutput")

        hidden_states = residual_mlp + mlp_output
        if debug_enabled:
            _debug_stats_fs(hidden_states, f"{prefix}_Output")

        return hidden_states
    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        return ParameterDict(
            norm1=self.norm1.export_weights(),
            attention=self.attention.export_weights(),
            norm2=self.norm2.export_weights(),
            mlp=self.mlp.export_weights(),
        )


@dataclass
class PatchMergerConfig:
    """Configuration for the patch merger in vision transformer."""
    
    precision: DType
    spatial_merge_size: int = 2
    has_biases: bool = False
    
    def random_init(
        self,
        context_dim: int,
        out_dim: int,
        *,
        key: PRNGKeyArray,
        expansion_factor: int = 4,
    ) -> "PatchMerger":
        """Initialize a PatchMerger with random weights."""
        norm_key, hidden_proj_key, out_proj_key = jax.random.split(key, 3)

        # Calculate hidden dimension size after spatial merging
        embed_dim_before_merge = context_dim * (self.spatial_merge_size ** 2)  # Eq. to HF's "context_dim * S²"
        mlp_hidden_dim = expansion_factor * embed_dim_before_merge

        # Initialize normalization layer
        norm = RMSNormConfig(
            scale_precision=self.precision,
            accumulation_precision=self.precision,
            epsilon=1e-6,
        ).init(context_dim)

        # Initialize linear projections
        linear_config = FullPrecisionLinearConfig(precision=self.precision)

        hidden_proj = linear_config.random_init(
            input_dim=embed_dim_before_merge,
            output_dims=(mlp_hidden_dim,),
            has_biases=self.has_biases,
            key=hidden_proj_key,
        )

        out_proj = linear_config.random_init(
            input_dim=mlp_hidden_dim,
            output_dims=(out_dim,),
            has_biases=self.has_biases,
            key=out_proj_key,
        )

        # Use JAX's GELU implementation
        gelu = jax.nn.gelu

        return PatchMerger(
            config=self,
            norm=norm,
            hidden_proj=hidden_proj,
            gelu=gelu,
            out_proj=out_proj,
        )


class PatchMerger(FartsovkaModule[PatchMergerConfig]):
    """Merges spatial patches to create a more compact representation."""
    
    norm: RMSNorm
    hidden_proj: LinearBase
    gelu: Callable[[Float[Array, "..."]], Float[Array, "..."]]
    out_proj: LinearBase
    
    def __call__(
        self, 
        x: Float[Array, "seq_len hidden_size"],
        debug_prefix: str | None = None
    ) -> Float[Array, "reduced_seq_len out_hidden_size"]:

        if debug_prefix:
            _debug_stats_fs(x, f"{debug_prefix}_Input")
        
        norm_out = vmap(self.norm, in_axes=0)(x)
        if debug_prefix:
            _debug_stats_fs(norm_out, f"{debug_prefix}_NormOutput")
            
        hidden_size_after_spatial_merge = self.config.spatial_merge_size ** 2 * norm_out.shape[-1]
        
        reshaped_for_mlp = norm_out.reshape(-1, hidden_size_after_spatial_merge)
        if debug_prefix:
            _debug_stats_fs(reshaped_for_mlp, f"{debug_prefix}_ReshapedInputToMLP")
            
        (hidden_proj_out,) = vmap(self.hidden_proj, in_axes=0)(reshaped_for_mlp)
        if debug_prefix:
            _debug_stats_fs(hidden_proj_out, f"{debug_prefix}_HiddenProjOutput")
        
        gelu_out = self.gelu(hidden_proj_out)
        if debug_prefix:
            _debug_stats_fs(gelu_out, f"{debug_prefix}_GeluOutput")
            
        (final_out,) = vmap(self.out_proj, in_axes=0)(gelu_out)
        
        if debug_prefix:
            _debug_stats_fs(final_out, f"{debug_prefix}_Output")
            
        return final_out
    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        return ParameterDict(
            norm=self.norm.export_weights(),
            hidden_proj=self.hidden_proj.export_weights(),
            out_proj=self.out_proj.export_weights(),
        )


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
        debug_layer_indices_map: dict[int, str] | None = None,
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
                
                current_layer_debug_prefix = None
                if debug_layer_indices_map and global_layer_idx in debug_layer_indices_map:
                    current_layer_debug_prefix = debug_layer_indices_map[global_layer_idx]
                
                cu_seqlens_for_this_layer = cu_seqlens_full_attention if is_full_attention_block else cu_window_seqlens
                hidden_states = layer_instance(
                    hidden_states,
                    position_embeddings_tuple=position_embeddings_tuple, 
                    cu_seqlens=cu_seqlens_for_this_layer,
                    debug_prefix=current_layer_debug_prefix
                )
                global_layer_idx += 1

            if stage_idx < len(self.inter_stage_mergers):
                merger_instance = self.inter_stage_mergers[stage_idx]
                hidden_states = merger_instance(hidden_states)
                
        print("global_layer_idx", global_layer_idx)
        
        final_merger_debug_prefix = None
        if debug_layer_indices_map is not None: 
            final_merger_debug_prefix = "FS_FinalMerger"

        if final_merger_debug_prefix:
            _debug_stats_fs(hidden_states, f"{final_merger_debug_prefix}_InputToMerger") 

        hidden_states = self.final_merger(hidden_states, debug_prefix=final_merger_debug_prefix)
        
        inv_window_index = jnp.argsort(window_index)
        hidden_states_after_unshuffle = hidden_states[inv_window_index, :] 
        if final_merger_debug_prefix:
             _debug_stats_fs(hidden_states_after_unshuffle, f"{final_merger_debug_prefix}_UnshuffledOutput")

        return VisionOutput(output=hidden_states_after_unshuffle)
    
    # ------------------------------------------------------------------  
    # Helper: local‑attention mask  
    # ------------------------------------------------------------------  
    def _build_cu_mask(  
        self,  
        cu_window_seqlens: Int[Array, "num_windows+1"],  
    ) -> Bool[Array, "1 1 win_len win_len"]:  
        window_len = int(cu_window_seqlens[1] - cu_window_seqlens[0])

        return jnp.ones((1, 1, window_len, window_len), dtype=jnp.bool_)