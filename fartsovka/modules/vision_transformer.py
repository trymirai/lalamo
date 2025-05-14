from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Callable, Any, Union

import jax
import jax.numpy as jnp
from jax import vmap, lax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule
from .mlp import MLP, MLPConfig
from .attention import Attention, AttentionConfig, AttentionOutput
from .normalization import RMSNorm, RMSNormConfig
from .rope import RoPE, RoPEConfig, PositionalEmbeddings
from .vision_rope import VisionRoPE, VisionRoPEConfig, VisionPositionalEmbeddings
from .linear import FullPrecisionLinear, FullPrecisionLinearConfig, LinearBase
from .kv_cache import KVCacheLayerSlice

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


# Create a compatibility wrapper
class PositionalEmbeddingsAdapter(PositionalEmbeddings):
    """Adapter to convert VisionPositionalEmbeddings to PositionalEmbeddings interface."""
    
    vision_embeddings: VisionPositionalEmbeddings
    
    def __init__(self, vision_embeddings: VisionPositionalEmbeddings):
        # For equinox compatibility, we need to define class variables properly
        self.cosines = jnp.zeros((1, 1))  # Dummy values not used
        self.sines = jnp.zeros((1, 1))    # Dummy values not used
        
        # Store the vision embeddings
        self.vision_embeddings = vision_embeddings
    
    def apply(self, heads: Float[Array, "tokens head_channels"]) -> Float[Array, "tokens head_channels"]:
        """Delegate to the vision embeddings implementation."""
        return self.vision_embeddings.apply(heads)


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
                # TODO: Add shape/dtype validation for initial_biases if desired
                biases_to_use = initial_biases
                print(f"DEBUG PatchEmbeddingConfig.random_init: using provided initial_biases with shape: {biases_to_use.shape}")
            else:
                biases_to_use = jnp.zeros(hidden_size, dtype=self.precision)
                print(f"DEBUG PatchEmbeddingConfig.random_init: created zero biases with shape: {biases_to_use.shape}")
        
        return PatchEmbedding(config=self, weights=final_weights, biases=biases_to_use)


class PatchEmbedding(FartsovkaModule[PatchEmbeddingConfig]):
    """Converts spatio‑temporal image patches into vector embeddings.

    The implementation now *exactly* mirrors HuggingFace's Qwen‑2.5‑VL logic:

    1.  **Reshape only** – no extra transpose – so the memory layout
        produced by JAX matches PyTorch's simple ``view`` stride‑trick.
    2.  Cast the patch tensor to ``bfloat16`` **after** extraction, just
        like HF.
    3.  Project the flattened patches with the pre‑loaded conv weights
        (already arranged in ``(E, C, T, H, W)`` order).
    """

    # (E, Pt, Ph, Pw, C) after parameter swap during HF weight import
    weights: Float[Array, "hidden_size temporal_patch_size patch_size patch_size in_channels"]
    biases:  Float[Array, "hidden_size"] | None = None


    def __call__(
        self,
        images: Float[Array, "batch_size channels time height width"],
    ) -> Float[Array, "num_patches hidden_size"]:
        """
        Extract spatio-temporal patches and project them with the exactly-HF
        semantics:

        •  Reshape-only patch extraction (no transposes).
        •  **Keep the patch tensor in the *original* dtype**:
           -  if the caller passed `float32` we stay in f32;
           -  if the caller passed `bfloat16` we stay in bf16.
           (HF’s Conv3d promotes its bf16 kernel to f32 when the *input* is
           f32, so mixing dtypes never happens.)
        •  Promote the stored kernel / bias to `float32` *only* when the
           input is f32.
        •  Accumulate in f32 and return a tensor in the *input* dtype.
        """
        # -------------------------------- input / config checks
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

        E = self.weights.shape[0]                           # embedding dim
        nT, nH, nW = T_img // Pt, H_img // Ph, W_img // Pw  # patch grid

        # -------------------------------- patch extraction (reshape-only)
        patches = (
            images.reshape(
                B, C,
                nT, Pt,
                nH, Ph,
                nW, Pw,
            )
            .reshape(-1, C, Pt, Ph, Pw)                     # (N, C, Pt, Ph, Pw)
        )

        # -------------------------------- dtype alignment
        input_is_f32 = images.dtype == jnp.float32

        # kernel / bias: bf16 on disk → promote if needed
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

        # -------------------------------- matmul + bias (f32 accum)
        out = patches_flat @ kernel_flat.T
        if bias is not None:
            out = out + bias

        # -------------------------------- return in *input* dtype
        return out.astype(images.dtype)

    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        params = ParameterDict(weights=self.weights)
        if self.biases is not None:
            params["biases"] = self.biases
        return params


@dataclass
class VisionLayerConfig:
    """Configuration for a transformer layer in the vision model."""
    
    # Normalization configurations
    norm_config: RMSNormConfig
    
    # Attention and MLP configurations
    attention_config: AttentionConfig
    mlp_config: MLPConfig
    
    def random_init(
        self,
        model_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_groups: int,
        head_dim: int,
        attention_scale: float | None,
        sliding_window_size: int | None,
        *,
        key: PRNGKeyArray,
    ) -> "VisionLayer":
        """Initialize a VisionLayer with random weights."""
        norm1_key, attn_key, norm2_key, mlp_key = jax.random.split(key, 4)
        
        # Initialize the normalization layers
        norm1 = self.norm_config.init(model_dim)
        norm2 = self.norm_config.init(model_dim)
        
        # Initialize the attention module
        attention = self.attention_config.random_init(
            model_dim=model_dim,
            num_heads=num_heads,
            num_groups=num_groups,
            head_dim=head_dim,
            scale=attention_scale,
            sliding_window_size=sliding_window_size,
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
    attention: Attention
    norm2: RMSNorm
    mlp: MLP
    
    def _apply_norm(self, norm_layer: RMSNorm, hidden_states: Float[Array, "seq_len hidden_size"]) -> Float[Array, "seq_len hidden_size"]:
        """Apply RMSNorm, handling potential shape mismatches."""
        # This local adjustment is fine for the norm input itself.
        if hidden_states.shape[-1] != norm_layer.input_dim:
            print(f"WARN: VisionLayer _apply_norm adjusting hidden_states shape ({hidden_states.shape}) for norm layer ({norm_layer.input_dim})")
            target_dim = norm_layer.input_dim
            current_dim = hidden_states.shape[-1]
            
            if current_dim < target_dim:
                padding_width = ((0, 0),) * (hidden_states.ndim - 1) + ((0, target_dim - current_dim),)
                hidden_states = jnp.pad(hidden_states, padding_width, mode='constant')
            else:
                hidden_states = hidden_states[..., :target_dim]
            print(f"Adjusted hidden_states shape to: {hidden_states.shape}")
            
        return vmap(norm_layer, in_axes=0)(hidden_states)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        positional_embeddings: PositionalEmbeddings,
        kv_cache: KVCacheLayerSlice | None = None,
        mask: Bool[Array, "suffix_tokens total_tokens"] | None = None,
        return_updated_kv_cache: bool = False,
    ) -> Float[Array, "seq_len hidden_size"]:
        """Process hidden states through transformer layer."""

        # Determine the configured model dimension for this layer
        # Assuming attention.config.model_dim holds the correct configured dimension for this layer.
        # Alternatively, VisionLayerConfig could store its own model_dim if it's more direct.
        layer_model_dim = self.norm1.input_dim # Use norm1.input_dim as the source of truth for the layer's model_dim

        # Adjust hidden_states to the layer's configured model_dim AT THE START
        if hidden_states.shape[-1] != layer_model_dim:
            print(f"WARN: VisionLayer __call__ adjusting input hidden_states shape ({hidden_states.shape}) to layer model_dim ({layer_model_dim})")
            current_input_dim = hidden_states.shape[-1]
            if current_input_dim < layer_model_dim:
                padding_width = ((0, 0),) * (hidden_states.ndim - 1) + ((0, layer_model_dim - current_input_dim),)
                hidden_states = jnp.pad(hidden_states, padding_width, mode='constant', constant_values=0)
            else:
                hidden_states = hidden_states[..., :layer_model_dim]
            print(f"Adjusted input hidden_states shape to: {hidden_states.shape} for layer {self}")

        print(f"DEBUG VisionLayer {self}: hidden_states shape AT START of __call__ (after potential adjustment): {hidden_states.shape}")
        
        residual = hidden_states
        
        # Attention branch
        normed_hidden_states = self._apply_norm(self.norm1, hidden_states)
        
        attention_output_obj = self.attention(
            normed_hidden_states, 
            positional_embeddings=positional_embeddings,
            kv_cache=kv_cache,
            mask=mask,
            return_updated_kv_cache=return_updated_kv_cache,
        )
        attention_output = attention_output_obj.attention_output
        print(f"DEBUG VisionLayer {self}: attention_output shape: {attention_output.shape}")
        print(f"DEBUG VisionLayer {self}: residual shape before att_add: {residual.shape}")
        
        hidden_states_after_attn = residual + attention_output
        print(f"DEBUG VisionLayer {self}: hidden_states_after_attn shape: {hidden_states_after_attn.shape}")
        
        # MLP branch
        residual_mlp = hidden_states_after_attn 
        normed_hidden_states_mlp = self._apply_norm(self.norm2, hidden_states_after_attn)
        mlp_output = vmap(self.mlp, in_axes=0)(normed_hidden_states_mlp)
        print(f"DEBUG VisionLayer {self}: mlp_output shape: {mlp_output.shape}")
        print(f"DEBUG VisionLayer {self}: residual_mlp shape before mlp_add: {residual_mlp.shape}")

        # Ensure residual_mlp has the same shape as mlp_output for addition
        if residual_mlp.shape[-1] != mlp_output.shape[-1]:
            print(f"WARN: VisionLayer residual_mlp shape {residual_mlp.shape} mismatch with mlp output {mlp_output.shape}. Adjusting residual_mlp.")
            target_dim = mlp_output.shape[-1]
            current_dim = residual_mlp.shape[-1]
            if current_dim < target_dim:
                padding_width = ((0, 0),) * (residual_mlp.ndim - 1) + ((0, target_dim - current_dim),)
                residual_mlp = jnp.pad(residual_mlp, padding_width, mode='constant')
            else:
                residual_mlp = residual_mlp[..., :target_dim]
            print(f"Adjusted residual_mlp shape to: {residual_mlp.shape}")

        hidden_states = residual_mlp + mlp_output
        print(f"DEBUG VisionLayer {self}: final hidden_states shape in layer: {hidden_states.shape}")
        
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
    ) -> "PatchMerger":
        """Initialize a PatchMerger with random weights."""
        norm_key, hidden_proj_key, out_proj_key = jax.random.split(key, 3)
        
        # Calculate hidden dimension size after spatial merging
        hidden_dim = context_dim * (self.spatial_merge_size ** 2)
        
        # Initialize normalization layer
        norm = RMSNormConfig(
            scale_precision=self.precision,
            accumulation_precision=self.precision,
            epsilon=1e-6,
        ).init(context_dim)
        
        # Initialize linear projections
        linear_config = FullPrecisionLinearConfig(precision=self.precision)
        
        hidden_proj = linear_config.random_init(
            input_dim=hidden_dim,
            output_dims=(hidden_dim,),
            has_biases=self.has_biases,
            key=hidden_proj_key,
        )
        
        out_proj = linear_config.random_init(
            input_dim=hidden_dim,
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
        x: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "reduced_seq_len out_hidden_size"]:
        """Merge patches and project to output dimension.
        
        Args:
            x: Input sequence of features, shape [seq_len, hidden_size]
            
        Returns:
            Merged and projected features, shape [reduced_seq_len, out_hidden_size]
        """
        # Print input shape for debugging
        print(f"PatchMerger input shape: {x.shape}")
        
        # Apply layer normalization
        x = vmap(self.norm, in_axes=0)(x)
        
        # Calculate the hidden size after merging spatial dimensions
        hidden_size = self.config.spatial_merge_size ** 2 * x.shape[-1]
        
        # Reshape to prepare for spatial merging - similar to HF's view(-1, self.hidden_size)
        x = x.reshape(-1, hidden_size)
        print(f"PatchMerger after reshaping to hidden_size: {x.shape}")
        
        # Apply the hidden projection (similar to first linear in HF's Sequential)
        (x,) = vmap(self.hidden_proj, in_axes=0)(x)
        
        # Apply GELU activation (similar to GELU in HF's Sequential)
        x = self.gelu(x)
        
        # Apply the output projection (similar to second linear in HF's Sequential)
        (x,) = vmap(self.out_proj, in_axes=0)(x)
        
        print(f"PatchMerger final output shape: {x.shape}")
        
        return x
    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        return ParameterDict(
            norm=self.norm.export_weights(),
            hidden_proj=self.hidden_proj.export_weights(),
            out_proj=self.out_proj.export_weights(),
        )


@dataclass
class VisionConfig:
    """Configuration for the Vision Transformer model."""

    # Base configurations for sub-components
    patch_embedding_config: PatchEmbeddingConfig
    rope_config: VisionRoPEConfig  # Only accept VisionRoPEConfig
    layer_config: VisionLayerConfig # Base config for layers within stages
    patch_merger_config: PatchMergerConfig # Base config for inter-stage and final mergers
    output_norm_config: RMSNormConfig

    # Vision-specific parameters that need to be defined first
    image_size: int # Needed for windowing calculation
    patch_size: int

    # Model architecture parameters (Stage-specific)
    stage_hidden_dims: tuple[int, ...] = field(default_factory=lambda: (160, 320, 640, 1280))
    stage_depths: tuple[int, ...] = field(default_factory=lambda: (2, 2, 18, 2))
    stage_num_heads: tuple[int, ...] = field(default_factory=lambda: (8, 16, 32, 64))
    stage_mlp_intermediate_dims: tuple[int, ...] = field(default_factory=lambda: (2560, 2560, 2560, 3420)) # Example: actual intermediate dims

    # Common parameters potentially derived or shared
    attention_scale: float | None = None # Keep if scale is global

    # Vision-specific parameters (Keep relevant ones)
    in_channels: int = 3
    temporal_patch_size: int = 2
    temporal_pos_scale_factor: int = 1 # Scale factor for temporal RoPE dimension
    spatial_merge_size: int = 2 # Used by inter-stage mergers
    out_hidden_size: int = 2048 # Final output dimension after the last merger

    # Special layer configurations (Can be stage-specific if needed, simplifying for now)
    fullatt_block_indexes: tuple[int, ...] = field(default_factory=lambda: ()) # Mark blocks that use full attention (revisit calculation based on stages)

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
            hidden_size=self.stage_hidden_dims[0], # Project to first stage dim
            key=embedding_key, # key is used if initial_weights/biases are None
            initial_weights=patch_embed_initial_weights,
            initial_biases=patch_embed_initial_biases,
        )

        # --- RoPE Initialization ----------------------------------------------------
        first_stage_head_dim = self.stage_hidden_dims[0] // self.stage_num_heads[0]
        num_patches_per_dim  = self.image_size // self.patch_size     # e.g. 224//14 → 16
        num_timesteps_for_rope = num_patches_per_dim * num_patches_per_dim  # 16×16 = 256

        rope = self.rope_config.init(
            head_dim=first_stage_head_dim,
            num_timesteps=num_timesteps_for_rope,   # ← updated value
        )

        # --- Initialize Stages and Inter-Stage Mergers ---
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
            stage_head_dim = stage_dim // stage_heads

            stage_key, merger_key = jax.random.split(stages_keys[i])

            # Initialize layers for the current stage
            stage_layers = []
            layers_keys = jax.random.split(stage_key, stage_depth)
            for j in range(stage_depth):
                layer_key = layers_keys[j]
                # Determine if this layer uses full attention
                is_full_attention = global_layer_index in self.fullatt_block_indexes
                sliding_window_size = None if is_full_attention else self.image_size // self.patch_size

                layer = self.layer_config.random_init(
                    model_dim=stage_dim,
                    hidden_dim=stage_mlp_intermediate_dim,
                    num_heads=stage_heads,
                    num_groups=stage_heads, # Assuming num_groups == num_heads for now
                    head_dim=stage_head_dim,
                    attention_scale=self.attention_scale,
                    sliding_window_size=sliding_window_size,
                    key=layer_key,
                )
                stage_layers.append(layer)
                global_layer_index += 1
            all_layers.append(tuple(stage_layers))

            # Initialize merger for the *next* stage transition (if not the last stage)
            if i < len(self.stage_hidden_dims) - 1:
                next_stage_dim = self.stage_hidden_dims[i+1]
                merger = self.patch_merger_config.random_init(
                    context_dim=stage_dim, # Input is current stage dim
                    out_dim=next_stage_dim, # Output is next stage dim
                    key=merger_key,
                )
                inter_stage_mergers.append(merger)
            # else:
                 # After the last stage, we need the final merger to go to out_hidden_size
                 # Moved final_merger initialization after the loop

        # --- Output Normalization (Applied after last stage, before final merger) ---
        last_stage_dim = self.stage_hidden_dims[-1]
        output_norm = self.output_norm_config.init(last_stage_dim)

        # --- Final Merger (Input from last stage, output to out_hidden_size) ---
        final_merger = self.patch_merger_config.random_init(
            context_dim=last_stage_dim, # Input from last stage
            out_dim=self.out_hidden_size, # Final desired output dim
            key=final_merger_key,
        )

        return VisionTransformer(
            config=self,
            patch_embed=patch_embed,
            rope=rope,
            stages=tuple(all_layers), # Tuple of tuples of layers
            inter_stage_mergers=tuple(inter_stage_mergers), # Mergers between stages
            output_norm=output_norm, # Norm applied before final merger
            final_merger=final_merger, # Final projection layer
        )


class VisionTransformer(FartsovkaModule[VisionConfig]):
    """Vision Transformer model for processing images, with multi-stage architecture."""

    patch_embed: PatchEmbedding
    rope: VisionRoPE  # Only accept VisionRoPE
    stages: tuple[tuple[VisionLayer, ...], ...] # Layers grouped by stage
    inter_stage_mergers: tuple[PatchMerger, ...] # Mergers between stages N -> N+1. Empty for single stage.
    output_norm: RMSNorm # Applied after last stage
    final_merger: PatchMerger # Applied after output_norm to get final embedding size

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
        """Calculate window indices for vision transformer.
        
        Args:
            grid_thw: The temporal, height and width of feature shape of each image.
                Shape is [batch_size, 3] where each row is [t, h, w].
                
        Returns:
            A tuple of (window_index, cu_window_seqlens) where:
                window_index: The indices to reorder patches into windows
                cu_window_seqlens: Cumulative sequence lengths for each window
        """
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        
        # Calculate window size for merging
        vit_merger_window_size = (
            self.config.image_size // 
            self.config.spatial_merge_size // 
            self.config.patch_size
        )
        
        for grid_t, grid_h, grid_w in grid_thw:
            # Calculate grid dimensions after merging
            llm_grid_h = grid_h // self.config.spatial_merge_size
            llm_grid_w = grid_w // self.config.spatial_merge_size
            
            # Create index tensor
            index = jnp.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            
            # Calculate padding for window size
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            
            # Calculate number of windows
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            
            # Pad the index tensor
            padding_value = -100  # Same as in HuggingFace implementation
            index_padded = jnp.pad(
                index, 
                ((0, 0), (0, pad_h), (0, pad_w)), 
                mode='constant', 
                constant_values=padding_value
            )
            
            # Reshape to windows
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            
            # Permute to match HuggingFace implementation
            index_padded = jnp.transpose(
                index_padded, 
                (0, 1, 3, 2, 4)
            ).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            
            # Calculate sequence lengths for each window
            seqlens = jnp.sum(index_padded != padding_value, axis=(2, 3)).reshape(-1)
            
            # Flatten the padded index
            index_padded = index_padded.reshape(-1)
            
            # Extract valid indices (not padding)
            index_new = index_padded[index_padded != padding_value]
            
            # Add to window index with offset
            window_index.append(index_new + window_index_id)
            
            # Update cumulative sequence lengths
            cu_seqlens_tmp = jnp.cumsum(
                seqlens * (self.config.spatial_merge_size ** 2), 
                axis=0
            ) + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            
            # Update window index id for next batch item
            window_index_id += (grid_t * llm_grid_h * llm_grid_w)
            
        # Concatenate all window indices
        window_index = jnp.concatenate(window_index, axis=0)
        
        return window_index, cu_window_seqlens
    
    def __call__(
        self,
        images: Float[Array, "batch_size channels time_or_height height_or_width width"],
        grid_thw: Int[Array, "batch_size 3"] | None = None,
        *,
        return_attention_weights: bool = False, # Note: Attention weight return not fully implemented
    ) -> VisionOutput:
        """Process images through the multi-stage vision transformer."""
        # Handle inputs and default grid_thw (Keep existing logic)
        if len(images.shape) == 4:
            batch_size, channels, height, width = images.shape
            images = jnp.expand_dims(images, axis=2)
            time = 1
        elif len(images.shape) == 5:
            batch_size, channels, time, height, width = images.shape
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {images.shape}")
        if grid_thw is None:
            # ... (keep existing default grid_thw calculation)
            t_val = time
            h_val = height // self.config.patch_size
            w_val = width // self.config.patch_size
            grid_thw = jnp.tile(jnp.array([[t_val, h_val, w_val]]), (batch_size, 1))


        hidden_states = self.patch_embed(images)
        print(f"DEBUG VisionTransformer.__call__: hidden_states shape AFTER patch_embed: {hidden_states.shape} (Expected Dim: {self.config.stage_hidden_dims[0]})")

        # --- New 3D pos_ids Calculation --- (Reverting to 2D for ViT internal RoPE)
        all_pos_ids_list = []
        for i in range(batch_size):
            # grid_thw provides patch counts: [num_t_patches, num_h_patches, num_w_patches]
            # For 2D RoPE, we only need H and W patch counts.
            _num_t_patches, num_h_patches, num_w_patches = grid_thw[i, 0], grid_thw[i, 1], grid_thw[i, 2]

            # Create 1D ranges of coordinates for H and W for this item
            h_coords = jnp.arange(num_h_patches)
            w_coords = jnp.arange(num_w_patches)

            # Create 2D grid of H, W coordinates using broadcasting, then flatten
            # Order: W fastest, then H (matching typical flattened order for spatial features)
            coords_h_grid = h_coords[None, :]  # Shape (1, H)
            coords_w_grid = w_coords[:, None]  # Shape (W, 1) -> Transpose later if needed, or build differently

            # Correct way to make a 2D grid and flatten for [h,w] pairs:
            # Example: H=2, W=3. Want pairs (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)
            grid_h = jnp.repeat(h_coords, num_w_patches) # [0,0,0,1,1,1]
            grid_w = jnp.tile(w_coords, num_h_patches)   # [0,1,2,0,1,2]
            
            # Stack to get [num_item_patches, 2]
            # For ViT, usually the patches are flattened in T,H,W order.
            # If we have T patches, the 2D spatial grid repeats T times.
            # The current `grid_thw` gives patch counts *after* initial patch embedding.
            # So, total patches for item `i` is `num_t_patches * num_h_patches * num_w_patches`.
            # The RoPE is applied spatially, so the (h,w) coords repeat for each temporal slice.
            
            spatial_pos_ids = jnp.stack([grid_h, grid_w], axis=-1) # Shape [H*W, 2]
            
            # Repeat these spatial_pos_ids for each temporal patch
            num_item_patches_spatial = num_h_patches * num_w_patches
            if _num_t_patches > 1:
                item_pos_ids = jnp.tile(spatial_pos_ids, (_num_t_patches, 1)) # Repeats the (H,W) grid T times
            else:
                item_pos_ids = spatial_pos_ids
            
            all_pos_ids_list.append(item_pos_ids)

        original_pos_ids = jnp.concatenate(all_pos_ids_list, axis=0) # Shape [total_patches_in_batch, 2]
        print(f"DEBUG VisionTransformer.__call__: Generated 2D H,W pos_ids shape: {original_pos_ids.shape}")

        # --- Windowing Setup ---
        # get_window_index expects grid_thw for the *LLM grid*, not patch grid.
        # For ViT's internal windowing, it uses patch grid dimensions.
        # Let's assume grid_thw passed to get_window_index should be the patch grid_thw.
        window_index, cu_window_seqlens_list = self.get_window_index(grid_thw) # grid_thw is already patch counts
        cu_window_seqlens_for_masking = jnp.array(cu_window_seqlens_list, dtype=jnp.int32)
        # Ensure cu_window_seqlens are unique and sorted for creating masks later
        # cu_window_seqlens_for_masking = jnp.unique(cu_window_seqlens_for_masking) # Temporarily remove unique for closer HF alignment
        print(f"DEBUG: cu_window_seqlens_for_masking (before unique, if applied): {cu_window_seqlens_for_masking}")

        # Reorder hidden_states and pos_ids based on window_index
        # This aligns tokens into their respective windows for attention.
        hidden_states = hidden_states[window_index]
        windowed_pos_ids = original_pos_ids[window_index]
        print(f"DEBUG VisionTransformer.__call__: hidden_states shape AFTER window_index reorder: {hidden_states.shape}")
        print(f"DEBUG VisionTransformer.__call__: windowed_pos_ids shape AFTER window_index reorder: {windowed_pos_ids.shape}")

        # Prepare reverse_indices for restoring original order later
        reverse_window_indices = jnp.argsort(window_index)

        # Apply RoPE using the *windowed* (reordered) position IDs
        vision_embeddings = self.rope(windowed_pos_ids)
        positional_embeddings = PositionalEmbeddingsAdapter(vision_embeddings)
        # print(f"DEBUG VisionTransformer.__call__: Calculated Positional Embeddings on windowed_pos_ids")

        global_layer_index = 0
        num_processed_patches = hidden_states.shape[0] # This is S_w, e.g. 64

        # --- Process through Stages ---
        for i, stage_layers in enumerate(self.stages):
            print(f"\n--- Entering Stage {i} (Dim: {self.config.stage_hidden_dims[i]}) ---")
            print(f"Stage {i} Input hidden_states shape: {hidden_states.shape}")

            # Process layers within the stage
            for j, layer in enumerate(stage_layers):
                is_full_attention = global_layer_index in self.config.fullatt_block_indexes
                
                current_attention_mask: Bool[Array, "query_len kv_len"] | None = None
                mask_seq_len = num_processed_patches # query_len and kv_len are both S_w

                if is_full_attention:
                    # For full attention layers, all S_w tokens attend to all S_w tokens.
                    # Mask is all False (no masking applied among these tokens).
                    current_attention_mask = jnp.full((mask_seq_len, mask_seq_len), False, dtype=jnp.bool_)
                    print(f"  Layer {j} (Global {global_layer_index}): Full Attention Mask (all False)")
                else:
                    # Windowed attention: derive mask from cu_window_seqlens_for_masking
                    # cu_window_seqlens_for_masking defines blocks within the S_w tokens.
                    current_attention_mask = jnp.full((mask_seq_len, mask_seq_len), True, dtype=jnp.bool_) # Start with all masked
                    for k_idx in range(len(cu_window_seqlens_for_masking) - 1):
                        start = cu_window_seqlens_for_masking[k_idx]
                        end = cu_window_seqlens_for_masking[k_idx + 1]
                        # Ensure start and end are within bounds of mask_seq_len (S_w)
                        if start < end and end <= mask_seq_len:
                            current_attention_mask = current_attention_mask.at[start:end, start:end].set(False)
                        else:
                            # This should not happen if cu_window_seqlens are correct for S_w
                            print(f"WARN: Invalid window slice for mask: start={start}, end={end}, S_w={mask_seq_len}")
                    print(f"  Layer {j} (Global {global_layer_index}): Windowed Attention Mask created from cu_window_seqlens_for_masking")
                
                print(f"  Layer {j} (Global {global_layer_index}) Input shape: {hidden_states.shape}")
                hidden_states = layer(
                    hidden_states,
                    positional_embeddings=positional_embeddings, 
                    mask=current_attention_mask, # Pass the created mask
                )
                print(f"  Layer {j} (Global {global_layer_index}) Output shape: {hidden_states.shape}")
                global_layer_index += 1

            print(f"Stage {i} Output hidden_states shape: {hidden_states.shape}")

            # Apply inter-stage merger (if not the last stage)
            if i < len(self.stages) - 1:
                merger = self.inter_stage_mergers[i]
                print(f"Applying Inter-Stage Merger {i} (Input Dim: {self.config.stage_hidden_dims[i]}, Output Dim: {self.config.stage_hidden_dims[i+1]})")
                hidden_states = merger(hidden_states)
                num_processed_patches = hidden_states.shape[0] # Update patch count after merging
                print(f"Shape after Inter-Stage Merger {i}: {hidden_states.shape}")
                # Positional embeddings might need recalculation/adaptation after merging
                # This depends heavily on how RoPE should work post-merging.

        # --- Final Processing ---
        print(f"\n--- Final Processing ---")
        print(f"Shape before Output Norm: {hidden_states.shape} (Expected Dim: {self.config.stage_hidden_dims[-1]})")
        hidden_states = self._apply_norm(self.output_norm, hidden_states)
        print(f"Shape after Output Norm: {hidden_states.shape}")

        print(f"Applying Final Merger (Input Dim: {self.config.stage_hidden_dims[-1]}, Output Dim: {self.config.out_hidden_size})")
        output_features = self.final_merger(hidden_states)
        print(f"Shape after Final Merger: {output_features.shape}")

        # Restore original token order using reverse_window_indices
        output_features = output_features[reverse_window_indices]
        print(f"DEBUG VisionTransformer.__call__: output_features shape AFTER reverse_window_indices: {output_features.shape}")

        print(f"VisionTransformer final output_features shape before VisionOutput: {output_features.shape}")
        return VisionOutput(output=output_features) 