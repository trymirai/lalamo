
from dataclasses import dataclass

import jax
from jaxtyping import Array, Float, PRNGKeyArray
from jax import vmap
from fartsovka.common import DType, ParameterDict
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.linear import LinearBase, FullPrecisionLinearConfig
from typing import Callable
from .common import FartsovkaModule


__all__ = [
    "PatchMergerConfig",
    "PatchMerger",
]

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

        embed_dim_before_merge = context_dim * (self.spatial_merge_size ** 2)
        mlp_hidden_dim = expansion_factor * embed_dim_before_merge

        norm = RMSNormConfig(
            scale_precision=self.precision,
            accumulation_precision=self.precision,
            epsilon=1e-6,
        ).init(context_dim)

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

        gelu = jax.nn.gelu

        return PatchMerger(
            config=self,
            norm=norm,
            hidden_proj=hidden_proj,
            gelu=gelu,
            out_proj=out_proj,
        )


class PatchMerger(FartsovkaModule[PatchMergerConfig]):    
    norm: RMSNorm
    hidden_proj: LinearBase
    gelu: Callable[[Float[Array, "..."]], Float[Array, "..."]]
    out_proj: LinearBase
    
    def __call__(
        self, 
        x: Float[Array, "seq_len hidden_size"]
    ) -> Float[Array, "reduced_seq_len out_hidden_size"]:        
        x_normed = vmap(self.norm, in_axes=0)(x)
        hidden_size_after_spatial_merge = self.config.spatial_merge_size ** 2 * x_normed.shape[-1]
        
        reshaped_for_mlp = x_normed.reshape(-1, hidden_size_after_spatial_merge)
        print(f"PatchMerger after reshaping to hidden_size: {reshaped_for_mlp.shape}") # Restore/remove
            
        (hidden_proj_out,) = vmap(self.hidden_proj, in_axes=0)(reshaped_for_mlp)
        gelu_out = self.gelu(hidden_proj_out)
        (final_out,) = vmap(self.out_proj, in_axes=0)(gelu_out)
        
        print(f"PatchMerger final output shape: {final_out.shape}") # Restore/remove
        return final_out
    
    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        return ParameterDict(
            norm=self.norm.export_weights(),
            hidden_proj=self.hidden_proj.export_weights(),
            out_proj=self.out_proj.export_weights(),
        )
