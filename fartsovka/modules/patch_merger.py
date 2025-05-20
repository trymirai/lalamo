from collections.abc import Callable
from dataclasses import dataclass

import jax
from jax import vmap
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import DType, ParameterDict
from fartsovka.modules.linear import FullPrecisionLinearConfig, LinearBase
from fartsovka.modules.normalization import RMSNorm, RMSNormConfig
from fartsovka.modules.activations import Activation

from .common import FartsovkaModule

__all__ = [
    "PatchMerger",
    "PatchMergerConfig",
]

@dataclass
class PatchMergerConfig:
    """Configuration for the patch merger in vision transformer."""

    norm_config: RMSNormConfig
    activation: Activation
    spatial_merge_size: int = 2
    has_biases: bool = False

    def random_init(
        self,
        context_dim: int,
        out_dim: int,
        *,
        expansion_factor: int = 4,
        key: PRNGKeyArray,
    ) -> "PatchMerger":
        norm_key, hidden_proj_key, out_proj_key = jax.random.split(key, 3)

        embed_dim_before_merge = context_dim * (self.spatial_merge_size ** 2)
        mlp_hidden_dim = expansion_factor * embed_dim_before_merge

        norm = self.norm_config.init(context_dim)

        linear_config = FullPrecisionLinearConfig(precision=self.norm_config.scale_precision)

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

        return PatchMerger(
            config=self,
            norm=norm,
            hidden_proj=hidden_proj,
            activation=self.activation,
            out_proj=out_proj,
        )


class PatchMerger(FartsovkaModule[PatchMergerConfig]):
    norm: RMSNorm
    hidden_proj: LinearBase
    activation: Activation
    out_proj: LinearBase

    def __call__(
        self,
        x: Float[Array, "seq_len hidden_size"],
    ) -> Float[Array, "reduced_seq_len out_hidden_size"]:
        x_normed = vmap(self.norm, in_axes=0)(x)
        hidden_size_after_spatial_merge = self.config.spatial_merge_size ** 2 * x_normed.shape[-1]

        reshaped_for_mlp = x_normed.reshape(-1, hidden_size_after_spatial_merge)

        (hidden_proj_out,) = vmap(self.hidden_proj, in_axes=0)(reshaped_for_mlp)
        gelu_out = self.activation(hidden_proj_out)
        (final_out,) = vmap(self.out_proj, in_axes=0)(gelu_out)

        return final_out

    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        return ParameterDict(
            norm=self.norm.export_weights(),
            hidden_proj=self.hidden_proj.export_weights(),
            out_proj=self.out_proj.export_weights(),
        )
