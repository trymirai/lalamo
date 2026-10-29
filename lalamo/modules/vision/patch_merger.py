from dataclasses import dataclass

import equinox as eqx
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.mlp import MLPForwardPassConfig, UngatedMLP, UngatedMLPConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.utils import call_vmapped
from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "PatchMerger",
    "PatchMergerConfig",
]


@dataclass(frozen=True)
class PatchMergerConfig(LalamoConfig):
    """Normalises patches, concatenates each ``merge x merge`` block, and projects it to the text model width."""

    norm_config: NormalizationConfig
    mlp_config: UngatedMLPConfig

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        output_dim: int,
        spatial_merge_size: int,
    ) -> "PatchMerger":
        merged_dim = model_dim * spatial_merge_size**2
        return PatchMerger(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=self.norm_config.init(initializer, model_dim),
            mlp=self.mlp_config.init_projection(initializer, merged_dim, merged_dim, output_dim),
        )


class PatchMerger(LalamoModule[PatchMergerConfig]):
    norm: Normalization
    mlp: UngatedMLP

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "patches channels"],
        spatial_merge_size: int,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "merged_tokens out_channels"]:
        # The preprocessor emits each block's patches contiguously, so fusing a block is a reshape.
        merged = rearrange(
            call_vmapped(self.norm, inputs),
            "(merged_tokens block) channels -> merged_tokens (block channels)",
            block=spatial_merge_size**2,
        )
        return call_vmapped(
            self.mlp.call_unbatched,
            merged,
            forward_pass_config=MLPForwardPassConfig(matmul_config=forward_pass_config),
            keychain=keychain,
        )
