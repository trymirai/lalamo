from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.utils import call_vmapped
from lalamo.weight_matrix import MatmulConfig

from .vision_mlp import VisionMLP, VisionMLPConfig

__all__ = [
    "PatchMerger",
    "PatchMergerConfig",
]


@dataclass(frozen=True)
class PatchMergerConfig(LalamoConfig):
    """Fuses each ``merge x merge`` block of patches into one token for the language model.

    Each patch is normalised, the patches of a block are concatenated into one wide vector,
    and an MLP projects that vector to the text model's hidden size. Blocks are already
    contiguous in the token sequence (see ``ImageGrid.patch_coordinates``), so concatenation
    is a reshape and no gather is needed.
    """

    norm_config: NormalizationConfig
    mlp_config: VisionMLPConfig

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
            mlp=self.mlp_config.init(initializer, merged_dim, merged_dim, output_dim),
        )


class PatchMerger(LalamoModule[PatchMergerConfig]):
    norm: Normalization
    mlp: VisionMLP

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "patches channels"],
        spatial_merge_size: int,
        forward_pass_config: MatmulConfig = MatmulConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "merged_tokens out_channels"]:
        # Under tensor parallelism the channels of a patch are spread across devices. Fusing a block
        # concatenates whole patches, so the normalised patches are gathered onto every device first.
        # Vision tokens are never sharded, so full replication is exactly what is needed.
        normalized = jax.sharding.reshard(
            call_vmapped(self.norm, inputs),
            self.sharding_config.make_sharding((None, None)),
        )
        merged = rearrange(
            normalized,
            "(merged_tokens block) channels -> merged_tokens (block channels)",
            block=spatial_merge_size**2,
        )
        return self.mlp(merged, forward_pass_config, keychain=keychain)
