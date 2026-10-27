from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.utils import call_vmapped
from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "PatchEmbedding",
    "PatchEmbeddingConfig",
]


@dataclass(frozen=True)
class PatchEmbeddingConfig(LalamoConfig):
    """Projects flattened pixel patches into the encoder width.

    HF implements this as a ``Conv3d`` whose kernel equals its stride, applied to inputs that
    the preprocessor has already cut into individual patches. A convolution that never overlaps
    and only ever sees one window is just a matrix multiply, so we store it as a ``Linear`` over
    the flattened ``(channels, temporal_patch, patch, patch)`` vector. This keeps the weights in
    the standard ``WeightMatrix`` format, which means quantisation and export work unchanged.
    """

    linear_config: LinearConfig
    in_channels: int
    temporal_patch_size: int
    patch_size: int

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size

    def init(self, initializer: Initializer, model_dim: int) -> "PatchEmbedding":
        return PatchEmbedding(
            config=self,
            sharding_config=initializer.sharding_config,
            projection=self.linear_config.init(initializer, self.patch_dim, (model_dim,), has_biases=True),
        )


class PatchEmbedding(LalamoModule[PatchEmbeddingConfig]):
    projection: Linear

    @property
    def model_dim(self) -> int:
        (result,) = self.projection.output_dims
        return result

    @eqx.filter_jit
    def __call__(
        self,
        patches: Float[Array, "patches patch_channels"],
        forward_pass_config: MatmulConfig = MatmulConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "patches channels"]:
        # Pixels arrive as float32 from the preprocessor; HF casts them to the weight dtype first.
        (result,) = call_vmapped(
            self.projection,
            patches.astype(self.projection.weights.dtype),
            forward_pass_config=forward_pass_config,
            keychain=keychain,
        )
        return result
