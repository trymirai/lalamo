from dataclasses import dataclass

import jax
import jax.numpy as jnp
from einops import rearrange, einsum
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule

__all__ = [
    "PatchEmbedding",
    "PatchEmbeddingConfig",
]

@dataclass
class PatchEmbeddingConfig:
    """Configuration for the patch embedding layer of vision transformer."""

    precision: DType
    patch_size: int
    temporal_patch_size: int = 2
    in_channels: int = 3
    has_biases: bool = False

    def random_init(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
    ) -> "PatchEmbedding":
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        kernel_shape = (hidden_size, *kernel_size, self.in_channels)

        fan_in = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
        stddev = (1 / fan_in) ** 0.5

        weights = jax.random.normal(key, kernel_shape, dtype=self.precision) * stddev

        biases: Float[Array, "hidden_size"] | None = None
        if self.has_biases:
            biases = jnp.zeros(hidden_size, dtype=self.precision)

        return PatchEmbedding(config=self, weights=weights, biases=biases)


class PatchEmbedding(FartsovkaModule[PatchEmbeddingConfig]):
    weights: Float[Array, "hidden_channels temporal_patches patch_height patch_width in_channels"]
    biases:  Float[Array, "hidden_channels"] | None = None


    def __call__(
        self,
        images: Float[Array, "batch_size channels time height width"],
    ) -> Float[Array, "num_patches hidden_channels"]:
        batch_size, in_channels_from_image, image_time_steps, image_height, image_width = images.shape
        patch_time_steps, patch_height_cfg, patch_width_cfg = (
            self.config.temporal_patch_size,
            self.config.patch_size,
            self.config.patch_size,
        )

        assert images.dtype == self.config.precision, (
            f"Input image dtype {images.dtype} does not match "
            f"configured precision {self.config.precision}"
        )
        assert self.config.in_channels == in_channels_from_image, (
            f"expected {self.config.in_channels} channels, got {in_channels_from_image}"
        )
        assert (
            image_time_steps % patch_time_steps == 0 and image_height % patch_height_cfg == 0 and image_width % patch_width_cfg == 0
        ), "image dims must be divisible by patch sizes"

        num_temporal_patches = image_time_steps // patch_time_steps
        num_height_patches = image_height // patch_height_cfg
        num_width_patches = image_width // patch_width_cfg

        _patches_intermediate = (
            images.reshape(
                batch_size, in_channels_from_image,
                num_temporal_patches, patch_time_steps,
                num_height_patches, patch_height_cfg,
                num_width_patches, patch_width_cfg,
            )
            .reshape(-1, in_channels_from_image, patch_time_steps, patch_height_cfg, patch_width_cfg)
        )
        patches_flat = _patches_intermediate.reshape(_patches_intermediate.shape[0], -1).astype(jnp.float32)

        kernel_permuted = rearrange(self.weights, "h t ph pw c -> h c t ph pw")
        kernel_flat = rearrange(kernel_permuted, "h c t ph pw -> h (c t ph pw)")

        out = einsum(patches_flat, kernel_flat, "n d, h d -> n h")
        if self.biases is not None:
            out = out + self.biases

        return out.astype(self.config.precision)


    def export_weights(self) -> ParameterDict:
        """Export model weights as a ParameterDict."""
        params = ParameterDict(weights=self.weights)
        if self.biases is not None:
            params["biases"] = self.biases
        return params
