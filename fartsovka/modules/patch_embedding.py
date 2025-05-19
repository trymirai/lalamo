
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

from .common import FartsovkaModule


__all__ = [
    "PatchEmbeddingConfig",
    "PatchEmbedding",
]

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
        final_weights: Float[Array, "hidden_size temporal_patch_size patch_size patch_size in_channels"]
        if initial_weights is not None:
            final_weights = initial_weights
        else:
            kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
            
            kernel_shape = (hidden_size, *kernel_size, self.in_channels)
            
            fan_in = self.in_channels * self.patch_size * self.patch_size * self.temporal_patch_size
            stddev = (1 / fan_in) ** 0.5
            
            final_weights = jax.random.normal(key, kernel_shape, dtype=self.precision) * stddev

        biases_to_use: Float[Array, "hidden_size"] | None = None
        if self.has_bias:
            if initial_biases is not None:
                biases_to_use = initial_biases
            else:
                biases_to_use = jnp.zeros(hidden_size, dtype=self.precision)
        
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
