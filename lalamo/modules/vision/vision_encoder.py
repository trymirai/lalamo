from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, Float

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.rope import PositionalEmbeddings

from .forward_pass_config import VisionForwardPassConfig
from .image_grid import ImageGrid
from .patch_embedding import PatchEmbedding, PatchEmbeddingConfig
from .patch_merger import PatchMerger, PatchMergerConfig
from .position_embedding import InterpolatedPositionEmbedding, InterpolatedPositionEmbeddingConfig
from .vision_layer import VisionLayer, VisionLayerConfig, VisionLayerResult
from .vision_rope import VisionRoPEConfig

__all__ = [
    "VisionEncoder",
    "VisionEncoderActivationTrace",
    "VisionEncoderConfig",
    "VisionEncoderResult",
]


class VisionEncoderActivationTrace(Exportable, eqx.Module):
    patch_embeddings: Float[Array, "patches channels"]
    position_embeddings: Float[Array, "patches channels"]
    positional_embeddings: PositionalEmbeddings
    layer_results: tuple[VisionLayerResult, ...]


class VisionEncoderResult(Exportable, eqx.Module):
    outputs: Float[Array, "merged_tokens out_channels"]
    activation_trace: VisionEncoderActivationTrace | None


@dataclass(frozen=True)
class VisionEncoderConfig(LalamoConfig):
    """Patch embedding, ViT blocks with 2D RoPE, and a patch merger producing language model embeddings."""

    patch_embedding_config: PatchEmbeddingConfig
    position_embedding_config: InterpolatedPositionEmbeddingConfig
    rope_config: VisionRoPEConfig
    layer_configs: tuple[VisionLayerConfig, ...]
    merger_config: PatchMergerConfig
    spatial_merge_size: int
    model_dim: int
    hidden_dim: int
    output_dim: int

    def __post_init__(self) -> None:
        mismatched_layers = [
            index
            for index, layer_config in enumerate(self.layer_configs)
            if layer_config.attention_config.head_dim != self.rope_config.head_dim
        ]
        if mismatched_layers:
            raise ValueError(f"Attention head_dim differs from RoPE head_dim in layers {mismatched_layers}.")

    def num_output_tokens(self, grids: Sequence[ImageGrid]) -> int:
        return sum(grid.num_merged_tokens(self.spatial_merge_size) for grid in grids)

    def init(self, initializer: Initializer) -> "VisionEncoder":
        if initializer.sharding_config.logical_to_physical:
            raise ValueError("Vision encoders cannot be sharded; use a replicated ShardingConfig.")
        return VisionEncoder(
            config=self,
            sharding_config=initializer.sharding_config,
            patch_embedding=self.patch_embedding_config.init(initializer, self.model_dim),
            position_embedding=self.position_embedding_config.init(initializer, self.model_dim),
            layers=tuple(
                layer_config.init(initializer, self.model_dim, self.hidden_dim) for layer_config in self.layer_configs
            ),
            merger=self.merger_config.init(initializer, self.model_dim, self.output_dim, self.spatial_merge_size),
        )


class VisionEncoder(LalamoModule[VisionEncoderConfig]):
    patch_embedding: PatchEmbedding
    position_embedding: InterpolatedPositionEmbedding
    layers: tuple[VisionLayer, ...]
    merger: PatchMerger

    @eqx.filter_jit
    def __call__(
        self,
        patches: Float[Array, "patches patch_channels"],
        grids: tuple[ImageGrid, ...],
        forward_pass_config: VisionForwardPassConfig = VisionForwardPassConfig(),
        *,
        return_activation_trace: bool = False,
        keychain: Keychain,
    ) -> VisionEncoderResult:
        """Encode the patches of several images, packed back to back; ``grids`` gives each image's size."""
        num_patches, _ = patches.shape
        assert num_patches == sum(grid.num_patches for grid in grids)

        merge_size = self.config.spatial_merge_size
        embedding_keychain, merger_keychain, *layer_keychains = keychain.split(len(self.layers) + 2)

        patch_embeddings = self.patch_embedding(
            patches,
            forward_pass_config.matmul_config,
            keychain=embedding_keychain,
        )
        position_embeddings = self.position_embedding(grids, merge_size).astype(patch_embeddings.dtype)
        positional_embeddings = self.config.rope_config.compute_positional_embeddings(grids, merge_size)

        hidden_states = patch_embeddings + position_embeddings
        layer_results = []
        for layer, layer_keychain in zip(self.layers, layer_keychains, strict=True):
            layer_result = layer(
                hidden_states,
                positional_embeddings,
                grids,
                forward_pass_config,
                return_activation_trace=return_activation_trace,
                keychain=layer_keychain,
            )
            hidden_states = layer_result.outputs
            layer_results.append(layer_result)

        outputs = self.merger(
            hidden_states,
            merge_size,
            forward_pass_config.matmul_config,
            keychain=merger_keychain,
        )

        if return_activation_trace:
            activation_trace = VisionEncoderActivationTrace(
                patch_embeddings=patch_embeddings,
                position_embeddings=position_embeddings,
                positional_embeddings=positional_embeddings,
                layer_results=tuple(layer_results),
            )
        else:
            activation_trace = None
        return VisionEncoderResult(outputs=outputs, activation_trace=activation_trace)
