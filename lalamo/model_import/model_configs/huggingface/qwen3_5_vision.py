import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Self

import cattrs

from lalamo.modules.activations import GELU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.vision import (
    ImagePreprocessingConfig,
    InterpolatedPositionEmbeddingConfig,
    PatchEmbeddingConfig,
    PatchMergerConfig,
    VisionAttentionConfig,
    VisionEncoderConfig,
    VisionLayerConfig,
    VisionMLPConfig,
    VisionRoPEConfig,
)

__all__ = [
    "HFQwen35ImageProcessorConfig",
    "HFQwen35VisionConfig",
]


@dataclass(frozen=True)
class HFQwen35VisionConfig:
    """Schema of the ``vision_config`` block in a Qwen3.5 ``config.json``.

    Dense and MoE checkpoints share the same tower and differ only in ``out_hidden_size``,
    which matches the text model width (1024 for 0.8B, 5120 for 27B, 2048 for 35B-A3B).
    """

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    model_type: Literal["qwen3_5", "qwen3_5_moe"]
    depth: int
    hidden_size: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    intermediate_size: int
    in_channels: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    deepstack_visual_indexes: list[int] = field(default_factory=list)
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        # Qwen3-VL feeds intermediate ViT layers into the decoder ("deepstack"). Qwen3.5 ships
        # with this switched off; refuse loudly if a checkpoint ever turns it on.
        if self.deepstack_visual_indexes:
            raise ValueError("Deepstack visual features are not supported.")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")

    @classmethod
    def from_model_config_json(cls, json_path: Path | str) -> Self:
        with Path(json_path).open() as config_file:
            return cls._converter.structure(json.load(config_file)["vision_config"], cls)

    def to_vision_encoder_config(self) -> VisionEncoderConfig:
        # HF hard-codes these instead of storing them in the config.
        layer_norm_epsilon = 1e-6
        rope_base = 10_000.0

        head_dim = self.hidden_size // self.num_heads
        linear_config = LinearConfig()
        # torch.nn.LayerNorm: mean-centred, learned scale and bias, computed in float32.
        layer_norm_config = NormalizationConfig(
            epsilon=layer_norm_epsilon,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=True,
            has_biases=True,
        )
        layer_config = VisionLayerConfig(
            pre_attention_norm_config=layer_norm_config,
            attention_config=VisionAttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                num_heads=self.num_heads,
                head_dim=head_dim,
                has_biases=True,
            ),
            pre_mlp_norm_config=layer_norm_config,
            mlp_config=VisionMLPConfig(
                linear_config=linear_config,
                activation=GELU(approximate=True),
                has_biases=True,
            ),
        )
        return VisionEncoderConfig(
            patch_embedding_config=PatchEmbeddingConfig(
                linear_config=linear_config,
                in_channels=self.in_channels,
                temporal_patch_size=self.temporal_patch_size,
                patch_size=self.patch_size,
            ),
            position_embedding_config=InterpolatedPositionEmbeddingConfig(
                num_positions=self.num_position_embeddings,
            ),
            rope_config=VisionRoPEConfig(base=rope_base, head_dim=head_dim),
            layer_configs=(layer_config,) * self.depth,
            merger_config=PatchMergerConfig(
                norm_config=layer_norm_config,
                # Unlike the blocks, the merger uses the exact (erf) GELU: torch.nn.GELU().
                mlp_config=VisionMLPConfig(
                    linear_config=linear_config,
                    activation=GELU(approximate=False),
                    has_biases=True,
                ),
            ),
            spatial_merge_size=self.spatial_merge_size,
            model_dim=self.hidden_size,
            hidden_dim=self.intermediate_size,
            output_dim=self.out_hidden_size,
        )


@dataclass(frozen=True)
class HFImageSizeLimits:
    shortest_edge: int
    longest_edge: int


@dataclass(frozen=True)
class HFQwen35ImageProcessorConfig:
    """Schema of ``preprocessor_config.json``.

    The ``size`` keys are named after edges, but HF uses them as pixel-count limits
    (``min_pixels`` and ``max_pixels``).
    """

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    size: HFImageSizeLimits
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with Path(json_path).open() as config_file:
            return cls._converter.structure(json.load(config_file), cls)

    def to_image_preprocessing_config(self) -> ImagePreprocessingConfig:
        return ImagePreprocessingConfig(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.merge_size,
            min_pixels=self.size.shortest_edge,
            max_pixels=self.size.longest_edge,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
