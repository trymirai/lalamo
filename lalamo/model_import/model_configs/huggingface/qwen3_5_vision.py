import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self

import cattrs

from lalamo.modules.activations import GELU
from lalamo.modules.linear import LinearConfig
from lalamo.modules.mlp import UngatedMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.vision import (
    ImagePreprocessingConfig,
    InterpolatedPositionEmbeddingConfig,
    VisionEncoderConfig,
    VisionLayerConfig,
    VisionRoPEConfig,
)

__all__ = ["HFQwen35ImageProcessorConfig", "HFQwen35VisionConfig"]


@dataclass(frozen=True)
class HFQwen35VisionConfig:
    """The ``vision_config`` block of a Qwen3.5 ``config.json``."""

    _converter: ClassVar[cattrs.Converter] = cattrs.Converter()

    model_type: Literal["qwen3_5"]
    depth: int
    hidden_size: int
    hidden_act: Literal["gelu_pytorch_tanh"]
    intermediate_size: int
    in_channels: Literal[3]
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    deepstack_visual_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        # Qwen3-VL feeds intermediate ViT layers into the decoder ("deepstack"); Qwen3.5 has it switched off.
        if self.deepstack_visual_indexes:
            raise ValueError("Deepstack visual features are not supported.")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")

    @classmethod
    def from_json(cls, json_path: Path | str) -> Self:
        with Path(json_path).open() as config_file:
            vision_config = json.load(config_file).get("vision_config")
        if vision_config is None:
            raise ValueError("The checkpoint has no vision encoder configuration.")
        return cls._converter.structure(vision_config, cls)

    def to_vision_encoder_config(self) -> VisionEncoderConfig:
        # HF's LayerNorm epsilon and default axial RoPE base.
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
            attention_config=AttentionConfig(
                qkvg_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=None,
                key_norm_config=None,
                num_heads=self.num_heads,
                num_groups=self.num_heads,
                head_dim=head_dim,
                is_causal=False,
                scale=None,
                sliding_window_size=None,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkvg_biases=True,
                has_out_biases=True,
            ),
            pre_mlp_norm_config=layer_norm_config,
            mlp_config=UngatedMLPConfig(
                linear_config=linear_config,
                activation=GELU(approximate=True),
                has_up_biases=True,
                has_down_biases=True,
            ),
        )
        return VisionEncoderConfig(
            patch_embedding_config=linear_config,
            position_embedding_config=InterpolatedPositionEmbeddingConfig(
                num_positions=self.num_position_embeddings,
            ),
            rope_config=VisionRoPEConfig(base=rope_base, head_dim=head_dim),
            layer_configs=(layer_config,) * self.depth,
            merger_norm_config=layer_norm_config,
            # Unlike the blocks, the merger uses the exact (erf) GELU: torch.nn.GELU().
            merger_mlp_config=UngatedMLPConfig(
                linear_config=linear_config,
                activation=GELU(approximate=False),
                has_up_biases=True,
                has_down_biases=True,
            ),
            in_channels=self.in_channels,
            temporal_patch_size=self.temporal_patch_size,
            patch_size=self.patch_size,
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
    """``preprocessor_config.json``. Despite their names, the ``size`` keys are HF's min and max pixel counts."""

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

    def to_image_preprocessing_config(self, encoder_config: VisionEncoderConfig) -> ImagePreprocessingConfig:
        if (self.patch_size, self.temporal_patch_size, self.merge_size) != (
            encoder_config.patch_size,
            encoder_config.temporal_patch_size,
            encoder_config.spatial_merge_size,
        ):
            raise ValueError("Preprocessor and vision tower disagree on patch dimensions.")
        return ImagePreprocessingConfig(
            min_pixels=self.size.shortest_edge,
            max_pixels=self.size.longest_edge,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
