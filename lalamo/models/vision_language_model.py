from dataclasses import dataclass, replace

from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.models.language_model import LanguageModel, LanguageModelConfig
from lalamo.modules.vision import ImagePreprocessingConfig, VisionEncoder, VisionEncoderConfig
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "VisionLanguageModel",
    "VisionLanguageModelConfig",
]


@dataclass(frozen=True, kw_only=True)
class VisionLanguageModelConfig(LanguageModelConfig):
    vision_encoder_config: VisionEncoderConfig
    image_preprocessing_config: ImagePreprocessingConfig
    image_token_id: int

    def __post_init__(self) -> None:
        encoder_width = self.vision_encoder_config.output_dim
        decoder_width = self.decoder_config.transformer_config.model_dim
        if encoder_width != decoder_width:
            raise ValueError(f"Vision embeddings ({encoder_width}) do not match the decoder width ({decoder_width}).")
        if self.image_preprocessing_config.spatial_merge_size != self.vision_encoder_config.spatial_merge_size:
            raise ValueError("Image preprocessing and the vision encoder disagree on the spatial merge size.")

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "VisionLanguageModel":
        language_model = super().init(tokenizer, initializer)
        return VisionLanguageModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=language_model.token_codec,
            decoder=language_model.decoder,
            # Vision encoders are never sharded, so they are replicated over the decoder's mesh.
            vision_encoder=self.vision_encoder_config.init(
                replace(initializer, sharding_config=ShardingConfig(mesh=initializer.sharding_config.mesh)),
            ),
        )


class VisionLanguageModel(LanguageModel):
    """A ``LanguageModel`` exported together with its vision tower.

    Text weights keep the exact names a plain ``LanguageModel`` exports; the tower is exported as ``vision_encoder.*``.
    """

    vision_encoder: VisionEncoder
