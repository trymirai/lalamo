from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.models.chat_codec import AssistantMessage, Message, ReasoningEffort, UserMessage
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules import DecoderForwardPassConfig
from lalamo.modules.vision import (
    ImagePositionScheme,
    ImagePreprocessingConfig,
    PreprocessedImages,
    VisionEncoder,
    VisionEncoderConfig,
    VisionForwardPassConfig,
)

__all__ = [
    "MultimodalPrompt",
    "VisionLanguageModel",
    "VisionLanguageModelConfig",
]


class MultimodalPrompt(NamedTuple):
    """A tokenized prompt whose image placeholders are already expanded to one token per image embedding."""

    token_ids: Int[np.ndarray, " tokens"]
    positions: Int[np.ndarray, " tokens *position_axes"]
    images: PreprocessedImages | None


class PromptInputs(NamedTuple):
    token_ids: Int[Array, "1 tokens"]
    embeddings: Float[Array, "1 tokens channels"]
    positions: Int[Array, "1 tokens *position_axes"]


@dataclass(frozen=True, kw_only=True)
class VisionLanguageModelConfig(LanguageModelConfig):
    """A language model plus everything needed to turn images into prompt embeddings.

    Each model-specific piece is a separate, swappable config: the vision tower, the host-side image
    preprocessing, and how image tokens get RoPE positions. Supporting a new VLM family means providing
    those three pieces; prompt construction and generation are shared.
    """

    vision_encoder_config: VisionEncoderConfig
    image_preprocessing_config: ImagePreprocessingConfig
    image_position_scheme: ImagePositionScheme
    image_token_id: int

    def __post_init__(self) -> None:
        encoder_width = self.vision_encoder_config.output_dim
        decoder_width = self.decoder_config.transformer_config.model_dim
        if encoder_width != decoder_width:
            raise ValueError(f"Vision embeddings ({encoder_width}) do not match the decoder width ({decoder_width}).")
        if self.image_preprocessing_config.spatial_merge_size != self.vision_encoder_config.spatial_merge_size:
            raise ValueError("Image preprocessing and the vision encoder disagree on the spatial merge size.")
        position_axes = self.image_position_scheme.num_position_axes
        mismatched_layers = [
            index
            for index, layer_config in enumerate(self.decoder_config.transformer_config.layer_configs)
            if layer_config.rope_config is not None and layer_config.rope_config.num_position_axes != position_axes
        ]
        if mismatched_layers:
            raise ValueError(
                f"The image position scheme produces {position_axes} positions per token, but the RoPE in"
                f" layers {mismatched_layers} expects a different number.",
            )

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "VisionLanguageModel":
        language_model = super().init(tokenizer, initializer)
        return VisionLanguageModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=language_model.token_codec,
            decoder=language_model.decoder,
            vision_encoder=self.vision_encoder_config.init(initializer),
        )


class VisionLanguageModel(LanguageModel):
    """A ``LanguageModel`` that also accepts images.

    It extends ``LanguageModel`` rather than wrapping it, so the text weights keep their exact export
    names (``decoder.*``) and every text-only code path (CLI, server, tracer) works unchanged. The
    vision tower is exported next to them as ``vision_encoder.*``.
    """

    vision_encoder: VisionEncoder

    @property
    def vision_config(self) -> VisionLanguageModelConfig:
        assert isinstance(self.config, VisionLanguageModelConfig)
        return self.config

    def build_prompt(
        self,
        messages: Iterable[Message],
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> MultimodalPrompt:
        """Render messages and expand each image placeholder to as many tokens as the image will produce."""
        messages = tuple(messages)
        images = tuple(image for message in messages if isinstance(message, UserMessage) for image in message.images)
        prompt = self.token_codec.render_request(messages, reasoning_effort=reasoning_effort)
        token_ids = np.asarray(self.token_codec.encode_text(prompt), dtype=np.int32)
        image_token_id = self.vision_config.image_token_id

        placeholder_indices = np.flatnonzero(token_ids == image_token_id)
        if len(placeholder_indices) != len(images):
            raise ValueError(f"The prompt has {len(placeholder_indices)} image placeholders for {len(images)} images.")
        preprocessed = self.vision_config.image_preprocessing_config.preprocess(images) if images else None
        grids = preprocessed.grids if preprocessed is not None else ()
        merge_size = self.vision_config.vision_encoder_config.spatial_merge_size
        repeats = np.ones_like(token_ids)
        repeats[placeholder_indices] = [grid.num_merged_tokens(merge_size) for grid in grids]
        expanded_token_ids = np.repeat(token_ids, repeats)

        positions = self.vision_config.image_position_scheme.positions(
            expanded_token_ids,
            image_token_id,
            grids,
            merge_size,
        )
        return MultimodalPrompt(token_ids=expanded_token_ids, positions=positions, images=preprocessed)

    def encode_images(
        self,
        images: PreprocessedImages,
        forward_pass_config: VisionForwardPassConfig | None = None,
        *,
        keychain: Keychain,
    ) -> Float[Array, "image_tokens channels"]:
        patches = jax.device_put(jnp.asarray(images.patches), self.sharding_config.make_sharding((None, None)))
        return self.vision_encoder(
            patches,
            images.grids,
            forward_pass_config or VisionForwardPassConfig.for_inference(),
            keychain=keychain,
        ).outputs

    def prompt_inputs(
        self,
        prompt: MultimodalPrompt,
        decoder_forward_pass_config: DecoderForwardPassConfig | None = None,
        vision_forward_pass_config: VisionForwardPassConfig | None = None,
        *,
        keychain: Keychain,
    ) -> PromptInputs:
        """Token ids, embeddings with image features spliced in, and positions, ready for prefill."""
        decoder_forward_pass_config = decoder_forward_pass_config or DecoderForwardPassConfig.for_inference()
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        token_ids = jax.device_put(
            jnp.asarray(prompt.token_ids)[None],
            self.sharding_config.make_sharding((batch_axis, None)),
        )
        positions = jax.device_put(
            jnp.asarray(prompt.positions)[None],
            self.sharding_config.make_sharding((batch_axis, *(None,) * prompt.positions.ndim)),
        )
        embedding_keychain, vision_keychain = keychain.split()
        embeddings = self.decoder.embedding.embed(
            token_ids,
            forward_pass_config=decoder_forward_pass_config.embedding_forward_pass_config,
            keychain=embedding_keychain,
        )
        if prompt.images is not None:
            image_embeddings = self.encode_images(prompt.images, vision_forward_pass_config, keychain=vision_keychain)
            image_token_indices = np.flatnonzero(prompt.token_ids == self.vision_config.image_token_id)
            embeddings = embeddings.at[0, image_token_indices].set(image_embeddings.astype(embeddings.dtype))
        return PromptInputs(token_ids=token_ids, embeddings=embeddings, positions=positions)

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        max_output_length: int = 8192,
        prefill_forward_pass_config: DecoderForwardPassConfig | None = None,
        decode_forward_pass_config: DecoderForwardPassConfig | None = None,
        vision_forward_pass_config: VisionForwardPassConfig | None = None,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        keychain: Keychain,
    ) -> AssistantMessage:
        messages = tuple(messages)
        generation_config = self.config.generation_config.override_with(generation_config or GenerationConfig())
        prompt = self.build_prompt(messages, reasoning_effort=reasoning_effort)
        inputs_keychain, generation_keychain = keychain.split()
        inputs = self.prompt_inputs(
            prompt,
            prefill_forward_pass_config,
            vision_forward_pass_config,
            keychain=inputs_keychain,
        )
        response_ids = self.generate_tokens(
            inputs.token_ids,
            generation_config=generation_config,
            max_output_length=max_output_length,
            prefill_forward_pass_config=prefill_forward_pass_config,
            decode_forward_pass_config=decode_forward_pass_config,
            prompt_embeddings=inputs.embeddings,
            prompt_positions=inputs.positions,
            keychain=generation_keychain,
        ).token_ids[0]
        rendered_prompt = self.token_codec.render_request(messages, reasoning_effort=reasoning_effort)
        return self.token_codec.parse_response(
            self.token_codec.decode_tokens(generation_config.trim_stop_tokens(response_ids.tolist())),
            prompt=rendered_prompt,
        )
