from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from lalamo.exportable import Exportable
from lalamo.initializer import RandomInitializer
from lalamo.model import Model
from lalamo.model_import.model_configs.huggingface.qwen3_5_vision import HFQwen35VisionConfig
from lalamo.models import GenerationConfig, LanguageModel, VisionLanguageModel, VisionLanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig, ImagePart, TextPart, UserMessage
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules import MRoPEConfig, MRoPELayout
from lalamo.modules.vision import ImagePreprocessingConfig, MRoPEImagePositions
from tests.helpers import build_tiny_attention_decoder_config

SHARDING = ShardingConfig.replicated(jax.devices("cpu")[:1])
IMAGE_TOKEN = "<|image_pad|>"
WORDS = ["describe", "this", "picture", IMAGE_TOKEN, *(f"w{index}" for index in range(28))]
# Mirrors how Qwen's template renders content parts, with spaces so a word-level tokenizer can split it.
TEMPLATE = (
    "{% for message in messages %}{% if message.content is string %}{{ message.content }}"
    "{% else %}{% for part in message.content %}"
    "{% if part.type == 'image' %} <|image_pad|> {% else %} {{ part.text }} {% endif %}"
    "{% endfor %}{% endif %}{% endfor %}"
)


def _vlm_config() -> VisionLanguageModelConfig:
    decoder_config = build_tiny_attention_decoder_config((None,))
    mrope = MRoPEConfig(
        base=10_000.0, max_sequence_length=64, head_dim=4, sections=(0, 1, 1), layout=MRoPELayout.CHUNKED
    )
    layers = tuple(replace(layer, rope_config=mrope) for layer in decoder_config.transformer_config.layer_configs)
    decoder_config = replace(
        decoder_config,
        transformer_config=replace(decoder_config.transformer_config, layer_configs=layers),
    )
    vision_config = HFQwen35VisionConfig._converter.structure(  # noqa: SLF001
        {
            "model_type": "qwen3_5",
            "depth": 1,
            "hidden_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "in_channels": 3,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": decoder_config.transformer_config.model_dim,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        HFQwen35VisionConfig,
    )
    return VisionLanguageModelConfig(
        token_codec_config=ChatCodecConfig(TEMPLATE, None, "system", "user", "assistant", None, None),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(temperature=0.0),
        vision_encoder_config=vision_config.to_vision_encoder_config(),
        image_preprocessing_config=ImagePreprocessingConfig(
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=16 * 16,
            max_pixels=64 * 64,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        image_position_scheme=MRoPEImagePositions(),
        image_token_id=WORDS.index(IMAGE_TOKEN),
    )


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="w0"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


@pytest.fixture(scope="module")
def model() -> VisionLanguageModel:
    return _vlm_config().init(_tokenizer(), RandomInitializer(jnp.float32, SHARDING, key=jax.random.key(0)))


def _message() -> UserMessage:
    pixels = np.random.default_rng(0).integers(0, 256, (24, 40, 3), dtype=np.uint8)
    return UserMessage((ImagePart(Image.fromarray(pixels)), TextPart("describe this picture")))


def test_prompt_expands_each_image_to_its_tokens(model: VisionLanguageModel) -> None:
    prompt = model.build_prompt([_message()])
    assert prompt.images is not None
    (grid,) = prompt.images.grids
    image_token_id = model.vision_config.image_token_id
    assert int((prompt.token_ids == image_token_id).sum()) == grid.num_merged_tokens(2)
    assert prompt.positions.shape == (len(prompt.token_ids), 3)
    # The text after the image resumes at image start + max(rows, cols) of the merged grid.
    rows, cols = grid.height_in_patches // 2, grid.width_in_patches // 2
    first_after_image = int(np.flatnonzero(prompt.token_ids == image_token_id)[-1]) + 1
    np.testing.assert_array_equal(prompt.positions[first_after_image], [max(rows, cols)] * 3)


def test_rejects_rope_that_cannot_take_image_positions() -> None:
    config = _vlm_config()
    plain_decoder = build_tiny_attention_decoder_config((None,))
    with pytest.raises(ValueError, match="positions per token"):
        replace(config, decoder_config=plain_decoder)


def test_rejects_placeholder_mismatch(model: VisionLanguageModel) -> None:
    with pytest.raises(ValueError, match="placeholders"):
        model.build_prompt([UserMessage(f"describe {IMAGE_TOKEN} {IMAGE_TOKEN}"), _message()])


def test_reply_runs_end_to_end(model: VisionLanguageModel) -> None:
    reply = model.reply([_message()], max_output_length=4, keychain=Keychain.init(0, sharding_config=SHARDING))
    assert isinstance(reply.response, str)


def test_save_load_round_trip_keeps_text_export_layout(model: VisionLanguageModel, tmp_path: Path) -> None:
    model.save(tmp_path)
    restored = Model.load(tmp_path, SHARDING)
    assert isinstance(restored, VisionLanguageModel)
    assert restored.config == model.config

    keychain = Keychain.init(0, sharding_config=SHARDING)
    prompt = model.build_prompt([_message()])
    original_inputs = model.prompt_inputs(prompt, keychain=keychain)
    restored_inputs = restored.prompt_inputs(prompt, keychain=keychain)
    assert jnp.array_equal(original_inputs.embeddings, restored_inputs.embeddings)

    # Uzu compatibility: the text weights export under exactly the names a plain LanguageModel uses.
    text_only = LanguageModelConfig(
        token_codec_config=model.config.token_codec_config,
        decoder_config=model.config.decoder_config,
        generation_config=model.config.generation_config,
    ).init(_tokenizer(), RandomInitializer(jnp.float32, SHARDING, key=jax.random.key(0)))
    assert isinstance(text_only, LanguageModel)
    vlm_keys = set(Exportable.export(model).arrays)
    text_keys = set(Exportable.export(text_only).arrays)
    assert text_keys <= vlm_keys
    assert all(key.startswith("vision_encoder.") for key in vlm_keys - text_keys)
