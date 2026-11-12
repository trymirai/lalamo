from dataclasses import replace
from pathlib import Path

import cattrs
import jax
import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.exportable import Exportable
from lalamo.initializer import RandomInitializer
from lalamo.model import Model
from lalamo.model_import.model_configs.huggingface.qwen3_5_vision import HFQwen35VisionConfig
from lalamo.models import GenerationConfig, LanguageModel, VisionLanguageModel, VisionLanguageModelConfig
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import LanguageModelConfig
from lalamo.module import ShardingConfig
from lalamo.modules import MRoPEConfig, MRoPELayout
from lalamo.modules.vision import ImagePreprocessingConfig
from tests.helpers import build_tiny_attention_decoder_config, make_test_sharding_config

SHARDING = ShardingConfig.replicated(jax.devices("cpu")[:1])
WORDS = ["describe", "this", "picture", "<|image_pad|>", *(f"w{index}" for index in range(28))]


def _vlm_config() -> VisionLanguageModelConfig:
    decoder_config = build_tiny_attention_decoder_config((None,))
    mrope = MRoPEConfig(
        base=10_000.0, max_sequence_length=64, head_dim=4, sections=(0, 1, 1), layout=MRoPELayout.INTERLEAVED
    )
    layers = tuple(replace(layer, rope_config=mrope) for layer in decoder_config.transformer_config.layer_configs)
    decoder_config = replace(
        decoder_config,
        transformer_config=replace(decoder_config.transformer_config, layer_configs=layers),
    )
    vision_config = cattrs.structure(
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
            "deepstack_visual_indexes": [],
        },
        HFQwen35VisionConfig,
    )
    return VisionLanguageModelConfig(
        token_codec_config=ChatCodecConfig("{{ messages }}", None, "system", "user", "assistant", None, None),
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
        image_token_id=WORDS.index("<|image_pad|>"),
    )


def _tokenizer() -> Tokenizer:
    return Tokenizer(WordLevel({word: index for index, word in enumerate(WORDS)}, unk_token="w0"))


def test_rejects_encoder_narrower_than_decoder() -> None:
    config = _vlm_config()
    narrow_encoder = replace(config.vision_encoder_config, output_dim=config.vision_encoder_config.output_dim + 1)
    with pytest.raises(ValueError, match="decoder width"):
        replace(config, vision_encoder_config=narrow_encoder)


def test_vision_encoder_stays_replicated_under_sharding() -> None:
    sharded = make_test_sharding_config()
    model = _vlm_config().init(_tokenizer(), RandomInitializer(jnp.float32, sharded, key=jax.random.key(0)))
    assert model.vision_encoder.sharding_config.mesh == sharded.mesh
    assert not model.vision_encoder.sharding_config.logical_to_physical


def test_save_load_round_trip_keeps_text_export_layout(tmp_path: Path) -> None:
    model = _vlm_config().init(_tokenizer(), RandomInitializer(jnp.float32, SHARDING, key=jax.random.key(0)))
    model.save(tmp_path)
    restored = Model.load(tmp_path, SHARDING)
    assert isinstance(restored, VisionLanguageModel)
    assert restored.config == model.config
    original_arrays = Exportable.export(model).arrays
    restored_arrays = Exportable.export(restored).arrays
    assert original_arrays.keys() == restored_arrays.keys()
    assert all(jnp.array_equal(original_arrays[name], restored_arrays[name]) for name in original_arrays)

    # Uzu compatibility: the text weights export under exactly the names a plain LanguageModel uses.
    text_only = LanguageModelConfig(
        token_codec_config=model.config.token_codec_config,
        decoder_config=model.config.decoder_config,
        generation_config=model.config.generation_config,
    ).init(_tokenizer(), RandomInitializer(jnp.float32, SHARDING, key=jax.random.key(0)))
    assert isinstance(text_only, LanguageModel)
    text_keys = set(Exportable.export(text_only).arrays)
    assert text_keys <= set(original_arrays)
    assert all(key.startswith("vision_encoder.") for key in set(original_arrays) - text_keys)
