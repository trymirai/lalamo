from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


@pytest.fixture
def model(fake_mesh: Mesh) -> LanguageModel:
    with jax.set_mesh(ShardingConfig.replicated(jax.devices("cpu")[:8]).mesh):
        decoder = build_tiny_attention_decoder((None,))
    transformer = decoder.config.transformer_config
    layer = transformer.layer_configs[0]
    assert isinstance(layer.mixer_config, AttentionConfig)
    transformer = replace(
        transformer,
        model_dim=128,
        layer_configs=(replace(layer, mixer_config=replace(layer.mixer_config, has_gate=True)),),
    )
    config = LanguageModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="{{ messages[0]['content'] }}",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        decoder_config=replace(decoder.config, transformer_config=transformer),
        generation_config=GenerationConfig(),
    )
    tokenizer = Tokenizer(WordLevel(vocab={f"token{i}": i for i in range(32)}, unk_token="token0"))
    with jax.set_mesh(fake_mesh):
        result = config.init(
            tokenizer,
            RandomInitializer(jnp.bfloat16, make_test_sharding_config(), key=jax.random.key(9)),
        )
        return jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if isinstance(value, jax.Array) else value, result
        )
