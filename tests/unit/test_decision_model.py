from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.model import Model
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.decision import DecisionConfig, DecisionModel, DecisionModelConfig
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.modules.embedding import UntiedEmbeddingConfig
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config


def test_standard_load_runs_masked_last_token_decisions(tmp_path: Path) -> None:
    sharding = ShardingConfig.replicated()
    decoder_config = replace(
        build_tiny_attention_decoder_config((None,)),
        embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
    )
    config = DecisionModelConfig(
        token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
        decoder_config=decoder_config,
        generation_config=GenerationConfig(),
        decision=DecisionConfig(("A", "B", "C", "D", "E"), temperature=0.7),
    )
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    original = config.init(tokenizer, RandomInitializer(jnp.float32, sharding, key=jax.random.key(1)))
    original.save(tmp_path)
    loaded = Model.load(tmp_path, sharding)
    assert isinstance(loaded, DecisionModel)
    token_ids = jnp.array([1, 4, 2], dtype=jnp.int32)
    keychain = Keychain.init(0, sharding_config=sharding)
    logits = loaded.decoder(
        token_ids[None],
        jnp.arange(token_ids.size, dtype=jnp.int32)[None],
        forward_pass_config=DecoderForwardPassConfig.for_inference(),
        keychain=keychain,
    ).logits[0, -1]
    assert logits.shape == (5,)
    for num_options in (2, 5):
        actual = loaded.probabilities(token_ids, num_options, keychain=keychain)
        expected = jax.nn.softmax(logits[:num_options].astype(jnp.float32) / config.decision.temperature)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        np.testing.assert_allclose(actual.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        loaded.probabilities(token_ids, 2, keychain=keychain),
        original.probabilities(token_ids, 2, keychain=keychain),
        atol=1e-6,
    )
