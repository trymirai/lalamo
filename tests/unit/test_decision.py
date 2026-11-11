from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.mark.parametrize("is_causal", [False, True])
def test_decision_head_probabilities_and_model_reload(tmp_path: Path, is_causal: bool) -> None:
    sharding = ShardingConfig.replicated(jax.devices("cpu")[:1])
    decoder = build_tiny_attention_decoder_config((None,))
    (layer,) = decoder.transformer_config.layer_configs
    attention = layer.mixer.body_config
    assert isinstance(attention, AttentionConfig)
    layer = replace(layer, mixer=replace(layer.mixer, body_config=replace(attention, is_causal=is_causal)))
    config = DecisionModelConfig(
        token_codec_config=ChatCodecConfig("", None, "system", "user", "assistant", None, None),
        decoder_config=replace(
            decoder,
            transformer_config=replace(decoder.transformer_config, layer_configs=(layer,)),
            embedding_config=UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
        ),
        generation_config=GenerationConfig(),
        decision=DecisionConfig(("A", "B", "C"), 0.7),
    )
    with jax.set_mesh(sharding.mesh):
        model = config.init(
            Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            RandomInitializer(jnp.float32, sharding, key=jax.random.key(7)),
        )
        assert model.decoder.embedding.readout_matrix.shape == (3, 8)
        tokens = jnp.array([1, 2, 3], dtype=jnp.int32)
        keychain = Keychain.init(0, sharding_config=sharding)
        decoder = replace(model.decoder, transformer=replace(model.decoder.transformer, output_norm=None))
        raw = decoder(
            tokens[None],
            jnp.arange(3, dtype=jnp.int32)[None],
            keychain=keychain,
            forward_pass_config=DecoderForwardPassConfig.for_inference(),
            return_activation_trace=True,
        ).activation_trace
        assert raw is not None
        hidden = np.asarray(raw.output_norm[0, -1], dtype=np.float32)
        norm = model.decoder.transformer.output_norm
        assert norm is not None
        normalized = hidden / np.sqrt(np.mean(hidden**2) + norm.config.epsilon) * np.asarray(norm.scales)
        weights = np.asarray(model.decoder.embedding.readout_matrix.decompress())
        logits = (weights @ normalized)[:2]
        expected = np.exp((logits - np.max(logits)) / 0.7)
        expected /= expected.sum()
        np.testing.assert_allclose(np.asarray(model.probabilities(tokens, 2, keychain=keychain)), expected, atol=1e-6)
        model.save(tmp_path)
        restored = Model.load(tmp_path, sharding)
        assert isinstance(restored, DecisionModel)
        np.testing.assert_array_equal(
            restored.probabilities(tokens, 3, keychain=keychain), model.probabilities(tokens, 3, keychain=keychain)
        )
        for count in (0, 4):
            with pytest.raises(ValueError, match="Option count"):
                restored.probabilities(tokens, count, keychain=keychain)
        with pytest.raises(ValueError, match="nonempty"):
            restored.probabilities(tokens[:0], 1, keychain=keychain)
        with pytest.raises(ValueError, match="context length"):
            restored.probabilities(jnp.zeros(17, dtype=jnp.int32), 1, keychain=keychain)
