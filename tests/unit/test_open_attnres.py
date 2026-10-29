from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.model_import.model_configs.huggingface.qwen3_attnres import HFQwen3AttnResConfig
from lalamo.models import GenerationConfig, LanguageModel
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.forward_pass import ModelForwardPassConfig
from tests.helpers import make_test_sharding_config


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize("variant", ["full", "block"])
def test_published_attnres_import_cache_suffix_and_native(variant: str, tmp_path: Path) -> None:
    # Golden outputs come from the author's implementation; provenance is stored in the fixture.
    with np.load(Path(__file__).parents[1] / "data" / f"open_attnres_{variant}_reference.npz") as reference:
        config_path = tmp_path / "foreign-config.json"
        config_path.write_text(str(reference["config"]))
        foreign = HFQwen3AttnResConfig.from_json(config_path)
        weights = {name: reference[name] for name in reference.files if name.startswith("model.")}
        reference_tokens = reference["tokens"]
        expected = reference["logits"]
    sharding = make_test_sharding_config()
    config = foreign.to_lalamo_config(
        ChatCodecConfig("", None, "system", "user", "assistant", None, None), GenerationConfig()
    )
    tokenizer = Tokenizer(WordLevel({f"t{index}": index for index in range(foreign.vocab_size)}, unk_token="t0"))
    model = foreign.load(
        config,
        tokenizer,
        jnp.float32,
        {path: jnp.asarray(value) for path, value in weights.items()},
        sharding_config=sharding,
    )
    assert isinstance(model, LanguageModel)
    token_sharding = sharding.resolve_sharding((LogicalAxis.BATCH, None))
    tokens = jax.device_put(jnp.asarray(reference_tokens, dtype=jnp.int32), token_sharding)
    positions = jax.device_put(jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), tokens.shape), token_sharding)
    keychain = Keychain.init(0, sharding_config=sharding)
    forward = ModelForwardPassConfig.for_tracer_tests()
    result = model.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain)
    np.testing.assert_allclose(result.logits, expected, atol=3e-5, rtol=3e-5)
    state = model.decoder.init_static_state(2, 8, jnp.float32)
    pieces = []
    for start, end in ((0, 2), (2, 4)):
        result = model.decoder(
            tokens[:, start:end],
            positions[:, start:end],
            state,
            return_updated_state=True,
            forward_pass_config=forward,
            keychain=keychain,
        )
        state = result.updated_state
        pieces.append(result.logits)
    np.testing.assert_allclose(jnp.concatenate(pieces, axis=1), expected, atol=3e-5, rtol=3e-5)
    for suffix in (1, 3):
        result = model.decoder(
            tokens, positions, return_suffix_tokens=suffix, forward_pass_config=forward, keychain=keychain
        )
        np.testing.assert_allclose(result.logits, expected[:, -suffix:], atol=3e-5, rtol=3e-5)
    result = model.decoder(
        tokens,
        positions,
        lengths_without_padding=jax.device_put(jnp.array([4, 2]), sharding.resolve_sharding((LogicalAxis.BATCH,))),
        return_suffix_tokens=1,
        forward_pass_config=forward,
        keychain=keychain,
    )
    np.testing.assert_allclose(
        result.logits, np.stack((expected[0, -1], expected[1, 1]))[:, None], atol=3e-5, rtol=3e-5
    )
    model.save(tmp_path)
    restored = LanguageModel.load(tmp_path, sharding)
    np.testing.assert_array_equal(
        restored.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain).logits,
        model.decoder(tokens, positions, forward_pass_config=forward, keychain=keychain).logits,
    )
