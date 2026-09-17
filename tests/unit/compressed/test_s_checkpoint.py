import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.model_import.loaders.s_checkpoint import load_s_checkpoint
from lalamo.models.chat_codec import ChatCodecConfig
from lalamo.models.language_model import GenerationConfig, LanguageModel, LanguageModelConfig
from lalamo.module import Keychain
from lalamo.modules.decoder import DecoderForwardPassConfig
from lalamo.modules.rope import SavedRoPE
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.safetensors import safe_write
from lalamo.utils.sharding import LogicalAxis, ShardingConfig
from tests.conftest import RunLalamo
from tests.helpers import build_tiny_attention_decoder, make_test_sharding_config


@pytest.fixture
def model(fake_mesh: Mesh) -> LanguageModel:
    with jax.set_mesh(ShardingConfig.replicated(jax.devices("cpu")[:8]).mesh):
        decoder = build_tiny_attention_decoder((None,))
    transformer = decoder.config.transformer_config
    layer = transformer.layer_configs[0]
    assert isinstance(layer.mixer_config, AttentionConfig)
    transformer = replace(
        transformer, layer_configs=(replace(layer, mixer_config=replace(layer.mixer_config, has_gate=True)),)
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


@pytest.mark.parametrize("legacy_attention", [False, True], ids=["muse_fused", "qwen_separate"])
def test_s_dense_export_preserves_parameters_and_execution(
    model: LanguageModel, tmp_path: Path, run_lalamo: RunLalamo, *, legacy_attention: bool
) -> None:
    exported = model.export()
    arrays, metadata = dict(exported.arrays), dict(exported.metadata)
    config = json.loads(json.dumps(model.config.to_json()))
    if legacy_attention:
        config["decoder_config"]["pard_token"] = None
        attention = config["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"]
        attention["qkv_projection_config"] = attention.pop("qkvg_projection_config")
        attention["gate_projection_config"] = attention["qkv_projection_config"]
        attention["has_qkv_biases"] = attention.pop("has_qkvg_biases")
        assert attention.pop("has_gate")
        prefix = "decoder.transformer.layers.0.mixer."
        qkvg = arrays.pop(prefix + "qkvg_projection.weights.weights")
        spec = metadata.pop(prefix + "qkvg_projection.weights.spec")
        qkv, gate = jnp.split(qkvg, [24])
        for name, weights in (("qkv_projection", qkv), ("gate_projection", gate)):
            arrays[prefix + name + ".weights.weights"] = weights
            metadata[prefix + name + ".weights.spec"] = spec

    (tmp_path / "config.json").write_text(json.dumps(config))
    model.token_codec.tokenizer.save(str(tmp_path / "tokenizer.json"))
    with (tmp_path / "model.safetensors").open("wb") as stream:
        safe_write(stream, arrays, metadata={key: json.dumps(value) for key, value in metadata.items()})

    with jax.set_mesh(model.sharding_config.mesh):
        restored = load_s_checkpoint(tmp_path, model.sharding_config)
        result = restored.export()
        assert result.metadata == exported.metadata
        assert result.arrays.keys() == exported.arrays.keys()
        for name, expected in exported.arrays.items():
            actual = result.arrays[name]
            assert actual.dtype == expected.dtype
            assert actual.sharding == expected.sharding
            np.testing.assert_array_equal(actual, expected)
        tokens = jnp.array([[1, 2, 3], [3, 2, 1]], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), tokens.shape)
        batch_sharding = model.sharding_config.resolve_sharding((LogicalAxis.BATCH, None))
        tokens, positions = jax.device_put((tokens, positions), batch_sharding)
        state = model.decoder.init_static_state(batch_size=2, capacity=4, dtype=jnp.bfloat16)
        keychain = Keychain.init(0, sharding_config=model.sharding_config)
        forward = DecoderForwardPassConfig.for_inference()
        expected = model.decoder(tokens, positions, state=state, keychain=keychain, forward_pass_config=forward)
        actual = restored.decoder(tokens, positions, state=state, keychain=keychain, forward_pass_config=forward)
        np.testing.assert_array_equal(actual.logits, expected.logits)

    # CLI loading happens before chat establishes a mesh, and must need no adapter flag.
    with jax.set_mesh(None):
        output = run_lalamo("chat", str(tmp_path), "--message", "token1", "--max-tokens", "2", "--temperature", "0")
        restored.save(tmp_path / "native")
        native = run_lalamo(
            "chat", str(tmp_path / "native"), "--message", "token1", "--max-tokens", "2", "--temperature", "0"
        )
    assert "token" in output
    assert output == native


def test_saved_rope_tables_survive_jit_import_and_native_reload(model: LanguageModel, tmp_path: Path) -> None:
    rope = model.decoder.transformer.ropes[0]
    with jax.set_mesh(model.sharding_config.mesh):
        positions = jnp.arange(rope.config.max_sequence_length, dtype=jnp.int32)
        tables = rope.config.compute_positional_embeddings(positions)
        tables = replace(tables, cosines=tables.cosines.at[0, 0].set(jnp.nextafter(jnp.float32(1), jnp.float32(0))))
        exported = model.export()
        arrays = dict(exported.arrays)
        arrays["decoder.transformer.ropes.0.cosines"] = tables.cosines
        arrays["decoder.transformer.ropes.0.sines"] = tables.sines
        config = model.config.to_json()
        assert isinstance(config, dict) and isinstance(config["decoder_config"], dict)
        config["decoder_config"]["pard_token"] = None
        (tmp_path / "config.json").write_text(json.dumps(config))
        model.token_codec.tokenizer.save(str(tmp_path / "tokenizer.json"))
        with (tmp_path / "model.safetensors").open("wb") as stream:
            safe_write(stream, arrays, metadata={key: json.dumps(value) for key, value in exported.metadata.items()})
        restored = LanguageModel.load(tmp_path, model.sharding_config)
        restored.save(tmp_path / "native")
        reloaded = LanguageModel.load(tmp_path / "native", model.sharding_config)
        converted = LanguageModel.load(tmp_path, model.sharding_config, dtype=jnp.float16)
        assert converted.export().arrays["decoder.embedding.embedding.weights"].dtype == jnp.float16
        for candidate in (restored, reloaded, converted):
            saved_rope = candidate.decoder.transformer.ropes[0]
            assert isinstance(saved_rope, SavedRoPE)
            actual = saved_rope(positions)
            np.testing.assert_array_equal(actual.cosines, tables.cosines)
            np.testing.assert_array_equal(actual.sines, tables.sines)
