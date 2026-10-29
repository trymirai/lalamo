import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.initializer import RandomInitializer
from lalamo.module import Keychain, ShardingConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.embedding import TiedEmbeddingConfig
from lalamo.modules.flow import Add, EmitConfig, SequenceConfig, SinkConfig, StepConfig, validate_program
from lalamo.modules.forward_pass import ModelForwardPassConfig
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig
from tests.helpers import tiny_attention_config, tiny_mlp_config, tiny_norm_config


def _decoder(steps: tuple[StepConfig, ...], taps: tuple[EmitConfig, ...] = ()) -> DecoderConfig:
    return DecoderConfig(
        embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
        program=SequenceConfig(steps),
        model_dim=8,
        vocab_size=32,
        taps=taps,
    )


def test_json_references_restore_shared_emitters_and_kv_owners() -> None:
    embedding = EmitConfig("embedding")
    owner = tiny_attention_config()
    hidden_states = EmitConfig("hidden_states")
    config = _decoder(
        (
            embedding,
            owner,
            SinkConfig((embedding,), Add()),
            tiny_attention_config(owner),
            SinkConfig((embedding,), Add()),
            hidden_states,
            tiny_norm_config(),
        ),
        taps=(hidden_states,),
    )
    raw = json.loads(json.dumps(config.to_json()))
    restored = DecoderConfig.from_json(raw)

    assert restored.to_json() == config.to_json()
    restored_embedding, restored_owner, first_sink, borrower, second_sink, restored_tap, _ = restored.program.steps
    assert isinstance(first_sink, SinkConfig) and isinstance(second_sink, SinkConfig)
    assert first_sink.sources[0] is restored_embedding
    assert second_sink.sources[0] is restored_embedding
    assert isinstance(borrower, AttentionConfig)
    assert borrower.kv_source is restored_owner
    assert restored.taps[0] is restored_tap
    # Shared objects are written once; later occurrences are ordinals derived at serialization time.
    assert raw["program"]["steps"][4]["sources"][0] == {"ref": 0}
    assert raw["program"]["steps"][3]["kv_source"] == {"ref": 1}


def test_json_restores_distinct_emitters_with_equal_fields_as_distinct() -> None:
    first, second = EmitConfig("branch"), EmitConfig("branch")
    config = _decoder((first, second, SinkConfig((first,), Add()), SinkConfig((second,), Add())))

    restored_first, restored_second, first_sink, second_sink = DecoderConfig.from_json(config.to_json()).program.steps

    assert restored_first is not restored_second
    assert isinstance(first_sink, SinkConfig) and isinstance(second_sink, SinkConfig)
    assert first_sink.sources[0] is restored_first
    assert second_sink.sources[0] is restored_second


def test_jit_preserves_kv_owner_identity_when_rerouting_a_program() -> None:
    first, second = tiny_attention_config(), tiny_attention_config()
    borrower = tiny_attention_config(first)
    config = _decoder((first, second, borrower))
    sharding = ShardingConfig.replicated(jax.devices("cpu")[:1])
    with jax.set_mesh(sharding.mesh):
        original = config.init(RandomInitializer(jnp.float32, sharding, key=jax.random.key(4)))
        redirected = replace(borrower, kv_source=second)
        redirected_config = _decoder((first, second, redirected))
        first_module, second_module, borrower_module = original.program.steps
        assert isinstance(borrower_module, Attention)
        rerouted = replace(
            original,
            config=redirected_config,
            program=replace(
                original.program,
                config=redirected_config.program,
                steps=(first_module, second_module, replace(borrower_module, config=redirected)),
            ),
        )
        tokens = jnp.array([[1, 2, 3]])
        positions = jnp.array([[0, 1, 2]])
        keychain = Keychain.init(0, sharding_config=sharding)
        forward = ModelForwardPassConfig.for_tracer_tests()
        original(tokens, positions, forward_pass_config=forward, keychain=keychain)
        actual = rerouted(tokens, positions, forward_pass_config=forward, keychain=keychain).logits
        jax.clear_caches()
        expected = rerouted(tokens, positions, forward_pass_config=forward, keychain=keychain).logits
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


def _invalid_programs() -> list[tuple[str, SequenceConfig, tuple[EmitConfig, ...]]]:
    emitter = EmitConfig("branch")
    return [
        ("A sink requires earlier emitters", SequenceConfig((SinkConfig((emitter,), Add()), emitter)), ()),
        ("occurs twice", SequenceConfig((emitter, emitter)), ()),
        (
            "must be a stateless primitive",
            SequenceConfig((emitter, SinkConfig((emitter,), SequenceConfig((tiny_mlp_config(),))))),
            (),
        ),
        (
            "must be a stateless primitive",
            SequenceConfig((emitter, SinkConfig((emitter,), tiny_attention_config()))),
            (),
        ),
        ("Every tap must be emitted", SequenceConfig((tiny_norm_config(),)), (EmitConfig("hidden_states"),)),
    ]


@pytest.mark.parametrize(("message", "program", "taps"), _invalid_programs())
def test_validate_program_rejects_malformed_graphs(
    message: str,
    program: SequenceConfig,
    taps: tuple[EmitConfig, ...],
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_program(program, taps)
