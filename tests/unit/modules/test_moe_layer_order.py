"""Routing traces and routing state of a mixture of experts under `layer_order`.

For routing, a layer applied several times per token is several independent layers: every application has
a trace of its own and an intervention state of its own. So the expectation is never computed from the
looped model. A decoder that applies layer 0 twice must behave exactly like an unrolled decoder holding two
copies of that layer, where each copy has its own state slot by construction.
"""

from dataclasses import replace
from itertools import accumulate, pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from lalamo.initializer import RandomInitializer
from lalamo.modules import (
    CacheConditionalRouting,
    Decoder,
    DecoderForwardPassConfig,
    EmbeddingForwardPassConfig,
    Keychain,
    LinearConfig,
    RoutingPhase,
    with_routing_intervention,
)
from lalamo.modules.mlp import MixtureOfExpertsConfig, SoftmaxRouting
from lalamo.utils.sharding import ShardingConfig
from tests.helpers import build_tiny_attention_decoder_config

CHUNK_LENGTHS = (5, 1, 4)
LAYER_ORDER = (0, 0, 1)
NUM_EXPERTS = 6
NUM_ACTIVE = 2
RULE = CacheConditionalRouting(
    phases=(RoutingPhase.CONTEXT, RoutingPhase.GENERATION), bias=0.7, forced_top=1, cache_capacity=3
)


def init_moe_decoder(layer_count: int, layer_order: tuple[int, ...] | None) -> Decoder:
    base = build_tiny_attention_decoder_config((None,) * layer_count, layer_order=layer_order)
    transformer_config = base.transformer_config
    layer_configs = tuple(
        replace(
            layer_config,
            mlp=replace(
                layer_config.mlp,
                body_config=MixtureOfExpertsConfig(
                    expert_config=layer_config.mlp.body_config,
                    router_config=LinearConfig(),
                    routing_function=SoftmaxRouting(),
                    num_routed_experts=NUM_EXPERTS,
                    num_active_routed_experts=NUM_ACTIVE,
                    router_has_biases=False,
                    num_shared_experts=0,
                    expert_hidden_dim=transformer_config.hidden_dim,
                ),
            ),
        )
        for layer_config in transformer_config.layer_configs
    )
    config = replace(base, transformer_config=replace(transformer_config, layer_configs=layer_configs))
    return config.init(
        RandomInitializer(
            default_dtype=jnp.float32,
            sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
            key=jax.random.key(4),
        ),
    )


def looped_and_unrolled_decoders() -> tuple[Decoder, Decoder]:
    looped = init_moe_decoder(2, LAYER_ORDER)
    first_layer, second_layer = looped.transformer.layers
    unrolled = eqx.tree_at(
        lambda decoder: (decoder.embedding, decoder.transformer.layers, decoder.transformer.output_norm),
        init_moe_decoder(3, None),
        (looped.embedding, (first_layer, first_layer, second_layer), looped.transformer.output_norm),
    )
    return looped, unrolled


def token_batch(vocab_size: int) -> Int[Array, "batch tokens"]:
    return jax.random.randint(jax.random.key(0), (2, sum(CHUNK_LENGTHS)), 0, vocab_size, dtype=jnp.int32)


def chunked_run(decoder: Decoder, token_ids: Int[Array, "batch tokens"]) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Logits, active experts as [applications, batch, tokens, active] and the final state, over chunks."""
    batch_size, num_tokens = token_ids.shape
    token_positions = jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), (batch_size, num_tokens))
    state = decoder.init_static_state(batch_size, num_tokens, jnp.float32)
    logits, experts = [], []
    for start, end in pairwise(accumulate(CHUNK_LENGTHS, initial=0)):
        result = decoder(
            token_ids[:, start:end],
            token_positions[:, start:end],
            state=state,
            return_updated_state=True,
            return_routing_traces=True,
            forward_pass_config=DecoderForwardPassConfig(
                embedding_forward_pass_config=EmbeddingForwardPassConfig(activation_dtype=jnp.float32),
            ),
            keychain=Keychain.init(1, sharding_config=decoder.sharding_config),
        )
        assert result.routing_traces is not None
        assert all(trace is not None for trace in result.routing_traces)
        state = result.updated_state
        logits.append(np.asarray(result.logits))
        experts.append(np.stack([np.asarray(trace.active_expert_indices) for trace in result.routing_traces]))
    return np.concatenate(logits, axis=1), np.concatenate(experts, axis=2), state


def test_a_looped_decoder_reports_one_routing_trace_per_application() -> None:
    # Spec: traces follow the applications of `layer_order`, not the stored layers. The unrolled decoder has
    # one trace per layer by definition, so the looped one must report the same three, in the same order.
    looped, unrolled = looped_and_unrolled_decoders()
    token_ids = token_batch(looped.vocab_size)

    _, looped_experts, _ = chunked_run(looped, token_ids)
    _, unrolled_experts, _ = chunked_run(unrolled, token_ids)

    assert looped_experts.shape == (len(LAYER_ORDER), *token_ids.shape, NUM_ACTIVE)
    np.testing.assert_array_equal(looped_experts, unrolled_experts)


def test_every_application_of_a_layer_keeps_its_own_intervention_state() -> None:
    # MR: a stateful routing rule on a looped decoder must act as it does on the unrolled copies, where each
    # copy owns a state slot. Shared state between the two applications of layer 0 would let the second read
    # the cache the first has just written, and its routing -- and the logits -- would differ.
    looped, unrolled = (with_routing_intervention(decoder, RULE) for decoder in looped_and_unrolled_decoders())
    token_ids = token_batch(looped.vocab_size)

    looped_logits, looped_experts, looped_state = chunked_run(looped, token_ids)
    unrolled_logits, unrolled_experts, _ = chunked_run(unrolled, token_ids)

    assert len(looped_state) == len(LAYER_ORDER)
    assert all(layer_state.routing is not None for layer_state in looped_state)
    np.testing.assert_array_equal(looped_experts, unrolled_experts)
    np.testing.assert_array_equal(looped_logits, unrolled_logits)


def test_the_rule_changes_routing_in_this_fixture() -> None:
    # Guard against a vacuous pass of the test above: with a bias of 0.7 over six experts and a cache of three
    # the rule must redirect at least one routing decision here, otherwise that test compares two base models.
    looped, _ = looped_and_unrolled_decoders()
    token_ids = token_batch(looped.vocab_size)

    _, base_experts, _ = chunked_run(looped, token_ids)
    _, ruled_experts, _ = chunked_run(with_routing_intervention(looped, RULE), token_ids)

    assert not np.array_equal(np.sort(base_experts, axis=-1), np.sort(ruled_experts, axis=-1))
