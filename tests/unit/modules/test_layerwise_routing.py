"""Tests for `LayerwiseRouting`: a routing rule of its own for every mixture of a model.

The container is a way to write per-layer rules down, so its expectations follow from what it stands for:
the model with rules (r_0, ..., r_n) is the model where layer i routes under r_i. With one rule repeated it is
the model under that rule, with no rules at all it is the model itself, and a rule placed on one layer must
not reach the others.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.modules import (
    CacheConditionalRouting,
    Decoder,
    LayerwiseRouting,
    MixtureOfExperts,
    RoutingIntervention,
    RoutingPhase,
    SoftmaxRouting,
    with_routing_intervention,
)
from lalamo.utils.surgery import select_nodes_of_type
from tests.unit.modules.test_moe_layer_order import LAYER_ORDER, chunked_run, init_moe_decoder, token_batch

BOTH_PHASES = (RoutingPhase.CONTEXT, RoutingPhase.GENERATION)
STRONG = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.7, forced_top=1, cache_capacity=3)
MILD = CacheConditionalRouting(phases=BOTH_PHASES, bias=0.2, forced_top=1, cache_capacity=4)


def rules_of(decoder: Decoder) -> list[RoutingIntervention | None]:
    return [mixture.config.routing_intervention for _, mixture in select_nodes_of_type(MixtureOfExperts, decoder)]


def run(decoder: Decoder) -> tuple[np.ndarray, np.ndarray]:
    logits, experts, _ = chunked_run(decoder, token_batch(decoder.config.vocab_size))
    return logits, experts


def test_every_layer_receives_its_own_entry_in_layer_order() -> None:
    # Spec: the entries go to the mixtures in the order the model holds them.
    decoder = with_routing_intervention(init_moe_decoder(3, None), LayerwiseRouting(rules=(STRONG, None, MILD)))
    assert rules_of(decoder) == [STRONG, None, MILD]


def test_one_rule_on_every_layer_is_the_model_under_that_rule() -> None:
    # Equivalent parametrisation: (r, r, r) layer by layer and r for the whole model are the same model.
    base = init_moe_decoder(3, None)
    layerwise = with_routing_intervention(base, LayerwiseRouting(rules=(STRONG, STRONG, STRONG)))
    uniform = with_routing_intervention(base, STRONG)
    assert rules_of(layerwise) == rules_of(uniform)
    np.testing.assert_array_equal(run(layerwise)[0], run(uniform)[0])
    np.testing.assert_array_equal(run(layerwise)[1], run(uniform)[1])


def test_no_rule_on_any_layer_is_the_model_itself() -> None:
    # Neutral element: None on every layer removes the rules, as with_routing_intervention(model, None) does.
    base = init_moe_decoder(3, None)
    cleared = with_routing_intervention(with_routing_intervention(base, STRONG), LayerwiseRouting(rules=(None,) * 3))
    assert rules_of(cleared) == [None, None, None]
    np.testing.assert_array_equal(run(cleared)[0], run(base)[0])


def test_a_rule_on_the_last_layer_leaves_the_routing_of_the_layers_before_it() -> None:
    # Causality across depth: layers below a rule never see it, so their routing traces are those of the
    # model without any rule, while the layer that carries the rule routes differently.
    base = init_moe_decoder(3, None)
    ruled = with_routing_intervention(base, LayerwiseRouting(rules=(None, None, STRONG)))
    _, base_experts = run(base)
    _, ruled_experts = run(ruled)
    np.testing.assert_array_equal(ruled_experts[:2], base_experts[:2])
    assert not np.array_equal(np.sort(ruled_experts[2], axis=-1), np.sort(base_experts[2], axis=-1))


def test_a_looped_layer_keeps_one_rule_for_all_its_applications() -> None:
    # Docstring: the entries are per stored layer. Under layer_order (0, 0, 1) there are two mixtures, the
    # first applied twice, and the rule on it acts on both applications while the second layer stays plain.
    base = init_moe_decoder(2, LAYER_ORDER)
    ruled = with_routing_intervention(base, LayerwiseRouting(rules=(STRONG, None)))
    assert rules_of(ruled) == [STRONG, None]
    everywhere = with_routing_intervention(base, STRONG)
    _, ruled_experts = run(ruled)
    _, everywhere_experts = run(everywhere)
    np.testing.assert_array_equal(ruled_experts[:2], everywhere_experts[:2])


@pytest.mark.parametrize("rules", [(STRONG,), (STRONG, None, MILD, None)])
def test_a_wrong_number_of_rules_is_refused(rules: tuple[RoutingIntervention | None, ...]) -> None:
    with pytest.raises(ValueError, match="mixtures"):
        with_routing_intervention(init_moe_decoder(3, None), LayerwiseRouting(rules=rules))


def test_the_container_survives_a_json_round_trip_with_empty_entries() -> None:
    # Candidate specs are JSON files; an entry without a rule must come back as None.
    container = LayerwiseRouting(rules=(STRONG, None, MILD))
    assert RoutingIntervention.from_json(container.to_json()) == container


def test_the_container_itself_never_routes() -> None:
    # Docstring: the mixtures never see the container; routing through it is a programming error.
    with pytest.raises(TypeError, match="with_routing_intervention"):
        LayerwiseRouting(rules=(STRONG,)).route(
            jnp.zeros((1, 6), dtype=jnp.float32), jnp.ones(1, dtype=jnp.bool_), None, SoftmaxRouting(), 2
        )
