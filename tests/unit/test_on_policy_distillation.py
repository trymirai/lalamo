from collections.abc import Callable
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from jaxtyping import Array, Float

from lalamo.compressed.low_rank import LowRankMatrix
from lalamo.distillation import (
    AdvantageBaseline,
    Rollouts,
    apply_gradients,
    distillation_advantages,
    init_optimizer_state,
    inject_low_rank_adapters,
    on_policy_distillation_loss,
    pack_rollouts,
    partition_adapters,
    policy_gradient_loss,
    policy_gradients,
    realign_response_values,
    sampled_token_log_probs,
)
from lalamo.initializer import RandomInitializer
from lalamo.modules import Decoder, DecoderForwardPassConfig, Keychain
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import select_nodes_of_type
from tests.common import assert_close, tolerance
from tests.helpers import build_tiny_attention_decoder_config

TRAINING_CONFIG = DecoderForwardPassConfig.for_training()
FORWARD_PASS_CONFIG = replace(
    TRAINING_CONFIG,
    embedding_forward_pass_config=replace(
        TRAINING_CONFIG.embedding_forward_pass_config,
        activation_dtype=jnp.float32,
    ),
)
PROMPT = (5, 17, 3, 9)
NUM_PADDING_TOKENS = 2


def tiny_decoder(seed: int) -> Decoder:
    initializer = RandomInitializer(
        default_dtype=jnp.float32,
        sharding_config=ShardingConfig.replicated(jax.devices("cpu")[:8]),
        key=jax.random.key(seed),
    )
    return build_tiny_attention_decoder_config((None, None)).init(initializer)


def keychain_for(decoder: Decoder) -> Keychain:
    return Keychain.init(0, sharding_config=decoder.sharding_config)


def enumerated_rollouts(vocab_size: int) -> Rollouts:
    """One rollout per possible single-token response to `PROMPT`, covering the whole next-token distribution."""
    prompts = jnp.tile(jnp.asarray(PROMPT, dtype=jnp.int32), (vocab_size, 1))
    responses = jnp.arange(vocab_size, dtype=jnp.int32)[:, None]
    padding = jnp.zeros((vocab_size, NUM_PADDING_TOKENS), dtype=jnp.int32)
    return Rollouts(
        token_ids=jnp.concatenate([prompts, responses, padding], axis=1),
        prompt_lengths=jnp.full(vocab_size, len(PROMPT), dtype=jnp.int32),
        response_lengths=jnp.ones(vocab_size, dtype=jnp.int32),
    )


def response_log_probs(decoder: Decoder, rollouts: Rollouts) -> Float[Array, "batch predictions"]:
    return sampled_token_log_probs(decoder, rollouts, FORWARD_PASS_CONFIG, keychain=keychain_for(decoder))


def next_token_log_probs(decoder: Decoder) -> Float[Array, " vocabulary"]:
    """Dense reference for the distribution over the token that follows `PROMPT`."""
    logits = decoder(
        jnp.asarray([PROMPT], dtype=jnp.int32),
        jnp.arange(len(PROMPT), dtype=jnp.int32)[None, :],
        forward_pass_config=FORWARD_PASS_CONFIG,
        keychain=keychain_for(decoder),
    ).logits
    return jax.nn.log_softmax(logits[0, -1].astype(jnp.float32))


def gradient_of(loss_fn: Callable[[Decoder], Float[Array, ""]], decoder: Decoder) -> Decoder:
    # Compiled like a real training step: an eager gradient cannot seed a loss that lives on a multi-device mesh.
    return eqx.filter_jit(eqx.filter_grad(loss_fn))(decoder)


def assert_gradients_close(result: Decoder, reference: Decoder) -> None:
    with tolerance(atol=1e-6, rtol=1e-3):
        jax.tree.map(
            lambda result_leaf, reference_leaf: assert_close(result=result_leaf, reference=reference_leaf),
            result,
            reference,
        )


def test_sampled_token_log_probs_match_dense_distribution() -> None:
    decoder = tiny_decoder(seed=0)
    log_probs = response_log_probs(decoder, enumerated_rollouts(decoder.vocab_size))

    response_prediction_index = len(PROMPT) - 1
    with tolerance(atol=1e-5, rtol=1e-4):
        assert_close(result=log_probs[:, response_prediction_index], reference=next_token_log_probs(decoder))
    assert jnp.all(jnp.delete(log_probs, response_prediction_index, axis=1) == 0)


@pytest.mark.parametrize("baseline", list(AdvantageBaseline))
def test_student_matching_teacher_gets_no_gradient(baseline: AdvantageBaseline) -> None:
    student = tiny_decoder(seed=0)
    rollouts = enumerated_rollouts(student.vocab_size)
    teacher_log_probs = response_log_probs(student, rollouts)

    def loss_fn(decoder: Decoder) -> Float[Array, ""]:
        return on_policy_distillation_loss(
            response_log_probs(decoder, rollouts),
            teacher_log_probs,
            rollouts.response_mask,
            baseline=baseline,
        )

    gradient = gradient_of(loss_fn, student)
    assert_gradients_close(gradient, jax.tree.map(jnp.zeros_like, gradient))


@pytest.mark.parametrize("baseline", list(AdvantageBaseline))
def test_expected_gradient_is_reverse_kl_gradient(baseline: AdvantageBaseline) -> None:
    student, teacher = tiny_decoder(seed=0), tiny_decoder(seed=1)
    rollouts = enumerated_rollouts(student.vocab_size)
    teacher_log_probs = response_log_probs(teacher, rollouts)

    def expected_loss(decoder: Decoder) -> Float[Array, ""]:
        student_log_probs = response_log_probs(decoder, rollouts)
        advantages = distillation_advantages(student_log_probs, teacher_log_probs, rollouts.response_mask, baseline)
        # Weighting every enumerated rollout by its sampling probability turns the batch mean into an expectation.
        sampling_weights = decoder.vocab_size * jnp.exp(student_log_probs)
        return policy_gradient_loss(student_log_probs, sampling_weights * advantages, rollouts.response_mask)

    assert_gradients_close(
        gradient_of(expected_loss, student),
        gradient_of(lambda decoder: reverse_kl(decoder, teacher), student),
    )


def expected_policy_gradients(student: Decoder, teacher: Decoder, rollouts: Rollouts) -> Decoder:
    """Adapter gradients averaged over the student's whole next-token distribution instead of over samples."""
    student_log_probs = response_log_probs(student, rollouts)
    advantages = distillation_advantages(
        student_log_probs,
        response_log_probs(teacher, rollouts),
        rollouts.response_mask,
        AdvantageBaseline.BATCH_MEAN,
    )
    sampling_weights = student.vocab_size * jnp.exp(student_log_probs)
    return policy_gradients(
        student,
        rollouts,
        sampling_weights * advantages,
        jnp.sum(rollouts.response_mask),
        FORWARD_PASS_CONFIG,
        keychain=keychain_for(student),
    )


def reverse_kl(student: Decoder, teacher: Decoder) -> Float[Array, ""]:
    student_distribution = next_token_log_probs(student)
    return jnp.sum(jnp.exp(student_distribution) * (student_distribution - next_token_log_probs(teacher)))


def test_pack_rollouts_joins_prompts_and_responses() -> None:
    rollouts = pack_rollouts(
        [(1, 2), (3,)],
        [(7, 8, 9), (5,)],
        num_tokens=6,
        sharding_config=tiny_decoder(seed=0).sharding_config,
    )

    assert rollouts.token_ids.tolist() == [[1, 2, 7, 8, 9, 0], [3, 5, 0, 0, 0, 0]]
    assert rollouts.response_mask.tolist() == [[False, True, True, True, False], [True, False, False, False, False]]


def test_realigned_values_follow_responses_behind_other_prompts() -> None:
    sharding_config = tiny_decoder(seed=0).sharding_config
    responses = [(7, 8, 9), (5,)]
    source = pack_rollouts([(1, 2, 3, 4), (1,)], responses, num_tokens=8, sharding_config=sharding_config)
    target = pack_rollouts([(1, 2), (1, 2, 3)], responses, num_tokens=6, sharding_config=sharding_config)
    source_values = jnp.where(source.response_mask, source.token_ids[:, 1:].astype(jnp.float32), 0.0)

    realigned = realign_response_values(source_values, source.prompt_lengths, target)

    assert realigned.tolist() == [[0.0, 7.0, 8.0, 9.0, 0.0], [0.0, 0.0, 5.0, 0.0, 0.0]]


def test_injected_adapters_leave_log_probs_unchanged() -> None:
    decoder = tiny_decoder(seed=0)
    adapted_decoder = inject_low_rank_adapters(decoder, rank=2, key=jax.random.key(0))
    rollouts = enumerated_rollouts(decoder.vocab_size)

    assert select_nodes_of_type(LowRankMatrix, adapted_decoder)
    assert jnp.array_equal(response_log_probs(adapted_decoder, rollouts), response_log_probs(decoder, rollouts))


def test_only_adapters_receive_gradients() -> None:
    student = inject_low_rank_adapters(tiny_decoder(seed=0), rank=2, key=jax.random.key(0))
    gradients = expected_policy_gradients(student, tiny_decoder(seed=1), enumerated_rollouts(student.vocab_size))

    adapter_gradients, other_gradients = partition_adapters(gradients)
    assert not jax.tree.leaves(other_gradients)
    assert any(jnp.any(gradient != 0) for gradient in jax.tree.leaves(adapter_gradients))


def test_optimizer_steps_lower_reverse_kl() -> None:
    teacher = tiny_decoder(seed=1)
    initial_student = inject_low_rank_adapters(tiny_decoder(seed=0), rank=2, key=jax.random.key(0))
    rollouts = enumerated_rollouts(initial_student.vocab_size)
    optimizer = optax.adam(1e-3)

    student, optimizer_state = initial_student, init_optimizer_state(initial_student, optimizer)
    for _ in range(5):
        gradients = expected_policy_gradients(student, teacher, rollouts)
        student, optimizer_state = apply_gradients(student, optimizer_state, gradients, optimizer)

    assert reverse_kl(student, teacher) < reverse_kl(initial_student, teacher)
