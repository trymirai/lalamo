import jax
import jax.numpy as jnp
import pytest

from lalamo.module import Keychain
from lalamo.sampling import SamplingPolicy


def _is_neginf(array: jax.Array) -> tuple[bool, ...]:
    return tuple(bool(value) for value in jnp.isneginf(array).tolist())


def test_init_uses_default_sampling_parameters() -> None:
    policy = SamplingPolicy.init()

    assert policy.temperature.shape == ()
    assert policy.temperature.dtype == jnp.float32
    assert policy.temperature.item() == 1.0
    assert policy.top_k.shape == ()
    assert policy.top_k.dtype == jnp.int32
    assert policy.top_k.item() == 0
    assert policy.top_p.shape == ()
    assert policy.top_p.dtype == jnp.float32
    assert policy.top_p.item() == 1.0
    assert policy.min_p.shape == ()
    assert policy.min_p.dtype == jnp.float32
    assert policy.min_p.item() == 0.0
    assert policy.banned_tokens.shape == (16,)
    assert policy.banned_tokens.dtype == jnp.int32
    assert jnp.array_equal(policy.banned_tokens, jnp.full((16,), -1, dtype=jnp.int32))
    assert policy.repetition_penalty.shape == ()
    assert policy.repetition_penalty.dtype == jnp.float32
    assert policy.repetition_penalty.item() == 1.0
    assert policy.presence_penalty.shape == ()
    assert policy.presence_penalty.dtype == jnp.float32
    assert policy.presence_penalty.item() == 0.0
    assert policy.frequency_penalty.shape == ()
    assert policy.frequency_penalty.dtype == jnp.float32
    assert policy.frequency_penalty.item() == 0.0
    assert policy.token_counts is None


def test_init_pads_banned_tokens() -> None:
    policy = SamplingPolicy.init(banned_tokens=(1, 3))

    assert jnp.array_equal(
        policy.banned_tokens,
        jnp.array([1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=jnp.int32),
    )


def test_init_rejects_too_many_banned_tokens() -> None:
    with pytest.raises(ValueError, match="At most 16 banned tokens"):
        SamplingPolicy.init(banned_tokens=range(17))


def test_init_rejects_negative_banned_tokens() -> None:
    with pytest.raises(ValueError, match="Banned tokens must be non-negative"):
        SamplingPolicy.init(banned_tokens=(-1,))


def test_init_rejects_non_positive_repetition_penalty() -> None:
    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        SamplingPolicy.init(repetition_penalty=0.0)


def test_init_batch_stacks_sampling_parameters() -> None:
    policy = SamplingPolicy.init_batch(
        temperature=(0.0, 0.5),
        top_k=(1, 2),
        top_p=(0.75, 0.9),
        min_p=(0.0, 0.1),
        banned_tokens=((1,), (2, 3)),
        repetition_penalty=(1.0, 1.2),
        presence_penalty=(0.0, 0.3),
        frequency_penalty=(0.0, 0.4),
    )

    assert jnp.array_equal(policy.temperature, jnp.array([0.0, 0.5], dtype=jnp.float32))
    assert jnp.array_equal(policy.top_k, jnp.array([1, 2], dtype=jnp.int32))
    assert jnp.array_equal(policy.top_p, jnp.array([0.75, 0.9], dtype=jnp.float32))
    assert jnp.array_equal(policy.min_p, jnp.array([0.0, 0.1], dtype=jnp.float32))
    assert jnp.array_equal(policy.repetition_penalty, jnp.array([1.0, 1.2], dtype=jnp.float32))
    assert jnp.array_equal(policy.presence_penalty, jnp.array([0.0, 0.3], dtype=jnp.float32))
    assert jnp.array_equal(policy.frequency_penalty, jnp.array([0.0, 0.4], dtype=jnp.float32))
    assert policy.banned_tokens.shape == (2, 16)
    assert jnp.array_equal(policy.banned_tokens[0, :3], jnp.array([1, -1, -1], dtype=jnp.int32))
    assert jnp.array_equal(policy.banned_tokens[1, :4], jnp.array([2, 3, -1, -1], dtype=jnp.int32))
    assert policy.token_counts is None


def test_init_batch_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        SamplingPolicy.init_batch(
            temperature=(1.0,),
            top_k=(0, 1),
            top_p=(1.0,),
            min_p=(0.0,),
            banned_tokens=((),),
        )


def test_default_policy_leaves_logits_unchanged() -> None:
    logits = jnp.array([1.0, -2.0, 3.5], dtype=jnp.float32)

    result = SamplingPolicy.init().process_logits(logits)

    assert jnp.array_equal(result, logits)


def test_banned_tokens_are_replaced_with_negative_infinity() -> None:
    logits = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(banned_tokens=(1, 3))

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (False, True, False, True)
    assert result[0].item() == 0.0
    assert result[2].item() == 2.0


def test_zero_temperature_keeps_only_best_token() -> None:
    logits = jnp.array([0.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(temperature=0.0)

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([-jnp.inf, 1.0, -jnp.inf], dtype=jnp.float32))


def test_nonzero_temperature_divides_logits() -> None:
    logits = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(temperature=2.0)

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32))


def test_top_k_keeps_highest_k_logits() -> None:
    logits = jnp.array([1.0, 5.0, 3.0, 4.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(top_k=2)

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (True, False, True, False)
    assert result[1].item() == 5.0
    assert result[3].item() == 4.0


def test_top_k_larger_than_vocabulary_keeps_all_logits() -> None:
    logits = jnp.array([1.0, 5.0, 3.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(top_k=8)

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, logits)


def test_top_p_keeps_at_least_the_highest_logit() -> None:
    logits = jnp.array([5.0, 4.0, 3.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(top_p=0.5)

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (False, True, True)
    assert result[0].item() == 5.0


def test_top_p_keeps_first_token_past_cumulative_threshold() -> None:
    logits = jnp.array([3.0, 2.0, 1.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(top_p=0.9)

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (False, False, True)


def test_min_p_keeps_logits_above_relative_probability_floor() -> None:
    logits = jnp.log(jnp.array([1.0, 0.25, 0.05], dtype=jnp.float32))
    policy = SamplingPolicy.init(min_p=0.2)

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (False, False, True)


def test_repetition_penalty_applies_to_seen_tokens() -> None:
    logits = jnp.array([4.0, -3.0, 8.0, 1.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(repetition_penalty=2.0).with_prompt_token_counts(
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array(2, dtype=jnp.int32),
        vocabulary_size=4,
    )

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([2.0, -6.0, 8.0, 1.0], dtype=jnp.float32))


def test_repetition_penalty_ignores_out_of_vocab_prompt_tokens() -> None:
    policy = SamplingPolicy.init(repetition_penalty=2.0).with_prompt_token_counts(
        jnp.array([1, -1, 9, 2], dtype=jnp.int32),
        jnp.array(4, dtype=jnp.int32),
        vocabulary_size=4,
    )

    assert policy.token_counts is not None
    assert jnp.array_equal(policy.token_counts, jnp.array([0, 1, 1, 0], dtype=jnp.int32))

    logits = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([1.0, 1.0, 2.0, 8.0], dtype=jnp.float32))


def test_repetition_penalty_ignores_out_of_vocab_generated_tokens() -> None:
    policy = SamplingPolicy.init(repetition_penalty=2.0).with_prompt_token_counts(
        jnp.array([1], dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
        vocabulary_size=4,
    )

    updated_policy = policy.with_next_token_count(jnp.array(99, dtype=jnp.int32))
    updated_policy = updated_policy.with_next_token_count(jnp.array(-1, dtype=jnp.int32))

    assert updated_policy.token_counts is not None
    assert jnp.array_equal(updated_policy.token_counts, jnp.array([0, 1, 0, 0], dtype=jnp.int32))


def test_presence_penalty_applies_once_to_seen_tokens() -> None:
    logits = jnp.array([4.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(presence_penalty=0.5).with_prompt_token_counts(
        jnp.array([0, 0, 2], dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        vocabulary_size=3,
    )

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([3.5, 3.0, 1.5], dtype=jnp.float32))


def test_frequency_penalty_scales_with_seen_token_count() -> None:
    logits = jnp.array([4.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(frequency_penalty=0.5).with_prompt_token_counts(
        jnp.array([0, 0, 2], dtype=jnp.int32),
        jnp.array(3, dtype=jnp.int32),
        vocabulary_size=3,
    )

    result = policy.process_logits(logits)

    assert jnp.array_equal(result, jnp.array([3.0, 3.0, 1.5], dtype=jnp.float32))


def test_token_counts_are_updated_with_generated_tokens() -> None:
    policy = SamplingPolicy.init(frequency_penalty=1.0).with_prompt_token_counts(
        jnp.array([1], dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
        vocabulary_size=3,
    )

    updated_policy = policy.with_next_token_count(jnp.array(1, dtype=jnp.int32))

    assert updated_policy.token_counts is not None
    assert jnp.array_equal(updated_policy.token_counts, jnp.array([0, 2, 0], dtype=jnp.int32))


def test_filters_are_applied_in_policy_order() -> None:
    logits = jnp.array([1.0, 4.0, 3.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(top_k=1, banned_tokens=(1,))

    result = policy.process_logits(logits)

    assert _is_neginf(result) == (True, True, False)
    assert result[2].item() == 3.0


def test_process_logits_rejects_batched_policy_without_vmap() -> None:
    policy = SamplingPolicy.init_batch(
        temperature=(1.0, 1.0),
        top_k=(0, 0),
        top_p=(1.0, 1.0),
        min_p=(0.0, 0.0),
        banned_tokens=((), ()),
    )
    logits = jnp.array([0.0, 1.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="Use vmap"):
        policy.process_logits(logits)


def test_batched_policy_processes_logits_under_vmap() -> None:
    policy = SamplingPolicy.init_batch(
        temperature=(0.0, 1.0),
        top_k=(0, 1),
        top_p=(1.0, 1.0),
        min_p=(0.0, 0.0),
        banned_tokens=((), ()),
    )
    logits = jnp.array(
        [
            [1.0, 3.0, 2.0],
            [1.0, 3.0, 2.0],
        ],
        dtype=jnp.float32,
    )

    result = jax.vmap(lambda policy_row, logits_row: policy_row.process_logits(logits_row))(policy, logits)

    assert jnp.array_equal(
        result,
        jnp.array(
            [
                [-jnp.inf, 1.0, -jnp.inf],
                [-jnp.inf, 3.0, -jnp.inf],
            ],
            dtype=jnp.float32,
        ),
    )


def test_call_samples_greedy_token_when_temperature_is_zero() -> None:
    keychain = Keychain.init(0)
    logits = jnp.array([0.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(temperature=0.0)

    result = policy(logits, keychain=keychain)

    assert result.shape == ()
    assert result.item() == 1
