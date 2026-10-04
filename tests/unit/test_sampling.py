from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from lalamo.models import GenerationConfig
from lalamo.module import Keychain
from lalamo.sampling import SamplingPolicy
from tests.helpers import make_test_sharding_config


def _assert_distribution(result: jax.Array, expected: jax.Array) -> None:
    assert jnp.array_equal(jnp.isneginf(result), jnp.isneginf(expected))
    assert jnp.allclose(jax.nn.softmax(result), jax.nn.softmax(expected), rtol=1e-6, atol=1e-7)


def _with_counts(
    policy: SamplingPolicy,
    tokens: tuple[int, ...],
    length: int,
    vocabulary_size: int,
) -> SamplingPolicy:
    policy = policy.with_empty_token_counts(vocabulary_size)
    for token in tokens[:length]:
        policy = policy.with_next_token_count(jnp.asarray(token, dtype=jnp.int32))
    return policy


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SamplingPolicy.init(banned_tokens=range(17)), "At most 16 banned tokens"),
        (lambda: SamplingPolicy.init(banned_tokens=(-1,)), "Banned tokens must be non-negative"),
        (lambda: SamplingPolicy.init(repetition_penalty=0.0), "repetition_penalty must be positive"),
        (
            lambda: SamplingPolicy.init_batch(
                temperature=(0.5,),
                top_k=(1, 2),
            ),
            "same length",
        ),
    ],
)
def test_init_rejects_invalid_arguments(call: Callable[[], SamplingPolicy], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    ("policy", "logits", "expected"),
    [
        (SamplingPolicy.init(), [1.0, -2.0, 3.5], [1.0, -2.0, 3.5]),
        (SamplingPolicy.init(banned_tokens=(1, 3)), [0.0, 1.0, 2.0, 3.0], [0.0, -jnp.inf, 2.0, -jnp.inf]),
        (SamplingPolicy.init(temperature=0.0), [0.0, 3.0, 2.0], [-jnp.inf, 1.0, -jnp.inf]),
        (
            SamplingPolicy.init(temperature=0.0, top_k=2, top_p=0.5, min_p=0.2, banned_tokens=(1,)),
            [99.0, 100.0, -100.0, -99.0, 0.0],
            [1.0, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf],
        ),
        (SamplingPolicy.init(temperature=2.0), [1.0, 2.0, 4.0], [0.5, 1.0, 2.0]),
        (SamplingPolicy.init(top_k=2), [1.0, 5.0, 3.0, 4.0], [-jnp.inf, 5.0, -jnp.inf, 4.0]),
        (SamplingPolicy.init(top_k=8), [1.0, 5.0, 3.0], [1.0, 5.0, 3.0]),
        (SamplingPolicy.init(top_k=1, top_p=0.5), [3.0, 3.0, 1.0], [3.0, -jnp.inf, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.5), [5.0, 4.0, 3.0], [5.0, -jnp.inf, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.9), [3.0, 2.0, 1.0], [3.0, 2.0, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.5), [0.0, 0.0], [0.0, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.0), [1.0, 4.0, 2.0], [-jnp.inf, 4.0, -jnp.inf]),
        (SamplingPolicy.init(top_p=0.0), [3.0, 3.0, 1.0], [3.0, -jnp.inf, -jnp.inf]),
        (SamplingPolicy.init(min_p=0.2), jnp.log(jnp.array([1.0, 0.25, 0.05])), [0.0, -1.3862944, -jnp.inf]),
        (
            _with_counts(SamplingPolicy.init(repetition_penalty=2.0), (0, 1), 2, 4),
            [4.0, -3.0, 8.0, 1.0],
            [2.0, -6.0, 8.0, 1.0],
        ),
        (
            _with_counts(SamplingPolicy.init(presence_penalty=0.5), (0, 0, 2), 3, 3),
            [4.0, 3.0, 2.0],
            [3.5, 3.0, 1.5],
        ),
        (
            _with_counts(SamplingPolicy.init(frequency_penalty=0.5), (0, 0, 2), 3, 3),
            [4.0, 3.0, 2.0],
            [3.0, 3.0, 1.5],
        ),
        (SamplingPolicy.init(top_k=1, banned_tokens=(1,)), [1.0, 4.0, 3.0], [-jnp.inf, -jnp.inf, 3.0]),
    ],
)
def test_process_logits(policy: SamplingPolicy, logits: jax.Array | list[float], expected: list[float]) -> None:
    result = policy.process_logits(jnp.asarray(logits, dtype=jnp.float32))

    _assert_distribution(result, jnp.array(expected, dtype=jnp.float32))


def test_top_k_keeps_exactly_k_tokens_at_cutoff_ties() -> None:
    result = SamplingPolicy.init(top_k=2).process_logits(jnp.array([5.0, 4.0, 4.0, 4.0, 3.0], dtype=jnp.float32))

    assert jnp.isfinite(result).sum().item() == 2
    assert result[0].item() == 5.0


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
def test_logit_bias_adjusts_penalized_distribution_and_sampling(temperature: float) -> None:
    policy = GenerationConfig(
        temperature=temperature,
        repetition_penalty=2.0,
        presence_penalty=0.5,
        logit_bias=((0, -100.0), (2, 8.0)),
    ).default_policy(vocabulary_size=3)
    policy = _with_counts(policy, (2, 2), 2, 3)
    logits = jnp.array([5.0, 2.0, -1.0], dtype=jnp.float32)
    expected = jnp.array([-95.0, 2.0, 5.5], dtype=jnp.float32)
    if temperature == 0.0:
        expected = jnp.array([-jnp.inf, -jnp.inf, 1.0], dtype=jnp.float32)
    else:
        expected /= temperature
    keychain = Keychain.init(42, sharding_config=make_test_sharding_config())
    with jax.set_mesh(keychain.sharding_config.mesh), jax.numpy_dtype_promotion("strict"):
        result = jax.jit(SamplingPolicy.process_logits)(policy, logits)
        sampled = jax.jit(lambda row: row(logits, keychain=keychain))(policy)
    _assert_distribution(result, expected)
    assert sampled == jax.random.categorical(keychain.vmapped_keys, expected)


def test_batched_biases_and_zero_nucleus_probability_apply_per_request() -> None:
    policy = SamplingPolicy.init_batch(
        logit_bias=(((0, -100.0),), ((1, 3.0), (2, -2.0)), None),
        vocabulary_size=3,
        top_p=(0.0, 1.0, 1.0),
    )
    logits = jnp.broadcast_to(jnp.array([1.0, 4.0, -2.0], dtype=jnp.float32), (3, 3))
    result = jax.jit(jax.vmap(SamplingPolicy.process_logits))(policy, logits)
    expected = jnp.array([[-jnp.inf, 4.0, -jnp.inf], [1.0, 7.0, -4.0], [1.0, 4.0, -2.0]])
    _assert_distribution(result, expected)


def test_bias_preserves_ranking_beside_large_equal_model_logits() -> None:
    policy = SamplingPolicy.init(logit_bias=((1, 1.25),), vocabulary_size=3)
    logits = jnp.full(3, 1e38, dtype=jnp.float32)
    result = jax.jit(SamplingPolicy.process_logits)(policy, logits)
    _assert_distribution(result, jnp.array([0.0, 1.25, 0.0], dtype=jnp.float32))


@pytest.mark.parametrize(
    "bias",
    [((-1, 1.0),), ((3, 1.0),), ((1, 1.0), (1, 2.0)), ((0, -100.1),), ((0, 100.1),), ((0, float("nan")),)],
)
def test_invalid_logit_bias_is_rejected(bias: tuple[tuple[int, float], ...]) -> None:
    with pytest.raises(ValueError, match="logit_bias"):
        SamplingPolicy.init(logit_bias=bias, vocabulary_size=3)


def test_logit_bias_requires_known_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocabulary_size"):
        GenerationConfig(logit_bias=((1, 1.0),)).default_policy()


def test_token_counts_ignore_out_of_vocab_tokens_and_update_generated_tokens() -> None:
    policy = _with_counts(SamplingPolicy.init(repetition_penalty=2.0), (1, -1, 9, 2), 4, 4)
    updated_policy = policy.with_next_token_count(jnp.array(2, dtype=jnp.int32))
    updated_policy = updated_policy.with_next_token_count(jnp.array(99, dtype=jnp.int32))
    updated_policy = updated_policy.with_next_token_count(jnp.array(-1, dtype=jnp.int32))

    _assert_distribution(
        policy.process_logits(jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)),
        jnp.array([1.0, 1.0, 2.0, 8.0], dtype=jnp.float32),
    )
    _assert_distribution(
        updated_policy.process_logits(jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)),
        jnp.array([1.0, 1.0, 2.0, 8.0], dtype=jnp.float32),
    )


@pytest.mark.parametrize(
    ("presence", "frequency", "expected"),
    [
        (1.0, 0.0, [3.0, 3.0, 2.0]),
        (-1.0, 0.0, [5.0, 3.0, 2.0]),
        (0.0, 0.5, [3.0, 3.0, 2.0]),
        (0.0, -0.5, [5.0, 3.0, 2.0]),
    ],
)
def test_additive_penalties_count_generated_tokens_only(
    presence: float, frequency: float, expected: list[float]
) -> None:
    logits = jnp.array([4.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(presence_penalty=presence, frequency_penalty=frequency).with_prompt_token_counts(
        jnp.array([0, 0, 2], dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32), vocabulary_size=3
    )
    _assert_distribution(policy.process_logits(logits), logits)
    for token in (0, 0):
        policy = policy.with_next_token_count(jnp.asarray(token, dtype=jnp.int32))
    _assert_distribution(policy.process_logits(logits), jnp.asarray(expected, dtype=jnp.float32))


def test_repetition_window_does_not_evict_additive_counts() -> None:
    logits = jnp.array([4.0, 3.0, 2.0], dtype=jnp.float32)
    policy = SamplingPolicy.init(
        repetition_penalty=2.0, presence_penalty=1.0, frequency_penalty=0.5, suffix_repetition_length=2
    ).with_prompt_token_counts(
        jnp.array([0, 0, 2], dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32), vocabulary_size=3
    )
    _assert_distribution(policy.process_logits(logits), jnp.array([2.0, 3.0, 1.0], dtype=jnp.float32))
    for token, expected in ((0, [0.5, 3.0, 1.0]), (1, [0.5, 0.0, 2.0]), (1, [2.5, -0.5, 2.0])):
        policy = policy.with_next_token_count(jnp.asarray(token, dtype=jnp.int32))
        _assert_distribution(policy.process_logits(logits), jnp.asarray(expected, dtype=jnp.float32))


def test_mixed_batched_penalties_keep_independent_prompt_and_generated_scopes() -> None:
    policy = SamplingPolicy.init_batch(
        repetition_penalty=(2.0, None, None), presence_penalty=(None, 1.0, -1.0), frequency_penalty=(None, 0.5, -0.5)
    )
    policy = jax.jit(jax.vmap(SamplingPolicy.with_prompt_token_counts, in_axes=(0, 0, 0, None)), static_argnums=3)(
        policy, jnp.array([[0, 0, 2], [0, 2, 1], [2, 0, 0]], dtype=jnp.int32), jnp.array([3, 1, 0], dtype=jnp.int32), 3
    )
    logits = jnp.broadcast_to(jnp.array([4.0, 3.0, 2.0], dtype=jnp.float32), (3, 3))
    processed = jax.jit(jax.vmap(SamplingPolicy.process_logits))(policy, logits)
    _assert_distribution(processed, jnp.array([[2.0, 3.0, 1.0], [4.0, 3.0, 2.0], [4.0, 3.0, 2.0]]))
    policy = jax.jit(jax.vmap(SamplingPolicy.with_next_token_count))(
        policy, jnp.array([1, 0, 2], dtype=jnp.int32), jnp.array([True, False, True])
    )
    policy = jax.jit(jax.vmap(SamplingPolicy.with_next_token_count))(policy, jnp.array([2, 0, 2], dtype=jnp.int32))
    processed = jax.jit(jax.vmap(SamplingPolicy.process_logits))(policy, logits)
    _assert_distribution(processed, jnp.array([[2.0, 1.5, 1.0], [2.5, 3.0, 2.0], [4.0, 3.0, 4.0]]))


def test_batched_policy_requires_vmap_and_processes_rows() -> None:
    policy = SamplingPolicy.init_batch(
        temperature=(0.0, 1.0),
        top_k=(0, 1),
        top_p=(1.0, 1.0),
        min_p=(0.0, 0.0),
        banned_tokens=((), ()),
    )
    logits = jnp.array([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]], dtype=jnp.float32)

    with pytest.raises(ValueError, match="Use vmap"):
        policy.process_logits(logits[0])

    result = jax.vmap(lambda policy_row, logits_row: policy_row.process_logits(logits_row))(policy, logits)

    _assert_distribution(result, jnp.array([[-jnp.inf, 1.0, -jnp.inf], [-jnp.inf, 3.0, -jnp.inf]], dtype=jnp.float32))


def test_call_samples_greedy_token_when_temperature_is_zero() -> None:
    result = SamplingPolicy.init(temperature=0.0)(
        jnp.array([0.0, 3.0, 2.0], dtype=jnp.float32),
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )

    assert result.shape == ()
    assert result.item() == 1


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize(("banned_tokens", "expected"), [(None, 64), ((64,), 63)])
def test_grammar_mask_precedes_bias_and_sampling_filters(
    temperature: float, banned_tokens: tuple[int, ...] | None, expected: int
) -> None:
    policy = replace(
        SamplingPolicy.init(
            temperature=temperature,
            top_k=1,
            top_p=0.01,
            min_p=1.0,
            banned_tokens=banned_tokens,
            logit_bias=((69, 100.0),),
            vocabulary_size=70,
        ),
        allowed_token_bitmask=jnp.asarray([-2147483648, -2147483647, 1], dtype=jnp.int32),
    )
    logits = jnp.arange(70, dtype=jnp.float32)
    keychain = Keychain.init(0, sharding_config=make_test_sharding_config())
    with jax.set_mesh(keychain.sharding_config.mesh):
        selected = jax.jit(lambda row: policy(row, keychain=keychain))(logits)
    assert int(selected) == expected


@pytest.mark.parametrize("top_k", [-1, 0, 2**31, 10**300])
def test_unrestricted_top_k_preserves_logits_and_sampling(top_k: int) -> None:
    logits = jnp.array([1.0, 4.0, -2.0], dtype=jnp.float32)
    keychain = Keychain.init(42, sharding_config=make_test_sharding_config())
    expected = SamplingPolicy.init()(logits, keychain=keychain)
    scalar = SamplingPolicy.init(top_k=top_k)
    batch = SamplingPolicy.init_batch(top_k=(top_k, None), temperature=(1.0, 1.0))
    assert jnp.array_equal(scalar.process_logits(logits), logits)
    assert scalar(logits, keychain=keychain) == expected
    processed = jax.vmap(lambda row: row.process_logits(logits), axis_size=2)(batch)
    sampled = jax.vmap(lambda row: row(logits, keychain=keychain), axis_size=2)(batch)
    _assert_distribution(processed, jnp.broadcast_to(logits, (2, 3)))
    assert jnp.array_equal(sampled, jnp.full(2, expected))


def test_sampling_construction_in_compiled_generation() -> None:
    compiled = jax.jit(lambda logits: SamplingPolicy.init(temperature=0.5, top_k=10**300).process_logits(logits))
    logits = jnp.array([1.0, 4.0, -2.0], dtype=jnp.float32)

    _assert_distribution(compiled(logits), logits * 2)


def test_disabled_probability_filters_preserve_sampling_for_finite_extremes() -> None:
    logits = jnp.array([1.0, 4.0, -2.0], dtype=jnp.float32)
    keychain = Keychain.init(42, sharding_config=make_test_sharding_config())
    expected = SamplingPolicy.init()(logits, keychain=keychain)
    scalar = SamplingPolicy.init(top_p=1e300, min_p=-1e300)
    batch = SamplingPolicy.init_batch(top_p=(None, 1e300), min_p=(None, -1e300))

    assert jnp.array_equal(scalar.process_logits(logits), logits)
    assert scalar(logits, keychain=keychain) == expected
    assert jnp.array_equal(
        jax.vmap(lambda row: row(logits, keychain=keychain), axis_size=2)(batch), jnp.full(2, expected)
    )


@pytest.mark.parametrize("value", [0.0, -1.0, 1e300, 1e-300, 1e-45, 1e-38, float("inf"), float("nan")])
@pytest.mark.parametrize(
    "construct",
    [
        lambda value: SamplingPolicy.init(repetition_penalty=value),
        lambda value: SamplingPolicy.init_batch(repetition_penalty=(None, value)),
    ],
    ids=["scalar", "batch"],
)
def test_repetition_penalty_rejects_nonpositive_or_unrepresentable_values(
    value: float, construct: Callable[[float], SamplingPolicy]
) -> None:
    with pytest.raises(ValueError, match="repetition_penalty"):
        construct(value)


@pytest.mark.parametrize(
    "construct",
    [
        lambda: SamplingPolicy.init(temperature=1e-300),
        lambda: SamplingPolicy.init_batch(temperature=(1.0, 1e-300)),
        lambda: SamplingPolicy.init(top_p=1e-300),
        lambda: SamplingPolicy.init_batch(top_p=(1.0, 1e-300)),
        lambda: SamplingPolicy.init(min_p=1e-300),
        lambda: SamplingPolicy.init_batch(min_p=(0.0, 1e-300)),
        lambda: SamplingPolicy.init(presence_penalty=1e300),
        lambda: SamplingPolicy.init_batch(presence_penalty=(0.0, 1e300)),
        lambda: SamplingPolicy.init(frequency_penalty=float("nan")),
        lambda: SamplingPolicy.init_batch(frequency_penalty=(0.0, float("nan"))),
    ],
)
def test_sampling_rejects_controls_that_cannot_be_interpreted_in_float32(
    construct: Callable[[], SamplingPolicy],
) -> None:
    with pytest.raises(ValueError, match="float32"):
        construct()


@pytest.mark.usefixtures("fake_mesh")
@pytest.mark.parametrize(
    ("policy", "tokens", "expected"),
    [
        pytest.param(SamplingPolicy.init(temperature=2e-38), (), [-1e38, 0.0, -1e38, -1e38, -1e38], id="temperature"),
        pytest.param(
            SamplingPolicy.init(repetition_penalty=2e-38),
            (0, 1),
            [-1e38, 0.0, -1e38, -1e38, -1e38],
            id="repetition",
        ),
        pytest.param(
            SamplingPolicy.init(frequency_penalty=-1e38),
            (0,) * 4 + (1,) * 5,
            [-1e38, 0.0, -1e38, -1e38, -1e38],
            id="frequency-counts",
        ),
        pytest.param(
            SamplingPolicy.init(frequency_penalty=-1e38),
            (0,) * 5 + (1,) * 5,
            [-1.0, 0.0, -1e38, -1e38, -1e38],
            id="frequency-ties",
        ),
        pytest.param(
            SamplingPolicy.init(presence_penalty=-1e38),
            (0, 1),
            [-1.0, 0.0, -1e38, -1e38, -1e38],
            id="presence-ties",
        ),
        pytest.param(
            SamplingPolicy.init(presence_penalty=-1e38),
            (2, 3),
            [-1e38, -1e38, -1.0, 0.0, -1e38],
            id="presence-negative-logits",
        ),
        pytest.param(
            SamplingPolicy.init(frequency_penalty=-1e38),
            (2,) * 5 + (3,) * 5,
            [-1e38, -1e38, -1.0, 0.0, -1e38],
            id="frequency-negative-logits",
        ),
        pytest.param(
            SamplingPolicy.init(frequency_penalty=1e38),
            (0, 1, 2, 3, 4) * 5,
            [-1.0, 0.0, -200.0, -199.0, -100.0],
            id="frequency-all-seen",
        ),
        pytest.param(
            SamplingPolicy.init(repetition_penalty=2e-38, temperature=2e38),
            (0, 1),
            [-0.25, 0.0, -25.0, -25.0, -25.0],
            id="repetition-temperature",
        ),
        pytest.param(
            SamplingPolicy.init(repetition_penalty=2e-38, banned_tokens=(1,)),
            (0, 1),
            [0.0, -jnp.inf, -1e38, -1e38, -1e38],
            id="repetition-banned",
        ),
        pytest.param(
            SamplingPolicy.init(temperature=0.0, frequency_penalty=-1e38),
            (0,) * 4 + (1,) * 5,
            [-jnp.inf, 0.0, -jnp.inf, -jnp.inf, -jnp.inf],
            id="greedy-frequency",
        ),
        pytest.param(
            SamplingPolicy.init(temperature=0.0, repetition_penalty=2e-38),
            (0, 1),
            [-jnp.inf, 0.0, -jnp.inf, -jnp.inf, -jnp.inf],
            id="greedy-repetition",
        ),
    ],
)
def test_extreme_controls_preserve_ranking_and_sampling(
    policy: SamplingPolicy, tokens: tuple[int, ...], expected: list[float]
) -> None:
    logits = jnp.array([99.0, 100.0, -100.0, -99.0, 0.0], dtype=jnp.float32)
    policy = _with_counts(policy, tokens, len(tokens), vocabulary_size=5)
    expected_logits = jnp.array(expected, dtype=jnp.float32)
    batch_logits = jnp.stack((logits, logits * 2))
    expected_batch_logits = jnp.stack((expected_logits, expected_logits * 2))
    keychain = Keychain.init(42, sharding_config=make_test_sharding_config())
    expected_token = jax.random.categorical(keychain.vmapped_keys, expected_logits)
    expected_batch_tokens = jax.vmap(lambda row: jax.random.categorical(keychain.vmapped_keys, row))(
        expected_batch_logits
    )
    x64_enabled = jax.config.jax_enable_x64
    with jax.numpy_dtype_promotion("strict"):
        processed = jax.jit(SamplingPolicy.process_logits)(policy, logits)
        processed_batch = jax.jit(jax.vmap(SamplingPolicy.process_logits))(policy.broadcast(2), batch_logits)
        sampled = jax.jit(lambda row: row(logits, keychain=keychain))(policy)
        sampled_batch = jax.jit(jax.vmap(lambda row, logits: row(logits, keychain=keychain)))(
            policy.broadcast(2), batch_logits
        )

    assert processed.dtype == jnp.float32
    assert jax.config.jax_enable_x64 == x64_enabled
    assert jnp.all(jnp.isfinite(processed) | jnp.isneginf(processed))
    assert jnp.all(jnp.isfinite(processed[~jnp.isneginf(expected_logits)]))
    assert jnp.argmax(processed) == jnp.argmax(expected_logits)
    assert jnp.allclose(jax.nn.softmax(processed), jax.nn.softmax(expected_logits), rtol=1e-6, atol=1e-7)
    assert jnp.allclose(jax.nn.softmax(processed_batch), jax.nn.softmax(expected_batch_logits), rtol=1e-6, atol=1e-7)
    assert sampled == expected_token
    assert jnp.array_equal(sampled_batch, expected_batch_tokens)
    if policy.banned_tokens is not None:
        assert jnp.isneginf(processed[1])
