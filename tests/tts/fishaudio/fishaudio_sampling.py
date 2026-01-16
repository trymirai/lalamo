from dataclasses import dataclass
from typing import Optional

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from numpy import isin

from lalamo.sampling import CompositePolicy, GreedyPolicy, SamplingPolicy, TemperaturePolicy, TopPPolicy


@dataclass(frozen=True)
class FishAudioSamplingParams:
    argmax_decoding: bool
    top_p: float
    temperature: float
    repetition_penalty: float


def logits_to_probs(
    logits: Float[Array, " vocabulary"],
    top_p: float,
    temperature: float,
    previous_tokens: Optional[Int[Array, " tokens"]] = None,
) -> Float[Array, " vocabulary"]:
    # NOTE: repetition_penalty is not implemented yet - stub it for API compatibility
    policies = []
    if top_p > 0 and top_p < 1.0:
        policies.append(TopPPolicy(p=top_p))
    if temperature > 0:
        policies.append(TemperaturePolicy(temperature=max(temperature, 1e-5)))

    if policies:
        policy = CompositePolicy(tuple(policies))
        processed_logits = policy.process_logits(logits)
    else:
        processed_logits = logits

    probs = jax.nn.softmax(processed_logits)
    return probs


def sample(
    logits: Float[Array, "batch tokens vocabulary"],
    key: PRNGKeyArray,
    sampling_params: FishAudioSamplingParams,
    previous_tokens: Optional[Int[Array, " tokens"]] = None,
) -> tuple[Int[Array, ""], Float[Array, " vocabulary"]]:
    # Take the last token's logits from first batch
    last_logits = logits[0, -1]

    probs = logits_to_probs(
        logits=last_logits,
        top_p=sampling_params.top_p,
        temperature=sampling_params.temperature,
        previous_tokens=previous_tokens,
    )

    idx_next = jax.random.categorical(key, jnp.log(probs + 1e-10))
    return idx_next, probs


def sampling_params_from_policy(
    policy: SamplingPolicy | None,
    repetition_penalty: float = 1.0,
) -> FishAudioSamplingParams:
    """
    Convert a SamplingPolicy to FishAudioSamplingParams.

    Extracts temperature and top_p values from the policy. If the policy is a
    GreedyPolicy, argmax_decoding is set to True.

    Args:
        policy: A SamplingPolicy instance (can be CompositePolicy, GreedyPolicy,
                TemperaturePolicy, TopPPolicy, or others).
        repetition_penalty: Repetition penalty value (not derived from policy
                           as SamplingPolicy doesn't support it).

    Returns:
        FishAudioSamplingParams with extracted values.
    """
    if policy is None:
        return FishAudioSamplingParams(
            argmax_decoding=True,
            top_p=0.0,
            temperature=0.0,
            repetition_penalty=0.0,
        )
    temperature = 1.0
    top_p = 1.0
    argmax_decoding = False

    policies_to_check: list[SamplingPolicy] = []

    if isinstance(policy, CompositePolicy):
        policies_to_check.extend(policy.policies)
    else:
        policies_to_check.append(policy)

    for p in policies_to_check:
        if isinstance(p, GreedyPolicy):
            argmax_decoding = True
        elif isinstance(p, TemperaturePolicy):
            temperature = p.temperature
        elif isinstance(p, TopPPolicy):
            top_p = p.p

    return FishAudioSamplingParams(
        argmax_decoding=argmax_decoding,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )
