from dataclasses import dataclass
from typing import Optional

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from lalamo.sampling import CompositePolicy, TemperaturePolicy, TopPPolicy


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
    # NOTE: repetition_penalty is not implemented yet - stub for API compatibility
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
