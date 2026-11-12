import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int

from lalamo.module import Keychain
from lalamo.modules import Decoder, DecoderForwardPassConfig

__all__ = [
    "Rollouts",
    "realign_response_values",
    "sampled_token_log_probs",
]


class Rollouts(eqx.Module):
    """Prompts followed by the responses a policy sampled for them, right-padded to a common length.

    Every prompt must contain at least one token, so that each response token has a prediction.
    """

    token_ids: Int[Array, "batch tokens"]
    prompt_lengths: Int[Array, " batch"]
    response_lengths: Int[Array, " batch"]

    @property
    def lengths_without_padding(self) -> Int[Array, " batch"]:
        return self.prompt_lengths + self.response_lengths

    @property
    def response_mask(self) -> Bool[Array, "batch predictions"]:
        """Marks the predictions of response tokens; the prediction at `index` is for the token at `index + 1`."""
        _, num_tokens = self.token_ids.shape
        predicted_token_indices = jnp.arange(1, num_tokens, dtype=jnp.int32)
        is_after_prompt = predicted_token_indices[None, :] >= self.prompt_lengths[:, None]
        is_before_padding = predicted_token_indices[None, :] < self.lengths_without_padding[:, None]
        return is_after_prompt & is_before_padding


@eqx.filter_jit
def sampled_token_log_probs(
    decoder: Decoder,
    rollouts: Rollouts,
    forward_pass_config: DecoderForwardPassConfig,
    *,
    keychain: Keychain,
) -> Float[Array, "batch predictions"]:
    """Log-probability under `decoder` of every sampled response token; zero outside `rollouts.response_mask`."""
    batch_size, num_tokens = rollouts.token_ids.shape
    token_positions = jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), (batch_size, num_tokens))
    logits = decoder(
        token_ids=rollouts.token_ids,
        token_positions=token_positions,
        lengths_without_padding=rollouts.lengths_without_padding,
        forward_pass_config=forward_pass_config,
        keychain=keychain,
    ).logits
    prediction_logits = logits[:, :-1, :].astype(jnp.float32)
    next_token_ids = rearrange(rollouts.token_ids[:, 1:], "batch predictions -> batch predictions 1")
    next_token_logits = rearrange(
        jnp.take_along_axis(prediction_logits, next_token_ids, axis=-1),
        "batch predictions 1 -> batch predictions",
    )
    log_probs = next_token_logits - jax.nn.logsumexp(prediction_logits, axis=-1)
    return jnp.where(rollouts.response_mask, log_probs, 0.0)


def realign_response_values(
    values: Float[Array, "batch source_predictions"],
    source_prompt_lengths: Int[Array, " batch"],
    target: Rollouts,
) -> Float[Array, "batch target_predictions"]:
    """Moves per-token values of responses to where the same responses sit in `target`.

    The source rollouts hold the same responses behind prompts of `source_prompt_lengths` tokens, which is how
    two models that render a conversation differently see one response.
    """
    _, num_source_predictions = values.shape
    _, num_target_tokens = target.token_ids.shape
    target_prediction_indices = jnp.arange(num_target_tokens - 1, dtype=jnp.int32)
    prompt_length_differences = source_prompt_lengths - target.prompt_lengths
    source_prediction_indices = jnp.clip(
        target_prediction_indices[None, :] + prompt_length_differences[:, None],
        0,
        num_source_predictions - 1,
    )
    realigned = jnp.take_along_axis(values, source_prediction_indices, axis=1)
    return jnp.where(target.response_mask, realigned, 0.0)
