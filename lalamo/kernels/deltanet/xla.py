import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def xla_recurrent_scan(
    queries: Float[Array, "tokens heads key_channels"],
    keys: Float[Array, "tokens heads key_channels"],
    values: Float[Array, "tokens heads value_channels"],
    decay_factor: Float[Array, "tokens heads"],
    beta: Float[Array, "tokens heads"],
    initial_state: Float[Array, "heads value_channels key_channels"],
    num_steps: Int[Array, ""] | int,
) -> tuple[
    Float[Array, "tokens heads value_channels"],
    Float[Array, "heads value_channels key_channels"],
]:
    def scan_fn(
        index_and_state: tuple[Int[Array, ""], Float[Array, "heads value_channels key_channels"]],
        step_inputs: tuple[
            Float[Array, "heads key_channels"],
            Float[Array, "heads key_channels"],
            Float[Array, "heads value_channels"],
            Float[Array, " heads"],
            Float[Array, " heads"],
        ],
    ) -> tuple[
        tuple[Int[Array, ""], Float[Array, "heads value_channels key_channels"]],
        Float[Array, "heads value_channels"],
    ]:
        index, carry_state = index_and_state
        query_t, key_t, value_t, decay_factor_t, beta_t = step_inputs

        decay = jnp.exp(decay_factor_t)[:, None, None]
        decayed_state = carry_state * decay
        value_delta = value_t - jnp.sum(decayed_state * key_t[:, None, :], axis=-1)
        value_delta = value_delta * beta_t[:, None]
        updated_state = decayed_state + value_delta[:, :, None] * key_t[:, None, :]
        output_t = einops.einsum(
            query_t,
            updated_state,
            "heads key_channels, heads value_channels key_channels -> heads value_channels",
        )

        propagated_state = jax.lax.cond(index < num_steps, lambda: updated_state, lambda: carry_state)
        return (index + 1, propagated_state), output_t

    (_, final_state), outputs = jax.lax.scan(
        scan_fn,
        (jnp.zeros((), dtype=jnp.int32), initial_state),
        (queries, keys, values, decay_factor, beta),
    )
    return outputs, final_state
