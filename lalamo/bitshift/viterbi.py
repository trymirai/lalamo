import jax.lax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .bitshift_codebook import BitShiftCodebook

__all__ = [
    "viterbi",
    "viterbi_backward",
    "viterbi_forward",
]


def compute_reconstruction_error(
    reconstructed_states: Float[Array, "chunk_size number_of_states"],
    target: Float[Array, "chunk_size number_of_blocks"],
) -> Float[Array, "number_of_blocks number_of_states"]:
    # reconstructed_states: (chunk_size, number_of_states) -> (chunk_size, 1, number_of_states)
    # target: (chunk_size, number_of_blocks) -> (chunk_size, number_of_blocks, 1)
    # broadcast: (chunk_size, number_of_blocks, number_of_states)
    #     sum over chunk_size -> (number_of_blocks, number_of_states)
    return jnp.sum(
        jnp.square(reconstructed_states[:, None, :] - target[:, :, None]),
        axis=0,
    )


def forward_step(
    cost: Float[Array, "number_of_blocks number_of_states"],
    target: Float[Array, "chunk_size number_of_blocks"],
    codebook: BitShiftCodebook,
) -> tuple[Float[Array, "number_of_blocks number_of_states"], Int[Array, "number_of_blocks number_of_reduced_states"]]:
    reconstruction_error = compute_reconstruction_error(codebook.reconstruct(codebook.states), target)

    candidate_cost = cost[:, codebook.transitions]
    best_transition_indices = jnp.argmin(candidate_cost, axis=-1)
    best_transition_cost = jnp.min(candidate_cost, axis=-1)

    reduced_state_indices = jnp.arange(codebook.config.number_of_reduced_states)[None, :]
    history_entry = codebook.transitions[reduced_state_indices, best_transition_indices]
    new_cost = reconstruction_error + jnp.repeat(best_transition_cost, codebook.config.transitions_per_state, axis=-1)

    return new_cost, history_entry


def viterbi_forward(
    array: Float[Array, "number_of_steps chunk_size number_of_blocks"],
    codebook: BitShiftCodebook,
) -> tuple[
    Float[Array, "number_of_blocks number_of_states"],
    Int[Array, "path_history_length number_of_blocks number_of_reduced_states"],
]:
    initial_cost = compute_reconstruction_error(codebook.reconstruct(codebook.states), array[0])

    def scan_fn(
        cost: Float[Array, "number_of_blocks number_of_states"],
        target: Float[Array, "chunk_size number_of_blocks"],
    ) -> tuple[
        Float[Array, "number_of_blocks number_of_states"],
        Int[Array, "number_of_blocks number_of_reduced_states"],
    ]:
        new_cost, history_entry = forward_step(cost, target, codebook)
        return new_cost, history_entry

    final_cost, path_history = jax.lax.scan(scan_fn, initial_cost, array[1:])
    return final_cost, path_history


def viterbi_backward(
    cost: Float[Array, "number_of_blocks number_of_states"],
    path_history: Int[Array, "path_history_length number_of_blocks number_of_reduced_states"],
    codebook: BitShiftCodebook,
) -> Int[Array, "number_of_steps number_of_blocks"]:
    number_of_blocks = cost.shape[0]
    final_states = jnp.argmin(cost, axis=-1)

    def scan_fn(
        state: Int[Array, " number_of_blocks"],
        history_entry: Int[Array, "number_of_blocks number_of_reduced_states"],
    ) -> tuple[Int[Array, " number_of_blocks"], Int[Array, " number_of_blocks"]]:
        reduced_state_index = state >> codebook.config.bits_per_step
        previous_state = history_entry[jnp.arange(number_of_blocks), reduced_state_index]
        return previous_state, state

    first_state, optimal_path = jax.lax.scan(scan_fn, final_states, path_history, reverse=True)
    return jnp.concatenate([first_state[None, :], optimal_path], axis=0)


def viterbi(
    array: Float[Array, "elements_per_block number_of_blocks"],
    codebook: BitShiftCodebook,
) -> Int[Array, "number_of_steps number_of_blocks"]:
    elements_per_block, number_of_blocks = array.shape
    assert elements_per_block % codebook.config.chunk_size == 0
    number_of_steps = elements_per_block // codebook.config.chunk_size
    array_chunked = array.reshape(number_of_steps, codebook.config.chunk_size, number_of_blocks)

    cost, path_history = viterbi_forward(array_chunked, codebook)
    return viterbi_backward(cost, path_history, codebook)
