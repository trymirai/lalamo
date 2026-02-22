import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .bitshift_codebook import BitShiftCodebook
from .bitshift_codebook_config import BitShiftCodebookConfig
from .viterbi import viterbi

__all__ = [
    "Quantizer",
]


class Quantizer(eqx.Module):
    codebook: BitShiftCodebook

    def quantize(
        self,
        array: Float[Array, "rows columns"],
        block_size: int,
    ) -> tuple[Float[Array, "rows columns"], Int[Array, "block_rows block_columns number_of_steps"]]:
        rows, columns = array.shape
        assert rows % block_size == 0
        assert columns % block_size == 0

        block_rows = rows // block_size
        block_columns = columns // block_size
        elements_per_block = block_size * block_size
        assert elements_per_block % self.codebook.config.chunk_size == 0
        number_of_steps = elements_per_block // self.codebook.config.chunk_size
        number_of_blocks = block_rows * block_columns

        blocks = array.reshape(block_rows, block_size, block_columns, block_size)
        blocks = blocks.transpose(0, 2, 1, 3)  # (block_rows, block_columns, block_size, block_size)
        blocks = blocks.reshape(block_rows, block_columns, elements_per_block)
        blocks = blocks.reshape(number_of_blocks, elements_per_block).T

        states = viterbi(blocks, self.codebook)
        array_reconstructed = self.codebook.reconstruct(states)
        array_reconstructed = array_reconstructed.transpose(1, 0, 2).reshape(elements_per_block, number_of_blocks)
        array_reconstructed = array_reconstructed.T.reshape(block_rows, block_columns, block_size, block_size)
        array_reconstructed = array_reconstructed.transpose(0, 2, 1, 3).reshape(rows, columns)
        states = states.T.reshape(block_rows, block_columns, number_of_steps)

        return array_reconstructed, states

    @classmethod
    def pack_states(
        cls,
        states: Int[Array, "block_rows block_columns number_of_steps"],
        config: BitShiftCodebookConfig,
    ) -> Int[Array, "number_of_blocks packed_size"]:
        block_rows, block_columns, number_of_steps = states.shape
        number_of_blocks = block_rows * block_columns
        states_flat = states.reshape(number_of_blocks, number_of_steps).T

        bits_per_step = config.bits_per_step
        state_bits = config.state_bits

        total_bits = state_bits + (number_of_steps - 1) * bits_per_step
        packed_size = (total_bits + 7) // 8

        first_state_bits = jnp.stack(
            [(states_flat[0] >> (state_bits - 1 - index)) & 1 for index in range(state_bits)],
            axis=0,
        )

        delta_bits_list = []
        for index in range(1, number_of_steps):
            for bit_index in range(bits_per_step):
                bit = (states_flat[index] >> (bits_per_step - 1 - bit_index)) & 1
                delta_bits_list.append(bit)

        if delta_bits_list:
            delta_bits = jnp.stack(delta_bits_list, axis=0)
            all_bits = jnp.concatenate([first_state_bits, delta_bits], axis=0)
        else:
            all_bits = first_state_bits

        pad_amount = packed_size * 8 - total_bits
        if pad_amount > 0:
            padding = jnp.zeros((pad_amount, number_of_blocks), dtype=all_bits.dtype)
            all_bits = jnp.concatenate([all_bits, padding], axis=0)

        all_bits = all_bits.reshape(packed_size, 8, number_of_blocks)
        powers = 2 ** jnp.arange(7, -1, -1, dtype=jnp.int32)
        packed = jnp.sum(all_bits * powers[:, None], axis=1).astype(jnp.int8)

        return packed.T

    @classmethod
    def unpack_states(
        cls,
        packed: Int[Array, "number_of_blocks packed_size"],
        number_of_steps: int,
        block_rows: int,
        block_columns: int,
        config: BitShiftCodebookConfig,
    ) -> Int[Array, "block_rows block_columns number_of_steps"]:
        number_of_blocks = packed.shape[0]
        packed_flat = packed.T

        bits_per_step = config.bits_per_step
        state_bits = config.state_bits
        state_mask = (1 << state_bits) - 1

        powers = 2 ** jnp.arange(7, -1, -1, dtype=jnp.int32)
        packed_int = packed_flat.astype(jnp.int32) & 0xFF
        bits = ((packed_int[:, :, None] // powers[None, None, :]) % 2).transpose(0, 2, 1).reshape(-1, number_of_blocks)

        first_state = jnp.sum(
            bits[:state_bits] * (2 ** jnp.arange(state_bits - 1, -1, -1))[:, None],
            axis=0,
        ).astype(jnp.int32)

        states_list = [first_state]
        for step in range(1, number_of_steps):
            bit_start = state_bits + (step - 1) * bits_per_step
            delta = jnp.sum(
                bits[bit_start : bit_start + bits_per_step] * (2 ** jnp.arange(bits_per_step - 1, -1, -1))[:, None],
                axis=0,
            ).astype(jnp.int32)
            previous_state = states_list[-1]
            new_state = ((previous_state << bits_per_step) & state_mask) + delta
            states_list.append(new_state)

        states_flat = jnp.stack(states_list, axis=0)
        return states_flat.T.reshape(block_rows, block_columns, number_of_steps)
