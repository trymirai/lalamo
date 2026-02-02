import equinox as eqx
from jaxtyping import Array, Float, Int

from .bitshift_codebook import BitShiftCodebook
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
