from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.bitshift import (
    BitShiftCodebook,
    BitShiftCodebookConfig,
    FixedLUTProvider,
    Quantizer,
)
from tests.helpers import read_test_json


@dataclass
class ViterbiTestCase:
    array: list[list[float]]
    array_rows: int
    array_columns: int
    state_bits: int
    bits_per_weight: int
    chunk_size: int
    block_size: int
    lut: list[list[float]]
    states: list[list[int]]

    @classmethod
    def from_dict(cls, data: dict) -> "ViterbiTestCase":
        return cls(
            array=data["array"],
            array_rows=data["array_rows"],
            array_columns=data["array_columns"],
            state_bits=data["state_bits"],
            bits_per_weight=data["bits_per_weight"],
            chunk_size=data["chunk_size"],
            block_size=data["block_size"],
            lut=data["lut"],
            states=data["states"],
        )

    @property
    def name(self) -> str:
        return f"{self.array_rows}x{self.array_columns}"

    @property
    def array_jnp(self) -> Float[Array, "array_rows array_columns"]:
        return jnp.array(self.array).reshape(self.array_rows, self.array_columns)

    @property
    def lut_jnp(self) -> Float[Array, "chunk_size number_of_states"]:
        number_of_states = 2**self.state_bits
        return jnp.array(self.lut).reshape(self.chunk_size, number_of_states)

    @property
    def states_jnp(self) -> Int[Array, "block_rows block_columns number_of_steps"]:
        block_rows = self.array_rows // self.block_size
        block_columns = self.array_columns // self.block_size
        elements_per_block = self.block_size * self.block_size
        number_of_steps = elements_per_block // self.chunk_size
        return jnp.array(self.states).T.reshape(block_rows, block_columns, number_of_steps)


def test_viterbi() -> None:
    for test_case in [ViterbiTestCase.from_dict(d) for d in read_test_json("viterbi")]:
        config = BitShiftCodebookConfig(
            state_bits=test_case.state_bits,
            bits_per_weight=test_case.bits_per_weight,
            chunk_size=test_case.chunk_size,
        )
        lut_provider = FixedLUTProvider.create_from_lut(test_case.lut_jnp)
        codebook = BitShiftCodebook.create(config=config, lut_provider=lut_provider)
        quantizer = Quantizer(codebook=codebook)

        array, expected_states = test_case.array_jnp, test_case.states_jnp
        array_reconstructed, actual_states = quantizer.quantize(array, test_case.block_size)
        reconstruction_error = jnp.mean(jnp.square(array - array_reconstructed))
        assert reconstruction_error < 0.01, f"Reconstruction error too high for {test_case.name}"
        assert actual_states.shape == expected_states.shape
        assert jnp.array_equal(actual_states, expected_states), f"States mismatch for {test_case.name}"

        block_rows, block_columns, number_of_steps = actual_states.shape
        packed_states = Quantizer.pack_states(actual_states, config)
        expected_packed_size = (config.state_bits + (number_of_steps - 1) * config.bits_per_step + 7) // 8
        assert packed_states.shape == (block_rows * block_columns, expected_packed_size), (
            f"Packed shape mismatch for {test_case.name}"
        )
        unpacked_states = Quantizer.unpack_states(packed_states, number_of_steps, block_rows, block_columns, config)
        assert jnp.array_equal(actual_states, unpacked_states), f"Pack/unpack mismatch for {test_case.name}"
