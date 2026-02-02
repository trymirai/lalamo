from dataclasses import dataclass

__all__ = [
    "BitShiftCodebookConfig",
]


@dataclass
class BitShiftCodebookConfig:
    state_bits: int
    bits_per_weight: int
    chunk_size: int

    @property
    def number_of_states(self) -> int:
        return 2**self.state_bits

    @property
    def bits_per_step(self) -> int:
        return self.bits_per_weight * self.chunk_size

    @property
    def transitions_per_state(self) -> int:
        return 2**self.bits_per_step

    @property
    def number_of_reduced_states(self) -> int:
        return self.number_of_states // self.transitions_per_state
