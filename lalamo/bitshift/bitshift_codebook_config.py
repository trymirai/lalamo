from dataclasses import dataclass

__all__ = [
    "BitShiftCodebookConfig",
]


@dataclass
class BitShiftCodebookConfig:
    state_bits: int
    bits_per_weight: int
    values_per_step: int

    @property
    def number_of_states(self) -> int:
        return 2**self.state_bits

    @property
    def bits_per_step(self) -> int:
        return self.bits_per_weight * self.values_per_step

    @property
    def number_of_deltas(self) -> int:
        return 2**self.bits_per_step
