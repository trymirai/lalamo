from dataclasses import dataclass

__all__ = [
    "BitshiftCodebookConfig",
]


@dataclass
class BitshiftCodebookConfig:
    state_bits: int
    bits_per_weight: int
    values_per_step: int

    @property
    def number_of_states(self) -> int:
        return 2**self.state_bits
