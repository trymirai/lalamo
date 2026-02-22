from dataclasses import dataclass

__all__ = [
    "BitShiftCodebookConfig",
]


@dataclass
class BitShiftCodebookConfig:
    state_bits: int
    bits_per_weight: int
    chunk_size: int

    def __post_init__(self) -> None:
        if self.bits_per_step > self.state_bits:
            raise ValueError(
                f"bits_per_step ({self.bits_per_step}) must be <= state_bits ({self.state_bits})",
            )

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
