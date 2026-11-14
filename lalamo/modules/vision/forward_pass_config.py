from dataclasses import dataclass, field
from typing import Literal, Self

from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "VisionForwardPassConfig",
]


@dataclass(frozen=True)
class VisionForwardPassConfig:
    matmul_config: MatmulConfig = field(default_factory=MatmulConfig)
    # cuDNN is used where supported; other inputs use XLA.
    attention_implementation: Literal["cudnn", "xla"] = "cudnn"
    # XLA attention processes this many queries at a time, which bounds the size of the logits tensor.
    attention_query_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.attention_query_chunk_size < 1:
            raise ValueError("attention_query_chunk_size must be positive.")

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(matmul_config=MatmulConfig.for_tracer_tests())
