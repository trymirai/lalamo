from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from jax.lax import DotAlgorithmPreset

from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "VisionAttentionImplementation",
    "VisionForwardPassConfig",
]


class VisionAttentionImplementation(StrEnum):
    # Used whenever cuDNN supports the inputs (GPU, float16/bfloat16); XLA otherwise.
    CUDNN = "cudnn"
    XLA = "xla"


@dataclass(frozen=True)
class VisionForwardPassConfig:
    matmul_config: MatmulConfig = field(default_factory=MatmulConfig)
    attention_implementation: VisionAttentionImplementation = VisionAttentionImplementation.CUDNN
    # XLA attention processes this many queries at a time, which bounds the size of the logits tensor.
    attention_query_chunk_size: int = 512

    def __post_init__(self) -> None:
        if self.attention_query_chunk_size < 1:
            raise ValueError("attention_query_chunk_size must be positive.")

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(matmul_config=MatmulConfig.for_tracer_tests())

    @classmethod
    def for_inference(cls, precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT) -> Self:
        return cls(matmul_config=MatmulConfig.for_inference(precision))
