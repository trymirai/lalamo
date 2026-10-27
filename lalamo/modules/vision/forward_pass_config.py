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
    # Portable implementation. Runs everywhere and is the reference for parity tests.
    XLA = "xla"
    # cuDNN flash attention. GPU only, requires float16/bfloat16 inputs, never materialises the
    # attention matrix, so it is the right choice for large images on NVIDIA hardware.
    CUDNN = "cudnn"


@dataclass(frozen=True)
class VisionForwardPassConfig:
    matmul_config: MatmulConfig = field(default_factory=MatmulConfig)
    attention_implementation: VisionAttentionImplementation = VisionAttentionImplementation.XLA
    # The XLA implementation materialises a (heads, queries, keys) logit tensor. A 16-megapixel
    # image is 65k patches, and the full tensor (188 GiB) does not fit even on a B200. Queries
    # are therefore processed in chunks, which caps the peak at heads * chunk * keys.
    # Measured on a 65k-patch frame against a float64 reference: accuracy degrades roughly
    # linearly with chunk size (fp32 relative error 5e-6 at 512, 1.2e-4 at 4096), while run time
    # barely moves (within ~12%). 512 keeps accuracy high and peak memory near 1.6 GB.
    attention_query_chunk_size: int | None = 512

    def __post_init__(self) -> None:
        if self.attention_query_chunk_size is not None and self.attention_query_chunk_size < 1:
            raise ValueError("attention_query_chunk_size must be positive.")

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(matmul_config=MatmulConfig.for_tracer_tests())

    @classmethod
    def for_inference(
        cls,
        attention_implementation: VisionAttentionImplementation = VisionAttentionImplementation.XLA,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            matmul_config=MatmulConfig.for_inference(precision),
            attention_implementation=attention_implementation,
        )
