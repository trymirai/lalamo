import jax.numpy as jnp
import torch
from bidict import bidict


class DTypeConvert:
    """Bidirectional JAX <-> PyTorch dtype converter."""

    _map = bidict(
        {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        },
    )

    @classmethod
    def to_torch(cls, in_type: str | jnp.dtype) -> torch.dtype:
        match in_type:
            case str():
                return cls._map[in_type]
            case _:
                return cls._map[jnp.dtype(in_type).name]

    @classmethod
    def to_jax(cls, in_type: str | torch.dtype) -> jnp.dtype:
        match in_type:
            case str():
                return jnp.dtype(in_type)
            case torch.dtype():
                return jnp.dtype(cls._map.inverse[in_type])
