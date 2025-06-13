from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from lalamo.model_import.configs import ForeignConfig
from lalamo.model_import.loaders import load_huggingface
from lalamo.modules import Decoder

__all__ = ["HuggingFaceConfig"]


@dataclass(frozen=True)
class HuggingFaceConfig(ForeignConfig):
    torch_dtype: Literal["bfloat16", "float16", "float32"]

    @property
    def default_precision(self) -> DTypeLike:
        return jnp.dtype(self.torch_dtype)

    @classmethod
    def _load_weights(
        cls,
        model: Decoder,
        weights_dict: dict[str, Array],
    ) -> Decoder:
        return load_huggingface(model, weights_dict)
