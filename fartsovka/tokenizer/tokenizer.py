from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence, Type, get_type_hints
from types import UnionType

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int

from fartsovka.common import ParameterDict
from fartsovka.modules.common import FartsovkaModule, config_converter, register_config_union


@dataclass
class TokenizerConfig:
    """Base configuration for tokenizers."""
    vocab_size: int
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    type: ClassVar[str] = "TokenizerConfig"


class Tokenizer(FartsovkaModule[TokenizerConfig], ABC):
    """Base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str | Sequence[str], add_special_tokens: bool = True) -> list[Int[Array, " tokens"]] | list[list[Int[Array, " tokens"]]]:
        """Encode text into token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: Int[Array, " tokens"] | Sequence[Int[Array, " tokens"]]) -> str | list[str]:
        """Decode token IDs into text."""
        pass
    
    def export_weights(self) -> ParameterDict:
        """Export tokenizer weights."""
        # Most tokenizers don't have weights in the JAX model
        return ParameterDict()
    
    @classmethod
    def from_config(cls, config: TokenizerConfig) -> "Tokenizer":
        """Create a tokenizer from a config."""
        from .hf_tokenizer import HFTokenizerConfig, HFTokenizer
        
        if isinstance(config, HFTokenizerConfig):
            return HFTokenizer(config)
        
        raise ValueError(f"Unknown tokenizer config type: {config.__class__.__name__}")