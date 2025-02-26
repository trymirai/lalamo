from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int
from tokenizers import Tokenizer as HFTokenizerImpl

from fartsovka.common import ParameterDict
from .tokenizer import Tokenizer, TokenizerConfig


@dataclass
class HFTokenizerConfig(TokenizerConfig):
    """Configuration for Hugging Face tokenizers."""
    tokenizer_path: str
    type: ClassVar[str] = "HFTokenizer"


class HFTokenizer(Tokenizer):
    """Adapter for Hugging Face tokenizers from the tokenizers library."""
    _tokenizer: HFTokenizerImpl = eqx.field(static=True)
    
    def __init__(self, config: HFTokenizerConfig):
        self.config = config
        self._tokenizer = HFTokenizerImpl.from_file(config.tokenizer_path)

    def encode(self, text: str | Sequence[str], add_special_tokens: bool = True) -> list[Int[Array, " tokens"]] | list[list[Int[Array, " tokens"]]]:
        if isinstance(text, str):
            encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return jnp.array(encoding.ids, dtype=jnp.int32)
        
        # Batch encoding
        encodings = self._tokenizer.encode_batch(list(text), add_special_tokens=add_special_tokens)
        return [jnp.array(enc.ids, dtype=jnp.int32) for enc in encodings]

    def decode(self, token_ids: Int[Array, " tokens"] | Sequence[Int[Array, " tokens"]]) -> str | list[str]:
        if isinstance(token_ids, Array):
            # Single sequence
            return self._tokenizer.decode(token_ids.tolist())
        
        # Batch decoding
        token_ids_lists = [ids.tolist() if isinstance(ids, Array) else ids for ids in token_ids]
        return self._tokenizer.decode_batch(token_ids_lists)

    @classmethod
    def from_file(cls, tokenizer_path: str | Path, **config_kwargs: Any) -> "HFTokenizer":
        """Create a tokenizer from a file."""
        tokenizer = HFTokenizerImpl.from_file(tokenizer_path)
        
        # Extract information from the tokenizer
        vocab_size = tokenizer.get_vocab_size()
        
        # Create the config
        config = HFTokenizerConfig(
            vocab_size=vocab_size,
            tokenizer_path=str(tokenizer_path),
            **config_kwargs,
        )
        
        return cls(config)