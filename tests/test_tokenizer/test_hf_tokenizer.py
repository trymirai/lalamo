import json
from pathlib import Path

import jax.numpy as jnp
import pytest
from tokenizers import Tokenizer as HFTokenizerImpl

from fartsovka.tokenizer import HFTokenizer, HFTokenizerConfig


@pytest.fixture
def sample_tokenizer_file(tmp_path):
    """Create a simple tokenizer and save it to a file for testing."""
    tokenizer = HFTokenizerImpl.from_pretrained("gpt2")
    
    # Save the tokenizer to a temporary file
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    return tokenizer_path


def test_hf_tokenizer_init(sample_tokenizer_file):
    """Test initializing the HF tokenizer."""
    vocab_size = HFTokenizerImpl.from_file(sample_tokenizer_file).get_vocab_size()
    
    # Create config and tokenizer
    config = HFTokenizerConfig(
        vocab_size=vocab_size,
        tokenizer_path=str(sample_tokenizer_file),
        bos_token_id=1,
        eos_token_id=2,
    )
    
    tokenizer = HFTokenizer(config)
    
    # Check the configuration
    assert tokenizer.config.vocab_size == vocab_size
    assert tokenizer.config.bos_token_id == 1
    assert tokenizer.config.eos_token_id == 2
    assert tokenizer.config.pad_token_id is None


def test_hf_tokenizer_encode_decode(sample_tokenizer_file):
    """Test encoding and decoding with the HF tokenizer."""
    # Create tokenizer
    tokenizer = HFTokenizer.from_file(str(sample_tokenizer_file))
    
    # Test single text encoding/decoding
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    
    # Check token array
    assert isinstance(tokens, jnp.ndarray)
    assert tokens.dtype == jnp.int32
    
    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text
    
    # Test batch encoding/decoding
    texts = ["Hello, world!", "This is a test."]
    token_arrays = tokenizer.encode(texts)
    
    # Check that we got a list of arrays
    assert isinstance(token_arrays, list)
    assert len(token_arrays) == 2
    assert all(isinstance(arr, jnp.ndarray) for arr in token_arrays)
    
    # Test batch decoding
    decoded_texts = tokenizer.decode(token_arrays)
    assert decoded_texts == texts


def test_tokenizer_from_file(sample_tokenizer_file):
    """Test creating a tokenizer from a file."""
    tokenizer = HFTokenizer.from_file(
        str(sample_tokenizer_file),
        bos_token_id=1,
        eos_token_id=2,
    )
    
    assert isinstance(tokenizer, HFTokenizer)
    assert tokenizer.config.bos_token_id == 1
    assert tokenizer.config.eos_token_id == 2