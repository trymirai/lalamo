import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Int
from tokenizers import Tokenizer as HFTokenizerImpl

from fartsovka.language_model import (
    LanguageModel,
    LanguageModelConfig,
    MessageFormatSpec,
    MessageFormatType,
)
from fartsovka.tokenizer import HFTokenizer, HFTokenizerConfig
from fartsovka.modules import DecoderConfig, Decoder
from fartsovka.modules.common import config_converter
from fartsovka.serialization import TokenizerConfigUnion


@pytest.fixture
def sample_tokenizer_file(tmp_path):
    """Create a simple tokenizer and save it to a file for testing."""
    tokenizer = HFTokenizerImpl.from_pretrained("gpt2")
    
    # Save the tokenizer to a temporary file
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    return tokenizer_path


@pytest.fixture
def sample_tokenizer(sample_tokenizer_file):
    """Create a sample tokenizer for testing."""
    vocab_size = HFTokenizerImpl.from_file(sample_tokenizer_file).get_vocab_size()
    config = HFTokenizerConfig(
        vocab_size=vocab_size,
        tokenizer_path=str(sample_tokenizer_file),
        bos_token_id=1,
        eos_token_id=2,
    )
    return HFTokenizer(config)


@pytest.fixture
def dummy_decoder(rng_key):
    """Create a dummy decoder for testing."""
    from tests.test_modules.common import create_dummy_decoder
    return create_dummy_decoder(rng_key, vocab_size=50257)  # GPT-2 vocab size


def test_language_model_init(sample_tokenizer, dummy_decoder):
    """Test initializing the language model."""
    # Create language model config
    lm_config = LanguageModelConfig(
        decoder_config=dummy_decoder.config,
        tokenizer_config=sample_tokenizer.config,
        message_format_type=MessageFormatType.LLAMA,
        message_format_spec=LanguageModelConfig.get_default_message_format_spec(MessageFormatType.LLAMA),
        model_name="Test Model",
    )
    
    # Create language model
    lm = LanguageModel(
        config=lm_config,
        decoder=dummy_decoder,
        tokenizer=sample_tokenizer,
    )
    
    # Check the configuration
    assert lm.config.model_name == "Test Model"
    assert lm.config.message_format_type == MessageFormatType.LLAMA
    assert isinstance(lm.decoder, Decoder)
    assert isinstance(lm.tokenizer, HFTokenizer)


def test_language_model_format_messages(sample_tokenizer, dummy_decoder):
    """Test formatting messages according to the model's format."""
    # Create language model with LLAMA format
    lm_config = LanguageModelConfig(
        decoder_config=dummy_decoder.config,
        tokenizer_config=sample_tokenizer.config,
        message_format_type=MessageFormatType.LLAMA,
        message_format_spec=LanguageModelConfig.get_default_message_format_spec(MessageFormatType.LLAMA),
        model_name="Test Model",
    )
    
    lm = LanguageModel(
        config=lm_config,
        decoder=dummy_decoder,
        tokenizer=sample_tokenizer,
    )
    
    # Test formatting messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]
    
    formatted = lm.format_messages(messages)
    
    # Check the format
    assert "<|system|>" in formatted
    assert "<|user|>" in formatted
    assert "<|assistant|>" in formatted
    assert "You are a helpful assistant." in formatted
    assert "Hello!" in formatted
    assert "Hi there! How can I help you?" in formatted
    
    # Test with a different format (GEMMA)
    lm.config = LanguageModelConfig(
        decoder_config=dummy_decoder.config,
        tokenizer_config=sample_tokenizer.config,
        message_format_type=MessageFormatType.GEMMA,
        message_format_spec=LanguageModelConfig.get_default_message_format_spec(MessageFormatType.GEMMA),
        model_name="Test Model",
    )
    
    formatted = lm.format_messages(messages)
    
    # Gemma doesn't support system messages by default
    assert "<start_of_turn>user" in formatted
    assert "<start_of_turn>model" in formatted
    assert "Hello!" in formatted
    assert "Hi there! How can I help you?" in formatted
    assert "You are a helpful assistant." not in formatted  # System message should be skipped


def test_language_model_config_serialization(sample_tokenizer, dummy_decoder):
    """Test serializing and deserializing language model config."""
    # Create language model config
    lm_config = LanguageModelConfig(
        decoder_config=dummy_decoder.config,
        tokenizer_config=sample_tokenizer.config,
        message_format_type=MessageFormatType.LLAMA,
        message_format_spec=LanguageModelConfig.get_default_message_format_spec(MessageFormatType.LLAMA),
        model_name="Test Model",
    )
    
    # Serialize config
    config_dict = config_converter.unstructure(lm_config)
    
    # Check serialized config
    assert config_dict["model_name"] == "Test Model"
    assert config_dict["message_format_type"] == "LLAMA"
    assert "decoder_config" in config_dict
    assert "tokenizer_config" in config_dict
    assert config_dict["tokenizer_config"]["type"] == "HFTokenizerConfig"
    
    # Deserialize config
    deserialized_config = config_converter.structure(config_dict, LanguageModelConfig)
    
    # Check deserialized config
    assert deserialized_config.model_name == "Test Model"
    assert deserialized_config.message_format_type == MessageFormatType.LLAMA
    assert isinstance(deserialized_config.tokenizer_config, HFTokenizerConfig)