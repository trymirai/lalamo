import jax
import pytest
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int
from tokenizers import Tokenizer as HFTokenizerImpl

from fartsovka.language_model import (
    LanguageModel, 
    LanguageModelConfig,
    MessageFormatType, 
    MessageFormatSpec,
    DecodingStrategy,
    DecodingConfig,
    decode_text,
    decode_batch,
)
from fartsovka.language_model.decoding import (
    _temperature_sample,
    _top_p_sample,
    _greedy_sample,
    _check_stop_tokens,
    _check_stop_strings,
    single_step_decode,
)
from fartsovka.tokenizer import HFTokenizer, HFTokenizerConfig


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
def mock_language_model(sample_tokenizer, rng_key, monkeypatch):
    """Create a mock language model for testing decoding."""
    from tests.test_modules.common import create_dummy_decoder
    
    # Create a dummy decoder
    dummy_decoder = create_dummy_decoder(rng_key, vocab_size=50257)  # GPT-2 vocab size
    
    # Create language model config
    lm_config = LanguageModelConfig(
        decoder_config=dummy_decoder.config,
        tokenizer_config=sample_tokenizer.config,
        message_format_type=MessageFormatType.PLAIN,
        message_format_spec=MessageFormatSpec(),
        model_name="Test Model",
    )
    
    # Create language model
    lm = LanguageModel(
        config=lm_config,
        decoder=dummy_decoder,
        tokenizer=sample_tokenizer,
    )
    
    # Mock the __call__ method to return predictable outputs for testing
    original_call = lm.__call__
    
    def mock_call(token_ids, token_positions, kv_cache=None, mask=None, return_updated_kv_cache=False):
        from fartsovka.modules import DecoderOutput
        
        # Create a simple output that returns predictable token IDs
        # This makes testing more reliable
        batch_size = token_ids.shape[0] if len(token_ids.shape) > 1 else 1
        vocab_size = lm.decoder.config.vocab_size
        
        # Create deterministic logits with the highest probability for token ID 42
        logits = jnp.zeros((batch_size, vocab_size)) - 100.0
        logits = logits.at[:, 42].set(100.0)  # Make token 42 very likely
        
        # Mock the KV cache update
        updated_kv_cache = [None] * len(lm.decoder.layers) if return_updated_kv_cache else None
        
        return DecoderOutput(output=logits, kv_cache=updated_kv_cache)
    
    monkeypatch.setattr(lm, "__call__", mock_call)
    
    return lm


def test_sampling_strategies():
    """Test basic token sampling strategies."""
    rng_key = jax.random.PRNGKey(42)
    
    # Create example logits
    logits = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test greedy sampling
    greedy_token = _greedy_sample(logits)
    assert greedy_token == 4  # Should select index of highest logit
    
    # Test temperature sampling
    temp_token = _temperature_sample(logits, temperature=1.0, key=rng_key)
    assert isinstance(temp_token, jax.Array)
    assert 0 <= temp_token < 5
    
    # With very low temperature, should approximate greedy
    low_temp_token = _temperature_sample(logits, temperature=0.01, key=rng_key)
    assert low_temp_token == 4
    
    # Test top-p sampling
    top_p_token = _top_p_sample(logits, p=0.9, temperature=1.0, key=rng_key)
    assert isinstance(top_p_token, jax.Array)
    assert 0 <= top_p_token < 5


def test_stop_conditions():
    """Test stop token and stop string detection."""
    # Test stop tokens
    tokens = jnp.array([1, 2, 3, 4, 5])
    
    # Should match when token is present
    assert _check_stop_tokens(tokens, [3, 10])
    
    # Should not match when token is absent
    assert not _check_stop_tokens(tokens, [10, 20])
    
    # Test stop strings
    text = "Hello, this is a test."
    
    # Should match when string is present
    assert _check_stop_strings(text, ["test", "not present"])
    
    # Should not match when string is absent
    assert not _check_stop_strings(text, ["xyz", "not present"])


def test_single_step_decode(mock_language_model, rng_key):
    """Test single step decoding."""
    model = mock_language_model
    
    # Prepare inputs
    token_ids = jnp.array([1, 2, 3])
    token_positions = jnp.arange(3)
    
    # Test greedy decoding
    next_token, kv_cache = single_step_decode(
        model=model,
        token_ids=token_ids,
        token_positions=token_positions,
        strategy=DecodingStrategy.GREEDY,
    )
    
    # Due to our mock, this should always be token 42
    assert next_token == 42
    assert kv_cache is not None
    
    # Test temperature sampling
    next_token, kv_cache = single_step_decode(
        model=model,
        token_ids=token_ids,
        token_positions=token_positions,
        strategy=DecodingStrategy.SAMPLE,
        temperature=1.0,
        key=rng_key,
    )
    
    # Since logits are heavily biased, this should also be 42
    assert next_token == 42
    
    # Test top-p sampling
    next_token, kv_cache = single_step_decode(
        model=model,
        token_ids=token_ids,
        token_positions=token_positions,
        strategy=DecodingStrategy.TOP_P,
        top_p=0.9,
        temperature=1.0,
        key=rng_key,
    )
    
    # Again, due to bias in mock model, should be 42
    assert next_token == 42


def test_decode_text(mock_language_model, rng_key, monkeypatch):
    """Test decoding a single text sequence."""
    model = mock_language_model
    
    # Mock tokenizer methods
    def mock_encode(self, text, add_special_tokens=True):
        if isinstance(text, str):
            # Return dummy token IDs
            return jnp.array([1, 2, 3], dtype=jnp.int32)
        return [jnp.array([1, 2, 3], dtype=jnp.int32) for _ in text]
    
    def mock_decode(self, token_ids):
        # Return dummy text
        if isinstance(token_ids, (list, tuple)) and not isinstance(token_ids[0], int):
            return ["Decoded text" for _ in token_ids]
        return "Decoded text"
    
    monkeypatch.setattr(HFTokenizer, "encode", mock_encode)
    monkeypatch.setattr(HFTokenizer, "decode", mock_decode)
    
    # Create decoding config
    config = DecodingConfig(
        strategy=DecodingStrategy.GREEDY,
        max_tokens=5,
    )
    
    # Decode text
    result = decode_text(
        model=model,
        text="Hello",
        config=config,
    )
    
    assert result == "Decoded text"
    
    # Test with stop tokens
    config = DecodingConfig(
        strategy=DecodingStrategy.GREEDY,
        max_tokens=5,
        stop_tokens=[42],  # This will trigger an early stop with our mock model
    )
    
    result = decode_text(
        model=model,
        text="Hello",
        config=config,
    )
    
    assert result == "Decoded text"


def test_decode_batch(mock_language_model, rng_key, monkeypatch):
    """Test decoding a batch of text sequences."""
    model = mock_language_model
    
    # Mock tokenizer methods as in previous test
    def mock_encode(self, text, add_special_tokens=True):
        if isinstance(text, str):
            # Return dummy token IDs
            return jnp.array([1, 2, 3], dtype=jnp.int32)
        return [jnp.array([1, 2, 3], dtype=jnp.int32) for _ in text]
    
    def mock_decode(self, token_ids):
        # Return dummy text
        if isinstance(token_ids, (list, tuple)) and not isinstance(token_ids[0], int):
            return ["Decoded text" for _ in token_ids]
        return "Decoded text"
    
    monkeypatch.setattr(HFTokenizer, "encode", mock_encode)
    monkeypatch.setattr(HFTokenizer, "decode", mock_decode)
    
    # Create decoding config
    config = DecodingConfig(
        strategy=DecodingStrategy.GREEDY,
        max_tokens=5,
    )
    
    # Decode batch of texts
    results = decode_batch(
        model=model,
        texts=["Hello", "World"],
        config=config,
    )
    
    assert len(results) == 2
    assert results[0] == "Decoded text"
    assert results[1] == "Decoded text"
    
    # Test with sampling and different parameters
    config = DecodingConfig(
        strategy=DecodingStrategy.SAMPLE,
        max_tokens=3,
        temperature=0.8,
    )
    
    results = decode_batch(
        model=model,
        texts=["Hello", "World", "Test"],
        config=config,
        key=rng_key,
    )
    
    assert len(results) == 3
    assert all(result == "Decoded text" for result in results)