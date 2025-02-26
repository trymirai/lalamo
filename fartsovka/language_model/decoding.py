from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Literal, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from fartsovka.language_model import LanguageModel

__all__ = [
    "DecodingStrategy",
    "DecodingConfig",
    "decode_text",
    "decode_batch",
]


class DecodingStrategy(Enum):
    """Strategy for decoding tokens."""
    GREEDY = auto()   # Argmax sampling
    SAMPLE = auto()   # Temperature-based sampling
    TOP_P = auto()    # Nucleus sampling


@dataclass
class DecodingConfig:
    """Configuration for text decoding."""
    strategy: DecodingStrategy = DecodingStrategy.GREEDY
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    stop_tokens: Optional[Sequence[int]] = None
    stop_strings: Optional[Sequence[str]] = None


def _temperature_sample(
    logits: Float[Array, "vocab_size"],
    temperature: float,
    key: PRNGKeyArray,
) -> Int[Array, ""]:
    """Sample from logits using temperature."""
    # Divide by temperature to control randomness
    scaled_logits = logits / jnp.maximum(temperature, 1e-7)
    # Convert to probabilities
    probs = jax.nn.softmax(scaled_logits)
    # Sample from the distribution
    return jax.random.categorical(key, probs)


def _top_p_sample(
    logits: Float[Array, "vocab_size"],
    p: float,
    temperature: float,
    key: PRNGKeyArray,
) -> Int[Array, ""]:
    """Sample from the top p logits (nucleus sampling)."""
    # Apply temperature
    scaled_logits = logits / jnp.maximum(temperature, 1e-7)
    
    # Convert to probabilities
    probs = jax.nn.softmax(scaled_logits)
    
    # Sort probabilities in descending order
    sorted_probs = jnp.sort(probs, axis=-1)[::-1]
    
    # Calculate cumulative probabilities
    cumulative_probs = jnp.cumsum(sorted_probs)
    
    # Get the cutoff index where cumulative prob exceeds p
    cutoff_idx = jnp.sum(cumulative_probs < p) + 1
    
    # Create a mask for the top-p probabilities
    mask = jnp.zeros_like(probs, dtype=bool)
    sorted_indices = jnp.argsort(probs, axis=-1)[::-1]
    mask = mask.at[sorted_indices[:cutoff_idx]].set(True)
    
    # Zero out probabilities outside the top-p
    filtered_probs = jnp.where(mask, probs, 0.0)
    
    # Renormalize probabilities
    normalized_probs = filtered_probs / jnp.maximum(jnp.sum(filtered_probs), 1e-10)
    
    # Sample from the distribution
    return jax.random.categorical(key, normalized_probs)


def _greedy_sample(
    logits: Float[Array, "vocab_size"],
) -> Int[Array, ""]:
    """Select the token with highest probability (argmax)."""
    return jnp.argmax(logits)


def _check_stop_tokens(
    tokens: Int[Array, "seq_len"],
    stop_tokens: Sequence[int],
) -> Bool[Array, ""]:
    """Check if any of the stop tokens appear in the token sequence."""
    # Check if sequence contains any stop token
    return jnp.any(jnp.isin(tokens, jnp.array(stop_tokens)))


def _check_stop_strings(
    text: str,
    stop_strings: Sequence[str],
) -> bool:
    """Check if any of the stop strings appear in the text."""
    return any(stop_string in text for stop_string in stop_strings)


def single_step_decode(
    model: LanguageModel,
    token_ids: Int[Array, "seq_len"],
    token_positions: Int[Array, "seq_len"],
    kv_cache: list[Array] | None = None,
    strategy: DecodingStrategy = DecodingStrategy.GREEDY,
    temperature: float = 1.0,
    top_p: float = 0.9,
    key: Optional[PRNGKeyArray] = None,
) -> tuple[Int[Array, ""], list[Array] | None]:
    """Run a single decoding step to generate the next token."""
    # Create causal mask for the sequence
    seq_len = token_ids.shape[0]
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    
    # Forward pass through the model
    output = model(
        token_ids=token_ids,
        token_positions=token_positions,
        kv_cache=kv_cache,
        mask=mask,
        return_updated_kv_cache=True,
    )
    
    # Get the logits for the last token
    logits = output.output[-1]
    
    # Sample the next token
    if strategy == DecodingStrategy.GREEDY:
        next_token = _greedy_sample(logits)
    elif strategy == DecodingStrategy.SAMPLE:
        assert key is not None, "Random key is required for temperature sampling"
        next_token = _temperature_sample(logits, temperature, key)
    elif strategy == DecodingStrategy.TOP_P:
        assert key is not None, "Random key is required for top-p sampling"
        next_token = _top_p_sample(logits, top_p, temperature, key)
    else:
        raise ValueError(f"Unknown decoding strategy: {strategy}")
    
    return next_token, output.kv_cache


def decode_text(
    model: LanguageModel,
    text: str,
    config: DecodingConfig,
    key: Optional[PRNGKeyArray] = None,
) -> str:
    """
    Decode text using the given model and configuration.
    
    Args:
        model: The language model to use for decoding
        text: The input text to decode
        config: The decoding configuration
        key: Random key for sampling
        
    Returns:
        The decoded text
    """
    # Encode the input text
    input_tokens = model.tokenizer.encode(text)
    
    # Prepare for decoding
    all_tokens = input_tokens
    positions = jnp.arange(len(all_tokens))
    kv_cache = None
    
    needs_random = config.strategy in [DecodingStrategy.SAMPLE, DecodingStrategy.TOP_P]
    if needs_random and key is None:
        key = jax.random.PRNGKey(0)
    
    # Iteratively decode tokens
    for i in range(config.max_tokens):
        # Get the next key if we're sampling
        next_key = None
        if needs_random:
            key, next_key = jax.random.split(key)
        
        # Decode the next token
        next_token, kv_cache = single_step_decode(
            model=model,
            token_ids=all_tokens,
            token_positions=positions,
            kv_cache=kv_cache,
            strategy=config.strategy,
            temperature=config.temperature,
            top_p=config.top_p,
            key=next_key,
        )
        
        # Append the new token
        all_tokens = jnp.append(all_tokens, next_token)
        positions = jnp.append(positions, positions[-1] + 1)
        
        # Check if we should stop
        if config.stop_tokens is not None and next_token in config.stop_tokens:
            break
        
        # Check if we have a stop string
        if config.stop_strings is not None:
            current_text = model.tokenizer.decode(all_tokens)
            if _check_stop_strings(current_text, config.stop_strings):
                break
    
    # Decode the tokens back to text
    return model.tokenizer.decode(all_tokens)


# For batch decoding we need additional support functions

def _batch_step_decode(
    model: LanguageModel,
    token_ids: Int[Array, "batch seq_len"],
    token_positions: Int[Array, "batch seq_len"],
    active_mask: Bool[Array, "batch"],
    kv_caches: list[list[Array]] | None = None,
    strategy: DecodingStrategy = DecodingStrategy.GREEDY,
    temperature: float = 1.0,
    top_p: float = 0.9,
    keys: Optional[PRNGKeyArray] = None,
) -> tuple[Int[Array, "batch"], list[list[Array]] | None, Bool[Array, "batch"]]:
    """Run a single decoding step for a batch of inputs."""
    # Create a batch of causal masks
    batch_size, seq_len = token_ids.shape
    batch_mask = jnp.tril(jnp.ones((1, seq_len, seq_len), dtype=bool)).repeat(batch_size, axis=0)
    
    # Create a mapping function to process each sequence in the batch
    def process_sequence(idx, args):
        tokens, positions, mask, kv_cache, rng_key = args
        
        # Skip inactive sequences
        should_process = active_mask[idx]
        
        def process():
            # Forward pass through the model
            output = model(
                token_ids=tokens,
                token_positions=positions,
                kv_cache=kv_cache,
                mask=mask[0],  # Single mask for this sequence
                return_updated_kv_cache=True,
            )
            
            # Get the logits for the last token
            logits = output.output[-1]
            
            # Sample the next token based on strategy
            if strategy == DecodingStrategy.GREEDY:
                next_token = _greedy_sample(logits)
            elif strategy == DecodingStrategy.SAMPLE:
                next_token = _temperature_sample(logits, temperature, rng_key)
            elif strategy == DecodingStrategy.TOP_P:
                next_token = _top_p_sample(logits, top_p, temperature, rng_key)
            
            return next_token, output.kv_cache
        
        # Only process active sequences
        next_token, new_kv_cache = jax.lax.cond(
            should_process,
            lambda _: process(),
            lambda _: (jnp.zeros((), dtype=jnp.int32), kv_cache),
            None
        )
        
        return next_token, new_kv_cache
    
    # Prepare inputs for each sequence
    batch_inputs = (
        token_ids, 
        token_positions, 
        batch_mask, 
        kv_caches if kv_caches is not None else [None] * batch_size,
        keys if keys is not None else [None] * batch_size
    )
    
    # Use vmap to process all sequences in parallel
    next_tokens, new_kv_caches = jax.vmap(process_sequence, in_axes=(0, (0, 0, 0, 0, 0)))(
        jnp.arange(batch_size), batch_inputs
    )
    
    # Check if any sequence should become inactive
    new_active_mask = active_mask
    
    return next_tokens, new_kv_caches, new_active_mask


def decode_batch(
    model: LanguageModel,
    texts: Sequence[str],
    config: DecodingConfig,
    key: Optional[PRNGKeyArray] = None,
) -> list[str]:
    """
    Decode a batch of texts using the given model and configuration.
    
    Args:
        model: The language model to use for decoding
        texts: The input texts to decode
        config: The decoding configuration
        key: Random key for sampling
        
    Returns:
        The decoded texts
    """
    batch_size = len(texts)
    
    # Encode the input texts
    encoded_inputs = model.tokenizer.encode(texts)
    
    # Pad sequences to the same length
    max_input_length = max(len(tokens) for tokens in encoded_inputs)
    
    # Prepare padded inputs and masks
    padded_inputs = []
    for tokens in encoded_inputs:
        padding = jnp.zeros(max_input_length - len(tokens), dtype=jnp.int32)
        padded_inputs.append(jnp.concatenate([tokens, padding]))
    
    # Stack into batch
    input_tokens = jnp.stack(padded_inputs)
    
    # Track active sequences (all active initially)
    active_mask = jnp.ones(batch_size, dtype=bool)
    
    # Positions for each sequence
    positions = jnp.tile(jnp.arange(max_input_length), (batch_size, 1))
    
    # Prepare the KV caches
    kv_caches = None
    
    # Prepare random keys if needed
    needs_random = config.strategy in [DecodingStrategy.SAMPLE, DecodingStrategy.TOP_P]
    if needs_random and key is not None:
        keys = jax.random.split(key, batch_size + config.max_tokens)
        batch_keys = keys[:batch_size]
        step_keys = keys[batch_size:]
    else:
        batch_keys = None
        step_keys = None
    
    # All generated tokens for each sequence
    all_tokens = input_tokens
    
    # Iteratively decode tokens
    for i in range(config.max_tokens):
        # Get the next key if we're sampling
        next_keys = None
        if needs_random:
            next_keys = step_keys[i]
        
        # Decode the next tokens for the whole batch
        next_tokens, kv_caches, active_mask = _batch_step_decode(
            model=model,
            token_ids=all_tokens,
            token_positions=positions,
            active_mask=active_mask,
            kv_caches=kv_caches,
            strategy=config.strategy,
            temperature=config.temperature,
            top_p=config.top_p,
            keys=next_keys,
        )
        
        # Append the new tokens
        all_tokens = jnp.concatenate([all_tokens, next_tokens[:, jnp.newaxis]], axis=1)
        positions = jnp.concatenate([positions, positions[:, -1:] + 1], axis=1)
        
        # If no sequences are active, we're done
        if not jnp.any(active_mask):
            break
    
    # Decode the tokens back to text
    results = []
    for tokens in all_tokens:
        # Remove padding tokens that might be added during batching
        valid_tokens = tokens[tokens != 0]
        results.append(model.tokenizer.decode(valid_tokens))
    
    return results