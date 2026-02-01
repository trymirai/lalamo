"""Test that _prefill produces consistent results regardless of chunk_size."""

import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel

from ..common import assert_close

# Tolerances for chunked prefill comparison.
# Different chunk sizes accumulate floating point errors differently,
# especially through attention softmax and layer norms.
# With logit RMS typically ~2-3, we use tolerances that catch major bugs
# while allowing expected numerical differences from chunking.
PREFILL_ATOL = 0.2  # Absolute tolerance (small relative to logit RMS ~2.7)
PREFILL_RTOL = 0.1  # 10% relative tolerance

# Very large chunk size that effectively means no chunking
NO_CHUNK_SIZE = 100_000


@pytest.fixture
def language_model() -> LanguageModel:
    model = import_model(REPO_TO_MODEL["Qwen/Qwen2.5-0.5B-Instruct"]).model
    assert isinstance(model, LanguageModel)
    return model


def run_prefill_with_chunk_size(
    model: LanguageModel,
    token_ids: jnp.ndarray,
    lengths_without_padding: jnp.ndarray | None,
    chunk_size: int,
) -> jnp.ndarray:
    """Run _prefill with a specific chunk_size and return last_token_logits."""
    result = model._prefill(
        token_ids,
        lengths_without_padding=lengths_without_padding,
        chunk_size=chunk_size,
    )
    return result.last_token_logits


class TestPrefillChunkingConsistency:
    """Test that different chunk sizes produce the same logits."""

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_sequence_random_length(
        self,
        language_model: LanguageModel,
        chunk_size: int,
        seed: int,
    ) -> None:
        """Test that a single sequence with random length produces same logits with chunking vs no chunking."""
        rng = np.random.default_rng(seed)
        # Random length that may span multiple chunks
        sequence_length = int(rng.integers(10, 300))

        # Create deterministic token ids
        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]  # Add batch dimension

        # Run with no chunking (reference)
        reference_logits = run_prefill_with_chunk_size(language_model, token_ids, None, NO_CHUNK_SIZE)

        # Compare with chunked version
        test_logits = run_prefill_with_chunk_size(language_model, token_ids, None, chunk_size)
        assert_close(
            result=test_logits,
            reference=reference_logits,
            atol=PREFILL_ATOL,
            rtol=PREFILL_RTOL,
            fraction_of_allowed_violations=0.05,
            operation_name=f"prefill chunk_size={chunk_size} vs no_chunk (seq_len={sequence_length})",
        )

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_batch_with_random_lengths(
        self,
        language_model: LanguageModel,
        chunk_size: int,
        seed: int,
    ) -> None:
        """Test batch where sequences have random lengths ending in different chunks."""
        rng = np.random.default_rng(seed)
        batch_size = 4

        # Generate random lengths - some will end in different chunks
        lengths = [int(x) for x in rng.integers(10, 200, size=batch_size)]
        max_length = max(lengths)

        # Create token ids and pad to max length
        batch_token_ids = []
        for length in lengths:
            tokens = jnp.arange(length, dtype=jnp.int32) % 1000
            padded = jnp.pad(tokens, (0, max_length - length), constant_values=0)
            batch_token_ids.append(padded)

        token_ids = jnp.stack(batch_token_ids)
        lengths_array = jnp.array(lengths, dtype=jnp.int32)

        # Run with no chunking (reference)
        reference_logits = run_prefill_with_chunk_size(language_model, token_ids, lengths_array, NO_CHUNK_SIZE)

        # Compare with chunked version
        test_logits = run_prefill_with_chunk_size(language_model, token_ids, lengths_array, chunk_size)

        assert_close(
            result=test_logits,
            reference=reference_logits,
            atol=PREFILL_ATOL,
            rtol=PREFILL_RTOL,
            fraction_of_allowed_violations=0.10,  # Allow 10% outliers for batch (more FP variation)
            operation_name=f"batch prefill chunk_size={chunk_size} vs no_chunk (lengths={lengths})",
        )

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    def test_batch_sequences_end_in_different_chunks(
        self,
        language_model: LanguageModel,
        chunk_size: int,
    ) -> None:
        """Test batch where sequences intentionally end in different chunks relative to chunk_size."""
        # Create lengths that end in different chunks for the given chunk_size
        # This ensures some sequences get their last token logit from chunk 0, some from chunk 1, etc.
        lengths = [
            chunk_size // 2,  # Ends in first chunk
            chunk_size - 1,  # Just before first chunk boundary
            chunk_size,  # Exactly at first chunk boundary
            chunk_size + 1,  # Just after first chunk boundary (in second chunk)
            chunk_size + chunk_size // 2,  # Middle of second chunk
            chunk_size * 2,  # At second chunk boundary
            chunk_size * 2 + 1,  # Just into third chunk
        ]
        max_length = max(lengths)

        # Create token ids and pad to max length
        batch_token_ids = []
        for length in lengths:
            tokens = jnp.arange(length, dtype=jnp.int32) % 1000
            padded = jnp.pad(tokens, (0, max_length - length), constant_values=0)
            batch_token_ids.append(padded)

        token_ids = jnp.stack(batch_token_ids)
        lengths_array = jnp.array(lengths, dtype=jnp.int32)

        # Run with no chunking (reference)
        reference_logits = run_prefill_with_chunk_size(language_model, token_ids, lengths_array, NO_CHUNK_SIZE)

        # Compare with chunked version
        test_logits = run_prefill_with_chunk_size(language_model, token_ids, lengths_array, chunk_size)

        assert_close(
            result=test_logits,
            reference=reference_logits,
            atol=PREFILL_ATOL,
            rtol=PREFILL_RTOL,
            fraction_of_allowed_violations=0.10,
            operation_name=f"batch prefill different_chunks chunk_size={chunk_size}",
        )

    @pytest.mark.parametrize(
        "length_offset",
        [-2, -1, 0, 1, 2],  # Positions relative to chunk boundary
    )
    def test_boundary_edge_cases(
        self,
        language_model: LanguageModel,
        length_offset: int,
    ) -> None:
        """Test sequences at various offsets from chunk boundary."""
        base_chunk_size = 64
        sequence_length = base_chunk_size + length_offset

        if sequence_length <= 0:
            pytest.skip("Invalid sequence length")

        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]

        # Compare small chunk size (will chunk) vs no chunking
        small_chunk = 32

        small_logits = run_prefill_with_chunk_size(language_model, token_ids, None, small_chunk)
        reference_logits = run_prefill_with_chunk_size(language_model, token_ids, None, NO_CHUNK_SIZE)

        assert_close(
            result=small_logits,
            reference=reference_logits,
            atol=PREFILL_ATOL,
            rtol=PREFILL_RTOL,
            fraction_of_allowed_violations=0.05,
            operation_name=f"boundary test offset={length_offset}",
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_state_capacity_with_chunking(
        self,
        language_model: LanguageModel,
        seed: int,
    ) -> None:
        """Test that state_capacity parameter works correctly with chunking and random lengths."""
        rng = np.random.default_rng(seed)
        sequence_length = int(rng.integers(50, 150))
        state_capacity = sequence_length + 100  # Extra capacity for generation

        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]

        # Run with no chunking (reference)
        reference_result = language_model._prefill(
            token_ids,
            state_capacity=state_capacity,
            chunk_size=NO_CHUNK_SIZE,
        )
        reference_logits = reference_result.last_token_logits

        # Compare with different chunk sizes
        for chunk_size in [32, 64, 128]:
            result = language_model._prefill(
                token_ids,
                state_capacity=state_capacity,
                chunk_size=chunk_size,
            )
            assert_close(
                result=result.last_token_logits,
                reference=reference_logits,
                atol=PREFILL_ATOL,
                rtol=PREFILL_RTOL,
                fraction_of_allowed_violations=0.05,
                operation_name=f"state_capacity test chunk_size={chunk_size} vs no_chunk (seq_len={sequence_length})",
            )


pytestmark = pytest.mark.xdist_group("heavy")
