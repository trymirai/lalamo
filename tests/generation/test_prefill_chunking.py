import jax.numpy as jnp
import numpy as np
import pytest

from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel

from ..common import assert_close

PREFILL_ATOL = 0.2
PREFILL_RTOL = 0.1
NO_CHUNK_SIZE = 100_000


@pytest.fixture
def language_model() -> LanguageModel:
    model = import_model(REPO_TO_MODEL["Qwen/Qwen2.5-0.5B-Instruct"]).model
    assert isinstance(model, LanguageModel)
    return model


class TestPrefillChunkingConsistency:
    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_single_sequence_random_length(
        self,
        language_model: LanguageModel,
        chunk_size: int,
        seed: int,
    ) -> None:
        random_generator = np.random.default_rng(seed)
        sequence_length = int(random_generator.integers(10, 300))
        state_capacity = sequence_length + 128
        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]

        reference_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=None,
            chunk_size=NO_CHUNK_SIZE,
        ).last_token_logits
        test_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=None,
            chunk_size=chunk_size,
        ).last_token_logits
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
        random_generator = np.random.default_rng(seed)
        batch_size = 4
        lengths = [int(value) for value in random_generator.integers(10, 200, size=batch_size)]
        max_length = max(lengths)
        state_capacity = max_length + 128
        token_ids_by_sequence = [
            jnp.pad(
                jnp.arange(length, dtype=jnp.int32) % 1000,
                (0, max_length - length),
                constant_values=0,
            )
            for length in lengths
        ]
        token_ids = jnp.stack(token_ids_by_sequence)
        lengths_without_padding = jnp.array(lengths, dtype=jnp.int32)
        reference_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=lengths_without_padding,
            chunk_size=NO_CHUNK_SIZE,
        ).last_token_logits
        test_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=lengths_without_padding,
            chunk_size=chunk_size,
        ).last_token_logits

        assert_close(
            result=test_logits,
            reference=reference_logits,
            atol=PREFILL_ATOL,
            rtol=PREFILL_RTOL,
            fraction_of_allowed_violations=0.10,
            operation_name=f"batch prefill chunk_size={chunk_size} vs no_chunk (lengths={lengths})",
        )

    @pytest.mark.parametrize("chunk_size", [32, 64, 128])
    def test_batch_sequences_end_in_different_chunks(
        self,
        language_model: LanguageModel,
        chunk_size: int,
    ) -> None:
        lengths = [
            chunk_size // 2,
            chunk_size - 1,
            chunk_size,
            chunk_size + 1,
            chunk_size + chunk_size // 2,
            chunk_size * 2,
            chunk_size * 2 + 1,
        ]
        max_length = max(lengths)
        state_capacity = max_length + 128
        token_ids_by_sequence = [
            jnp.pad(
                jnp.arange(length, dtype=jnp.int32) % 1000,
                (0, max_length - length),
                constant_values=0,
            )
            for length in lengths
        ]
        token_ids = jnp.stack(token_ids_by_sequence)
        lengths_without_padding = jnp.array(lengths, dtype=jnp.int32)
        reference_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=lengths_without_padding,
            chunk_size=NO_CHUNK_SIZE,
        ).last_token_logits
        test_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=lengths_without_padding,
            chunk_size=chunk_size,
        ).last_token_logits

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
        base_chunk_size = 64
        sequence_length = base_chunk_size + length_offset
        state_capacity = sequence_length + 128

        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]

        small_chunk = 32
        small_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=None,
            chunk_size=small_chunk,
        ).last_token_logits
        reference_logits = language_model._prefill(
            token_ids,
            state_capacity,
            lengths_without_padding=None,
            chunk_size=NO_CHUNK_SIZE,
        ).last_token_logits

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
        random_generator = np.random.default_rng(seed)
        sequence_length = int(random_generator.integers(50, 150))
        state_capacity = sequence_length + 100

        token_ids = jnp.arange(sequence_length, dtype=jnp.int32) % 1000
        token_ids = token_ids[None, :]

        reference_result = language_model._prefill(
            token_ids,
            state_capacity,
            chunk_size=NO_CHUNK_SIZE,
        )
        reference_logits = reference_result.last_token_logits

        for chunk_size in [32, 64, 128]:
            result = language_model._prefill(
                token_ids,
                state_capacity,
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
