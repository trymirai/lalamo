from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Int
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.models.language_model import GenerationConfig, LanguageModel
from lalamo.models.raw_text_codec import RawTextCodecConfig
from lalamo.module import Keychain
from lalamo.modules.embedding import EmbeddingBase
from lalamo.speculator import NoSpeculatorConfig, NoSpeculatorState, Speculator, TreeProposal
from lalamo.utils.sharding import ShardingConfig
from tests.unit.speculator.conftest import VOCAB_SIZE

GREEDY_CONFIG = GenerationConfig(temperature=0.0)
NUM_TREE_NODES = 1 + 2 * VOCAB_SIZE


def tree_parent_indices() -> Int[Array, " nodes"]:
    children_of_root = jnp.zeros(VOCAB_SIZE, dtype=jnp.int32)
    grandchild_parents = jnp.arange(1, VOCAB_SIZE + 1, dtype=jnp.int32)
    return jnp.concatenate([jnp.asarray([-1], dtype=jnp.int32), children_of_root, grandchild_parents])


def tree_depths() -> Int[Array, " nodes"]:
    return jnp.concatenate(
        [
            jnp.zeros(1, dtype=jnp.int32),
            jnp.ones(VOCAB_SIZE, dtype=jnp.int32),
            jnp.full(VOCAB_SIZE, 2, dtype=jnp.int32),
        ],
    )


class FullCoverTreeSpeculator(Speculator[NoSpeculatorState, NoSpeculatorConfig]):
    requires_activation_trace: ClassVar[bool] = False

    @classmethod
    def build(cls, sharding_config: ShardingConfig) -> "FullCoverTreeSpeculator":
        tokenizer = Tokenizer(WordLevel({f"tok{i}": i for i in range(VOCAB_SIZE)}, unk_token="tok0"))
        token_codec_config = RawTextCodecConfig()
        return cls(
            config=NoSpeculatorConfig(token_codec_config=token_codec_config),
            sharding_config=sharding_config,
            token_codec=token_codec_config.init(tokenizer),
        )

    def init_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> NoSpeculatorState:
        _ = (batch_size, context_capacity, dtype)
        return NoSpeculatorState()

    @property
    def max_proposal_tokens(self) -> int:
        return NUM_TREE_NODES

    def empty_proposal(
        self,
        token_positions: Int[Array, "batch nodes"],
        token_dtype: DTypeLike,
    ) -> TreeProposal:
        return TreeProposal.empty(
            token_positions,
            token_dtype,
            jnp.broadcast_to(tree_parent_indices()[None, :], token_positions.shape),
            max_depth=2,
        )

    def draft(
        self,
        state: NoSpeculatorState,
        last_token_ids: Int[Array, " batch"],
        last_token_indices: Int[Array, " batch"],
        target_embedding: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> TreeProposal:
        _ = (state, target_embedding, keychain)
        (batch_size,) = last_token_ids.shape
        child_tokens = jnp.arange(VOCAB_SIZE, dtype=last_token_ids.dtype)
        grandchild_tokens = (child_tokens * 7 + 3) % VOCAB_SIZE
        token_ids = jnp.concatenate(
            [
                last_token_ids[:, None],
                jnp.broadcast_to(child_tokens[None, :], (batch_size, VOCAB_SIZE)),
                jnp.broadcast_to(grandchild_tokens[None, :], (batch_size, VOCAB_SIZE)),
            ],
            axis=1,
        )
        token_positions = last_token_indices[:, None] + 1 + tree_depths()[None, :]
        return TreeProposal(
            token_ids=token_ids,
            token_positions=token_positions.astype(last_token_indices.dtype),
            parent_indices=jnp.broadcast_to(tree_parent_indices()[None, :], token_ids.shape),
            draft_logprobs=jnp.zeros_like(token_ids, dtype=jnp.float32),
            lengths=jnp.full_like(last_token_ids, NUM_TREE_NODES),
            max_depth=2,
        )


def make_ragged_batch() -> tuple[jax.Array, jax.Array]:
    lengths = [5, 12, 9]
    max_length = max(lengths)
    token_ids = jnp.stack(
        [
            jnp.pad(jnp.arange(3 * index, 3 * index + length) % (VOCAB_SIZE - 1), (0, max_length - length))
            for index, length in enumerate(lengths)
        ],
    ).astype(jnp.int32)
    return token_ids, jnp.asarray(lengths, dtype=jnp.int32)


def generate(
    language_model: LanguageModel,
    speculator: FullCoverTreeSpeculator | None,
    max_output_length: int = 16,
) -> tuple[jax.Array, jax.Array]:
    token_ids, lengths = make_ragged_batch()
    with jax.set_mesh(language_model.sharding_config.mesh):
        results = language_model.generate_tokens(
            token_ids,
            generation_config=GREEDY_CONFIG,
            prompt_lengths_without_padding=lengths,
            max_output_length=max_output_length,
            keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
            speculator=speculator,
        )
    return results.token_ids, results.step_lengths


def test_tree_speculator_matches_baseline(language_model: LanguageModel) -> None:
    baseline_token_ids, _ = generate(language_model, None)
    tree_speculator = FullCoverTreeSpeculator.build(language_model.sharding_config)
    tree_token_ids, tree_step_lengths = generate(language_model, tree_speculator)

    assert jnp.array_equal(baseline_token_ids, tree_token_ids)
    assert int(tree_step_lengths.max()) >= 2


def test_tree_speculator_matches_baseline_for_hybrid_model(hybrid_language_model: LanguageModel) -> None:
    baseline_token_ids, _ = generate(hybrid_language_model, None)
    tree_speculator = FullCoverTreeSpeculator.build(hybrid_language_model.sharding_config)
    tree_token_ids, tree_step_lengths = generate(hybrid_language_model, tree_speculator)

    assert jnp.array_equal(baseline_token_ids, tree_token_ids)
    assert int(tree_step_lengths.max()) >= 2
