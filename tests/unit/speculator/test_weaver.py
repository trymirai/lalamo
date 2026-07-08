import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.initializer import RandomInitializer
from lalamo.models.language_model import GenerationConfig, LanguageModel
from lalamo.models.raw_text_codec import RawTextCodecConfig
from lalamo.module import Keychain
from lalamo.modules import LinearConfig, NormalizationConfig
from lalamo.modules.normalization import UpcastMode
from lalamo.speculator import WeaverConfig, WeaverDraftState, WeaverSpeculator, WeaverSpeculatorConfig
from lalamo.utils.sharding import ShardingConfig
from tests.unit.speculator.conftest import BLOCK_SIZE, MODEL_DIM, VOCAB_SIZE, make_draft_config

TREE_BUDGET = 24


@pytest.fixture(scope="module")
def weaver_speculator(sharding_config: ShardingConfig) -> WeaverSpeculator:
    draft_config = make_draft_config()
    weaver_config = WeaverConfig(
        d_model=MODEL_DIM,
        d_embed=MODEL_DIM,
        d_rank=16,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
        k=BLOCK_SIZE - 1,
        candidate_pool_size=16,
        linear_config=LinearConfig(),
        norm_config=NormalizationConfig(
            epsilon=1e-6,
            scale_offset=None,
            upcast_mode=UpcastMode.FULL_LAYER,
            subtract_mean=False,
            has_biases=True,
        ),
    )
    speculator_config = WeaverSpeculatorConfig(
        token_codec_config=RawTextCodecConfig(),
        draft_config=draft_config,
        weaver_config=weaver_config,
        tree_budget=TREE_BUDGET,
    )
    tokenizer = Tokenizer(WordLevel({f"tok{i}": i for i in range(VOCAB_SIZE)}, unk_token="tok0"))
    return WeaverSpeculator(
        config=speculator_config,
        sharding_config=sharding_config,
        token_codec=speculator_config.token_codec_config.init(tokenizer),
        draft_model=draft_config.init(
            RandomInitializer(
                default_dtype=jnp.float32,
                sharding_config=sharding_config,
                key=jax.random.key(1),
            ),
        ),
        weaver=weaver_config.init(
            RandomInitializer(
                default_dtype=jnp.float32,
                sharding_config=sharding_config,
                key=jax.random.key(2),
            ),
        ),
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
    speculator: WeaverSpeculator | None,
    generation_config: GenerationConfig,
    max_output_length: int = 16,
) -> tuple[jax.Array, jax.Array]:
    token_ids, lengths = make_ragged_batch()
    with jax.set_mesh(language_model.sharding_config.mesh):
        results = language_model.generate_tokens(
            token_ids,
            generation_config=generation_config,
            prompt_lengths_without_padding=lengths,
            max_output_length=max_output_length,
            keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
            speculator=speculator,
        )
    return results.token_ids, results.step_lengths


def test_weaver_greedy_matches_baseline(
    language_model: LanguageModel,
    weaver_speculator: WeaverSpeculator,
) -> None:
    config = GenerationConfig(temperature=0.0)
    baseline_tokens, _ = generate(language_model, None, config)
    weaver_tokens, step_lengths = generate(language_model, weaver_speculator, config)
    assert baseline_tokens.tolist() == weaver_tokens.tolist()
    assert step_lengths.sum(axis=1).tolist() == [16, 16, 16]


def test_weaver_greedy_matches_baseline_for_hybrid_model(
    hybrid_language_model: LanguageModel,
    weaver_speculator: WeaverSpeculator,
) -> None:
    config = GenerationConfig(temperature=0.0)
    baseline_tokens, _ = generate(hybrid_language_model, None, config)
    weaver_tokens, _ = generate(hybrid_language_model, weaver_speculator, config)
    assert baseline_tokens.tolist() == weaver_tokens.tolist()


def test_weaver_sampled_generation_is_well_formed(
    language_model: LanguageModel,
    weaver_speculator: WeaverSpeculator,
) -> None:
    config = GenerationConfig(temperature=1.0)
    weaver_tokens, step_lengths = generate(language_model, weaver_speculator, config)
    assert step_lengths.sum(axis=1).tolist() == [16, 16, 16]
    assert bool(jnp.all((weaver_tokens >= 0) & (weaver_tokens < VOCAB_SIZE)))


def test_weaver_draft_builds_valid_tree(
    language_model: LanguageModel,
    weaver_speculator: WeaverSpeculator,
) -> None:
    token_ids, lengths = make_ragged_batch()
    with jax.set_mesh(language_model.sharding_config.mesh):
        prefill = language_model.prefill_tokens(
            token_ids,
            64,
            lengths,
            None,
            keychain=Keychain.init(0, sharding_config=language_model.sharding_config),
            speculator=weaver_speculator,
        )
        batch_indices = jnp.arange(token_ids.shape[0], dtype=jnp.int32)
        last_token_ids = token_ids[batch_indices, lengths - 1]
        speculator_state = prefill.speculator_state
        assert isinstance(speculator_state, WeaverDraftState)
        proposal = weaver_speculator.draft(
            speculator_state,
            last_token_ids,
            prefill.last_token_indices,
            language_model.decoder.embedding,
            keychain=Keychain.init(3, sharding_config=language_model.sharding_config),
        )

    tokens = np.asarray(proposal.token_ids)
    parents = np.asarray(proposal.parent_indices)
    positions = np.asarray(proposal.token_positions)
    node_lengths = np.asarray(proposal.lengths)
    depth_limit = weaver_speculator.tree_depth
    last_indices = np.asarray(prefill.last_token_indices)

    assert tokens.shape == (3, TREE_BUDGET + 1)
    for row in range(tokens.shape[0]):
        num_nodes = int(node_lengths[row])
        assert 1 <= num_nodes <= TREE_BUDGET + 1
        assert parents[row, 0] == -1
        assert positions[row, 0] == last_indices[row] + 1
        for node in range(1, num_nodes):
            parent = int(parents[row, node])
            assert 0 <= parent < node
            assert positions[row, node] == positions[row, parent] + 1
            assert positions[row, node] - positions[row, 0] <= depth_limit
        assert bool(np.all((tokens[row, :num_nodes] >= 0) & (tokens[row, :num_nodes] < VOCAB_SIZE)))
