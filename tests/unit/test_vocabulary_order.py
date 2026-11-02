import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from lalamo.initializer import RandomInitializer
from lalamo.model_import.vocabulary_order import frequency_order, reorder_embedding, reorder_tokenizer
from lalamo.module import Keychain
from lalamo.modules.embedding import EmbeddingConfig, TiedEmbeddingConfig, UntiedEmbeddingConfig
from tests.helpers import make_test_sharding_config


def _tokenizer() -> Tokenizer:
    pieces = ["a", "b", "c", "ab", "abc", "id"]
    tokenizer = Tokenizer(BPE({piece: i for i, piece in enumerate(pieces)}, [("a", "b"), ("ab", "c")]))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<eos>"])
    eos = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = TemplateProcessing(single="$A <eos>", special_tokens=[("<eos>", eos)])
    return tokenizer


def test_reordered_tokenizer_encodes_to_renumbered_ids() -> None:
    tokenizer = _tokenizer()  # 7 tokens; the 8th row is padding
    order = frequency_order(np.array([0, 3, 0, 0, 7, 7, 1, 9]), tokenizer)
    # <eos> (id 6) is an added token, then tokens by count with ties in id order; padding id 7 stays despite its count.
    assert order.tolist() == [6, 4, 5, 1, 0, 2, 3, 7]
    new_ids = np.argsort(order)
    reordered = reorder_tokenizer(tokenizer, new_ids)
    text = "abc id ab<eos>"
    original_ids = tokenizer.encode(text).ids
    assert reordered.encode(text).ids == [int(new_ids[i]) for i in original_ids]
    assert reordered.decode(reordered.encode(text).ids, skip_special_tokens=False) == tokenizer.decode(
        original_ids,
        skip_special_tokens=False,
    )


@pytest.mark.parametrize(
    "config",
    [
        TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
        UntiedEmbeddingConfig(input_scale=None, logit_soft_cap=None),
    ],
)
def test_reordered_embedding_computes_the_same_function(config: EmbeddingConfig) -> None:
    vocab_size, model_dim = 8, 4
    sharding_config = make_test_sharding_config()
    initializer = RandomInitializer(default_dtype=jnp.float32, sharding_config=sharding_config, key=jax.random.key(0))
    embedding = config.init(initializer, model_dim=model_dim, vocab_size=vocab_size)
    order = np.array([6, 4, 5, 1, 0, 2, 3, 7])
    reordered = reorder_embedding(embedding, order)
    keychain = Keychain.init(0, sharding_config=sharding_config)

    assert np.array_equal(
        reordered.embed(jnp.arange(vocab_size), keychain=keychain),
        embedding.embed(jnp.asarray(order), keychain=keychain),
    )
    hidden = jnp.arange(model_dim, dtype=jnp.float32)
    assert np.array_equal(
        reordered.readout(hidden, keychain=keychain),
        np.asarray(embedding.readout(hidden, keychain=keychain))[order],
    )
