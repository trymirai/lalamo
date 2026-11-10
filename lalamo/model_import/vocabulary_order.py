import json
import warnings
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from lalamo.models import LanguageModel
from lalamo.modules.embedding import EmbeddingBase, TiedEmbedding, UntiedEmbedding
from lalamo.weight_matrix import FullPrecisionMatrix, WeightMatrix

__all__ = ["frequency_ordered"]


def training_mixture_counts(tokenizer: Tokenizer, vocab_size: int) -> np.ndarray:
    """Token counts over the message texts of one trymirai/training-mixture shard."""
    shard = hf_hub_download("trymirai/training-mixture", "train-00000-of-00256.parquet", repo_type="dataset")
    with warnings.catch_warnings():
        # The tool call fields are JSON extension columns, which polars warns about; only the texts are read.
        warnings.filterwarnings("ignore", message="Extension type 'arrow.json' is not registered")
        messages = pl.read_parquet(shard, columns=["messages"])["messages"].explode().struct.unnest()
    texts = pl.concat([messages["content"], messages["reasoning"]]).drop_nulls().to_list()
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    token_ids = np.fromiter((token_id for encoding in encodings for token_id in encoding.ids), dtype=np.int64)
    return np.bincount(token_ids, minlength=vocab_size)


def frequency_order(counts: np.ndarray, tokenizer: Tokenizer) -> np.ndarray:
    """New id -> old id. Added tokens come first: chat templates emit them (`<|im_end|>`, `<think>`, ...) but message
    texts rarely contain them. Then the other tokens by count. Padding rows past the tokenizer keep their ids, so ids
    that only the model uses (such as a DFlash mask) stay valid.
    """
    num_tokens = tokenizer.get_vocab_size(with_added_tokens=True)
    assert num_tokens <= len(counts), f"The tokenizer has {num_tokens} tokens, the model only {len(counts)}"
    is_added = np.zeros(num_tokens, dtype=bool)
    is_added[list(tokenizer.get_added_tokens_decoder())] = True
    token_order = np.lexsort((np.arange(num_tokens), -counts[:num_tokens], ~is_added))
    return np.concatenate([token_order, np.arange(num_tokens, len(counts))])


def reorder_post_processor(post_processor: dict | None, new_ids: np.ndarray) -> dict | None:
    if post_processor is None:
        return None
    match post_processor["type"]:
        case "ByteLevel":
            return post_processor
        case "TemplateProcessing":
            special_tokens = {
                name: {**token, "ids": [int(new_ids[i]) for i in token["ids"]]}
                for name, token in post_processor["special_tokens"].items()
            }
            return {**post_processor, "special_tokens": special_tokens}
        case "Sequence":
            processors = [reorder_post_processor(processor, new_ids) for processor in post_processor["processors"]]
            return {**post_processor, "processors": processors}
        case unknown:
            raise ValueError(f"Cannot reorder the token ids of a {unknown} post-processor")


def reorder_tokenizer(tokenizer: Tokenizer, new_ids: np.ndarray) -> Tokenizer:
    content = json.loads(tokenizer.to_str())
    model = content["model"]
    assert model["type"] in ("BPE", "WordPiece", "WordLevel"), f"Cannot reorder a {model['type']} vocabulary"
    added_tokens = [{**token, "id": int(new_ids[token["id"]])} for token in content["added_tokens"]]
    # `tokenizers` numbers added tokens after the model vocabulary unless the vocabulary lists them.
    vocab = {piece: int(new_ids[i]) for piece, i in model["vocab"].items()}
    vocab |= {token["content"]: token["id"] for token in added_tokens}
    content |= {
        "model": {**model, "vocab": vocab},
        "added_tokens": added_tokens,
        "post_processor": reorder_post_processor(content["post_processor"], new_ids),
    }
    return Tokenizer.from_str(json.dumps(content))


def reorder_matrix(matrix: WeightMatrix, order: np.ndarray, vocabulary_axis: int) -> FullPrecisionMatrix:
    # Permuting a quantized matrix would requantize it, so only full precision vocabularies are reordered.
    assert isinstance(matrix, FullPrecisionMatrix), f"Cannot reorder the vocabulary of a {type(matrix).__name__}"
    # On the host: the vocabulary axis may be sharded, and a sharded gather needs an explicit output sharding.
    weights = jnp.asarray(np.take(np.asarray(matrix.decompress()), order, axis=vocabulary_axis))
    return matrix.spec.compress(weights, sharding_config=matrix.sharding_config, is_sharded=matrix.is_sharded)


def reorder_embedding(embedding: EmbeddingBase, order: np.ndarray) -> TiedEmbedding | UntiedEmbedding:
    # Embeddings are (channels, vocabulary) in output-input terms; an untied readout is (vocabulary, channels).
    config = replace(embedding.config, permute_indices=True)
    index_permutation = jnp.asarray(order, dtype=jnp.int32)
    match embedding:
        case TiedEmbedding():
            matrix = reorder_matrix(embedding.embedding, order, -1)
            return replace(embedding, config=config, embedding=matrix, index_permutation=index_permutation)
        case UntiedEmbedding():
            return replace(
                embedding,
                config=config,
                input_embedding=reorder_matrix(embedding.input_embedding, order, -1),
                output_embedding=reorder_matrix(embedding.output_embedding, order, -2),
                index_permutation=index_permutation,
            )
        case unknown:
            raise ValueError(f"Cannot reorder the vocabulary of a {type(unknown).__name__}")


def frequency_ordered(model: LanguageModel) -> LanguageModel:
    """The same model with the most frequent tokens renumbered to the smallest ids."""
    assert not model.config.decoder_config.embedding_config.permute_indices
    assert model.decoder.per_layer_embedding is None, "Per-layer embeddings have their own vocabulary"
    tokenizer = model.token_codec.tokenizer
    order = frequency_order(training_mixture_counts(tokenizer, model.decoder.vocab_size), tokenizer)
    new_ids = np.argsort(order)
    generation_config = model.config.generation_config
    stop_token_ids = tuple(int(new_ids[i]) for i in generation_config.stop_token_ids)
    generation_config = replace(generation_config, stop_token_ids=stop_token_ids)
    if generation_config.banned_tokens is not None:
        banned_tokens = tuple(int(new_ids[i]) for i in generation_config.banned_tokens)
        generation_config = replace(generation_config, banned_tokens=banned_tokens)
    embedding = reorder_embedding(model.decoder.embedding, order)
    decoder_config = replace(model.config.decoder_config, embedding_config=embedding.config)
    config = replace(model.config, generation_config=generation_config, decoder_config=decoder_config)
    return replace(
        model,
        config=config,
        token_codec=config.token_codec_config.init(reorder_tokenizer(tokenizer, new_ids)),
        decoder=replace(model.decoder, config=decoder_config, embedding=embedding),
    )
