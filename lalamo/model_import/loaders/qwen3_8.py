from collections.abc import Mapping
from itertools import accumulate

import jax
import numpy as np
from jaxtyping import Array

from lalamo.modules.gated_residual import GatedResidual
from lalamo.modules.ngram import NGramInjection
from lalamo.modules.token_mixers.qsa import QwenSparseAttention
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import array_from_host
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation, FullPrecisionMatrix, FullPrecisionSpec, Layout

from .huggingface import load_attention, load_linear, load_rmsnorm


def load_gated_residual(
    module: GatedResidual,
    weights: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,
) -> GatedResidual:
    return load_as_at(
        lambda m: (m.norm, m.down, m.up, m.inject),
        module,
        (
            load_rmsnorm(module.norm, weights, path / "hc_norm"),
            load_linear(module.down, weights, path / "input_mix_weight_down", implementation=implementation),
            load_linear(module.up, weights, path / "input_mix_weight_up", implementation=implementation),
            load_linear(module.inject, weights, path / "block_inject_weight", implementation=implementation)
            if module.inject is not None
            else None,
        ),
    )


def load_sparse_attention(
    module: QwenSparseAttention,
    weights: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,
) -> QwenSparseAttention:
    indexer_path = path / "indexer"
    return load_as_at(
        lambda m: (m.attention, m.indexer_projection, m.query_norm, m.key_norm),
        module,
        (
            load_attention(module.attention, weights, path, implementation=implementation),
            load_linear(
                module.indexer_projection, weights, indexer_path / "index_qk_proj", implementation=implementation
            ),
            load_rmsnorm(module.query_norm, weights, indexer_path / "q_layernorm"),
            load_rmsnorm(module.key_norm, weights, indexer_path / "k_layernorm"),
        ),
    )


def load_ngram(
    module: NGramInjection,
    weights: Mapping[str, Array],
    path: ParameterPath,
    *,
    parts: int,
    implementation: CompressionImplementation,
) -> NGramInjection:
    embedding_path = path / "ple_embedding"
    # Checkpoint stores signed int64 hashes; JAX normally narrows these at the file boundary.
    with jax.enable_x64():
        for name, expected in (
            ("layer_multipliers", module.config.multipliers),
            ("ngram_heads_vocab_sizes", module.config.head_vocab_sizes),
            ("ngram_heads_offsets", tuple(accumulate(module.config.head_vocab_sizes, initial=0))[:-1]),
        ):
            if not np.array_equal(np.asarray(weights[embedding_path / name]), expected):
                raise ValueError(f"Checkpoint N-gram {name} differs from the configured Qwen hash.")
    # Assemble on the host and place only final shards, avoiding a GPU-sized table intermediate.
    table = np.concatenate(
        [
            np.asarray(weights[embedding_path / "ngram_embedding" / f"shard_{index}" / "weight"])
            for index in range(parts)
        ],
        axis=0,
    )
    if table.shape != module.embedding.shape:
        raise ValueError(f"N-gram table shape {table.shape} differs from {module.embedding.shape}.")
    sharding = module.embedding.sharding_config.resolve_sharding(
        Layout.INPUT_OUTPUT.weight_partition(0, is_sharded=module.embedding.is_sharded)
    )
    embedding = FullPrecisionMatrix(
        spec=FullPrecisionSpec(Layout.INPUT_OUTPUT),
        sharding_config=module.embedding.sharding_config,
        is_sharded=module.embedding.is_sharded,
        weights=array_from_host(table, sharding, dtype=module.embedding.dtype),
    )
    return load_as_at(
        lambda m: (m.embedding, m.key, m.value, m.key_norm, m.query_norm, m.conv_norm, m.conv_weights),
        module,
        (
            embedding,
            load_linear(module.key, weights, path / "key_proj", implementation=implementation),
            load_linear(module.value, weights, path / "value_proj", implementation=implementation),
            load_rmsnorm(module.key_norm, weights, path / "norm_key"),
            load_rmsnorm(module.query_norm, weights, path / "norm_query"),
            load_rmsnorm(module.conv_norm, weights, path / "norm_conv"),
            weights[path / "conv1d" / "weight"][:, 0, :],
        ),
    )
