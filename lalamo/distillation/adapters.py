import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Key

from lalamo.compressed.hybrid import HybridMatrix, HybridSpec
from lalamo.compressed.low_rank import LowRankMatrix, LowRankSpec
from lalamo.modules import Decoder
from lalamo.utils.surgery import map_nodes_of_type, map_nodes_of_type_with_path, select_nodes_of_type
from lalamo.weight_matrix import FullPrecisionMatrix, Layout

__all__ = [
    "inject_low_rank_adapters",
    "partition_adapters",
]


def _with_zero_adapter(weight_matrix: FullPrecisionMatrix, rank: int, *, key: Key[Array, ""]) -> HybridMatrix:
    match weight_matrix.spec.layout:
        case Layout.OUTPUT_INPUT:
            output_dim, input_dim = weight_matrix.shape
        case Layout.INPUT_OUTPUT:
            input_dim, output_dim = weight_matrix.shape

    sharding_config = weight_matrix.sharding_config
    up_projection_axes = Layout.OUTPUT_INPUT.weight_partition(0, is_sharded=weight_matrix.is_sharded)
    # The adapter starts as an exact zero, so injecting it leaves the outputs of the model unchanged.
    up_projection = jnp.zeros((output_dim, rank), dtype=jnp.float32)
    down_projection = jax.random.normal(key, (rank, input_dim), dtype=jnp.float32) / jnp.sqrt(
        jnp.asarray(input_dim, dtype=jnp.float32),
    )
    adapter = LowRankMatrix(
        spec=LowRankSpec(rank=rank),
        sharding_config=sharding_config,
        is_sharded=weight_matrix.is_sharded,
        down_projection=jax.device_put(down_projection, sharding_config.resolve_sharding((None, None))),
        up_projection=jax.device_put(up_projection, sharding_config.resolve_sharding(up_projection_axes)),
    )
    return HybridMatrix(
        spec=HybridSpec(quantization_spec=weight_matrix.spec, adapter_spec=adapter.spec, incoherence_spec=None),
        sharding_config=sharding_config,
        is_sharded=weight_matrix.is_sharded,
        quantized=weight_matrix,
        adapter=adapter,
        incoherence_signs=None,
    )


def inject_low_rank_adapters(decoder: Decoder, rank: int, *, key: Key[Array, ""]) -> Decoder:
    """Adds a zero-initialized low-rank adapter to every full-precision matrix of the transformer.

    Embeddings are left alone: tied embeddings read their matrix transposed, which adapters do not support.
    """
    adapted_paths = [
        path
        for path, weight_matrix in select_nodes_of_type(FullPrecisionMatrix, decoder.transformer)
        if weight_matrix.ndim == 2
    ]
    if not adapted_paths:
        raise ValueError("The transformer has no full-precision matrices to adapt.")
    path_to_key = dict(zip(adapted_paths, jax.random.split(key, len(adapted_paths)), strict=True))

    def add_adapter(path: tuple[str, ...], weight_matrix: FullPrecisionMatrix) -> FullPrecisionMatrix | HybridMatrix:
        if path not in path_to_key:
            return weight_matrix
        return _with_zero_adapter(weight_matrix, rank, key=path_to_key[path])

    transformer = map_nodes_of_type_with_path(FullPrecisionMatrix, add_adapter, decoder.transformer)
    return eqx.tree_at(lambda module: module.transformer, decoder, transformer)


def partition_adapters(decoder: Decoder) -> tuple[Decoder, Decoder]:
    """Splits a decoder into its adapter weights, which are trained, and everything else, which stays frozen."""
    nothing_trainable = jtu.tree_map(lambda _: False, decoder)
    filter_spec = map_nodes_of_type(
        LowRankMatrix,
        lambda adapter: jtu.tree_map(lambda _: True, adapter),
        nothing_trainable,
    )
    return eqx.partition(decoder, filter_spec)
