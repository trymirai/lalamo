from collections.abc import Callable

import jax
from jax.lax import DotAlgorithmPreset
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array

from lalamo.utils.sharding import sharding_of, with_sharding


def row_batched_dot(
    decode_row: Callable[[tuple[Array, Array, Array]], Array],
    rows: tuple[Array, Array, Array],
    vector: Array,
    precision: DotAlgorithmPreset,
) -> Array:
    sharding = sharding_of(vector)
    # FSDP shares the batch and matrix axis. Gather packed rows, so token
    # batching owns that axis; only 128 rows are ever decoded at once.
    replicated_rows = tuple(
        with_sharding(array, NamedSharding(sharding.mesh, PartitionSpec(*((None,) * array.ndim)))) for array in rows
    )

    def dot(row: tuple[Array, Array, Array]) -> Array:
        return jax.lax.dot_general(
            decode_row(row).astype(vector.dtype),
            vector,
            dimension_numbers=(((0,), (0,)), ((), ())),
            precision=precision,
            out_sharding=NamedSharding(sharding.mesh, PartitionSpec()),
        )

    return jax.lax.map(dot, replicated_rows, batch_size=128)
