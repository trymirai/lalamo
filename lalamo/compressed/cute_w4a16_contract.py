from functools import cache
from importlib.util import find_spec

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array

from lalamo.weight_matrix import Layout, MatmulConfig

GROUP_SIZE = 32
REDUCE_THREADS = 32
MICRO_VALUES = 16
MICRO_PACKED = MICRO_VALUES // 2
CHANNEL_MULTIPLE = REDUCE_THREADS * MICRO_VALUES
MLX_ROWS_PER_CTA = 16
AWQ_ROWS_PER_CTA = 4
PACKED_PAIR_PREFILL_ROWS = 4096


def use_packed_w4a16_route(batch: int, rows: int) -> bool:
    return batch == 2 and rows == PACKED_PAIR_PREFILL_ROWS


@cache
def cute_w4a16_runtime_available() -> bool:
    return (
        find_spec("cutlass") is not None
        and find_spec("cuda") is not None
        and find_spec("cuda.bindings") is not None
        and find_spec("cuda.bindings.driver") is not None
    )


def can_use_cute_w4a16_dot(
    *,
    bits: int,
    group_size: int,
    layout: Layout,
    row_count: int,
    row_multiple: int,
    vector: Array,
    forward_pass_config: MatmulConfig,
    transposed: bool,
) -> bool:
    return (
        jax.default_backend() == "gpu"
        and cute_w4a16_runtime_available()
        and bits == 4
        and group_size == GROUP_SIZE
        and layout == Layout.OUTPUT_INPUT
        and vector.ndim == 1
        and vector.dtype in (jnp.float16, jnp.bfloat16)
        and vector.shape[0] % CHANNEL_MULTIPLE == 0
        and row_count % row_multiple == 0
        and forward_pass_config.precision == DotAlgorithmPreset.DEFAULT
        and not transposed
    )
