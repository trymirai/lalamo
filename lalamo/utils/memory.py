import functools
import os

import jax

__all__ = [
    "get_available_bytes_on_default_device",
    "get_usable_bytes_from_available_bytes",
]


@functools.cache
def get_available_bytes_on_default_device() -> int | None:
    dynamic_allocate = False

    preallocate = os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE", "")
    dynamic_allocate |= preallocate.strip().lower() in {"0", "false", "no", "off"}

    allocator = os.getenv("XLA_PYTHON_CLIENT_ALLOCATOR", "")
    dynamic_allocate |= allocator.strip().lower() in {"platform", "cuda_malloc_async"}

    if dynamic_allocate:
        return None

    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or "bytes_limit" not in memory_stats:
        return None

    # 500mb is seemingly the usually observed overhead
    return memory_stats["bytes_limit"] - (500 * 1000 * 1000)


def get_usable_bytes_from_available_bytes(available_bytes: int) -> int:
    return int(available_bytes * 0.95)
