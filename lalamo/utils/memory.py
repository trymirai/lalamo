import ctypes
import functools
import os
from importlib.resources import files

import jax

__all__ = [
    "get_available_bytes_on_default_device",
    "get_free_bytes",
    "get_usable_bytes_from_available_bytes",
]


def get_free_bytes(device: jax.Device) -> int | None:
    stats = device.memory_stats()
    if stats is not None and stats.get("bytes_limit", 0) > 0:
        return stats["bytes_limit"] - stats["bytes_in_use"]
    if device.platform != "gpu":
        return None
    # PJRT prefixes the runtime version with its API name. cuda_async reports no allocator limit, so query CUDA.
    platform, version = device.client.platform_version.splitlines()[-1].split()
    if platform != "cuda":
        return None
    library_name = f"libcudart.so.{int(version) // 1000}"
    (library,) = (
        package / "lib" / library_name
        for package in files("nvidia").iterdir()
        if (package / "lib" / library_name).is_file()
    )
    runtime = ctypes.CDLL(str(library))
    runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    runtime.cudaGetDevice.restype = runtime.cudaSetDevice.restype = runtime.cudaMemGetInfo.restype = ctypes.c_int
    current_device = ctypes.c_int()
    if error := runtime.cudaGetDevice(ctypes.byref(current_device)):
        raise RuntimeError(f"cudaGetDevice failed with CUDA error {error}.")
    try:
        if error := runtime.cudaSetDevice(device.local_hardware_id):
            raise RuntimeError(f"cudaSetDevice failed with CUDA error {error}.")
        free_bytes, total_bytes = ctypes.c_size_t(), ctypes.c_size_t()
        if error := runtime.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)):
            raise RuntimeError(f"cudaMemGetInfo failed with CUDA error {error}.")
        return free_bytes.value
    finally:
        if error := runtime.cudaSetDevice(current_device.value):
            raise RuntimeError(f"Restoring the CUDA device failed with CUDA error {error}.")


@functools.cache
def get_available_bytes_on_default_device() -> int | None:
    dynamic_allocate = False

    preallocate = os.getenv("XLA_PYTHON_CLIENT_PREALLOCATE", "")
    dynamic_allocate |= preallocate.strip().lower() in {"0", "false", "no", "off"}

    allocator = os.getenv("XLA_PYTHON_CLIENT_ALLOCATOR", "")
    dynamic_allocate |= allocator.strip().lower() in {"platform", "cuda_async"}

    if dynamic_allocate:
        return None

    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or memory_stats.get("bytes_limit", 0) <= 0:
        return None

    # 500mb is seemingly the usually observed overhead
    return memory_stats["bytes_limit"] - (500 * 1000 * 1000)


def get_usable_bytes_from_available_bytes(available_bytes: int) -> int:
    return int(available_bytes * 0.95)
