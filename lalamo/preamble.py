import os


def init_jax() -> None:
    # Must run before importing jax / tensorflow, this hides the XLA optimization logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    os.environ.setdefault("JAX_COMPILER_ENABLE_REMAT_PASS", "false")
    os.environ.setdefault("JAX_NUMPY_DTYPE_PROMOTION", "strict")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".95")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
