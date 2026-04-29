import os


def init_jax() -> None:
    # Must run before importing jax / tensorflow, this hides the XLA optimization logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Persistent JAX compilation cache. Any of these can be overridden by exporting the env var beforehand
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.01")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
