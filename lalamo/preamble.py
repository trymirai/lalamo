import os
import sys

from absl import flags, logging


def init_jax() -> None:
    # Must run before importing jax / tensorflow, this hides the XLA optimization logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    os.environ.setdefault("JAX_COMPILER_ENABLE_REMAT_PASS", "false")
    os.environ.setdefault("JAX_NUMPY_DTYPE_PROMOTION", "strict")

    # Tokamax lazily parses absl flags from sys.argv; pre-parse only argv[0] so pytest/Typer flags do not fail later.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(sys.argv[:1])

    # Disable tokamax logs
    logging.set_verbosity(logging.ERROR)
    logging.set_stderrthreshold("error")
