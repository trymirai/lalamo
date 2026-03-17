from collections.abc import Iterator
from contextlib import ExitStack

import jax
from pytest import fixture


@fixture(autouse=True)
def _stabilize_qwen3_jax_runtime() -> Iterator[None]:
    cpu = jax.devices("cpu")[0]
    with ExitStack() as stack:
        stack.enter_context(jax.default_device(cpu))
        yield
