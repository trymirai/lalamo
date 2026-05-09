import jax
import jax.numpy as jnp

from lalamo.preconditioner import Preconditioner
from tests.common import assert_close


def _assert_close(result: jax.Array, reference: jax.Array) -> None:
    assert_close(result=jnp.asarray(jax.device_get(result)), reference=jnp.asarray(jax.device_get(reference)))


def test_preconditioner_preserves_symmetric_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )

    preconditioner = Preconditioner.init(input_block=input_block, output_block=output_block)

    assert preconditioner.input_block_tril is not None
    assert preconditioner.input_block_tril.shape == (6,)
    assert preconditioner.output_block_tril is not None
    assert preconditioner.output_block_tril.shape == (3,)
    assert preconditioner.input_block is not None
    assert preconditioner.output_block is not None
    _assert_close(result=preconditioner.input_block, reference=input_block)
    _assert_close(result=preconditioner.output_block, reference=output_block)
    _assert_close(
        result=jnp.asarray(preconditioner.magnitude),
        reference=jnp.trace(input_block) * jnp.trace(output_block),
    )


def test_preconditioner_identity_has_no_blocks() -> None:
    preconditioner = Preconditioner.identity()

    assert preconditioner.input_block is None
    assert preconditioner.output_block is None
    _assert_close(result=preconditioner.magnitude, reference=jnp.asarray(1.0))


def test_preconditioner_magnitude_uses_only_present_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )

    input_preconditioner = Preconditioner.init(input_block=input_block)
    output_preconditioner = Preconditioner.init(output_block=output_block)

    _assert_close(result=jnp.asarray(input_preconditioner.magnitude), reference=jnp.trace(input_block))
    _assert_close(result=jnp.asarray(output_preconditioner.magnitude), reference=jnp.trace(output_block))


def test_preconditioner_supports_batched_symmetric_blocks() -> None:
    input_block = jnp.array(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 3.0, 4.0],
            [0.5, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    output_block = jnp.array(
        [
            [7.0, 2.0],
            [2.0, 11.0],
        ],
        dtype=jnp.float32,
    )
    input_blocks = jnp.stack([input_block, input_block + jnp.identity(3, dtype=jnp.float32)])
    output_blocks = jnp.stack([output_block, output_block + jnp.identity(2, dtype=jnp.float32)])

    preconditioner = Preconditioner.init(input_block=input_blocks, output_block=output_blocks)

    assert preconditioner.input_block_tril is not None
    assert preconditioner.input_block_tril.shape == (2, 6)
    assert preconditioner.output_block_tril is not None
    assert preconditioner.output_block_tril.shape == (2, 3)
    assert preconditioner.input_block is not None
    assert preconditioner.output_block is not None
    _assert_close(result=preconditioner.input_block, reference=input_blocks)
    _assert_close(result=preconditioner.output_block, reference=output_blocks)
    _assert_close(
        result=jnp.asarray(preconditioner.magnitude),
        reference=jnp.mean(
            jnp.trace(input_blocks, axis1=-2, axis2=-1) * jnp.trace(output_blocks, axis1=-2, axis2=-1),
        ),
    )
