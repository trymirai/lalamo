import jax
import jax.numpy as jnp
import pytest
from jaxtyping import DTypeLike

from lalamo.compressed import quant_params
from lalamo.compressed.utils.packing import pack_uint_to_uint8, packed_last_axis_dim
from lalamo.weight_matrix import Layout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


@pytest.mark.parametrize(
    ("bits", "dtype"),
    [(4, jnp.uint8), (8, jnp.uint8), (16, jnp.bfloat16), (32, jnp.float32)],
)
@pytest.mark.parametrize(
    ("layout", "columns", "stored_shape"),
    [
        (Layout.OUTPUT_INPUT, 4, (3, 4)),
        (Layout.OUTPUT_INPUT, 5, (3, 8)),
        (Layout.INPUT_OUTPUT, 5, (5, 3)),
    ],
)
def test_storage_roundtrip(
    bits: int, dtype: DTypeLike, layout: Layout, columns: int, stored_shape: tuple[int, int]
) -> None:
    sharding_config = make_test_sharding_config()
    groups = 3
    source = jnp.arange(columns * groups, dtype=dtype).reshape(columns, groups)
    source = jax.device_put(source, sharding_config.resolve_sharding((None, None)))
    packed = bits in (4, 8)
    if packed:
        source = pack_uint_to_uint8(source, bits, sharding_config=sharding_config)

    stored = quant_params.for_export(
        source,
        shape=(columns, groups),
        layout=layout,
        bits=bits,
        sharding_config=sharding_config,
    )
    restored = quant_params.from_export(
        stored,
        like=source,
        shape=(columns, groups),
        layout=layout,
        bits=bits,
        sharding_config=sharding_config,
    )

    stored_rows, stored_columns = stored_shape
    if packed:
        stored_columns = packed_last_axis_dim(stored_columns, bits)
    assert stored.shape[-2:] == (stored_rows, stored_columns)
    assert jnp.array_equal(restored, source)

    if layout == Layout.OUTPUT_INPUT and bits == 4 and columns == 5:
        partial_byte = packed_last_axis_dim(columns, bits) - 1
        unused_nibble_mask = ((1 << bits) - 1) << bits
        assert jnp.all((stored[..., partial_byte] & unused_nibble_mask) == 0)
        assert jnp.all(stored[..., partial_byte + 1 :] == 0)
