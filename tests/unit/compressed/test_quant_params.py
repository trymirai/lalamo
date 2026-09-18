import jax
import jax.numpy as jnp
import pytest

from lalamo.compressed import quant_params
from lalamo.compressed.utils.packing import pack_uint_to_uint8
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import QuantParamsLayout
from tests.helpers import make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _shard(array: jax.Array, sharding_config: ShardingConfig) -> jax.Array:
    return jax.device_put(array, sharding_config.resolve_sharding((None,) * array.ndim))


@pytest.mark.parametrize("bits", [4, 8, 16, 32])
@pytest.mark.parametrize("columns", [4, 5])
def test_params_layout_roundtrip_with_padding(bits: int, columns: int) -> None:
    sharding_config = make_test_sharding_config()
    groups = 3
    if bits in (4, 8):
        logical = jnp.arange(columns * groups, dtype=jnp.uint8).reshape(columns, groups)
        source = pack_uint_to_uint8(
            _shard(logical, sharding_config),
            bits,
            sharding_config=sharding_config,
        )
    else:
        dtype = jnp.bfloat16 if bits == 16 else jnp.float32
        source = jnp.arange(columns * groups, dtype=dtype).reshape(columns, groups)
    source = _shard(source, sharding_config)

    group_output = quant_params.for_export(
        source,
        shape=(columns, groups),
        layout=QuantParamsLayout.GROUP_OUTPUT,
        bits=bits,
        sharding_config=sharding_config,
    )
    stride = (columns + 3) // 4 * 4
    expected_width = stride if bits in (16, 32) else (stride * bits + 7) // 8
    assert group_output.shape[-2:] == (groups, expected_width)

    restored = quant_params.from_export(
        group_output,
        like=source,
        shape=(columns, groups),
        layout=QuantParamsLayout.GROUP_OUTPUT,
        bits=bits,
        sharding_config=sharding_config,
    )
    assert jnp.array_equal(restored, source)

    if bits == 4 and columns % 4:
        first_partial_byte = columns // 2
        assert jnp.all((group_output[..., first_partial_byte] & 0xF0) == 0)
        assert jnp.all(group_output[..., first_partial_byte + 1 :] == 0)
