import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.compressed import AWQMatrix, AWQSpec, LoRAMatrix, LoRASpec, MLXMatrix, MLXSpec
from lalamo.exportable import Exportable
from lalamo.initializer import EmptyInitializer
from lalamo.weight_matrix import Layout, WeightMatrix, WeightMatrixSpec
from tests.common import assert_close


class MatrixContainer(Exportable, eqx.Module):
    layer: WeightMatrix


def test_awq_spec_json_round_trip() -> None:
    spec = AWQSpec(bits=4, group_size=4, float_dtype=jnp.float32, layout=Layout.INPUT_OUTPUT)
    restored = WeightMatrixSpec.from_json(spec.to_json())
    assert restored == spec


def test_mlx_spec_json_round_trip() -> None:
    spec = MLXSpec(bits=8, group_size=4, float_dtype=jnp.float32, layout=Layout.OUTPUT_INPUT)
    restored = WeightMatrixSpec.from_json(spec.to_json())
    assert restored == spec


def test_awq_export_round_trip() -> None:
    spec = AWQSpec(bits=4, group_size=4, float_dtype=jnp.float32)
    original = MatrixContainer(layer=spec.compress(jax.random.normal(jax.random.key(0), (4, 8))))
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(mesh=None, dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, AWQMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_mlx_export_round_trip() -> None:
    spec = MLXSpec(bits=4, group_size=4, float_dtype=jnp.float32)
    original = MatrixContainer(layer=spec.compress(jax.random.normal(jax.random.key(1), (4, 8))))
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(mesh=None, dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, MLXMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_lora_export_round_trip() -> None:
    spec = LoRASpec(rank=2)
    original = MatrixContainer(
        layer=LoRAMatrix(
            spec=spec,
            down=jnp.ones((4, 2)),
            up=jnp.arange(16, dtype=jnp.float32).reshape(2, 8),
        ),
    )
    skeleton = MatrixContainer(layer=spec.init(EmptyInitializer(mesh=None, dtype=jnp.float32), (), 4, 8))

    restored = skeleton.load_exported(original.export())

    assert isinstance(restored.layer, LoRAMatrix)
    assert_close(result=restored.layer.decompress(), reference=original.layer.decompress())


def test_awq_input_output_layout() -> None:
    weights = jax.random.normal(jax.random.key(2), (5, 8))
    matrix = AWQSpec(bits=4, group_size=4, float_dtype=jnp.float32, layout=Layout.INPUT_OUTPUT).compress(weights)
    token_index = 3
    vector = jax.random.normal(jax.random.key(3), (8,))

    assert_close(
        result=matrix.lookup_embedding(token_index, dequant_key=jax.random.key(4)),
        reference=matrix.decompress()[:, token_index],
    )
    assert_close(
        result=matrix.dot(vector, dequant_key=jax.random.key(5)),
        reference=vector @ matrix.decompress(),
    )


def test_mlx_input_output_layout() -> None:
    weights = jax.random.normal(jax.random.key(6), (5, 8))
    matrix = MLXSpec(bits=4, group_size=4, float_dtype=jnp.float32, layout=Layout.INPUT_OUTPUT).compress(weights)
    token_index = 1
    vector = jax.random.normal(jax.random.key(7), (8,))

    assert_close(
        result=matrix.lookup_embedding(token_index, dequant_key=jax.random.key(8)),
        reference=matrix.decompress()[:, token_index],
    )
    assert_close(
        result=matrix.dot(vector, dequant_key=jax.random.key(9)),
        reference=vector @ matrix.decompress(),
    )
