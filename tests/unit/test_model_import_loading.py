from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.int import IntMatrixForInference, IntMatrixForTraining, IntSpec
from lalamo.compressed.mlx import MLXMatrixForInference, MLXMatrixForTraining
from lalamo.initializer import EmptyInitializer, Initializer
from lalamo.model import Model, ModelConfig
from lalamo.model_import.loaders.huggingface import load_linear
from lalamo.model_import.model_configs.foreign_config import ForeignConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation, FullPrecisionSpec, Layout, WeightMatrix
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")


def _pack_int32(values: Array, bits: int) -> Array:
    values_per_word = 32 // bits
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // values_per_word, values_per_word).astype(jnp.uint32)
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(bits)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint32).astype(jnp.int32)


def _linear_template(dtype: DTypeLike, layout: Layout = Layout.OUTPUT_INPUT) -> Linear:
    result = LinearConfig().init(
        initializer=EmptyInitializer(dtype=dtype, sharding_config=make_test_sharding_config()),
        input_dim=4,
        output_dims=(4,),
        has_biases=False,
    )
    if layout == Layout.OUTPUT_INPUT:
        return result

    weights = FullPrecisionSpec(layout=layout).compress(
        dummy_array((4, 4), dtype, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )
    return eqx.tree_at(lambda module: module.weights, result, weights)


def _mlx_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    return {
        path / "weight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((4, 2), dtype=jnp.float32),
        path / "biases": jnp.zeros((4, 2), dtype=jnp.float32),
    }


def _int_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(16, dtype=jnp.int32).reshape(4, 4)
    unpacked_zero_points = jnp.zeros((2, 4), dtype=jnp.int32)
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "qzeros": _pack_int32(unpacked_zero_points, bits=8),
        path / "scales": jnp.ones((2, 4), dtype=jnp.float32),
    }


def _symmetric_int_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(16, dtype=jnp.int32).reshape(4, 4) + 128
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((2, 4), dtype=jnp.float32),
    }


@pytest.mark.parametrize(
    ("weights_factory", "implementation", "expected_type"),
    [
        (_mlx_weights, CompressionImplementation.INFERENCE, MLXMatrixForInference),
        (_mlx_weights, CompressionImplementation.TRAINING, MLXMatrixForTraining),
        (_int_weights, CompressionImplementation.INFERENCE, IntMatrixForInference),
        (_int_weights, CompressionImplementation.TRAINING, IntMatrixForTraining),
    ],
)
@pytest.mark.parametrize("template_layout", [Layout.OUTPUT_INPUT, Layout.INPUT_OUTPUT])
def test_load_linear_quantized_checkpoint_uses_requested_dtype_and_implementation(
    weights_factory: Callable[[ParameterPath], Mapping[str, Array]],
    implementation: CompressionImplementation,
    expected_type: type[MLXMatrixForInference | MLXMatrixForTraining | IntMatrixForInference | IntMatrixForTraining],
    template_layout: Layout,
) -> None:
    path = ParameterPath("layer")

    loaded = load_linear(
        _linear_template(jnp.bfloat16, layout=template_layout),
        weights_factory(path),
        path,
        implementation=implementation,
    )

    assert isinstance(loaded.weights, expected_type)
    assert loaded.weights.spec.layout == Layout.OUTPUT_INPUT
    assert loaded.weights.dtype == jnp.bfloat16


def test_load_linear_symmetric_int_without_qzeros_uses_symmetric_spec() -> None:
    path = ParameterPath("layer")

    loaded = load_linear(
        _linear_template(jnp.bfloat16),
        _symmetric_int_weights(path),
        path,
        implementation=CompressionImplementation.INFERENCE,
    )

    assert isinstance(loaded.weights, IntMatrixForInference)
    assert loaded.weights.spec.is_symmetric
    assert loaded.weights.packed_zero_points is None


@dataclass(frozen=True)
class TinyConfig(LalamoConfig):
    def init(self, initializer: Initializer) -> "TinyModule":
        return TinyModule(
            config=self,
            sharding_config=make_test_sharding_config(),
            matrix=initializer.weight_matrix(output_dim=4, input_dim=4),
        )


class TinyModule(LalamoModule[TinyConfig]):
    matrix: WeightMatrix


@dataclass(frozen=True)
class TinyModelConfig(ModelConfig[ChatCodecConfig]):
    module_config: TinyConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "TinyModel":
        return TinyModel(
            config=self,
            sharding_config=make_test_sharding_config(),
            token_codec=self.token_codec_config.init(tokenizer),
            module=self.module_config.init(initializer),
        )


class TinyModel(Model[ChatCodecConfig, TinyModelConfig, ChatCodec]):
    token_codec: ChatCodec
    module: TinyModule


@dataclass(frozen=True)
class TinyForeignConfig(ForeignConfig[TinyModelConfig]):
    @property
    def default_dtype(self) -> DTypeLike:
        return jnp.float32

    def _load_weights(
        self,
        model: Model,
        weights_dict: Mapping[str, Array],
        *,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    ) -> Model:
        assert isinstance(model, TinyModel)
        return TinyModel(
            config=model.config,
            sharding_config=make_test_sharding_config(),
            token_codec=model.token_codec,
            module=TinyModule(
                config=model.module.config,
                sharding_config=make_test_sharding_config(),
                matrix=IntSpec(bits=4, group_size=2).compress(
                    weights_dict["matrix"].astype(model.module.matrix.dtype),
                    implementation=implementation,
                    sharding_config=make_test_sharding_config(),
                ),
            ),
        )


def test_foreign_config_load_initializes_model_with_requested_dtype_and_implementation() -> None:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))

    loaded = TinyForeignConfig().load(
        config=TinyModelConfig(
            token_codec_config=ChatCodecConfig(
                prompt_template="",
                output_parser_regex=None,
                system_role_name="system",
                user_role_name="user",
                assistant_role_name="assistant",
                eos_token=None,
                bos_token=None,
            ),
            module_config=TinyConfig(),
        ),
        tokenizer=tokenizer,
        dtype=jnp.bfloat16,
        weights_dict={"matrix": jnp.arange(16, dtype=jnp.float32).reshape(4, 4)},
        implementation=CompressionImplementation.TRAINING,
        sharding_config=make_test_sharding_config(),
    )

    assert isinstance(loaded, TinyModel)
    assert loaded.token_codec.tokenizer is tokenizer
    assert isinstance(loaded.module.matrix, IntMatrixForTraining)
    assert loaded.module.matrix.dtype == jnp.bfloat16


def test_model_export_load_uses_saved_weight_matrix_spec_with_shape_dtype_template(tmp_path: Path) -> None:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    config = TinyModelConfig(
        token_codec_config=ChatCodecConfig(
            prompt_template="",
            output_parser_regex=None,
            system_role_name="system",
            user_role_name="user",
            assistant_role_name="assistant",
            eos_token=None,
            bos_token=None,
        ),
        module_config=TinyConfig(),
    )
    original = TinyModel(
        config=config,
        sharding_config=make_test_sharding_config(),
        token_codec=config.token_codec_config.init(tokenizer),
        module=TinyModule(
            config=TinyConfig(),
            sharding_config=make_test_sharding_config(),
            matrix=IntSpec(bits=8, group_size=2).compress(
                jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
                sharding_config=make_test_sharding_config(),
            ),
        ),
    )

    original.save(tmp_path)
    restored = TinyModel.load(tmp_path, sharding_config=make_test_sharding_config())
    restored_float32 = TinyModel.load(
        tmp_path,
        dtype=jnp.float32,
        sharding_config=make_test_sharding_config(),
    )

    assert isinstance(restored.module.matrix, IntMatrixForInference)
    assert restored.module.matrix.spec == original.module.matrix.spec
    assert restored.module.matrix.dtype == jnp.bfloat16
    assert restored_float32.module.matrix.dtype == jnp.float32
