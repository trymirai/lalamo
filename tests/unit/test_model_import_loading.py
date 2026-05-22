from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, DTypeLike
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from lalamo.compressed.int import IntMatrixForInference, IntMatrixForTraining, IntSpec
from lalamo.compressed.mlx import MLXMatrixForInference, MLXMatrixForTraining
from lalamo.initializer import EmptyInitializer, Initializer
from lalamo.model import Model, ModelConfig
from lalamo.model_import.loaders.huggingface import load_huggingface_classifier, load_linear
from lalamo.model_import.model_configs.foreign_config import ForeignConfig
from lalamo.model_import.model_configs.huggingface import ModernBERTConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.classifier import Classifier
from lalamo.modules.embedding import TiedEmbedding
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.mlp import DenseMLP
from lalamo.modules.token_mixers.attention import Attention
from lalamo.utils.dummy_array import dummy_array
from lalamo.utils.parameter_path import ParameterPath
from lalamo.weight_matrix import CompressionImplementation, FullPrecisionSpec, Layout, WeightMatrix
from tests.helpers import make_sharding, make_test_sharding_config

pytestmark = pytest.mark.usefixtures("fake_mesh")

INPUT_DIM = 8
OUTPUT_DIM = 4
NUM_GROUPS = 2
CLASSIFIER_VOCAB_SIZE = 16
CLASSIFIER_HIDDEN_SIZE = 4
CLASSIFIER_INTERMEDIATE_SIZE = 8
CLASSIFIER_NUM_HEADS = 2
CLASSIFIER_NUM_LABELS = 2


def _pack_int32(values: Array, bits: int) -> Array:
    values_per_word = 32 // bits
    rows, cols = values.shape
    grouped = values.reshape(rows, cols // values_per_word, values_per_word).astype(jnp.uint32)
    shifts = jnp.arange(values_per_word, dtype=jnp.uint32) * jnp.uint32(bits)
    return jnp.sum(grouped << shifts, axis=-1, dtype=jnp.uint32).astype(jnp.int32)


def _linear_template(dtype: DTypeLike, layout: Layout = Layout.OUTPUT_INPUT) -> Linear:
    result = LinearConfig().init(
        initializer=EmptyInitializer(default_dtype=dtype, sharding_config=make_test_sharding_config()),
        input_dim=INPUT_DIM,
        output_dims=(OUTPUT_DIM,),
        has_biases=False,
    )
    if layout == Layout.OUTPUT_INPUT:
        return result

    weights = FullPrecisionSpec(layout=layout).compress(
        dummy_array((OUTPUT_DIM, INPUT_DIM), dtype, make_sharding((None, None))),
        sharding_config=make_test_sharding_config(),
    )
    return eqx.tree_at(lambda module: module.weights, result, weights)


def _mlx_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(OUTPUT_DIM * INPUT_DIM, dtype=jnp.int32).reshape(OUTPUT_DIM, INPUT_DIM)
    return {
        path / "weight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((OUTPUT_DIM, NUM_GROUPS), dtype=jnp.float32),
        path / "biases": jnp.zeros((OUTPUT_DIM, NUM_GROUPS), dtype=jnp.float32),
    }


def _awq_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(INPUT_DIM * OUTPUT_DIM, dtype=jnp.int32).reshape(INPUT_DIM, OUTPUT_DIM)
    unpacked_zero_points = jnp.zeros((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.int32)
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "qzeros": _pack_int32(unpacked_zero_points, bits=8),
        path / "scales": jnp.ones((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.float32),
    }


def _symmetric_awq_weights(path: ParameterPath) -> Mapping[str, Array]:
    unpacked_weights = jnp.arange(INPUT_DIM * OUTPUT_DIM, dtype=jnp.int32).reshape(INPUT_DIM, OUTPUT_DIM) + 128
    return {
        path / "qweight": _pack_int32(unpacked_weights, bits=8),
        path / "scales": jnp.ones((NUM_GROUPS, OUTPUT_DIM), dtype=jnp.float32),
    }


def _classifier_tensor(shape: tuple[int, ...]) -> Array:
    return jnp.arange(prod(shape), dtype=jnp.float32).reshape(shape)


def _classifier_template() -> Classifier:
    config = ModernBERTConfig(
        architectures=["ModernBertForSequenceClassification"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        classifier_activation="gelu",
        classifier_bias=False,
        classifier_dropout=0.0,
        classifier_pooling="mean",
        cls_token_id=1,
        decoder_bias=False,
        deterministic_flash_attn=False,
        embedding_dropout=0.0,
        eos_token_id=2,
        global_attn_every_n_layers=1,
        global_rope_theta=10000.0,
        gradient_checkpointing=False,
        hidden_activation="gelu",
        hidden_size=CLASSIFIER_HIDDEN_SIZE,
        initializer_cutoff_factor=2.0,
        initializer_range=0.02,
        intermediate_size=CLASSIFIER_INTERMEDIATE_SIZE,
        layer_norm_eps=1e-5,
        local_attention=4,
        local_rope_theta=10000.0,
        max_position_embeddings=8,
        mlp_bias=False,
        mlp_dropout=0.0,
        model_type="modernbert",
        norm_bias=False,
        norm_eps=1e-5,
        num_attention_heads=CLASSIFIER_NUM_HEADS,
        num_hidden_layers=1,
        pad_token_id=3,
        position_embedding_type="absolute",
        sep_token_id=4,
        transformers_version="test",
        vocab_size=CLASSIFIER_VOCAB_SIZE,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    return config.to_classifier_config(context_length=8).init(
        EmptyInitializer(default_dtype=jnp.float32, sharding_config=make_test_sharding_config()),
    )


def _classifier_weights(classifier: Classifier) -> Mapping[str, Array]:
    assert isinstance(classifier.embedding, TiedEmbedding)
    layer = classifier.transformer.layers[0]
    assert isinstance(layer.mixer, Attention)
    assert isinstance(layer.mlp, DenseMLP)
    assert layer.pre_mlp_norm is not None

    base_path = ParameterPath()
    decoder_path = base_path / "model"
    head_path = base_path / "head"
    classifier_path = base_path / "classifier"

    return {
        decoder_path / "embeddings" / "tok_embeddings" / "weight": _classifier_tensor(
            (CLASSIFIER_VOCAB_SIZE, CLASSIFIER_HIDDEN_SIZE),
        ),
        decoder_path / "embeddings" / "norm" / "weight": _classifier_tensor(classifier.embedding_norm.scales.shape),
        decoder_path / "layers" / 0 / "attn" / "Wqkv" / "weight": _classifier_tensor(
            layer.mixer.qkv_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "attn" / "Wo" / "weight": _classifier_tensor(
            layer.mixer.out_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "mlp_norm" / "weight": _classifier_tensor(layer.pre_mlp_norm.scales.shape),
        decoder_path / "layers" / 0 / "mlp" / "Wi" / "weight": _classifier_tensor(
            layer.mlp.up_projection.weights.shape,
        ),
        decoder_path / "layers" / 0 / "mlp" / "Wo" / "weight": _classifier_tensor(
            layer.mlp.down_projection.weights.shape,
        ),
        decoder_path / "final_norm" / "weight": _classifier_tensor(classifier.transformer.output_norm.scales.shape),
        head_path / "dense" / "weight": _classifier_tensor(classifier.prediction_head.dense.weights.shape),
        head_path / "norm" / "weight": _classifier_tensor(classifier.prediction_head.norm.scales.shape),
        classifier_path / "weight": _classifier_tensor(classifier.prediction_head.readout.weights.shape),
        classifier_path / "bias": _classifier_tensor((CLASSIFIER_NUM_LABELS,)),
    }


@pytest.mark.parametrize(
    ("weights_factory", "implementation", "expected_type"),
    [
        (_mlx_weights, CompressionImplementation.INFERENCE, MLXMatrixForInference),
        (_mlx_weights, CompressionImplementation.TRAINING, MLXMatrixForTraining),
        (_awq_weights, CompressionImplementation.INFERENCE, IntMatrixForInference),
        (_awq_weights, CompressionImplementation.TRAINING, IntMatrixForTraining),
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


def test_load_linear_symmetric_awq_without_qzeros_uses_symmetric_spec() -> None:
    path = ParameterPath("layer")

    loaded = load_linear(
        _linear_template(jnp.bfloat16),
        _symmetric_awq_weights(path),
        path,
        implementation=CompressionImplementation.INFERENCE,
    )

    assert isinstance(loaded.weights, IntMatrixForInference)
    assert loaded.weights.spec.is_symmetric
    assert loaded.weights.packed_zero_points is None


def test_load_huggingface_classifier_uses_hf_embedding_layout() -> None:
    classifier = _classifier_template()
    weights = _classifier_weights(classifier)

    loaded = load_huggingface_classifier(classifier, weights)

    assert isinstance(loaded.embedding, TiedEmbedding)
    assert isinstance(classifier.embedding, TiedEmbedding)
    assert loaded.embedding.embedding.shape == classifier.embedding.embedding.shape
    token_embedding = loaded.embedding.embedding.lookup_embedding(
        0,
        keychain=Keychain.init(0, sharding_config=make_test_sharding_config()),
    )
    expected_embedding = weights["model.embeddings.tok_embeddings.weight"][0].astype(token_embedding.dtype)
    np.testing.assert_array_equal(token_embedding, expected_embedding)


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
