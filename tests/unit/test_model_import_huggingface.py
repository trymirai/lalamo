import jax.numpy as jnp

from lalamo.common import ParameterPath
from lalamo.model_import.loaders.huggingface import load_linear, load_mlx_quantized_tied_embedding
from lalamo.modules.embedding import MLXQuantizedTiedEmbeddingConfig
from lalamo.modules.linear import MLXQuantizedLinearConfig
from lalamo.quantization import QuantizationMode


def test_load_mlx_quantized_tied_embedding_updates_detected_quantization_mode() -> None:
    module = MLXQuantizedTiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=4,
        embedding_quantization_mode=QuantizationMode.UINT8,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=8)

    loaded = load_mlx_quantized_tied_embedding(
        module,
        {
            ParameterPath("embedding") / "weight": jnp.zeros((2, 1), dtype=jnp.int32),
            ParameterPath("embedding") / "scales": jnp.ones((2, 2), dtype=jnp.float32),
            ParameterPath("embedding") / "biases": jnp.zeros((2, 2), dtype=jnp.float32),
        },
        ParameterPath("embedding"),
    )

    assert loaded.config.embedding_quantization_mode == QuantizationMode.UINT4


def test_load_mlx_quantized_linear_updates_detected_quantization_mode() -> None:
    module = MLXQuantizedLinearConfig(
        group_size=4,
        weight_quantization_mode=QuantizationMode.UINT8,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(8, (8,), has_biases=False)

    loaded = load_linear(
        module,
        {
            ParameterPath("linear") / "weight": jnp.zeros((8, 1), dtype=jnp.int32),
            ParameterPath("linear") / "scales": jnp.ones((8, 2), dtype=jnp.float32),
            ParameterPath("linear") / "biases": jnp.zeros((8, 2), dtype=jnp.float32),
        },
        ParameterPath("linear"),
    )

    assert loaded.config.weight_quantization_mode == QuantizationMode.UINT4
