import jax.numpy as jnp

from lalamo.modules.embedding import MLXQuantizedTiedEmbedding, MLXQuantizedTiedEmbeddingConfig
from lalamo.quantization import QuantizationMode


def test_mlx_quantized_tied_embedding_export_import_roundtrip() -> None:
    config = MLXQuantizedTiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    )
    embedding = MLXQuantizedTiedEmbedding(
        config=config,
        weights=jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.float32),
        scales=jnp.ones((2, 2), dtype=jnp.float32),
        biases=jnp.zeros((2, 2), dtype=jnp.float32),
    )

    exported = embedding.export_weights()
    loaded = embedding.import_weights(exported)

    assert exported["weights"].dtype == jnp.uint8
    assert jnp.array_equal(loaded.weights, embedding.weights)
    assert jnp.array_equal(loaded.scales, embedding.scales)
    assert jnp.array_equal(loaded.biases, embedding.biases)
