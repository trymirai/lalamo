import jax.numpy as jnp
from einops import rearrange

from lalamo.modules import TiedEmbeddingConfig, quantize_tied_embedding_to_mlx
from lalamo.quantization import QuantizationMode


def test_quantize_tied_embedding_to_mlx_preserves_configuration_fields() -> None:
    embedding = TiedEmbeddingConfig(
        input_scale=2.0,
        logit_soft_cap=5.0,
        precision=jnp.float32,
    ).empty(vocab_size=4, model_dim=4)
    embedding = embedding.import_weights(
        {
            "weights": jnp.array(
                [
                    [-1.0, 0.5, 2.0, 3.25],
                    [1.25, 1.5, -2.5, 4.0],
                    [0.25, 0.75, -1.0, -0.5],
                    [3.0, 2.0, 1.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
        },
    )

    quantized_embedding = quantize_tied_embedding_to_mlx(
        embedding,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT8,
    )

    assert quantized_embedding.weights.shape == embedding.weights.shape
    assert quantized_embedding.scales.shape == (embedding.vocab_size, 2)
    assert quantized_embedding.biases.shape == (embedding.vocab_size, 2)
    assert quantized_embedding.config.input_scale == embedding.config.input_scale
    assert quantized_embedding.config.logit_soft_cap == embedding.config.logit_soft_cap
    assert quantized_embedding.config.embedding_quantization_mode == QuantizationMode.UINT8


def test_quantize_tied_embedding_to_mlx_reconstructs_two_value_groups_exactly() -> None:
    embedding = TiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=4)
    original_weights = jnp.array(
        [
            [-1.0, 0.5, 2.0, 3.25],
            [1.25, 1.5, -2.5, 4.0],
        ],
        dtype=jnp.float32,
    )
    embedding = embedding.import_weights({"weights": original_weights})

    quantized_embedding = quantize_tied_embedding_to_mlx(
        embedding,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT8,
    )

    grouped_weights = rearrange(
        quantized_embedding.int_weights.astype(quantized_embedding.activation_precision),
        "vocab (groups elements) -> vocab groups elements",
        elements=quantized_embedding.config.group_size,
    )
    reconstructed_weights = rearrange(
        grouped_weights * quantized_embedding.scales[:, :, None] + quantized_embedding.biases[:, :, None],
        "vocab groups elements -> vocab (groups elements)",
    )

    assert jnp.allclose(reconstructed_weights, original_weights, atol=1e-6)

    exported_weights = quantized_embedding.export_weights()
    restored_embedding = quantized_embedding.import_weights(exported_weights)

    assert jnp.array_equal(restored_embedding.weights, quantized_embedding.weights)
    assert jnp.array_equal(restored_embedding.scales, quantized_embedding.scales)
    assert jnp.array_equal(restored_embedding.biases, quantized_embedding.biases)
