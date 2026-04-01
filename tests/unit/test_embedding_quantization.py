import equinox as eqx
import jax.numpy as jnp
import pytest
from einops import rearrange

from lalamo.modules import (
    MLXQuantizedUntiedEmbeddingConfig,
    MLXSemiQuantizedUntiedEmbeddingConfig,
    TiedEmbeddingConfig,
    UntiedEmbeddingConfig,
    quantize_tied_embedding_to_mlx,
)
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
    exported_weights = quantized_embedding.export_weights()

    grouped_weights = rearrange(
        exported_weights["weights"].astype(quantized_embedding.activation_precision),
        "vocab (groups elements) -> vocab groups elements",
        elements=quantized_embedding.config.group_size,
    )
    reconstructed_weights = rearrange(
        grouped_weights * quantized_embedding.scales[:, :, None] + quantized_embedding.biases[:, :, None],
        "vocab groups elements -> vocab (groups elements)",
    )

    assert jnp.allclose(reconstructed_weights, original_weights, atol=1e-6)

    restored_embedding = quantized_embedding.import_weights(exported_weights)

    assert jnp.array_equal(restored_embedding.weights, quantized_embedding.weights)
    assert jnp.array_equal(restored_embedding.scales, quantized_embedding.scales)
    assert jnp.array_equal(restored_embedding.biases, quantized_embedding.biases)


def test_quantize_tied_embedding_to_mlx_packs_uint1_weights() -> None:
    embedding = TiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=8)
    embedding = embedding.import_weights(
        {
            "weights": jnp.array(
                [
                    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
        },
    )

    quantized_embedding = quantize_tied_embedding_to_mlx(
        embedding,
        group_size=8,
        embedding_quantization_mode=QuantizationMode.UINT1,
    )
    exported_weights = quantized_embedding.export_weights()
    restored_embedding = quantized_embedding.import_weights(exported_weights)

    assert exported_weights["weights"].dtype == jnp.uint8
    assert exported_weights["weights"].shape == (2, 1)
    assert jnp.array_equal(restored_embedding.weights, quantized_embedding.weights)


def test_mlx_tied_embedding_import_weights_rejects_wrong_packed_dtype() -> None:
    embedding = quantize_tied_embedding_to_mlx(
        TiedEmbeddingConfig(
            input_scale=None,
            logit_soft_cap=None,
            precision=jnp.float32,
        )
        .empty(vocab_size=2, model_dim=4)
        .import_weights(
            {"weights": jnp.arange(8, dtype=jnp.float32).reshape(2, 4)},
        ),
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
    )
    exported_weights = embedding.export_weights()

    with pytest.raises(AssertionError):
        embedding.import_weights(
            {
                **exported_weights,
                "weights": exported_weights["weights"].astype(jnp.int32),
            },
        )


def test_untied_embedding_import_weights_rejects_shape_changes() -> None:
    embedding = UntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        precision=jnp.float32,
    ).empty(vocab_size=4, model_dim=4)

    with pytest.raises(AssertionError):
        embedding.import_weights(
            {
                "input_weights": jnp.ones((4, 3), dtype=jnp.float32),
                "output_weights": jnp.ones((4, 4), dtype=jnp.float32),
            },
        )


def test_mlx_untied_embedding_import_weights_rejects_wrong_packed_dtype() -> None:
    embedding = MLXQuantizedUntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=4)
    embedding = eqx.tree_at(
        lambda current: (
            current.input_weights,
            current.input_scales,
            current.input_biases,
            current.output_weights,
            current.output_scales,
            current.output_biases,
        ),
        embedding,
        (
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
        ),
    )
    exported_weights = embedding.export_weights()

    with pytest.raises(AssertionError):
        embedding.import_weights(
            {
                **exported_weights,
                "output_weights": exported_weights["output_weights"].astype(jnp.int32),
            },
        )


def test_mlx_semi_quantized_embedding_import_weights_rejects_wrong_packed_dtype() -> None:
    embedding = MLXSemiQuantizedUntiedEmbeddingConfig(
        input_scale=None,
        logit_soft_cap=None,
        group_size=2,
        embedding_quantization_mode=QuantizationMode.UINT4,
        activation_quantization_mode=None,
        activation_precision=jnp.float32,
    ).empty(vocab_size=2, model_dim=4)
    embedding = eqx.tree_at(
        lambda current: (
            current.input_weights,
            current.output_weights,
            current.output_scales,
            current.output_biases,
        ),
        embedding,
        (
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.ones((2, 4), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.float32),
        ),
    )
    exported_weights = embedding.export_weights()

    with pytest.raises(AssertionError):
        embedding.import_weights(
            {
                **exported_weights,
                "output_weights": exported_weights["output_weights"].astype(jnp.int32),
            },
        )
