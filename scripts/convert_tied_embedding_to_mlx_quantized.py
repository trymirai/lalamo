import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from lalamo.model_import import ModelMetadata
from lalamo.models import LanguageModelConfig
from lalamo.modules import EmbeddingBase, TiedEmbeddingConfig, config_converter, quantize_tied_embedding_to_mlx
from lalamo.quantization import QuantizationMode
from lalamo.safetensors import safe_read, safe_write


def _compute_validation_metrics(
    original_embedding: EmbeddingBase,
    quantized_embedding: EmbeddingBase,
) -> dict[str, float]:
    hidden_states = jax.random.normal(jax.random.key(0), (16, original_embedding.model_dim), dtype=jnp.float32)
    original_readout = jax.vmap(original_embedding.readout)(hidden_states).astype(jnp.float32)
    quantized_readout = jax.vmap(quantized_embedding.readout)(hidden_states).astype(jnp.float32)
    original_log_probs = jax.nn.log_softmax(original_readout, axis=-1)
    quantized_log_probs = jax.nn.log_softmax(quantized_readout, axis=-1)
    original_probs = jnp.exp(original_log_probs)
    token_ids = jnp.arange(original_embedding.vocab_size, dtype=jnp.int32)
    original_embed = original_embedding.embed(token_ids).astype(jnp.float32)
    quantized_embed = quantized_embedding.embed(token_ids).astype(jnp.float32)
    return {
        "embedding_mean_abs_error": float(jnp.mean(jnp.abs(original_embed - quantized_embed))),
        "embedding_max_abs_error": float(jnp.max(jnp.abs(original_embed - quantized_embed))),
        "readout_mean_abs_error": float(jnp.mean(jnp.abs(original_readout - quantized_readout))),
        "readout_max_abs_error": float(jnp.max(jnp.abs(original_readout - quantized_readout))),
        "readout_mean_kl": float(
            jnp.mean(jnp.sum(original_probs * (original_log_probs - quantized_log_probs), axis=-1))
        ),
        "readout_top1_agreement": float(
            jnp.mean(jnp.argmax(original_readout, axis=-1) == jnp.argmax(quantized_readout, axis=-1))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("input-config", "input-weights", "output-config", "output-weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()

    metadata = config_converter.structure(json.loads(args.input_config.read_text()), ModelMetadata)
    assert isinstance(metadata.model_config, LanguageModelConfig)
    embedding_config = metadata.model_config.model_config.embedding_config
    assert isinstance(embedding_config, TiedEmbeddingConfig)

    with args.input_weights.open("rb") as weights_file:
        flat_weights = dict(safe_read(weights_file)[1].items())
    embedding_weights = flat_weights["embedding.weights"]
    original_embedding = embedding_config.empty(*embedding_weights.shape).import_weights(
        {"weights": embedding_weights}
    )

    quantized_embedding = quantize_tied_embedding_to_mlx(
        original_embedding,
        group_size=args.group_size,
        embedding_quantization_mode=QuantizationMode.from_num_bits(args.num_bits),
        activation_quantization_mode=None,
        activation_precision=original_embedding.activation_precision,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
    )

    updated_metadata = replace(
        metadata,
        model_config=replace(
            metadata.model_config,
            model_config=replace(metadata.model_config.model_config, embedding_config=quantized_embedding.config),
        ),
    )
    for key, value in quantized_embedding.export_weights().items():
        flat_weights[f"embedding.{key}"] = value

    for path in (args.output_config, args.output_weights):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config_converter.unstructure(updated_metadata, ModelMetadata), indent=4))
    with args.output_weights.open("wb") as weights_file:
        safe_write(weights_file, flat_weights)

    print(json.dumps(_compute_validation_metrics(original_embedding, quantized_embedding), indent=4))


if __name__ == "__main__":
    main()
