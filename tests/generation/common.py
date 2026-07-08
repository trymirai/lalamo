from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from lalamo.models import LanguageModel
from lalamo.module import ForwardPassMode, LogicalAxis
from lalamo.modules import DecoderForwardPassConfig


def stable_generation_forward_pass_configs() -> tuple[DecoderForwardPassConfig, DecoderForwardPassConfig]:
    prefill_forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    decode_forward_pass_config = DecoderForwardPassConfig.for_tracer_tests()
    decode_transformer_config = decode_forward_pass_config.transformer_forward_pass_config
    decode_forward_pass_config = replace(
        decode_forward_pass_config,
        transformer_forward_pass_config=replace(
            decode_transformer_config,
            mlp_forward_pass_config=replace(
                decode_transformer_config.mlp_forward_pass_config,
                mode=ForwardPassMode.SINGLE_TOKEN,
            ),
        ),
    )
    return prefill_forward_pass_config, decode_forward_pass_config


def _batch_axis_size(language_model: LanguageModel) -> int:
    batch_axis = language_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    if batch_axis is None:
        return 1
    return language_model.sharding_config.mesh.shape[batch_axis]


def sharded_generation_batch(
    language_model: LanguageModel,
    token_ids: jax.Array,
    lengths_without_padding: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_size, sequence_length = token_ids.shape
    batch_axis_size = _batch_axis_size(language_model)
    remainder = batch_size % batch_axis_size
    if remainder == 0:
        padded_token_ids = token_ids
        padded_lengths = lengths_without_padding
    else:
        num_padding_rows = batch_axis_size - remainder
        padded_token_ids = jnp.concatenate(
            [
                token_ids,
                jnp.zeros((num_padding_rows, sequence_length), dtype=token_ids.dtype),
            ],
        )
        padded_lengths = jnp.concatenate(
            [
                lengths_without_padding,
                jnp.ones((num_padding_rows,), dtype=lengths_without_padding.dtype),
            ],
        )

    batch_axis = language_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
    return (
        jax.device_put(
            padded_token_ids,
            language_model.sharding_config.make_sharding((batch_axis, None)),
        ),
        jax.device_put(
            padded_lengths,
            language_model.sharding_config.make_sharding((batch_axis,)),
        ),
    )


def take_batch_prefix(language_model: LanguageModel, values: jax.Array, batch_size: int) -> np.ndarray:
    full_mesh_replicated_sharding = language_model.sharding_config.make_sharding((None,) * values.ndim)
    prefix = values.at[:batch_size].get(out_sharding=full_mesh_replicated_sharding)
    return np.asarray(prefix)


def take_first_batch_row(language_model: LanguageModel, values: jax.Array) -> np.ndarray:
    return take_batch_prefix(language_model, values, 1).squeeze(0)
