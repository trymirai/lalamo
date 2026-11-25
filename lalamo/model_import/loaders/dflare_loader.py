from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float

from lalamo.initializer import EmptyInitializer
from lalamo.model_import.common import _combine_weight_shards
from lalamo.model_import.model_configs.huggingface.dflare import HFDFlareConfig
from lalamo.model_import.origins import LocalOrigin, WeightFormat
from lalamo.modules.linear import Linear
from lalamo.modules.speculators.dflare import DFlareLayerContext, DFlareSpeculator
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at

from .common import load_full_precision
from .huggingface import load_rmsnorm, load_transformer_layer
from .inkling import f32, load_linear_matrix, pad_rows

__all__ = [
    "load_dflare_speculator",
    "load_hf_dflare_speculator",
]


def load_affine(module: Linear, matrix: Array, biases: Array) -> Linear:
    return load_as_at(
        lambda linear: (linear.weights, linear.biases),
        module,
        (load_full_precision(module.weights, matrix), biases),
    )


def load_dflare_speculator(
    speculator: DFlareSpeculator,
    weights_dict: Mapping[str, Array],
    final_norm_scales: Float[Array, " channels"],
    unpadded_vocab_size: int,
) -> DFlareSpeculator:
    draft_model = speculator.draft_model
    dspark_head = speculator.dspark_head
    draft_config = draft_model.config
    tap_scales = jnp.stack(
        [
            final_norm_scales if layer_id == draft_config.num_target_layers - 1 else jnp.ones_like(final_norm_scales)
            for layer_id in draft_config.target_layer_ids
        ],
    )
    # The reference rounds the mix weights back to the bfloat16 of the checkpoint.
    mix_weights = jax.nn.softmax(f32(weights_dict, ParameterPath("fuse_weight")), axis=-1)
    mix_weights = mix_weights.astype(jnp.bfloat16).astype(jnp.float32)
    *layer_tap_weights, head_tap_weights = mix_weights[:, :, None] * tap_scales
    layer_paths = tuple(ParameterPath("layers") / layer_index for layer_index in range(len(draft_model.layers)))
    markov_path = ParameterPath("markov_head")
    rank = dspark_head.config.markov_rank
    return load_as_at(
        lambda module: (
            module.draft_model.mask_embedding,
            module.draft_model.contexts,
            module.draft_model.layers,
            module.draft_model.output_norm.scales,
            module.dspark_head.context_tap_weights,
            module.dspark_head.context_norm,
            module.dspark_head.token_embeddings,
            module.dspark_head.recurrence_projection,
            module.dspark_head.output_projection,
        ),
        speculator,
        (
            weights_dict[ParameterPath("mask_embed")],
            tuple(
                DFlareLayerContext(
                    tap_weights=tap_weights,
                    norm=load_rmsnorm(context.norm, weights_dict, layer_path / "dflare_norm"),
                    key_value_projection=load_linear_matrix(
                        context.key_value_projection,
                        jnp.concatenate(
                            [
                                weights_dict[layer_path / "self_attn" / name / "weight"]
                                for name in ("k_proj_ctx", "v_proj_ctx")
                            ],
                        ),
                    ),
                    key_norm=load_rmsnorm(context.key_norm, weights_dict, layer_path / "self_attn" / "k_norm_ctx"),
                )
                for context, tap_weights, layer_path in zip(
                    draft_model.contexts,
                    layer_tap_weights,
                    layer_paths,
                    strict=True,
                )
            ),
            tuple(
                load_transformer_layer(
                    layer,
                    weights_dict,
                    layer_path,
                    layer_path,
                    "self_attn",
                    "mlp",
                    "input_layernorm",
                    "post_attention_layernorm",
                    "up_proj",
                    "gate_proj",
                    "down_proj",
                    permute_conv=False,
                )
                for layer, layer_path in zip(draft_model.layers, layer_paths, strict=True)
            ),
            f32(weights_dict, ParameterPath("norm.weight")) / final_norm_scales,
            head_tap_weights,
            load_rmsnorm(dspark_head.context_norm, weights_dict, markov_path / "dflare_norm"),
            pad_rows(
                f32(weights_dict, markov_path / "markov_w1" / "weight")[:unpadded_vocab_size],
                draft_config.vocab_size,
            ),
            load_affine(
                dspark_head.recurrence_projection,
                f32(weights_dict, markov_path / "rnn_proj" / "weight")
                .at[:, 2 * rank : 2 * rank + draft_config.model_dim]
                .multiply(final_norm_scales),
                f32(weights_dict, markov_path / "rnn_proj" / "bias"),
            ),
            load_affine(
                dspark_head.output_projection,
                f32(weights_dict, markov_path / "hidden_out" / "weight") / final_norm_scales[:, None],
                f32(weights_dict, markov_path / "hidden_out" / "bias") / final_norm_scales,
            ),
        ),
    )


def load_hf_dflare_speculator(
    hf_model_dir: Path | str,
    config: HFDFlareConfig,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
) -> DFlareSpeculator:
    hf_model_dir = Path(hf_model_dir)
    origin = LocalOrigin(
        root=str(hf_model_dir),
        weight_files=tuple(path.name for path in sorted(hf_model_dir.glob(f"*{WeightFormat.SAFETENSORS.value}"))),
        weight_format=WeightFormat.SAFETENSORS,
    )
    with ExitStack() as stack:
        weights = _combine_weight_shards(
            tuple(stack.enter_context(weight_shard) for weight_shard in origin.get_weights()),
        ).weights
        return load_dflare_speculator(
            config.to_speculator_config().init(EmptyInitializer(dtype, sharding_config)),
            weights,
            weights[ParameterPath("model.llm.norm.weight")].astype(jnp.float32),
            config.target_config.unpadded_vocab_size,
        )
