import math
from collections.abc import Mapping

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

from lalamo.modules.attention_residual import (
    BlockAttnResConnection,
    BlockAttnResHead,
    BlockAttnResQKVConnection,
    MessageGate,
    MessageGates,
    rms_normalize,
)
from lalamo.modules.decoder import Decoder
from lalamo.modules.embedding import UntiedEmbedding
from lalamo.modules.linear import Linear
from lalamo.modules.mlp import MixtureOfExperts
from lalamo.modules.token_mixers.attention import Attention, ProjectionLayout
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv
from lalamo.modules.transformer_layer import TransformerLayer
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at

from .common import load_full_precision
from .huggingface import load_untied_embedding
from .utils import dequantize_experts

__all__ = ["load_inkling2_loop_decoder", "load_message_gates"]


def f32(weights_dict: Mapping[str, Array], path: ParameterPath) -> Float[Array, "..."]:
    return weights_dict[path].astype(jnp.float32)


def pad_rows(matrix: Float[Array, "..."], rows: int) -> Float[Array, "..."]:
    return jnp.pad(matrix, ((0, rows - matrix.shape[0]), *((0, 0),) * (matrix.ndim - 1)))


def load_linear_matrix(module: Linear, matrix: Array) -> Linear:
    return load_as_at(lambda linear: (linear.weights,), module, (load_full_precision(module.weights, matrix),))


def stack_parts(module: Linear, parts: tuple[Float[Array, "_ inputs"], ...]) -> Linear:
    return load_linear_matrix(
        module,
        jnp.concatenate(
            [pad_rows(part, padded_rows) for part, padded_rows in zip(parts, module.output_dims, strict=True)],
        ),
    )


def fold_static_conv(conv: SeparableCausalConv, hf_kernel: Float[Array, "channels 1 kernel"]) -> SeparableCausalConv:
    _, _, kernel_size = hf_kernel.shape
    taps = hf_kernel[:, 0].at[:, kernel_size - 1].add(1.0) / math.sqrt(kernel_size + 1)
    return load_as_at(lambda module: (module.weights,), conv, (taps,))


def fold_dynamic_conv(
    conv: SeparableCausalConv,
    lag_first_static_taps: Float[Array, "kernel channels"],
) -> SeparableCausalConv:
    kernel_size, _ = lag_first_static_taps.shape
    taps = jnp.flip((lag_first_static_taps.at[0].add(1.0) / math.sqrt(2 * kernel_size + 1)).T, axis=1)
    return load_as_at(lambda module: (module.weights,), conv, (taps,))


def oldest_first_dynamic_taps(
    lag_first_projection: Float[Array, "taps channels"],
    kernel_size: int,
) -> Float[Array, "taps channels"]:
    return rearrange(
        jnp.flip(
            rearrange(lag_first_projection, "(groups kernel) channels -> groups kernel channels", kernel=kernel_size),
            axis=1,
        ),
        "groups kernel channels -> (groups kernel) channels",
    ) / math.sqrt(2 * kernel_size + 1)


def load_inkling_attention(
    attention: Attention,
    weights_dict: Mapping[str, Array],
    layer_path: ParameterPath,
) -> Attention:
    config = attention.config
    stacked_config = config.stacked_projection_config
    head_dim = config.head_dim
    if stacked_config is None or attention.key_conv is None or attention.value_conv is None:
        raise TypeError("Inkling attention expects stacked projections with key/value convolutions.")
    kernel_size = attention.key_conv.kernel_size
    attention_path = layer_path / "attn"
    norm_names = (
        ("attn_norm",) * 3
        if stacked_config.layout == ProjectionLayout.STACKED_QUERY_FIRST
        else ("q_input_norm", "k_input_norm", "v_input_norm")
    )
    query_gamma, key_gamma, value_gamma = (f32(weights_dict, layer_path / name / "weight") for name in norm_names)
    query_scales = rearrange(
        f32(weights_dict, attention_path / "q_l2_clip" / "scale"), "heads channels -> (heads channels)"
    )
    relative_scales = rearrange(
        f32(weights_dict, attention_path / "rel_l2_clip" / "scale"),
        "heads channels -> (heads channels)",
    )
    query_parts = tuple(
        matrix * query_gamma
        for matrix in (
            f32(weights_dict, attention_path / "wq_du" / "weight") * query_scales[:, None] * head_dim**-0.5,
            f32(weights_dict, attention_path / "gate_attn" / "weight"),
            f32(weights_dict, attention_path / "wr_du" / "weight") * relative_scales[:, None],
            f32(weights_dict, attention_path / "attn_sink_proj" / "weight"),
        )
    )
    key_rows = f32(weights_dict, attention_path / "wk_dv" / "weight") * key_gamma
    value_rows = f32(weights_dict, attention_path / "wv_dv" / "weight") * value_gamma
    match stacked_config.layout:
        case ProjectionLayout.STACKED_QUERY_FIRST:
            projections = (stack_parts(attention.qkvg_projection, (*query_parts, key_rows, value_rows)),)
            key_conv = fold_static_conv(attention.key_conv, f32(weights_dict, attention_path / "k_sconv" / "weight"))
            value_conv = fold_static_conv(
                attention.value_conv, f32(weights_dict, attention_path / "v_sconv" / "weight")
            )
        case ProjectionLayout.GROUPED_WITH_CONV_TAPS:
            key_projection, value_projection = attention.key_value_projections
            key_taps = f32(weights_dict, attention_path / "k_sconv" / "weight_proj") * key_gamma
            value_taps = f32(weights_dict, attention_path / "v_sconv" / "weight_proj") * value_gamma
            projections = (
                stack_parts(attention.qkvg_projection, query_parts),
                stack_parts(key_projection, (key_rows, oldest_first_dynamic_taps(key_taps, kernel_size))),
                stack_parts(
                    value_projection,
                    (value_rows, oldest_first_dynamic_taps(value_taps, kernel_size)),
                ),
            )
            key_conv = fold_dynamic_conv(
                attention.key_conv, f32(weights_dict, attention_path / "k_sconv" / "static_w")
            )
            value_conv = fold_dynamic_conv(
                attention.value_conv,
                f32(weights_dict, attention_path / "v_sconv" / "static_w"),
            )
    qkvg_projection, *key_value_projections = projections
    return load_as_at(
        lambda module: (
            module.qkvg_projection,
            module.key_value_projections,
            module.out_projection,
            module.relative_profiles,
            module.key_conv,
            module.value_conv,
        ),
        attention,
        (
            qkvg_projection,
            tuple(key_value_projections),
            load_linear_matrix(attention.out_projection, weights_dict[attention_path / "wo_ud" / "weight"]),
            weights_dict[attention_path / "rel_logits_proj" / "proj"],
            key_conv,
            value_conv,
        ),
    )


def load_inkling_moe(
    moe: MixtureOfExperts,
    weights_dict: Mapping[str, Array],
    layer_path: ParameterPath,
) -> MixtureOfExperts:
    mlp_path = layer_path / "mlp"
    gamma = f32(weights_dict, layer_path / "mlp_norm" / "weight")
    gate_up = dequantize_experts(weights_dict, mlp_path / "experts" / "w13_weight") * gamma
    return load_as_at(
        lambda module: (
            module.router,
            module.routed_experts.up_projection,
            module.routed_experts.down_projection,
            module.expert_bias,
        ),
        moe,
        (
            load_linear_matrix(moe.router, f32(weights_dict, mlp_path / "gate" / "weight") * gamma),
            load_linear_matrix(
                moe.routed_experts.up_projection,
                jnp.concatenate([gate_up[:, 1::2], gate_up[:, 0::2]], axis=1),
            ),
            load_linear_matrix(
                moe.routed_experts.down_projection,
                dequantize_experts(weights_dict, mlp_path / "experts" / "w2_weight"),
            ),
            weights_dict[mlp_path / "gate" / "bias"],
        ),
    )


def load_message_gate(
    gate: MessageGate,
    weights_dict: Mapping[str, Array],
    controller_path: ParameterPath,
    query: Float[Array, " channels"],
) -> MessageGate:
    channels = gate.message_projection.input_dim
    return load_as_at(
        lambda module: (module.scaled_query, module.message_projection, module.forget_gain_projection),
        gate,
        (
            query.astype(jnp.float32) * channels**-0.5,
            load_linear_matrix(
                gate.message_projection, f32(weights_dict, controller_path / "down_weight")[:, channels:]
            ),
            load_linear_matrix(
                gate.forget_gain_projection,
                jnp.concatenate(
                    [
                        f32(weights_dict, controller_path / "forget_weight"),
                        f32(weights_dict, controller_path / "gain_weight"),
                    ],
                ),
            ),
        ),
    )


def load_message_gates(
    gates: MessageGates,
    weights_dict: Mapping[str, Array],
    controller_paths: tuple[ParameterPath, ...],
    queries: tuple[Float[Array, " channels"], ...],
) -> MessageGates:
    per_gate_matrices = tuple(
        f32(weights_dict, path / "down_weight")[:, : gate.message_projection.input_dim]
        for gate, path in zip(gates.gates, controller_paths, strict=True)
    )
    hidden_matrices = (
        (jnp.concatenate(per_gate_matrices),) if len(gates.hidden_projections) == 1 else per_gate_matrices
    )
    return load_as_at(
        lambda module: (module.hidden_projections, module.gates),
        gates,
        (
            tuple(
                load_linear_matrix(projection, matrix)
                for projection, matrix in zip(gates.hidden_projections, hidden_matrices, strict=True)
            ),
            tuple(
                load_message_gate(gate, weights_dict, path, query)
                for gate, path, query in zip(gates.gates, controller_paths, queries, strict=True)
            ),
        ),
    )


def load_inkling_layer(
    layer: TransformerLayer,
    weights_dict: Mapping[str, Array],
    model_path: ParameterPath,
    layer_index: int,
    loop_start: int,
) -> TransformerLayer:
    layer_path = model_path / "layers" / layer_index
    queries_path = model_path / "attn_res"
    mixer_connection = layer.mixer.connection
    mlp_connection = layer.mlp.connection
    attention = layer.mixer.body
    moe = layer.mlp.body
    if not isinstance(mixer_connection, BlockAttnResQKVConnection) or not isinstance(attention, Attention):
        raise TypeError(f"Layer {layer_index} is not an Inkling attention sublayer.")
    if not isinstance(mlp_connection, BlockAttnResConnection) or not isinstance(moe, MixtureOfExperts):
        raise TypeError(f"Layer {layer_index} is not an Inkling mixture-of-experts sublayer.")
    if layer.mixer.output_conv is None or layer.mlp.output_conv is None:
        raise TypeError(f"Layer {layer_index} lacks the Inkling output convolutions.")
    query_scale = attention.model_dim**-0.5

    if mixer_connection.message_gates is None or mlp_connection.message_gates is None:
        mixer_gates = mixer_connection.message_gates
        mlp_gates = mlp_connection.message_gates
    else:
        loop_index = layer_index - loop_start
        cross_pass_queries = weights_dict[model_path / "middle_cycle_cross_pass_queries"]
        cross_pass_key_value_queries = weights_dict[model_path / "middle_cycle_cross_pass_kv_queries"]
        controllers_path = layer_path / "controllers"
        mixer_gates = load_message_gates(
            mixer_connection.message_gates,
            weights_dict,
            tuple(controllers_path / name for name in ("attn", "k", "v")),
            (
                cross_pass_queries[2 * loop_index],
                cross_pass_key_value_queries[2 * loop_index],
                cross_pass_key_value_queries[2 * loop_index + 1],
            ),
        )
        mlp_gates = load_message_gates(
            mlp_connection.message_gates,
            weights_dict,
            (controllers_path / "mlp",),
            (cross_pass_queries[2 * loop_index + 1],),
        )
    mixer_queries = (
        f32(weights_dict, queries_path / "attn_queries" / (layer_index - 1)).reshape(
            mixer_connection.scaled_queries.shape
        )
        * query_scale
        if mixer_connection.scaled_queries is not None
        else None
    )
    return load_as_at(
        lambda module: (
            module.mixer.connection,
            module.mixer.body,
            module.mixer.output_conv,
            module.mlp.connection,
            module.mlp.body,
            module.mlp.output_conv,
        ),
        layer,
        (
            load_as_at(
                lambda connection: (connection.scaled_queries, connection.message_gates),
                mixer_connection,
                (mixer_queries, mixer_gates),
            ),
            load_inkling_attention(attention, weights_dict, layer_path),
            fold_static_conv(layer.mixer.output_conv, f32(weights_dict, layer_path / "attn_sconv" / "weight")),
            load_as_at(
                lambda connection: (connection.scaled_query, connection.message_gates),
                mlp_connection,
                (f32(weights_dict, queries_path / "mlp_queries" / layer_index) * query_scale, mlp_gates),
            ),
            load_inkling_moe(moe, weights_dict, layer_path),
            fold_static_conv(layer.mlp.output_conv, f32(weights_dict, layer_path / "mlp_sconv" / "weight")),
        ),
    )


def load_inkling2_loop_decoder(
    decoder: Decoder,
    weights_dict: Mapping[str, Array],
    loop_start: int,
    unpadded_vocab_size: int,
) -> Decoder:
    model_path = ParameterPath("model.llm")
    embedding = decoder.embedding
    head = decoder.transformer.residual_head
    if not isinstance(embedding, UntiedEmbedding) or not isinstance(head, BlockAttnResHead):
        raise TypeError("Inkling decoders need an untied embedding and a block attention residual head.")
    embeddings = f32(weights_dict, model_path / "embed" / "weight")[:unpadded_vocab_size]
    normalized_embeddings = rms_normalize(embeddings, head.config.epsilon) * f32(
        weights_dict,
        model_path / "embed_norm" / "weight",
    )
    unembedding = f32(weights_dict, model_path / "unembed" / "weight")[:unpadded_vocab_size] * f32(
        weights_dict,
        model_path / "norm" / "weight",
    )
    embedding_weights: dict[str, Array] = {
        ParameterPath("embed.weight"): pad_rows(normalized_embeddings, decoder.vocab_size),
        ParameterPath("unembed.weight"): pad_rows(unembedding, decoder.vocab_size),
    }
    (channels,) = head.scaled_output_query.shape
    return load_as_at(
        lambda module: (module.embedding, module.transformer.layers, module.transformer.residual_head),
        decoder,
        (
            load_untied_embedding(embedding, embedding_weights, ParameterPath("embed"), ParameterPath("unembed")),
            tuple(
                load_inkling_layer(layer, weights_dict, model_path, layer_index, loop_start)
                for layer_index, layer in enumerate(decoder.transformer.layers)
            ),
            load_as_at(
                lambda residual_head: (residual_head.scaled_output_query,),
                head,
                (f32(weights_dict, model_path / "attn_res" / "output_query") * channels**-0.5,),
            ),
        ),
    )
