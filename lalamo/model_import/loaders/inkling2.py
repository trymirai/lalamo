"""Loads Inkling 2 Mini from the mlx-lm layout of `mlx-community/Inkling-2-Mini-4bit`.

The routed experts keep their NVFP4 weights, except the MXFP8 experts of the last layers, which are requantized to
affine 4-bit.
"""

from collections.abc import Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from lalamo.compressed import MicrofloatSpec, MLXSpec
from lalamo.compressed.microfloat import MicrofloatScaleMode
from lalamo.compressed.utils.rounding import e8m0_scale_values
from lalamo.modules.decoder import Decoder
from lalamo.modules.embedding import UntiedEmbedding
from lalamo.modules.linear import Linear
from lalamo.modules.mlp import MixtureOfExperts
from lalamo.modules.normalization import Normalization
from lalamo.modules.residual import BlockAttentionConnection, BlockAttentionHead, MessageGate
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.convolutions import CausalConv
from lalamo.modules.transformer_layer import TransformerLayer
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation, WeightMatrix

from .common import load_full_precision
from .huggingface import _update_linear, load_linear, load_rmsnorm, load_untied_embedding

__all__ = ["load_inkling2_decoder"]


# mlx-lm keeps the NVFP4 tensor amax; the scale applied to the E4M3 group scales is amax / (E4M3 max * E2M1 max).
_NVFP4_GLOBAL_SCALE_DIVISOR = 448.0 * 6.0
_MXFP8_GROUP_SIZE = 32
# Every expert is stored in MLX affine 4-bit, the format of the rest of the checkpoint: the NVFP4 and MXFP8 experts
# are decoded to their exact values and quantized again.
_EXPERT_REQUANTIZATION = MLXSpec(bits=4, group_size=64)

_TEXT_MODEL = ParameterPath("model.language_model")


def _gate_up_to_up_gate_rows(num_rows: int) -> Array:
    # The checkpoints interleave gate and up rows; lalamo stacks all up rows before all gate rows.
    return jnp.concatenate([jnp.arange(1, num_rows, 2), jnp.arange(0, num_rows, 2)])


def _load_nvfp4_matrix(
    template: WeightMatrix,
    packed_weights: Array,
    packed_scales: Array,
    global_scale: Array,
    *,
    implementation: CompressionImplementation,
) -> WeightMatrix:
    if packed_weights.dtype != jnp.uint32 or packed_scales.dtype != jnp.uint8:
        raise ValueError(f"Expected U32 NVFP4 codes and U8 scales, got {packed_weights.dtype}, {packed_scales.dtype}.")
    code_bytes = jax.lax.bitcast_convert_type(packed_weights, jnp.uint8)
    return MicrofloatSpec(group_size=16, scale_mode=MicrofloatScaleMode.NVFP4).from_packed_parameters(
        packed_weights=rearrange(code_bytes, "... packed_words word_bytes -> ... (packed_words word_bytes)"),
        packed_scales=jax.lax.bitcast_convert_type(packed_scales, jnp.float8_e4m3fn),
        global_scale=global_scale.astype(jnp.float32) / _NVFP4_GLOBAL_SCALE_DIVISOR,
        dtype=template.dtype,
        implementation=implementation,
        sharding_config=template.sharding_config,
        is_sharded=template.is_sharded,
    )


def _dequantize_mxfp8(packed_weights: Array, packed_scales: Array) -> Array:
    if packed_weights.dtype != jnp.uint32 or packed_scales.dtype != jnp.uint8:
        raise ValueError(f"Expected U32 MXFP8 codes and U8 scales, got {packed_weights.dtype}, {packed_scales.dtype}.")
    code_bytes = rearrange(
        jax.lax.bitcast_convert_type(packed_weights, jnp.uint8),
        "... packed_words word_bytes -> ... (packed_words word_bytes)",
    )
    values = jax.lax.bitcast_convert_type(code_bytes, jnp.float8_e4m3fn).astype(jnp.float32)
    scales = e8m0_scale_values(packed_scales, jnp.float32)
    grouped = rearrange(values, "... (groups group) -> ... groups group", group=_MXFP8_GROUP_SIZE)
    return rearrange(grouped * scales[..., None], "... groups group -> ... (groups group)")


def _load_experts(
    module: MixtureOfExperts,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,
) -> MixtureOfExperts:
    router = _update_linear(
        module.router,
        load_full_precision(module.router.weights, weights_dict[path / "gate.weight"]),
        None,
    )
    expert_bias = weights_dict[path / "gate.e_score_correction_bias"].astype(jnp.float32)

    up_template = module.routed_experts.up_projection
    down_template = module.routed_experts.down_projection
    gate_up_rows = _gate_up_to_up_gate_rows(sum(up_template.output_dims))
    nvfp4_path = path / "experts"
    gate_up_weights = weights_dict.get(nvfp4_path / "gate_up_proj.weight")
    if gate_up_weights is not None and jnp.issubdtype(gate_up_weights.dtype, jnp.floating):
        up_projection = _update_linear(
            up_template,
            load_full_precision(up_template.weights, gate_up_weights[:, gate_up_rows]),
            None,
        )
        down_projection = _update_linear(
            down_template,
            load_full_precision(down_template.weights, weights_dict[nvfp4_path / "down_proj.weight"]),
            None,
        )
    elif (nvfp4_path / "gate_up_proj.global_scale") in weights_dict:

        def load_nvfp4(template: Linear, name: str, rows: Array | None) -> Linear:
            weights = weights_dict[nvfp4_path / name / "weight"]
            scales = weights_dict[nvfp4_path / name / "scales"]
            if rows is not None:
                weights, scales = weights[:, rows], scales[:, rows]
            nvfp4 = _load_nvfp4_matrix(
                template.weights,
                weights,
                scales,
                weights_dict[nvfp4_path / name / "global_scale"],
                implementation=implementation,
            )
            # Decoded in the model dtype: its rounding is far below the 4-bit step and keeps the scales in that dtype.
            matrix = _EXPERT_REQUANTIZATION.compress(
                nvfp4.decompress(),
                implementation=implementation,
                sharding_config=template.weights.sharding_config,
                is_sharded=template.weights.is_sharded,
            )
            return _update_linear(template, matrix, None)

        up_projection = load_nvfp4(up_template, "gate_up_proj", gate_up_rows)
        down_projection = load_nvfp4(down_template, "down_proj", None)
    elif (nvfp4_path / "gate_up_proj.scales") in weights_dict:

        def load_requantized_mxfp8(template: Linear, name: str, rows: Array | None) -> Linear:
            weights = _dequantize_mxfp8(
                weights_dict[nvfp4_path / name / "weight"],
                weights_dict[nvfp4_path / name / "scales"],
            )
            if rows is not None:
                weights = weights[:, rows]
            # MXFP8 values are exact in bfloat16, so casting to the model dtype first loses nothing.
            matrix = _EXPERT_REQUANTIZATION.compress(
                weights.astype(template.weights.dtype),
                implementation=implementation,
                sharding_config=template.weights.sharding_config,
                is_sharded=template.weights.is_sharded,
            )
            return _update_linear(template, matrix, None)

        up_projection = load_requantized_mxfp8(up_template, "gate_up_proj", gate_up_rows)
        down_projection = load_requantized_mxfp8(down_template, "down_proj", None)
    else:
        raise ValueError(f"No experts found at {nvfp4_path}.")
    routed_experts = load_as_at(
        lambda experts: (experts.up_projection, experts.down_projection),
        module.routed_experts,
        (up_projection, down_projection),
    )
    return load_as_at(
        lambda moe: (moe.router, moe.routed_experts, moe.expert_bias),
        module,
        (router, routed_experts, expert_bias),
    )


def _load_conv(module: CausalConv, weights: Array) -> CausalConv:
    return load_as_at(lambda conv: (conv.weights,), module, (weights.astype(module.weights.dtype),))


class _Overlay(Mapping[str, Array]):
    """The entries of `overrides`, then those of `base` that it does not replace."""

    def __init__(self, overrides: Mapping[str, Array], base: Mapping[str, Array]) -> None:
        self._overrides = overrides
        self._base = base

    def __getitem__(self, key: str) -> Array:
        if key in self._overrides:
            return self._overrides[key]
        return self._base[key]

    def __contains__(self, key: object) -> bool:
        return key in self._overrides or key in self._base

    def __iter__(self) -> Iterator[str]:
        yield from self._overrides
        yield from (key for key in self._base if key not in self._overrides)

    def __len__(self) -> int:
        return sum(1 for _ in self)


def _reverse_taps(array: Array, kernel_size: int) -> Array:
    # Tap rows are ordered by lag (the current token first); CausalConv weights the current token last.
    return rearrange(
        jnp.flip(rearrange(array, "(groups taps) ... -> groups taps ...", taps=kernel_size), axis=1),
        "groups taps ... -> (groups taps) ...",
    )


def _load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,
) -> Attention:
    assert module.query_projection is not None
    assert module.key_projection is not None
    assert module.value_projection is not None
    assert module.key_conv is not None and module.value_conv is not None
    query_projection = load_linear(
        module.query_projection,
        weights_dict,
        path,
        ["q_proj", "r_proj", "gate_proj", "sink_proj"],
        implementation=implementation,
    )
    out_projection = load_linear(module.out_projection, weights_dict, path / "o_proj", implementation=implementation)

    def load_key_value_side(projection: Linear, conv: CausalConv, name: str) -> tuple[Linear, CausalConv]:
        conv_path = path / f"{name}_sconv"
        if conv.config.num_dynamic_groups is None:
            static_weights = rearrange(weights_dict[conv_path / "conv1d.weight"], "channels taps 1 -> channels taps")
            loaded = load_linear(projection, weights_dict, path, [f"{name}_proj"], implementation=implementation)
            return loaded, _load_conv(conv, static_weights)
        kernel_size = conv.kernel_size
        taps_path = conv_path / "taps"
        reordered_taps: dict[str, Array] = {
            taps_path / part: _reverse_taps(weights_dict[conv_path / "weight_proj" / part], kernel_size)
            for part in ("weight", "scales", "biases")
            if (conv_path / "weight_proj" / part) in weights_dict
        }
        loaded = load_linear(
            projection,
            _Overlay(reordered_taps, weights_dict),
            path,
            [f"{name}_proj", f"{name}_sconv.taps"],
            implementation=implementation,
        )
        static_weights = jnp.flip(
            rearrange(weights_dict[conv_path / "static_weight"], "taps channels -> channels taps"), axis=1
        )
        return loaded, _load_conv(conv, static_weights)

    key_projection, key_conv = load_key_value_side(module.key_projection, module.key_conv, "k")
    value_projection, value_conv = load_key_value_side(module.value_projection, module.value_conv, "v")
    return load_as_at(
        lambda attention: (
            attention.query_projection,
            attention.key_projection,
            attention.value_projection,
            attention.out_projection,
            attention.query_clip_scale,
            attention.relative_clip_scale,
            attention.relative_position_profiles,
            attention.key_conv,
            attention.value_conv,
        ),
        module,
        (
            query_projection,
            key_projection,
            value_projection,
            out_projection,
            weights_dict[path / "q_l2_clip.scale"].astype(jnp.float32),
            weights_dict[path / "rel_l2_clip.scale"].astype(jnp.float32),
            weights_dict[path / "rel_logits_proj.proj"].astype(jnp.float32),
            key_conv,
            value_conv,
        ),
    )


def _load_message_gate(gate: MessageGate, weights_dict: Mapping[str, Array], path: ParameterPath) -> MessageGate:
    def load(projection: Linear, name: str) -> Linear:
        weights = load_full_precision(projection.weights, weights_dict[path / name / "weight"])
        return _update_linear(projection, weights, None)

    return load_as_at(
        lambda module: (module.down_projection, module.forget_projection, module.gain_projection),
        gate,
        (
            load(gate.down_projection, "down_proj"),
            load(gate.forget_projection, "forget_gate_proj"),
            load(gate.gain_projection, "message_gate_proj"),
        ),
    )


def _load_connection(
    connection: BlockAttentionConnection,
    weights_dict: Mapping[str, Array],
    query_path: ParameterPath,
    message_queries: Array | None,
    gate_paths: Sequence[ParameterPath],
) -> BlockAttentionConnection:
    if connection.queries is not None:
        queries = weights_dict[query_path].astype(jnp.float32).reshape(connection.queries.shape)
        connection = load_as_at(lambda module: (module.queries,), connection, (queries,))
    if connection.message_gates is None:
        return connection
    assert connection.message_queries is not None and message_queries is not None
    gates = tuple(
        _load_message_gate(gate, weights_dict, gate_path)
        for gate, gate_path in zip(connection.message_gates, gate_paths, strict=True)
    )
    return load_as_at(
        lambda module: (module.message_queries, module.message_gates),
        connection,
        (message_queries.astype(jnp.float32), gates),
    )


def _load_input_norm(
    norm: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    separate_input_norms: bool,
) -> Normalization:
    if not separate_input_norms:
        return load_rmsnorm(norm, weights_dict, path / "input_layernorm")
    assert norm.scales is not None
    scales = jnp.stack([weights_dict[path / f"{role}_input_layernorm.weight"] for role in ("q", "k", "v")])
    return load_as_at(lambda module: (module.scales,), norm, (scales.astype(norm.scales.dtype),))


def _load_layer(
    layer: TransformerLayer,
    weights_dict: Mapping[str, Array],
    layer_index: int,
    message_queries: Array | None,
    *,
    implementation: CompressionImplementation,
) -> TransformerLayer:
    path = _TEXT_MODEL / "layers" / layer_index
    mixer, mlp = layer.mixer, layer.mlp
    assert isinstance(mixer.connection, BlockAttentionConnection)
    assert isinstance(mlp.connection, BlockAttentionConnection)
    assert isinstance(mixer.body, Attention)
    assert isinstance(mlp.body, MixtureOfExperts)
    assert mixer.pre_norm is not None and mlp.pre_norm is not None
    assert mixer.output_conv is not None and mlp.output_conv is not None

    mixer_connection = _load_connection(
        mixer.connection,
        weights_dict,
        path / "attention_input.query",
        message_queries[:3] if message_queries is not None else None,
        tuple(path / f"{role}_message_gate" for role in ("q", "k", "v")),
    )
    mlp_connection = _load_connection(
        mlp.connection,
        weights_dict,
        path / "mlp_input.query",
        message_queries[3:] if message_queries is not None else None,
        (path / "mlp_message_gate",),
    )
    return load_as_at(
        lambda module: (
            module.mixer.connection,
            module.mixer.pre_norm,
            module.mixer.body,
            module.mixer.output_conv,
            module.mlp.connection,
            module.mlp.pre_norm,
            module.mlp.body,
            module.mlp.output_conv,
        ),
        layer,
        (
            mixer_connection,
            _load_input_norm(
                mixer.pre_norm,
                weights_dict,
                path,
                separate_input_norms=mixer.config.separate_input_norms,
            ),
            _load_attention(mixer.body, weights_dict, path / "self_attn", implementation=implementation),
            _load_conv(
                mixer.output_conv,
                rearrange(
                    weights_dict[path / "post_attention_convolution.conv1d.weight"],
                    "channels taps 1 -> channels taps",
                ),
            ),
            mlp_connection,
            load_rmsnorm(mlp.pre_norm, weights_dict, path / "post_attention_layernorm"),
            _load_experts(mlp.body, weights_dict, path / "mlp", implementation=implementation),
            _load_conv(
                mlp.output_conv,
                rearrange(
                    weights_dict[path / "post_mlp_convolution.conv1d.weight"], "channels taps 1 -> channels taps"
                ),
            ),
        ),
    )


def _truncated_vocabulary(
    weights_dict: Mapping[str, Array],
    paths: Sequence[ParameterPath],
    vocab_size: int,
) -> Mapping[str, Array]:
    rows: dict[str, Array] = {
        path / part: weights_dict[path / part][:vocab_size]
        for path in paths
        for part in ("weight", "scales", "biases")
        if (path / part) in weights_dict
    }
    return _Overlay(rows, weights_dict)


def load_inkling2_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Decoder:
    transformer = module.transformer
    loop_config = transformer.config.loop_config
    if loop_config is None:
        raise ValueError("Inkling 2 Mini runs its middle layers in a loop.")
    assert isinstance(module.embedding, UntiedEmbedding)
    assert isinstance(transformer.residual_head, BlockAttentionHead)
    assert module.embedding_norm is not None

    embedding_path = _TEXT_MODEL / "embed_tokens"
    lm_head_path = ParameterPath("lm_head")
    embedding = load_untied_embedding(
        module.embedding,
        _truncated_vocabulary(weights_dict, (embedding_path, lm_head_path), module.vocab_size),
        embedding_path,
        lm_head_path,
        implementation=implementation,
    )
    all_message_queries = weights_dict[_TEXT_MODEL / "message_input.query"]
    layers = tuple(
        _load_layer(
            layer,
            weights_dict,
            layer_index,
            (
                all_message_queries[layer_index - loop_config.start_layer]
                if loop_config.is_looped(layer_index)
                else None
            ),
            implementation=implementation,
        )
        for layer_index, layer in enumerate(transformer.layers)
    )
    final_query = weights_dict[_TEXT_MODEL / "final_input.query"].astype(jnp.float32)
    residual_head = load_as_at(
        lambda head: (head.query,),
        transformer.residual_head,
        (final_query.reshape(transformer.residual_head.query.shape),),
    )
    assert transformer.output_norm is not None
    return load_as_at(
        lambda decoder: (
            decoder.embedding,
            decoder.embedding_norm,
            decoder.transformer.layers,
            decoder.transformer.residual_head,
            decoder.transformer.output_norm,
        ),
        module,
        (
            embedding,
            load_rmsnorm(module.embedding_norm, weights_dict, _TEXT_MODEL / "embed_norm"),
            layers,
            residual_head,
            load_rmsnorm(transformer.output_norm, weights_dict, _TEXT_MODEL / "norm"),
        ),
    )
