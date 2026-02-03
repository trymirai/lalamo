from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, DTypeLike

from lalamo.common import ParameterPath
from lalamo.modules import (
    Attention,
    AttentionConfig,
    Decoder,
    DeltaNetAttention,
    DeltaNetAttentionConfig,
    DenseMLP,
    FullPrecisionLinear,
    GroupQuantizedLinear,
    LinearBase,
    Mamba2,
    Mamba2Config,
    MLXQuantizedLinear,
    MLXQuantizedTiedEmbedding,
    MLXQuantizedTiedEmbeddingConfig,
    MLXSemiQuantizedUntiedEmbedding,
    Normalization,
    SeparableCausalConv,
    ShortConv,
    ShortConvConfig,
    TiedEmbedding,
    TransformerLayer,
    UntiedEmbedding,
)
from lalamo.modules.classifier import Classifier
from lalamo.modules.embedding import MLXQuantizedUntiedEmbedding
from lalamo.modules.mlp import MixtureOfExperts, MLPBase
from lalamo.quantization import QuantizationMode

from .common import load_parameters
from .utils import decode_mxfp4, deinterleave_pairwise_columns

__all__ = ["load_huggingface_decoder"]


AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _reverse_uint4_order(array: Array, reverse_order: Array) -> Array:
    """Reverses the AWQ packing order to get the logical order of channels for INT4."""
    pack_factor = 32 // 4
    *_, last_dim = array.shape
    if last_dim % pack_factor != 0:
        return array

    array_reshaped = rearrange(
        array,
        "... (group pack_factor) -> ... group pack_factor",
        pack_factor=pack_factor,
    )
    array_reordered = array_reshaped[..., reverse_order]
    return rearrange(array_reordered, "... group pack_factor -> ... (group pack_factor)")


def unpack_int32(packed_weights: Array, mode: QuantizationMode) -> Array:
    assert packed_weights.dtype in (
        jnp.int32,
        jnp.uint32,
    ), f"Expected packed_weights to be of dtype jnp.(u)int32, got {packed_weights.dtype}"
    assert 32 % mode.bits == 0

    shifts = jnp.arange(0, 32, mode.bits)
    mask = (2**mode.bits) - 1
    unpacked = jnp.bitwise_and(jnp.right_shift(packed_weights[:, :, None], shifts[None, None, :]), mask)
    unpacked = rearrange(
        unpacked,
        "out_channels packed_groups packed_values -> out_channels (packed_groups packed_values)",
    )

    return unpacked


def _process_quantized_tensor(
    quantized: Array,
    weight_quantization: QuantizationMode,
    activation_precision: DTypeLike,
    reverse_order: Array | None = None,
) -> Array:
    unpacked = unpack_int32(quantized, weight_quantization)
    if reverse_order is not None:
        assert weight_quantization == QuantizationMode.UINT4, "reverse order only supported on uint4 quant type"
        unpacked = _reverse_uint4_order(unpacked, reverse_order)

    return unpacked.astype(activation_precision)


def _maybe_reorder(array: Array, reorder: tuple[Array, int] | None) -> Array:
    if reorder is None:
        return array
    perm, axis = reorder
    return jnp.take(array, perm, axis=axis)


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    *,
    param_name: str = "weight",
    reorder: tuple[Array, int] | None = None,
) -> Array:
    if sublayers_to_fuse is None:
        return _maybe_reorder(weights_dict[path / param_name], reorder)

    weights = [weights_dict[path / layer_name / param_name] for layer_name in sublayers_to_fuse]
    fused = jnp.concatenate(weights, axis=0)
    return _maybe_reorder(fused, reorder)


@dataclass(frozen=True)
class QuantizedParamLayout:
    weight: str
    scale: str
    bias: str
    transposed: bool


AWQ_QUANTIZED_WEIGHT_LAYOUT = QuantizedParamLayout("qweight", "scales", "qzeros", transposed=True)
MLX_QUANTIZED_WEIGHT_LAYOUT = QuantizedParamLayout("weight", "scales", "biases", transposed=False)


def _build_qkv_gate_reorder(
    q_len: int,
    k_len: int,
    v_len: int,
    q_perm: Array,
    q_output_dim: int,
) -> Array:
    if q_perm.shape[0] != q_len:
        raise ValueError(f"q_perm length {q_perm.shape[0]} does not match q_proj length {q_len}.")
    q_indices = q_perm[:q_output_dim]
    gate_indices = q_perm[q_output_dim:]
    k_indices = jnp.arange(k_len, dtype=jnp.int32) + q_len
    v_indices = jnp.arange(v_len, dtype=jnp.int32) + q_len + k_len
    return jnp.concatenate([q_indices, k_indices, v_indices, gate_indices], axis=0)


def _fuse_quantized_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    quantized_param_layout: QuantizedParamLayout,
    *,
    reorder: tuple[Array, int] | None = None,
) -> tuple[Array, Array, Array]:
    # Note that AWQ quantized weights are stored transposed relative to full-precision weights

    if sublayers_to_fuse is None:
        qweights = weights_dict[path / quantized_param_layout.weight]
        qzeros = weights_dict[path / quantized_param_layout.bias]
        scales = weights_dict[path / quantized_param_layout.scale]
        return (
            _maybe_reorder(qweights, reorder),
            _maybe_reorder(qzeros, reorder),
            _maybe_reorder(scales, reorder),
        )

    qweights = [weights_dict[path / layer_name / quantized_param_layout.weight] for layer_name in sublayers_to_fuse]
    qzeros = [weights_dict[path / layer_name / quantized_param_layout.bias] for layer_name in sublayers_to_fuse]
    scales = [weights_dict[path / layer_name / quantized_param_layout.scale] for layer_name in sublayers_to_fuse]

    fused_qweights = jnp.concatenate(qweights, axis=int(quantized_param_layout.transposed))
    fused_qzeros = jnp.concatenate(qzeros, axis=int(quantized_param_layout.transposed))
    fused_scales = jnp.concatenate(scales, axis=int(quantized_param_layout.transposed))

    return (
        _maybe_reorder(fused_qweights, reorder),
        _maybe_reorder(fused_qzeros, reorder),
        _maybe_reorder(fused_scales, reorder),
    )


def load_linear(
    module: LinearBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None = None,
) -> LinearBase:
    """Loads a linear layer, optionally fusing weights from sublayers."""
    if not module.has_biases:
        if sublayers_to_fuse:
            paths_to_check = [path / proj / "bias" for proj in sublayers_to_fuse]
        else:
            paths_to_check = path / "bias"
        for p in paths_to_check:
            if p in weights_dict:
                raise ValueError(f"Bias tensor found at {p} but module does not support it.")
        bias = None
    elif sublayers_to_fuse is None:
        bias = weights_dict[path / "bias"]
    else:
        bias = jnp.concatenate(
            [weights_dict[path / proj_name / "bias"] for proj_name in sublayers_to_fuse],
            axis=0,
        )

    if isinstance(module, FullPrecisionLinear):
        weights = _fuse_full_precision_weights(weights_dict, path, sublayers_to_fuse)
        return load_parameters(lambda m: (m.weights, m.biases), module, (weights, bias))

    if isinstance(module, GroupQuantizedLinear):
        qweights, qzeros, scales = _fuse_quantized_weights(
            weights_dict,
            path,
            sublayers_to_fuse,
            AWQ_QUANTIZED_WEIGHT_LAYOUT,
        )
        weight_quantization = module.config.weight_quantization_mode
        activation_precision = module.activation_precision

        if weight_quantization == QuantizationMode.UINT4:
            reverse_order = AWQ_UINT4_REVERSE_ORDER
        else:
            reverse_order = None

        weights = _process_quantized_tensor(
            qweights,
            weight_quantization,
            activation_precision,
            reverse_order,
        )
        zeros = _process_quantized_tensor(
            qzeros,
            weight_quantization,
            activation_precision,
            reverse_order,
        )
        scales = scales.astype(activation_precision)

        return load_parameters(
            lambda m: (m.weights, m.scales, m.zero_points, m.biases),
            module,
            (weights.T, scales.T, zeros.T, bias),
        )

    if isinstance(module, MLXQuantizedLinear):
        qweights, deq_biases, scales = _fuse_quantized_weights(
            weights_dict,
            path,
            sublayers_to_fuse,
            MLX_QUANTIZED_WEIGHT_LAYOUT,
        )
        weight_quantization = module.config.weight_quantization_mode
        activation_precision = module.activation_precision

        # MLX models can have per-layer quantization (e.g., 8-bit for MoE router, 4-bit elsewhere).
        # Detect actual quantization from weight shape: UINT4 packs 8 values per int32, UINT8 packs 4.
        expected_in_dim = module.weights.shape[1]
        packed_dim = qweights.shape[1]
        if packed_dim * 8 == expected_in_dim:
            actual_quantization = QuantizationMode.UINT4
        elif packed_dim * 4 == expected_in_dim:
            actual_quantization = QuantizationMode.UINT8
        else:
            actual_quantization = weight_quantization

        weights = _process_quantized_tensor(
            qweights,
            actual_quantization,
            activation_precision,
            None,
        )
        scales = scales.astype(activation_precision)
        deq_biases = deq_biases.astype(activation_precision)

        return load_parameters(
            lambda m: (m.weights, m.scales, m.deq_biases, m.biases),
            module,
            (weights, scales, deq_biases, bias),
        )

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_mlp(
    module: MLPBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
) -> MLPBase:
    if isinstance(module, DenseMLP):
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear(
            module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=[up_proj_key, gate_proj_key],
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / down_proj_key)
        return load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module,
            (up_projection, down_projection),
        )

    if isinstance(module, MixtureOfExperts):
        return load_moe(module, weights_dict, path)

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_moe(module: MixtureOfExperts, weights_dict: Mapping[str, Array], path: ParameterPath) -> MixtureOfExperts:
    # Load router via the standard linear loader.
    # Qwen-MoE often names the router layer "gate" in HF weights.
    if (path / "router.weight") in weights_dict or (path / "router.qweight") in weights_dict:
        router_path = path / "router"
    elif (path / "gate.weight") in weights_dict or (path / "gate.qweight") in weights_dict:
        router_path = path / "gate"
    else:
        router_path = path / "router"
    router = load_linear(module.router, weights_dict, router_path)

    num_routed = module.num_routed_experts
    num_shared = module.num_shared_experts
    has_up_biases = module.experts.up_projection.has_biases
    has_down_biases = module.experts.down_projection.has_biases

    experts_path = path / "experts"

    # Debug: print keys that contain "expert" for the first layer
    if "layers.0" in str(path):
        expert_keys = [k for k in weights_dict if "expert" in str(k).lower() and "layers.0" in str(k)]
        print(f"DEBUG: Expert keys for layer 0 (first 20): {expert_keys[:20]}")
        print(f"DEBUG: Total expert keys for layer 0: {len(expert_keys)}")
        # Also check what the expected paths look like
        print(f"DEBUG: experts_path = {experts_path}")
        print(f"DEBUG: Looking for: {experts_path / 'gate_up_proj.weight'}")
        print(f"DEBUG: Looking for: {experts_path / 'gate_up_proj' / 'weight'}")

    # GPT-OSS uses fused MXFP4 expert weights; detect and decode those.
    if (experts_path / "gate_up_proj_blocks") in weights_dict:
        fused = decode_mxfp4(
            weights_dict[experts_path / "gate_up_proj_blocks"],
            weights_dict[experts_path / "gate_up_proj_scales"],
            dtype=module.activation_precision,
            flatten=False,
        )
        fused_eio = rearrange(fused, "e o ib ie -> e (ib ie) o")
        up_weights, gate_weights = deinterleave_pairwise_columns(fused_eio, first="odd")
        combined_up_gate_weights = jnp.swapaxes(
            jnp.concatenate([up_weights, gate_weights], axis=-1), -1, -2,
        )

        gate_up_bias = weights_dict[experts_path / "gate_up_proj_bias"]
        if gate_up_bias.ndim == 1:
            gate_up_bias = jnp.broadcast_to(
                gate_up_bias, (combined_up_gate_weights.shape[0], gate_up_bias.shape[0]),
            )
        up_bias, gate_bias = deinterleave_pairwise_columns(gate_up_bias, first="odd")
        combined_up_gate_biases = jnp.concatenate([up_bias + 1.0, gate_bias], axis=-1)

        up_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.up_projection,
            (combined_up_gate_weights, combined_up_gate_biases),
        )

        down_weights = decode_mxfp4(
            weights_dict[experts_path / "down_proj_blocks"],
            weights_dict[experts_path / "down_proj_scales"],
            dtype=module.activation_precision,
            flatten=False,
        )
        down_weights = rearrange(down_weights, "e o ib ie -> e o (ib ie)")
        down_biases = weights_dict[experts_path / "down_proj_bias"]
        if down_biases.ndim == 1:
            down_biases = jnp.broadcast_to(down_biases, (*down_weights.shape[:-1], down_biases.shape[0]))

        down_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.down_projection,
            (down_weights, down_biases),
        )

        experts = load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module.experts,
            (up_projection, down_projection),
        )
    elif (
        (experts_path / "gate_up_proj.weight") in weights_dict
        or (experts_path / "gate_up_proj" / "weight") in weights_dict
        or any(str(k).startswith(str(experts_path) + ".gate_up_proj") for k in weights_dict)
    ):
        # MLX/Qwen2Moe batched expert format: gate_up_proj fused, shape (num_experts, hidden*2, model_dim)
        # Check for both flat and nested key formats
        gate_up_path = experts_path / "gate_up_proj"
        down_path = experts_path / "down_proj"
        if (gate_up_path / "weight") in weights_dict:
            gate_up_weights = weights_dict[gate_up_path / "weight"]
            down_weights = weights_dict[down_path / "weight"]
        elif (experts_path / "gate_up_proj.weight") in weights_dict:
            gate_up_weights = weights_dict[experts_path / "gate_up_proj.weight"]
            down_weights = weights_dict[experts_path / "down_proj.weight"]
        else:
            # Find the actual key format
            gate_up_key = next(
                (k for k in weights_dict if str(k).startswith(str(gate_up_path))),
                None,
            )
            if gate_up_key is None:
                raise KeyError(f"Could not find gate_up_proj weights under {gate_up_path}")
            # Infer the weight key suffix
            suffix = str(gate_up_key)[len(str(gate_up_path)):]
            gate_up_weights = weights_dict[gate_up_key]
            down_key = str(down_path) + suffix
            down_weights = weights_dict[ParameterPath(down_key)]

        # gate_up_proj is [num_experts, intermediate_size*2, hidden_size] - split into gate and up
        intermediate_size_2 = gate_up_weights.shape[1]
        intermediate_size = intermediate_size_2 // 2

        # Split gate and up: first half is gate, second half is up (or vice versa depending on model)
        gate_weights = gate_up_weights[:, :intermediate_size, :]
        up_weights = gate_up_weights[:, intermediate_size:, :]

        # Combine up and gate for our format: (num_experts, hidden*2, model_dim)
        combined_up_gate_weights = jnp.concatenate([up_weights, gate_weights], axis=1)

        up_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.up_projection,
            (combined_up_gate_weights, None),
        )

        down_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.down_projection,
            (down_weights, None),
        )

        experts = load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module.experts,
            (up_projection, down_projection),
        )
    else:
        # Collect expert weight paths: routed experts first, then shared experts.
        expert_paths: list[ParameterPath] = [experts_path / str(idx) for idx in range(num_routed)]

        if num_shared > 0:
            if (path / "shared_expert" / "up_proj.weight") in weights_dict:
                if num_shared != 1:
                    raise ValueError("Single shared expert path found but num_shared_experts != 1.")
                expert_paths.append(path / "shared_expert")
            elif (path / "shared_experts" / "0" / "up_proj.weight") in weights_dict:
                for idx in range(num_shared):
                    expert_paths.append(path / "shared_experts" / str(idx))
            else:
                raise KeyError("Could not find shared expert weights in HF checkpoint.")

        up_weight_list: list[Array] = []
        gate_weight_list: list[Array] = []
        down_weight_list: list[Array] = []
        up_bias_list: list[Array] | None = [] if has_up_biases else None
        gate_bias_list: list[Array] | None = [] if has_up_biases else None
        down_bias_list: list[Array] | None = [] if has_down_biases else None

        for expert_path in expert_paths:
            up_weight_list.append(weights_dict[expert_path / "up_proj.weight"])
            gate_weight_list.append(weights_dict[expert_path / "gate_proj.weight"])
            down_weight_list.append(weights_dict[expert_path / "down_proj.weight"])
            if up_bias_list is not None:
                assert gate_bias_list is not None
                up_bias_list.append(weights_dict[expert_path / "up_proj.bias"])
                gate_bias_list.append(weights_dict[expert_path / "gate_proj.bias"])
            if down_bias_list is not None:
                down_bias_list.append(weights_dict[expert_path / "down_proj.bias"])

        stacked_up = jnp.stack(up_weight_list, axis=0)
        stacked_gate = jnp.stack(gate_weight_list, axis=0)
        combined_up_gate_weights = jnp.concatenate([stacked_up, stacked_gate], axis=1)
        if up_bias_list is None:
            combined_up_gate_biases = None
        else:
            assert gate_bias_list is not None
            stacked_up_biases = jnp.stack(up_bias_list, axis=0)
            stacked_gate_biases = jnp.stack(gate_bias_list, axis=0)
            combined_up_gate_biases = jnp.concatenate([stacked_up_biases, stacked_gate_biases], axis=1)

        up_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.up_projection,
            (combined_up_gate_weights, combined_up_gate_biases),
        )

        stacked_down = jnp.stack(down_weight_list, axis=0)
        stacked_down_biases = jnp.stack(down_bias_list, axis=0) if down_bias_list is not None else None
        down_projection = load_parameters(
            lambda m: (m.weights, m.biases),
            module.experts.down_projection,
            (stacked_down, stacked_down_biases),
        )

        experts = load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module.experts,
            (up_projection, down_projection),
        )

    gate = None
    if module.gate is not None:
        gate_path = None
        for candidate in ("shared_expert_gate", "shared_experts_gate", "shared_gate"):
            if (path / candidate / "weight") in weights_dict or (path / candidate / "qweight") in weights_dict:
                gate_path = path / candidate
                break
        if gate_path is None:
            raise KeyError("Could not find shared expert gate weights in HF checkpoint.")
        gate = load_linear(module.gate, weights_dict, gate_path)

    return load_parameters(
        lambda m: (m.router, m.experts, m.gate),
        module,
        (router, experts, gate),
    )


def load_rmsnorm(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization:
    scales = weights_dict[path / "weight"]
    return load_parameters(lambda m: (m.scales,), module, (scales,))


def load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    reorder_q_proj_gate: bool = True,
) -> Attention:
    if (path / "o_proj.weight") in weights_dict or (path / "o_proj.qweight") in weights_dict:
        o_proj_name = "o_proj"
    elif (path / "out_proj.weight") in weights_dict or (path / "out_proj.qweight") in weights_dict:
        o_proj_name = "out_proj"
    else:
        raise NotImplementedError("Can't determine attention output projection name")

    if module.config.has_gate:
        q_output_dim, k_output_dim, v_output_dim = module.qkv_projection.output_dims[:3]
        q_len = q_output_dim * 2
        head_dim = module.head_dim
        num_heads = module.num_heads
        if reorder_q_proj_gate:
            base = jnp.arange(num_heads, dtype=jnp.int32) * (2 * head_dim)
            q = base[:, None] + jnp.arange(head_dim, dtype=jnp.int32)[None, :]
            gate = base[:, None] + head_dim + jnp.arange(head_dim, dtype=jnp.int32)[None, :]
            q_perm = jnp.concatenate([q.reshape(-1), gate.reshape(-1)], axis=0)
        else:
            q_perm = jnp.arange(q_len, dtype=jnp.int32)
        reorder = (_build_qkv_gate_reorder(q_len, k_output_dim, v_output_dim, q_perm, q_output_dim), 0)

        def fuse_bias() -> Array | None:
            if not module.qkv_projection.has_biases:
                for proj in ("q_proj", "k_proj", "v_proj"):
                    if (path / proj / "bias") in weights_dict:
                        raise ValueError(
                            f"Bias tensor found at {path / proj / 'bias'} but module does not support it.",
                        )
                return None
            return _fuse_full_precision_weights(
                weights_dict,
                path,
                ["q_proj", "k_proj", "v_proj"],
                param_name="bias",
                reorder=reorder,
            )

        if isinstance(module.qkv_projection, FullPrecisionLinear):
            weights = _fuse_full_precision_weights(
                weights_dict,
                path,
                ["q_proj", "k_proj", "v_proj"],
                reorder=reorder,
            )
            bias = fuse_bias()

            qkv_projection = load_parameters(
                lambda m: (m.weights, m.biases),
                module.qkv_projection,
                (weights, bias),
            )
        elif isinstance(module.qkv_projection, GroupQuantizedLinear):
            layout = AWQ_QUANTIZED_WEIGHT_LAYOUT
            axis = int(layout.transposed)
            reorder_with_axis = (reorder[0], axis)
            fused_qweights, fused_qzeros, fused_scales = _fuse_quantized_weights(
                weights_dict,
                path,
                ["q_proj", "k_proj", "v_proj"],
                layout,
                reorder=reorder_with_axis,
            )
            bias = fuse_bias()

            weight_quantization = module.qkv_projection.config.weight_quantization_mode
            activation_precision = module.qkv_projection.activation_precision
            reverse_order = AWQ_UINT4_REVERSE_ORDER if weight_quantization == QuantizationMode.UINT4 else None

            weights = _process_quantized_tensor(
                fused_qweights,
                weight_quantization,
                activation_precision,
                reverse_order,
            )
            zeros = _process_quantized_tensor(
                fused_qzeros,
                weight_quantization,
                activation_precision,
                reverse_order,
            )
            scales = fused_scales.astype(activation_precision)

            qkv_projection = load_parameters(
                lambda m: (m.weights, m.scales, m.zero_points, m.biases),
                module.qkv_projection,
                (weights.T, scales.T, zeros.T, bias),
            )
        elif isinstance(module.qkv_projection, MLXQuantizedLinear):
            layout = MLX_QUANTIZED_WEIGHT_LAYOUT
            axis = int(layout.transposed)
            reorder_with_axis = (reorder[0], axis)
            fused_qweights, fused_qzeros, fused_scales = _fuse_quantized_weights(
                weights_dict,
                path,
                ["q_proj", "k_proj", "v_proj"],
                layout,
                reorder=reorder_with_axis,
            )
            bias = fuse_bias()

            weight_quantization = module.qkv_projection.config.weight_quantization_mode
            activation_precision = module.qkv_projection.activation_precision

            weights = _process_quantized_tensor(
                fused_qweights,
                weight_quantization,
                activation_precision,
                None,
            )
            deq_biases = _process_quantized_tensor(
                fused_qzeros,
                weight_quantization,
                activation_precision,
                None,
            )
            scales = fused_scales.astype(activation_precision)

            qkv_projection = load_parameters(
                lambda m: (m.weights, m.scales, m.deq_biases, m.biases),
                module.qkv_projection,
                (weights, scales, deq_biases, bias),
            )
        else:
            raise NotImplementedError("Unsupported qkv projection type for gated attention.")
    else:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path,
            sublayers_to_fuse=["q_proj", "k_proj", "v_proj"],
        )

    out_projection = load_linear(module.out_projection, weights_dict, path / o_proj_name)

    if module.query_norm is not None:
        if (path / "q_norm.weight") in weights_dict:
            q_norm_name = "q_norm"
        elif (path / "q_layernorm.weight") in weights_dict:
            q_norm_name = "q_layernorm"
        else:
            raise NotImplementedError("Can't determine attention query projection parameter name")

        query_norm = load_rmsnorm(module.query_norm, weights_dict, path / q_norm_name)
    else:
        query_norm = None

    if module.key_norm is not None:
        if (path / "k_norm.weight") in weights_dict:
            k_norm_name = "k_norm"
        elif (path / "k_layernorm.weight") in weights_dict:
            k_norm_name = "k_layernorm"
        else:
            raise NotImplementedError("Can't determine attention key projection parameter name")

        key_norm = load_rmsnorm(module.key_norm, weights_dict, path / k_norm_name)
    else:
        key_norm = None

    # GPT-OSS adds per-head attention sinks; load them if present.
    if (path / "sinks") in weights_dict:
        sinks = weights_dict[path / "sinks"]
    else:
        sinks = module.sinks

    return load_parameters(
        lambda m: (
            m.qkv_projection,
            m.out_projection,
            m.query_norm,
            m.key_norm,
            m.sinks,
        ),
        module,
        (qkv_projection, out_projection, query_norm, key_norm, sinks),
    )


def _load_conv(
    conv_module: SeparableCausalConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> SeparableCausalConv:
    weight_path = path / "conv1d" / "weight"
    if weight_path not in weights_dict:
        weight_path = path / "conv_weight"
    if weight_path not in weights_dict:
        weight_path = path / "conv.weight"
    if weight_path not in weights_dict:
        weight_path = None

    if weight_path is not None:
        raw = weights_dict[weight_path]
        if permute_conv:
            raw = jnp.matrix_transpose(raw)
        if raw.ndim == 3:
            # PyTorch Conv1d: (channels, 1, kernel) -> squeeze axis 1
            # MLX Conv1d: (channels, kernel, 1) -> squeeze axis 2
            axis_to_squeeze = 1 if raw.shape[1] == 1 else 2
            conv_weight = raw.squeeze(axis_to_squeeze)
        else:
            conv_weight = raw
    else:
        conv_weight = conv_module.weights

    bias_path = path / "conv1d" / "bias"
    if bias_path not in weights_dict:
        bias_path = path / "conv_bias"
    if bias_path not in weights_dict:
        bias_path = path / "conv.bias"
    if bias_path not in weights_dict:
        bias_path = None

    if bias_path is not None and conv_module.biases is not None:
        conv_bias = weights_dict[bias_path]
    else:
        conv_bias = conv_module.biases

    return load_parameters(
        lambda m: (m.weights, m.biases),
        conv_module,
        (conv_weight, conv_bias),
    )


def load_mamba2(
    module: Mamba2,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> Mamba2:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj")
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj")
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    skip_connection_weight_path = path / "D"
    if skip_connection_weight_path in weights_dict:
        skip_connection_weight = weights_dict[skip_connection_weight_path]
    else:
        skip_connection_weight = module.skip_connection_weight

    gate_bias_path = path / "z_bias"
    if gate_bias_path in weights_dict:
        gate_bias = weights_dict[gate_bias_path]
    else:
        gate_bias = module.gate_bias

    return load_parameters(
        lambda m: (
            m.in_projection,
            m.out_projection,
            m.conv,
            m.skip_connection_weight,
            m.gate_bias,
        ),
        module,
        (in_projection, out_projection, conv, skip_connection_weight, gate_bias),
    )


def load_short_conv(
    module: ShortConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> ShortConv:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj")
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj")
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    return load_parameters(
        lambda m: (m.in_projection, m.out_projection, m.conv),
        module,
        (in_projection, out_projection, conv),
    )


def load_delta_net_attention(
    module: DeltaNetAttention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> DeltaNetAttention:
    def _delta_net_qkvz_perm() -> Array:
        v_per_k = module.num_heads // module.num_groups
        key_block = 2 * module.head_dim
        value_block = module.value_head_dim * v_per_k
        per_group = key_block + 2 * value_block

        base = jnp.arange(module.num_groups, dtype=jnp.int32) * per_group
        q = base[:, None] + jnp.arange(module.head_dim, dtype=jnp.int32)[None, :]
        k = base[:, None] + module.head_dim + jnp.arange(module.head_dim, dtype=jnp.int32)[None, :]
        v = base[:, None] + key_block + jnp.arange(value_block, dtype=jnp.int32)[None, :]
        z = base[:, None] + key_block + value_block + jnp.arange(value_block, dtype=jnp.int32)[None, :]
        return jnp.concatenate([q.reshape(-1), k.reshape(-1), v.reshape(-1), z.reshape(-1)], axis=0)

    def _delta_net_ba_perm() -> Array:
        v_per_k = module.num_heads // module.num_groups
        per_group = 2 * v_per_k
        base = jnp.arange(module.num_groups, dtype=jnp.int32) * per_group
        b = base[:, None] + jnp.arange(v_per_k, dtype=jnp.int32)[None, :]
        a = base[:, None] + v_per_k + jnp.arange(v_per_k, dtype=jnp.int32)[None, :]
        return jnp.concatenate([b.reshape(-1), a.reshape(-1)], axis=0)

    in_proj_path = path / "in_proj"
    in_proj_weight_path = in_proj_path / "weight"
    if in_proj_weight_path in weights_dict:
        in_proj = load_linear(module.in_proj, weights_dict, in_proj_path)
    else:
        qkvz_path = path / "in_proj_qkvz"
        ba_path = path / "in_proj_ba"
        qkvz_weight_path = qkvz_path / "weight"
        ba_weight_path = ba_path / "weight"
        if not (qkvz_weight_path in weights_dict and ba_weight_path in weights_dict):
            raise ValueError("Expected in_proj or in_proj_qkvz/in_proj_ba weights for DeltaNetAttention.")

        qkvz_perm = _delta_net_qkvz_perm()
        ba_perm = _delta_net_ba_perm()

        def _reorder(array: Array, perm: Array, axis: int) -> Array:
            return jnp.take(array, perm, axis=axis)

        if isinstance(module.in_proj, FullPrecisionLinear):
            qkvz_weight = _fuse_full_precision_weights(weights_dict, qkvz_path, None)
            ba_weight = _fuse_full_precision_weights(weights_dict, ba_path, None)
            qkvz_weight = _reorder(qkvz_weight, qkvz_perm, axis=0)
            ba_weight = _reorder(ba_weight, ba_perm, axis=0)
            merged = jnp.concatenate([qkvz_weight, ba_weight], axis=0)
            in_proj = load_parameters(lambda m: (m.weights, m.biases), module.in_proj, (merged, None))
        elif isinstance(module.in_proj, GroupQuantizedLinear):
            qweights, qzeros, qscales = _fuse_quantized_weights(
                weights_dict,
                qkvz_path,
                None,
                AWQ_QUANTIZED_WEIGHT_LAYOUT,
            )
            bweights, bzeros, bscales = _fuse_quantized_weights(
                weights_dict,
                ba_path,
                None,
                AWQ_QUANTIZED_WEIGHT_LAYOUT,
            )
            axis = int(AWQ_QUANTIZED_WEIGHT_LAYOUT.transposed)
            qweights = _reorder(qweights, qkvz_perm, axis=axis)
            qzeros = _reorder(qzeros, qkvz_perm, axis=axis)
            qscales = _reorder(qscales, qkvz_perm, axis=axis)
            bweights = _reorder(bweights, ba_perm, axis=axis)
            bzeros = _reorder(bzeros, ba_perm, axis=axis)
            bscales = _reorder(bscales, ba_perm, axis=axis)

            fused_qweights = jnp.concatenate([qweights, bweights], axis=axis)
            fused_qzeros = jnp.concatenate([qzeros, bzeros], axis=axis)
            fused_scales = jnp.concatenate([qscales, bscales], axis=axis)

            weight_quantization = module.in_proj.config.weight_quantization_mode
            activation_precision = module.in_proj.activation_precision
            reverse_order = AWQ_UINT4_REVERSE_ORDER if weight_quantization == QuantizationMode.UINT4 else None

            weights = _process_quantized_tensor(
                fused_qweights,
                weight_quantization,
                activation_precision,
                reverse_order,
            )
            zeros = _process_quantized_tensor(
                fused_qzeros,
                weight_quantization,
                activation_precision,
                reverse_order,
            )
            scales = fused_scales.astype(activation_precision)

            in_proj = load_parameters(
                lambda m: (m.weights, m.scales, m.zero_points, m.biases),
                module.in_proj,
                (weights.T, scales.T, zeros.T, None),
            )
        elif isinstance(module.in_proj, MLXQuantizedLinear):
            qweights, qdeq_biases, qscales = _fuse_quantized_weights(
                weights_dict,
                qkvz_path,
                None,
                MLX_QUANTIZED_WEIGHT_LAYOUT,
            )
            bweights, bdeq_biases, bscales = _fuse_quantized_weights(
                weights_dict,
                ba_path,
                None,
                MLX_QUANTIZED_WEIGHT_LAYOUT,
            )
            axis = int(MLX_QUANTIZED_WEIGHT_LAYOUT.transposed)
            qweights = _reorder(qweights, qkvz_perm, axis=axis)
            qdeq_biases = _reorder(qdeq_biases, qkvz_perm, axis=axis)
            qscales = _reorder(qscales, qkvz_perm, axis=axis)
            bweights = _reorder(bweights, ba_perm, axis=axis)
            bdeq_biases = _reorder(bdeq_biases, ba_perm, axis=axis)
            bscales = _reorder(bscales, ba_perm, axis=axis)

            fused_qweights = jnp.concatenate([qweights, bweights], axis=axis)
            fused_deq_biases = jnp.concatenate([qdeq_biases, bdeq_biases], axis=axis)
            fused_scales = jnp.concatenate([qscales, bscales], axis=axis)

            weight_quantization = module.in_proj.config.weight_quantization_mode
            activation_precision = module.in_proj.activation_precision

            weights = _process_quantized_tensor(
                fused_qweights,
                weight_quantization,
                activation_precision,
                None,
            )
            scales = fused_scales.astype(activation_precision)
            deq_biases = fused_deq_biases.astype(activation_precision)

            in_proj = load_parameters(
                lambda m: (m.weights, m.scales, m.deq_biases, m.biases),
                module.in_proj,
                (weights, scales, deq_biases, None),
            )
        else:
            raise TypeError(f"Unsupported DeltaNetAttention in_proj type: {type(module.in_proj)}")
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)
    out_proj = load_linear(module.out_proj, weights_dict, path / "out_proj")
    norm = load_rmsnorm(module.norm, weights_dict, path / "norm")

    dt_bias_path = path / "dt_bias"
    if dt_bias_path in weights_dict:
        dt_bias = weights_dict[dt_bias_path]
    else:
        dt_bias = module.dt_bias

    a_log_path = path / "A_log"
    if a_log_path in weights_dict:
        a_log = weights_dict[a_log_path]
    else:
        a_log = module.a_log

    return load_parameters(
        lambda m: (
            m.in_proj,
            m.conv,
            m.out_proj,
            m.norm,
            m.dt_bias,
            m.a_log,
        ),
        module,
        (in_proj, conv, out_proj, norm, dt_bias, a_log),
    )


def load_transformer_layer(
    module: TransformerLayer,
    weights_dict: Mapping[str, Array],
    mixer_path: ParameterPath,
    mlp_path: ParameterPath,
    mixer_key: str,
    mlp_key: str,
    pre_mixer_norm_key: str,
    pre_mlp_norm_key: str,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
    permute_conv: bool,
    *,
    reorder_q_proj_gate: bool = True,
) -> TransformerLayer:
    if module.pre_mixer_norm is not None:
        pre_attention_norm = load_rmsnorm(
            module.pre_mixer_norm,
            weights_dict,
            mixer_path / pre_mixer_norm_key,
        )

    else:
        pre_attention_norm = None
    # Load mixer (attention or mamba)
    if isinstance(module.mixer, Attention):
        mixer = load_attention(
            module.mixer,
            weights_dict,
            mixer_path / mixer_key,
            reorder_q_proj_gate=reorder_q_proj_gate,
        )
    elif isinstance(module.mixer, DeltaNetAttention):
        mixer = load_delta_net_attention(module.mixer, weights_dict, mixer_path / mixer_key, permute_conv)
    elif isinstance(module.mixer, Mamba2):
        mixer = load_mamba2(module.mixer, weights_dict, mixer_path / mixer_key, permute_conv)
    elif isinstance(module.mixer, ShortConv):
        mixer = load_short_conv(module.mixer, weights_dict, mixer_path / mixer_key, permute_conv)
    else:
        mixer = module.mixer

    if module.post_mixer_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_mixer_norm,
            weights_dict,
            mixer_path / "post_attention_layernorm",
        )

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            mlp_path / "pre_feedforward_layernorm",
        )
    else:
        post_attention_norm = None

        pre_mlp_norm = load_rmsnorm(
            module.pre_mlp_norm,
            weights_dict,
            mlp_path / pre_mlp_norm_key,
        )

    mlp = load_mlp(
        module.mlp,
        weights_dict,
        mlp_path / mlp_key,
        up_proj_key,
        gate_proj_key,
        down_proj_key,
    )

    if module.post_mlp_norm is not None:
        post_mlp_norm = load_rmsnorm(
            module.post_mlp_norm,
            weights_dict,
            mlp_path / "post_feedforward_layernorm",
        )
    else:
        post_mlp_norm = None

    return load_parameters(
        lambda m: (
            m.pre_mixer_norm,
            m.mixer,
            m.post_mixer_norm,
            m.pre_mlp_norm,
            m.mlp,
            m.post_mlp_norm,
        ),
        module,
        (
            pre_attention_norm,
            mixer,
            post_attention_norm,
            pre_mlp_norm,
            mlp,
            post_mlp_norm,
        ),
    )


def load_tied_embedding(
    module: TiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
) -> TiedEmbedding:
    weights = weights_dict[embedding_path / "weight"]
    return load_parameters(lambda m: (m.weights,), module, (weights,))


def load_mlx_quantized_tied_embedding(
    module: MLXQuantizedTiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
) -> MLXQuantizedTiedEmbedding:
    qweights = weights_dict[embedding_path / "weight"]
    qscales = weights_dict[embedding_path / "scales"]
    qbiases = weights_dict[embedding_path / "biases"]

    weights = _process_quantized_tensor(
        qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    scales = qscales.astype(module.activation_precision)
    biases = qbiases.astype(module.activation_precision)

    return load_parameters(lambda m: (m.weights, m.scales, m.biases), module, (weights, scales, biases))


def load_mlx_quantized_untied_embedding(
    module: MLXQuantizedUntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> MLXQuantizedUntiedEmbedding:
    input_qweights = weights_dict[embedding_path / "weight"]
    input_qscales = weights_dict[embedding_path / "scales"]
    input_qbiases = weights_dict[embedding_path / "biases"]
    output_qweights = weights_dict[lm_head_path / "weight"]
    output_qscales = weights_dict[lm_head_path / "scales"]
    output_qbiases = weights_dict[lm_head_path / "biases"]

    input_weights = _process_quantized_tensor(
        input_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    input_scales = input_qscales.astype(module.activation_precision)
    input_biases = input_qbiases.astype(module.activation_precision)

    output_weights = _process_quantized_tensor(
        output_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    output_scales = output_qscales.astype(module.activation_precision)
    output_biases = output_qbiases.astype(module.activation_precision)

    return load_parameters(
        lambda m: (
            m.input_weights,
            m.input_scales,
            m.input_biases,
            m.output_weights,
            m.output_scales,
            m.output_biases,
        ),
        module,
        (input_weights, input_scales, input_biases, output_weights, output_scales, output_biases),
    )


def load_mlx_semi_quantized_untied_embedding(
    module: MLXSemiQuantizedUntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> MLXSemiQuantizedUntiedEmbedding:
    input_weights = weights_dict[embedding_path / "weight"]

    output_qweights = weights_dict[lm_head_path / "weight"]
    output_qscales = weights_dict[lm_head_path / "scales"]
    output_qbiases = weights_dict[lm_head_path / "biases"]

    output_weights = _process_quantized_tensor(
        output_qweights,
        module.config.embedding_quantization_mode,
        module.activation_precision,
        None,
    )
    output_scales = output_qscales.astype(module.activation_precision)
    output_biases = output_qbiases.astype(module.activation_precision)

    return load_parameters(
        lambda m: (m.input_weights, m.output_weights, m.output_scales, m.output_biases),
        module,
        (input_weights, output_weights, output_scales, output_biases),
    )


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
) -> UntiedEmbedding:
    input_weights = weights_dict[embedding_path / "weight"]
    output_weights = weights_dict[lm_head_path / "weight"]
    return load_parameters(
        lambda m: (m.input_weights, m.output_weights),
        module,
        (input_weights, output_weights),
    )


def load_huggingface_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
    *,
    reorder_q_proj_gate: bool = True,
) -> Decoder:
    if any(key.startswith("language_model.") for key in weights_dict):
        base_path = ParameterPath("language_model")
    else:
        base_path = ParameterPath()

    is_llamba_full_precision = any(key.startswith("backbone.") for key in weights_dict)
    is_llamba_mlx = any(key.startswith("embedding.encoder.") for key in weights_dict)
    is_lfm2 = any(key.startswith("model.layers.0.operator_norm.weight") for key in weights_dict)
    if is_llamba_full_precision:
        decoder_path = base_path / "backbone"
        embedding_path = decoder_path / "embedding"
        pre_mixer_norm_key = "input_layernorm"
        mixer_key = {Mamba2Config: "mixer"}
        permute_conv = False
        pre_mlp_norm_key = "post_attention_layernorm"
        mlp_key = "mlp"
        up_proj_key = "up_proj"
        gate_proj_key = "gate_proj"
        down_proj_key = "down_proj"
        alternating_layers = False
        norm_key = "final_layernorm"
        lm_head_path = base_path / "lm_head"
    elif is_llamba_mlx:
        decoder_path = base_path / "model"
        embedding_path = base_path / "embedding.encoder"
        pre_mixer_norm_key = "norm"
        mixer_key = {Mamba2Config: "layer"}
        permute_conv = False
        pre_mlp_norm_key = "norm"
        mlp_key = "layer"
        up_proj_key = "gate_proj"
        gate_proj_key = "in_proj"
        down_proj_key = "out_proj"
        alternating_layers = True
        norm_key = "norm"
        lm_head_path = base_path / "head.linear"
    elif is_lfm2:
        decoder_path = base_path / "model"
        embedding_path = decoder_path / "embed_tokens"
        pre_mixer_norm_key = "operator_norm"
        mixer_key = {ShortConvConfig: "conv", AttentionConfig: "self_attn"}
        permute_conv = isinstance(module.config.embedding_config, MLXQuantizedTiedEmbeddingConfig)
        pre_mlp_norm_key = "ffn_norm"
        mlp_key = "feed_forward"
        up_proj_key = "w3"
        gate_proj_key = "w1"
        down_proj_key = "w2"
        alternating_layers = False
        norm_key = "embedding_norm"
        lm_head_path = base_path / "lm_head"
    else:
        decoder_path = base_path / "model"
        embedding_path = decoder_path / "embed_tokens"
        pre_mixer_norm_key = "input_layernorm"
        mixer_key = {AttentionConfig: "self_attn", DeltaNetAttentionConfig: "linear_attn"}
        permute_conv = False
        pre_mlp_norm_key = "post_attention_layernorm"
        mlp_key = "mlp"
        up_proj_key = "up_proj"
        gate_proj_key = "gate_proj"
        down_proj_key = "down_proj"
        alternating_layers = False
        norm_key = "norm"
        lm_head_path = base_path / "lm_head"

    if isinstance(module.embedding, TiedEmbedding):
        embedding = load_tied_embedding(module.embedding, weights_dict, embedding_path)
    elif isinstance(module.embedding, MLXQuantizedTiedEmbedding):
        embedding = load_mlx_quantized_tied_embedding(module.embedding, weights_dict, embedding_path)
    elif isinstance(module.embedding, MLXQuantizedUntiedEmbedding):
        embedding = load_mlx_quantized_untied_embedding(module.embedding, weights_dict, embedding_path, lm_head_path)
    elif isinstance(module.embedding, MLXSemiQuantizedUntiedEmbedding):
        embedding = load_mlx_semi_quantized_untied_embedding(
            module.embedding,
            weights_dict,
            embedding_path,
            lm_head_path,
        )
    elif isinstance(module.embedding, UntiedEmbedding):
        embedding = load_untied_embedding(module.embedding, weights_dict, embedding_path, lm_head_path)
    else:
        raise TypeError(f"Unsupported embedding type: {type(module.embedding)}")

    decoder_layers = tuple(
        load_transformer_layer(
            layer,
            weights_dict,
            decoder_path / "layers" / ((i * 2) if alternating_layers else i),
            decoder_path / "layers" / ((i * 2 + 1) if alternating_layers else i),
            mixer_key[type(layer.config.mixer_config)],
            mlp_key,
            pre_mixer_norm_key,
            pre_mlp_norm_key,
            up_proj_key,
            gate_proj_key,
            down_proj_key,
            permute_conv,
            reorder_q_proj_gate=reorder_q_proj_gate,
        )
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / norm_key)
    return load_parameters(
        lambda m: (m.embedding, m.transformer.layers, m.transformer.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )


def load_huggingface_classifier(
    module: Classifier,
    weights_dict: Mapping[str, Array],
) -> Classifier:
    def load_tied_embedding_local(
        module: TiedEmbedding,
        weights_dict: Mapping[str, Array],
        decoder_path: ParameterPath,
    ) -> TiedEmbedding:
        input_weights = weights_dict[decoder_path / "embeddings" / "tok_embeddings" / "weight"]
        return load_parameters(lambda m: (m.weights,), module, (input_weights,))

    def load_linear_with_reshufling(
        module: LinearBase,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> LinearBase:
        """Loads a linear layer and reshufle some weights in resulting matrix to meet
        requirements of downstream 'split' in MLP layer in attention."""

        assert not module.has_biases, "Expecting no biases in FullPrecisionLinear"
        assert isinstance(module, FullPrecisionLinear), "Expecting FullPrecisionLinear module as input"

        weights = weights_dict[path / "weight"]
        rows, _ = weights.shape
        shuffled_weights = jnp.vstack((weights[rows // 2 :, :], weights[: rows // 2, :]))
        return load_parameters(lambda m: (m.weights, m.biases), module, (shuffled_weights, None))

    def load_attention_local(
        module: Attention,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> Attention:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path / "Wqkv",
            sublayers_to_fuse=None,
        )
        out_projection = load_linear(module.out_projection, weights_dict, path / "Wo")

        if module.query_norm is not None:
            query_norm = load_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
        else:
            query_norm = None

        if module.key_norm is not None:
            key_norm = load_rmsnorm(module.key_norm, weights_dict, path / "k_norm")
        else:
            key_norm = None

        return load_parameters(
            lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
            module,
            (qkv_projection, out_projection, query_norm, key_norm),
        )

    def load_mlp_local(module: MLPBase, weights_dict: Mapping[str, Array], path: ParameterPath) -> MLPBase:
        assert isinstance(module, DenseMLP)
        up_projection = load_linear_with_reshufling(
            module.up_projection,
            weights_dict,
            path / "Wi",
        )
        down_projection = load_linear(module.down_projection, weights_dict, path / "Wo")
        return load_parameters(
            lambda m: (m.up_projection, m.down_projection),
            module,
            (up_projection, down_projection),
        )

    def load_transformer_layer_local(
        module: TransformerLayer,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> TransformerLayer:
        if module.pre_mixer_norm is not None:
            pre_attention_norm = load_rmsnorm(
                module.pre_mixer_norm,
                weights_dict,
                path / "attn_norm",
            )
        else:
            pre_attention_norm = None

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(module.mixer, weights_dict, path / "attn")
        if module.post_mixer_norm is not None:
            post_attention_norm = load_rmsnorm(
                module.post_mixer_norm,
                weights_dict,
                path / "post_attention_layernorm",
            )

            pre_mlp_norm = load_rmsnorm(
                module.pre_mlp_norm,
                weights_dict,
                path / "pre_feedforward_layernorm",
            )
        else:
            post_attention_norm = None

            pre_mlp_norm = load_rmsnorm(
                module.pre_mlp_norm,
                weights_dict,
                path / "mlp_norm",
            )

        mlp = load_mlp_local(module.mlp, weights_dict, path / "mlp")
        if module.post_mlp_norm is not None:
            post_mlp_norm = load_rmsnorm(
                module.post_mlp_norm,
                weights_dict,
                path / "post_feedforward_layernorm",
            )
        else:
            post_mlp_norm = None
        return load_parameters(
            lambda m: (
                m.pre_mixer_norm,
                m.mixer,
                m.post_mixer_norm,
                m.pre_mlp_norm,
                m.mlp,
                m.post_mlp_norm,
            ),
            module,
            (
                pre_attention_norm,
                attention,
                post_attention_norm,
                pre_mlp_norm,
                mlp,
                post_mlp_norm,
            ),
        )

    base_path = ParameterPath()
    decoder_path = base_path / "model"
    head_path = base_path / "head"
    classifier_path = base_path / "classifier"
    assert isinstance(module.embedding, TiedEmbedding)
    embedding = load_tied_embedding_local(module.embedding, weights_dict, decoder_path)
    embedding_norm = load_rmsnorm(module.embedding_norm, weights_dict, base_path / "model" / "embeddings" / "norm")

    decoder_layers = tuple(
        load_transformer_layer_local(layer, weights_dict, decoder_path / "layers" / i)
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / "final_norm")
    head_dense = load_linear(module.prediction_head.dense, weights_dict, head_path / "dense")
    head_norm = load_rmsnorm(module.prediction_head.norm, weights_dict, head_path / "norm")
    head_readout = load_linear(module.prediction_head.readout, weights_dict, classifier_path)
    return load_parameters(
        lambda m: (
            m.embedding,
            m.embedding_norm,
            m.transformer.layers,
            m.transformer.output_norm,
            m.prediction_head.dense,
            m.prediction_head.norm,
            m.prediction_head.readout,
        ),
        module,
        (
            embedding,
            embedding_norm,
            decoder_layers,
            output_norm,
            head_dense,
            head_norm,
            head_readout,
        ),
    )
