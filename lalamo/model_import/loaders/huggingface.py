from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from lalamo.compressed import IntMatrix, IntSpec, MLXMatrix, MLXSpec
from lalamo.compressed.utils.packing import pack_uint_to_uint8
from lalamo.modules.classifier import Classifier
from lalamo.modules.decoder import Decoder
from lalamo.modules.embedding import TiedEmbedding, UntiedEmbedding
from lalamo.modules.linear import Linear
from lalamo.modules.mlp import DenseMLP, MixtureOfExperts, MLPBase, ParallelMLP
from lalamo.modules.normalization import Normalization
from lalamo.modules.token_mixers.attention import Attention, AttentionConfig, AttentionProjectionMode
from lalamo.modules.token_mixers.convolutions import SeparableCausalConv
from lalamo.modules.token_mixers.deltanet import DeltaNet, DeltaNetConfig
from lalamo.modules.token_mixers.mamba import Mamba2, Mamba2Config
from lalamo.modules.token_mixers.short_conv import ShortConv, ShortConvConfig
from lalamo.modules.transformer_layer import TransformerLayer
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation, Layout, WeightMatrix

from .common import load_full_precision
from .utils import decode_mxfp4, deinterleave_pairwise_columns

__all__ = ["load_huggingface_decoder", "load_linear", "load_rmsnorm"]


AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _update_linear(module: Linear, weights: WeightMatrix, biases: Array | None) -> Linear:
    return eqx.tree_at(lambda m: (m.weights, m.biases), module, (weights, biases))


def _dense_mlp_projections(module: DenseMLP) -> tuple[Linear, Linear]:
    return module.up_projection, module.down_projection


def _parallel_mlp_parts(module: ParallelMLP) -> tuple[MLPBase, Normalization, MLPBase, Normalization]:
    return module.primary_mlp, module.primary_output_norm, module.parallel_mlp, module.parallel_output_norm


def _first_path(weights_dict: Mapping[str, Array], paths: Sequence[ParameterPath]) -> ParameterPath | None:
    return next((path for path in paths if path in weights_dict), None)


def _has_prefix(weights_dict: Mapping[str, Array], path: ParameterPath) -> bool:
    if not path:
        return bool(weights_dict)
    return any(key.startswith(f"{path}.") for key in weights_dict)


def _projection_path(weights_dict: Mapping[str, Array], path: ParameterPath, names: Sequence[str]) -> ParameterPath:
    for name in names:
        candidate = path / name
        if (candidate / "weight") in weights_dict or (candidate / "qweight") in weights_dict:
            return candidate
    raise ValueError(f"Cannot find projection under {path}; tried {', '.join(names)}")


def _supported_quantization_bits(bits: int) -> Literal[4, 8]:
    if bits == 4:
        return 4
    if bits == 8:
        return 8
    raise ValueError(f"Unsupported quantization bit width: {bits}")


def _reverse_uint4_order(array: Array, reverse_order: Array) -> Array:
    """Reverses the AutoAWQ packing order to get the logical order of channels for INT4."""
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


def unpack_int32(packed_weights: Array, bits: int) -> Array:
    assert packed_weights.dtype in (jnp.int32, jnp.uint32)
    assert 32 % bits == 0

    shifts = jnp.arange(0, 32, bits, dtype=jnp.uint32)
    mask = jnp.asarray((2**bits) - 1, dtype=jnp.uint32)
    packed_unsigned = packed_weights.astype(jnp.uint32)
    unpacked = jnp.bitwise_and(jnp.right_shift(packed_unsigned[:, :, None], shifts[None, None, :]), mask)
    return rearrange(
        unpacked,
        "rows packed_groups packed_values -> rows (packed_groups packed_values)",
    )


def _fuse_full_precision_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    *,
    param_name: str = "weight",
) -> Array:
    if sublayers_to_fuse is None:
        return weights_dict[path / param_name]

    weights = [weights_dict[path / layer_name / param_name] for layer_name in sublayers_to_fuse]
    return jnp.concatenate(weights, axis=0)


def _fuse_awq_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> tuple[Array, Array | None, Array]:
    if sublayers_to_fuse is None:
        qzeros_path = path / "qzeros"
        qzeros = weights_dict.get(qzeros_path)
        return weights_dict[path / "qweight"], qzeros, weights_dict[path / "scales"]

    qzeros_paths = [path / layer_name / "qzeros" for layer_name in sublayers_to_fuse]
    if all(qzeros_path in weights_dict for qzeros_path in qzeros_paths):
        qzeros = jnp.concatenate([weights_dict[qzeros_path] for qzeros_path in qzeros_paths], axis=1)
    elif any(qzeros_path in weights_dict for qzeros_path in qzeros_paths):
        raise ValueError("Cannot fuse AWQ layers with mixed symmetric and asymmetric zero-point parameters.")
    else:
        qzeros = None

    return (
        jnp.concatenate([weights_dict[path / layer_name / "qweight"] for layer_name in sublayers_to_fuse], axis=1),
        qzeros,
        jnp.concatenate([weights_dict[path / layer_name / "scales"] for layer_name in sublayers_to_fuse], axis=1),
    )


def _fuse_mlx_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> tuple[Array, Array, Array]:
    if sublayers_to_fuse is None:
        return (
            weights_dict[path / "weight"],
            weights_dict[path / "biases"],
            weights_dict[path / "scales"],
        )
    return (
        jnp.concatenate([weights_dict[path / layer_name / "weight"] for layer_name in sublayers_to_fuse], axis=0),
        jnp.concatenate([weights_dict[path / layer_name / "biases"] for layer_name in sublayers_to_fuse], axis=0),
        jnp.concatenate([weights_dict[path / layer_name / "scales"] for layer_name in sublayers_to_fuse], axis=0),
    )


def _load_bias(
    module: Linear,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
) -> Array | None:
    if not module.has_biases:
        paths_to_check = [path / proj / "bias" for proj in sublayers_to_fuse] if sublayers_to_fuse else [path / "bias"]
        for p in paths_to_check:
            if p in weights_dict:
                raise ValueError(f"Bias tensor found at {p} but module does not support it.")
        return None
    if sublayers_to_fuse is None:
        bias = weights_dict[path / "bias"]
    else:
        bias = jnp.concatenate(
            [weights_dict[path / proj_name / "bias"] for proj_name in sublayers_to_fuse],
            axis=0,
        )
    return bias.astype(module.weights.dtype)


def _is_awq(weights_dict: Mapping[str, Array], path: ParameterPath, sublayers_to_fuse: list[str] | None) -> bool:
    probe = path / sublayers_to_fuse[0] if sublayers_to_fuse else path
    return (probe / "qweight") in weights_dict


def _is_mlx(weights_dict: Mapping[str, Array], path: ParameterPath, sublayers_to_fuse: list[str] | None) -> bool:
    probe = path / sublayers_to_fuse[0] if sublayers_to_fuse else path
    return (probe / "scales") in weights_dict


def _load_matrix(
    template: WeightMatrix,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    *,
    layout: Layout,
    expected_grouped_channels: int,
    full_precision_weights: Callable[[], Array],
    implementation: CompressionImplementation,
) -> WeightMatrix:
    sharding_config = template.sharding_config
    if _is_awq(weights_dict, path, sublayers_to_fuse):
        return _load_awq_array(
            weights_dict,
            path,
            sublayers_to_fuse,
            template=template,
            layout=layout,
            implementation=implementation,
            sharding_config=sharding_config,
        )
    if _is_mlx(weights_dict, path, sublayers_to_fuse):
        return _load_mlx_matrix(
            weights_dict,
            path,
            sublayers_to_fuse,
            expected_grouped_channels,
            template=template,
            layout=layout,
            implementation=implementation,
            sharding_config=sharding_config,
        )
    return load_full_precision(template, full_precision_weights())


def _load_awq_array(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    *,
    template: WeightMatrix,
    layout: Layout,
    implementation: CompressionImplementation,
    sharding_config: ShardingConfig,
) -> IntMatrix:
    packed_qweights, packed_qzeros, scales = _fuse_awq_weights(weights_dict, path, sublayers_to_fuse)
    # AutoAWQ HF layout: qweight [in_channels, out_packed], scales [num_groups, out_channels]
    out_channels = scales.shape[1]
    num_groups = scales.shape[0]
    pack_factor = out_channels // packed_qweights.shape[1]
    bits = 32 // pack_factor
    group_size = packed_qweights.shape[0] // num_groups

    unpacked_weights = unpack_int32(packed_qweights, bits)
    unpacked_zeros = None
    if packed_qzeros is not None:
        unpacked_zeros = unpack_int32(packed_qzeros, bits)
    if bits == 4:
        unpacked_weights = _reverse_uint4_order(unpacked_weights, AWQ_UINT4_REVERSE_ORDER)
        if unpacked_zeros is not None:
            unpacked_zeros = _reverse_uint4_order(unpacked_zeros, AWQ_UINT4_REVERSE_ORDER)

    weight_sharding = sharding_config.resolve_sharding(
        layout.weight_partition(unpacked_weights.ndim - 2, is_sharded=template.is_sharded),
    )
    weight_values = jax.device_put(unpacked_weights.T.astype(template.dtype), weight_sharding)
    scale_values = jax.device_put(scales.T.astype(template.dtype), weight_sharding)
    if unpacked_zeros is None:
        zero_point_values = None
    else:
        zero_point_values = jax.device_put(unpacked_zeros.T.astype(template.dtype), weight_sharding)

    spec = IntSpec(
        bits=_supported_quantization_bits(bits),
        group_size=group_size,
        is_symmetric=zero_point_values is None,
        layout=layout,
    )
    if zero_point_values is None:
        packed_zero_points = None
    else:
        packed_zero_points = pack_uint_to_uint8(zero_point_values, bits, sharding_config=sharding_config)
    return spec.from_packed_parameters(
        packed_weights=pack_uint_to_uint8(weight_values, bits, sharding_config=sharding_config),
        scales=scale_values,
        packed_zero_points=packed_zero_points,
        implementation=implementation,
        sharding_config=sharding_config,
        is_sharded=template.is_sharded,
    )


def _load_mlx_matrix(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None,
    expected_grouped_channels: int,
    *,
    template: WeightMatrix,
    layout: Layout,
    implementation: CompressionImplementation,
    sharding_config: ShardingConfig,
) -> MLXMatrix:
    packed_weights, deq_biases, scales = _fuse_mlx_weights(weights_dict, path, sublayers_to_fuse)
    return _load_packed_mlx_matrix(
        packed_weights,
        deq_biases,
        scales,
        expected_grouped_channels,
        template=template,
        layout=layout,
        implementation=implementation,
        sharding_config=sharding_config,
    )


def _load_packed_mlx_matrix(
    packed_weights: Array,
    deq_biases: Array,
    scales: Array,
    expected_grouped_channels: int,
    *,
    template: WeightMatrix,
    layout: Layout,
    implementation: CompressionImplementation,
    sharding_config: ShardingConfig,
) -> MLXMatrix:
    # MLX HF layout: weight [rows, packed_cols], scales [rows, num_groups].
    packed_in = packed_weights.shape[-1]
    num_groups = scales.shape[-1]
    for bits in (1, 4, 8):
        if packed_in * (32 // bits) == expected_grouped_channels:
            break
    else:
        raise ValueError(f"Cannot infer MLX bits: packed_in={packed_in}, expected_cols={expected_grouped_channels}")

    group_size = expected_grouped_channels // num_groups
    unpacked_weights = unpack_int32(packed_weights, bits)

    weight_sharding = sharding_config.resolve_sharding(
        layout.weight_partition(unpacked_weights.ndim - 2, is_sharded=template.is_sharded),
    )
    weight_values = jax.device_put(unpacked_weights.astype(template.dtype), weight_sharding)
    scale_values = jax.device_put(scales.astype(template.dtype), weight_sharding)
    bias_values = jax.device_put(deq_biases.astype(template.dtype), weight_sharding)

    spec = MLXSpec(bits=bits, group_size=group_size, layout=layout)
    return spec.from_packed_parameters(
        packed_weights=pack_uint_to_uint8(weight_values, bits, sharding_config=sharding_config),
        scales=scale_values,
        biases=bias_values,
        implementation=implementation,
        sharding_config=sharding_config,
        is_sharded=template.is_sharded,
    )


def load_linear(
    module: Linear,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    sublayers_to_fuse: list[str] | None = None,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Linear:
    bias = _load_bias(module, weights_dict, path, sublayers_to_fuse)

    weights = _load_matrix(
        module.weights,
        weights_dict,
        path,
        sublayers_to_fuse,
        layout=Layout.OUTPUT_INPUT,
        expected_grouped_channels=module.input_dim,
        full_precision_weights=lambda: _fuse_full_precision_weights(weights_dict, path, sublayers_to_fuse),
        implementation=implementation,
    )
    return _update_linear(module, weights, bias)


def load_mlp(
    module: MLPBase,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> MLPBase:
    if isinstance(module, DenseMLP):
        dense_module: DenseMLP = module
        # Standard dense MLP with separate sublayers.
        up_projection = load_linear(
            dense_module.up_projection,
            weights_dict,
            path,
            sublayers_to_fuse=[up_proj_key, gate_proj_key],
            implementation=implementation,
        )
        down_projection = load_linear(
            dense_module.down_projection,
            weights_dict,
            path / down_proj_key,
            implementation=implementation,
        )
        return load_as_at(
            _dense_mlp_projections,
            dense_module,
            (up_projection, down_projection),
        )

    if isinstance(module, MixtureOfExperts):
        return load_moe(module, weights_dict, path, implementation=implementation)

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_parallel_mlp(
    module: ParallelMLP,
    weights_dict: Mapping[str, Array],
    layer_path: ParameterPath,
    primary_mlp_key: str,
    up_proj_key: str,
    gate_proj_key: str,
    down_proj_key: str,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> ParallelMLP:
    primary_mlp = load_mlp(
        module.primary_mlp,
        weights_dict,
        layer_path / primary_mlp_key,
        up_proj_key,
        gate_proj_key,
        down_proj_key,
        implementation=implementation,
    )
    primary_output_norm = load_rmsnorm(
        module.primary_output_norm,
        weights_dict,
        layer_path / "post_feedforward_layernorm_1",
    )
    parallel_mlp = load_mlp(
        module.parallel_mlp,
        weights_dict,
        layer_path / "parallel_mlp",
        up_proj_key,
        gate_proj_key,
        down_proj_key,
        implementation=implementation,
    )
    parallel_output_norm = load_rmsnorm(
        module.parallel_output_norm,
        weights_dict,
        layer_path / "post_feedforward_layernorm_2",
    )
    return load_as_at(
        _parallel_mlp_parts,
        module,
        (primary_mlp, primary_output_norm, parallel_mlp, parallel_output_norm),
    )


def load_moe(
    module: MixtureOfExperts,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> MixtureOfExperts:
    # Load router via the standard linear loader.
    # Qwen-MoE often names the router layer "gate" in HF weights.
    if (path / "router.weight") in weights_dict or (path / "router.qweight") in weights_dict:
        router_path = path / "router"
    elif (path / "gate.weight") in weights_dict or (path / "gate.qweight") in weights_dict:
        router_path = path / "gate"
    else:
        router_path = path / "router"
    router = load_linear(module.router, weights_dict, router_path, implementation=implementation)

    num_routed = module.config.num_routed_experts
    num_shared = module.config.num_shared_experts
    has_up_biases = module.experts.up_projection.has_biases
    has_down_biases = module.experts.down_projection.has_biases

    experts_path = path / "experts"

    # GPT-OSS uses fused MXFP4 expert weights; detect and decode those.
    if (experts_path / "gate_up_proj_blocks") in weights_dict:
        fused = decode_mxfp4(
            weights_dict[experts_path / "gate_up_proj_blocks"],
            weights_dict[experts_path / "gate_up_proj_scales"],
            dtype=module.experts.up_projection.weights.dtype,
            flatten=False,
        )
        fused_eio = rearrange(fused, "e o ib ie -> e (ib ie) o")
        up_weights, gate_weights = deinterleave_pairwise_columns(fused_eio, first="odd")
        combined_up_gate_weights = jnp.swapaxes(
            jnp.concatenate([up_weights, gate_weights], axis=-1),
            -1,
            -2,
        )

        gate_up_bias = weights_dict[experts_path / "gate_up_proj_bias"]
        if gate_up_bias.ndim == 1:
            gate_up_bias = jnp.broadcast_to(
                gate_up_bias,
                (combined_up_gate_weights.shape[0], gate_up_bias.shape[0]),
            )
        up_bias, gate_bias = deinterleave_pairwise_columns(gate_up_bias, first="odd")
        combined_up_gate_biases = jnp.concatenate([up_bias + 1.0, gate_bias], axis=-1)

        up_projection = _update_linear(
            module.experts.up_projection,
            load_full_precision(
                module.experts.up_projection.weights,
                combined_up_gate_weights,
            ),
            combined_up_gate_biases,
        )

        down_weights = decode_mxfp4(
            weights_dict[experts_path / "down_proj_blocks"],
            weights_dict[experts_path / "down_proj_scales"],
            dtype=module.experts.down_projection.weights.dtype,
            flatten=False,
        )
        down_weights = rearrange(down_weights, "e o ib ie -> e o (ib ie)")
        down_biases = weights_dict[experts_path / "down_proj_bias"]
        if down_biases.ndim == 1:
            down_biases = jnp.broadcast_to(down_biases, (*down_weights.shape[:-1], down_biases.shape[0]))

        down_projection = _update_linear(
            module.experts.down_projection,
            load_full_precision(
                module.experts.down_projection.weights,
                down_weights,
            ),
            down_biases,
        )

        experts = load_as_at(
            lambda m: (m.up_projection, m.down_projection),
            module.experts,
            (up_projection, down_projection),
        )
    elif (
        (experts_path / "gate_up_proj.weight") in weights_dict
        or (experts_path / "gate_up_proj" / "weight") in weights_dict
        or _has_prefix(weights_dict, experts_path / "gate_up_proj")
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
                (k for k in weights_dict if k.startswith(f"{gate_up_path}.")),
                None,
            )
            if gate_up_key is None:
                raise KeyError(f"Could not find gate_up_proj weights under {gate_up_path}")
            # Infer the weight key suffix
            suffix = gate_up_key[len(gate_up_path) :]
            gate_up_weights = weights_dict[gate_up_key]
            down_weights = weights_dict[down_path + suffix]

        # gate_up_proj is [num_experts, intermediate_size*2, hidden_size] - split into gate and up
        intermediate_size_2 = gate_up_weights.shape[1]
        intermediate_size = intermediate_size_2 // 2

        # Split gate and up: first half is gate, second half is up (or vice versa depending on model)
        gate_weights = gate_up_weights[:, :intermediate_size, :]
        up_weights = gate_up_weights[:, intermediate_size:, :]

        # Combine up and gate for our format: (num_experts, hidden*2, model_dim)
        combined_up_gate_weights = jnp.concatenate([up_weights, gate_weights], axis=1)
        if num_shared > 0:
            if num_shared != 1:
                raise ValueError("Single shared expert path found but num_shared_experts != 1.")
            shared_expert_path = path / "shared_expert"
            shared_up_gate_weights = jnp.concatenate(
                [
                    weights_dict[shared_expert_path / "up_proj.weight"],
                    weights_dict[shared_expert_path / "gate_proj.weight"],
                ],
                axis=0,
            )
            combined_up_gate_weights = jnp.concatenate(
                [combined_up_gate_weights, shared_up_gate_weights[None, ...]],
                axis=0,
            )
            down_weights = jnp.concatenate(
                [down_weights, weights_dict[shared_expert_path / "down_proj.weight"][None, ...]],
                axis=0,
            )

        up_projection = _update_linear(
            module.experts.up_projection,
            load_full_precision(
                module.experts.up_projection.weights,
                combined_up_gate_weights,
            ),
            None,
        )

        down_projection = _update_linear(
            module.experts.down_projection,
            load_full_precision(
                module.experts.down_projection.weights,
                down_weights,
            ),
            None,
        )

        experts = load_as_at(
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
                expert_paths.extend(path / "shared_experts" / str(idx) for idx in range(num_shared))
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

        up_projection = _update_linear(
            module.experts.up_projection,
            load_full_precision(
                module.experts.up_projection.weights,
                combined_up_gate_weights,
            ),
            combined_up_gate_biases,
        )

        stacked_down = jnp.stack(down_weight_list, axis=0)
        stacked_down_biases = jnp.stack(down_bias_list, axis=0) if down_bias_list is not None else None
        down_projection = _update_linear(
            module.experts.down_projection,
            load_full_precision(
                module.experts.down_projection.weights,
                stacked_down,
            ),
            stacked_down_biases,
        )

        experts = load_as_at(
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
        gate = load_linear(module.gate, weights_dict, gate_path, implementation=implementation)

    return load_as_at(
        lambda m: (m.router, m.experts, m.gate),
        module,
        (router, experts, gate),
    )


def load_rmsnorm(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization:
    scales = weights_dict[path / "weight"].astype(module.scales.dtype)
    return load_as_at(lambda m: (m.scales,), module, (scales,))


def _load_optional_rmsnorm(
    module: Normalization | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization | None:
    if module is None:
        return None
    return load_rmsnorm(module, weights_dict, path)


def _load_named_rmsnorm(
    module: Normalization | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    names: Sequence[str],
) -> Normalization | None:
    if module is None:
        return None
    scale_path = _first_path(weights_dict, tuple(path / name / "weight" for name in names))
    if scale_path is None:
        raise ValueError(f"Cannot find normalization under {path}; tried {', '.join(names)}")
    return load_rmsnorm(module, weights_dict, ParameterPath(scale_path.removesuffix(".weight")))


def _split_q_gate_tensor(
    tensor: Array,
    num_heads: int,
    head_dim: int,
) -> tuple[Array, Array]:
    rest = tensor.shape[1:]
    reshaped = tensor.reshape(num_heads, 2 * head_dim, *rest)
    q = reshaped[:, :head_dim].reshape(num_heads * head_dim, *rest)
    gate = reshaped[:, head_dim:].reshape(num_heads * head_dim, *rest)
    return q, gate


def _extract_gate_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    num_heads: int,
    head_dim: int,
) -> tuple[dict[str, Array], dict[str, Array]]:
    q_proj_path = path / "q_proj"
    q_proj_prefix = f"{q_proj_path}."
    kv_proj_prefixes = (f"{path / 'k_proj'}.", f"{path / 'v_proj'}.")
    gate_path = path / "gate_projection"
    q_weights: dict[str, Array] = {}
    gate_weights: dict[str, Array] = {}

    for key in weights_dict:
        if key.startswith(q_proj_prefix):
            suffix = key[len(q_proj_prefix) :]
            tensor = weights_dict[key]
            q_part, gate_part = _split_q_gate_tensor(tensor, num_heads, head_dim)
            q_weights[key] = q_part
            gate_weights[gate_path / suffix] = gate_part
        elif key.startswith(kv_proj_prefixes):
            q_weights[key] = weights_dict[key]
    return q_weights, gate_weights


def load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Attention:
    match module.config.projection_mode:
        case AttentionProjectionMode.QKV:
            qkv_sublayers = ["q_proj", "k_proj", "v_proj"]
        case AttentionProjectionMode.KEY_SAME_AS_VALUE:
            qkv_sublayers = ["q_proj", "k_proj"]
        case AttentionProjectionMode.BORROWED_KV:
            qkv_sublayers = ["q_proj"]

    if module.gate_projection is not None:
        num_heads, head_dim = module.config.num_heads, module.config.head_dim

        qkv_weights, gate_weights = _extract_gate_weights(
            weights_dict,
            path,
            num_heads,
            head_dim,
        )

        qkv_projection = load_linear(
            module.qkv_projection,
            qkv_weights,
            path,
            sublayers_to_fuse=qkv_sublayers,
            implementation=implementation,
        )

        gate_projection = load_linear(
            module.gate_projection,
            gate_weights,
            path / "gate_projection",
            implementation=implementation,
        )
    else:
        qkv_projection = load_linear(
            module.qkv_projection,
            weights_dict,
            path,
            sublayers_to_fuse=qkv_sublayers,
            implementation=implementation,
        )
        gate_projection = None

    out_projection = load_linear(
        module.out_projection,
        weights_dict,
        _projection_path(weights_dict, path, ("o_proj", "out_proj")),
        implementation=implementation,
    )

    query_norm = _load_named_rmsnorm(module.query_norm, weights_dict, path, ("q_norm", "q_layernorm"))
    key_norm = _load_named_rmsnorm(module.key_norm, weights_dict, path, ("k_norm", "k_layernorm"))
    sinks = weights_dict.get(path / "sinks", module.sinks)

    return load_as_at(
        lambda m: (
            m.qkv_projection,
            m.gate_projection,
            m.out_projection,
            m.query_norm,
            m.key_norm,
            m.sinks,
        ),
        module,
        (qkv_projection, gate_projection, out_projection, query_norm, key_norm, sinks),
    )


def _load_conv(
    conv_module: SeparableCausalConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
) -> SeparableCausalConv:
    parameter_dtype = conv_module.weights.dtype
    if conv_module.weights.weak_type:
        parameter_dtype = jnp.float32

    weight_path = _first_path(
        weights_dict,
        (path / "conv1d" / "weight", path / "conv_weight", path / "conv.weight"),
    )

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
        conv_weight = conv_weight.astype(parameter_dtype)
    else:
        conv_weight = conv_module.weights

    bias_path = _first_path(
        weights_dict,
        (path / "conv1d" / "bias", path / "conv_bias", path / "conv.bias"),
    )

    if bias_path is not None and conv_module.biases is not None:
        conv_bias = weights_dict[bias_path].astype(parameter_dtype)
    else:
        conv_bias = conv_module.biases

    return load_as_at(
        lambda m: (m.weights, m.biases),
        conv_module,
        (conv_weight, conv_bias),
    )


def load_mamba2(
    module: Mamba2,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Mamba2:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj", implementation=implementation)
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj", implementation=implementation)
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    skip_connection_weight = weights_dict.get(path / "D", module.skip_connection_weight).astype(jnp.float32)
    gate_bias = weights_dict.get(path / "z_bias", module.gate_bias).astype(jnp.float32)

    return load_as_at(
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
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> ShortConv:
    in_projection = load_linear(module.in_projection, weights_dict, path / "in_proj", implementation=implementation)
    out_projection = load_linear(module.out_projection, weights_dict, path / "out_proj", implementation=implementation)
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)

    return load_as_at(
        lambda m: (m.in_projection, m.out_projection, m.conv),
        module,
        (in_projection, out_projection, conv),
    )


def load_delta_net_attention(
    module: DeltaNet,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    permute_conv: bool,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DeltaNet:
    def _permute_rows(array: Array, permutation: Array | None) -> Array:
        if permutation is None:
            return array
        return jnp.take(array, permutation, axis=0)

    def _delta_net_qkvz_perm() -> Array:
        v_per_k = module.config.num_heads // module.config.num_groups
        key_block = 2 * module.config.head_dim
        value_block = module.config.value_head_dim * v_per_k
        per_group = key_block + 2 * value_block

        base = jnp.arange(module.config.num_groups, dtype=jnp.int32) * per_group
        q = base[:, None] + jnp.arange(module.config.head_dim, dtype=jnp.int32)[None, :]
        k = base[:, None] + module.config.head_dim + jnp.arange(module.config.head_dim, dtype=jnp.int32)[None, :]
        v = base[:, None] + key_block + jnp.arange(value_block, dtype=jnp.int32)[None, :]
        z = base[:, None] + key_block + value_block + jnp.arange(value_block, dtype=jnp.int32)[None, :]
        return jnp.concatenate([q.reshape(-1), k.reshape(-1), v.reshape(-1), z.reshape(-1)], axis=0)

    def _delta_net_ba_perm() -> Array:
        v_per_k = module.config.num_heads // module.config.num_groups
        per_group = 2 * v_per_k
        base = jnp.arange(module.config.num_groups, dtype=jnp.int32) * per_group
        b = base[:, None] + jnp.arange(v_per_k, dtype=jnp.int32)[None, :]
        a = base[:, None] + v_per_k + jnp.arange(v_per_k, dtype=jnp.int32)[None, :]
        return jnp.concatenate([b.reshape(-1), a.reshape(-1)], axis=0)

    in_proj_path = path / "in_proj"
    in_proj_weight_path = in_proj_path / "weight"
    if in_proj_weight_path in weights_dict:
        in_proj = load_linear(module.in_proj, weights_dict, in_proj_path, implementation=implementation)
    else:
        qkv_path = path / "in_proj_qkv"
        z_path = path / "in_proj_z"
        b_path = path / "in_proj_b"
        a_path = path / "in_proj_a"
        has_qwen35_split_projections = all(
            projection_path / "weight" in weights_dict for projection_path in (qkv_path, z_path, b_path, a_path)
        )
        if has_qwen35_split_projections:
            projection_branches: list[tuple[ParameterPath, Array | None]] = [
                (qkv_path, None),
                (z_path, None),
                (b_path, None),
                (a_path, None),
            ]
        else:
            qkvz_path = path / "in_proj_qkvz"
            ba_path = path / "in_proj_ba"
            if not ((qkvz_path / "weight") in weights_dict and (ba_path / "weight") in weights_dict):
                raise ValueError(
                    "Expected in_proj, in_proj_qkvz/in_proj_ba, or "
                    "in_proj_qkv/in_proj_z/in_proj_b/in_proj_a weights for DeltaNet.",
                )
            projection_branches = [
                (qkvz_path, _delta_net_qkvz_perm()),
                (ba_path, _delta_net_ba_perm()),
            ]

        (first_branch_path, _), *_ = projection_branches
        if _is_awq(weights_dict, first_branch_path, None):
            raise ValueError("DeltaNet does not support AWQ quantization.")
        if not _is_mlx(weights_dict, first_branch_path, None):
            merged = jnp.concatenate(
                [
                    _permute_rows(_fuse_full_precision_weights(weights_dict, bp, None), perm)
                    for bp, perm in projection_branches
                ],
                axis=0,
            )
            new_weights = load_full_precision(module.in_proj.weights, merged)
        else:
            per_branch = [
                tuple(_permute_rows(tensor, perm) for tensor in _fuse_mlx_weights(weights_dict, bp, None))
                for bp, perm in projection_branches
            ]
            fused_qweights = jnp.concatenate([qw for qw, _, _ in per_branch], axis=0)
            fused_deq_biases = jnp.concatenate([db for _, db, _ in per_branch], axis=0)
            fused_scales = jnp.concatenate([s for _, _, s in per_branch], axis=0)
            new_weights = _load_packed_mlx_matrix(
                fused_qweights,
                fused_deq_biases,
                fused_scales,
                module.in_proj.input_dim,
                template=module.in_proj.weights,
                layout=Layout.OUTPUT_INPUT,
                implementation=implementation,
                sharding_config=module.in_proj.weights.sharding_config,
            )
        in_proj = _update_linear(module.in_proj, new_weights, None)
    conv = _load_conv(module.conv, weights_dict, path, permute_conv)
    out_proj = load_linear(module.out_proj, weights_dict, path / "out_proj", implementation=implementation)
    norm = load_rmsnorm(module.norm, weights_dict, path / "norm")

    dt_bias = weights_dict[path / "dt_bias"].astype(module.dt_bias)
    a_log = weights_dict[path / "A_log"].astype(module.a_log)

    return load_as_at(
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
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> TransformerLayer:
    pre_attention_norm = _load_optional_rmsnorm(module.pre_mixer_norm, weights_dict, mixer_path / pre_mixer_norm_key)
    # Load mixer (attention or mamba)
    if isinstance(module.mixer, Attention):
        mixer = load_attention(
            module.mixer,
            weights_dict,
            mixer_path / mixer_key,
            implementation=implementation,
        )
    elif isinstance(module.mixer, DeltaNet):
        mixer = load_delta_net_attention(
            module.mixer,
            weights_dict,
            mixer_path / mixer_key,
            permute_conv,
            implementation=implementation,
        )
    elif isinstance(module.mixer, Mamba2):
        mixer = load_mamba2(
            module.mixer,
            weights_dict,
            mixer_path / mixer_key,
            permute_conv,
            implementation=implementation,
        )
    elif isinstance(module.mixer, ShortConv):
        mixer = load_short_conv(
            module.mixer,
            weights_dict,
            mixer_path / mixer_key,
            permute_conv,
            implementation=implementation,
        )
    else:
        mixer = module.mixer

    assert isinstance(module.pre_mlp_norm, Normalization)
    if module.post_mixer_norm is not None:
        post_attention_norm = load_rmsnorm(
            module.post_mixer_norm, weights_dict, mixer_path / "post_attention_layernorm"
        )
        pre_mlp_norm = load_rmsnorm(module.pre_mlp_norm, weights_dict, mlp_path / "pre_feedforward_layernorm")
    else:
        post_attention_norm = None
        pre_mlp_norm = load_rmsnorm(module.pre_mlp_norm, weights_dict, mlp_path / pre_mlp_norm_key)

    if isinstance(module.mlp, ParallelMLP):
        mlp = load_parallel_mlp(
            module.mlp,
            weights_dict,
            mlp_path,
            mlp_key,
            up_proj_key,
            gate_proj_key,
            down_proj_key,
            implementation=implementation,
        )
    else:
        mlp = load_mlp(
            module.mlp,
            weights_dict,
            mlp_path / mlp_key,
            up_proj_key,
            gate_proj_key,
            down_proj_key,
            implementation=implementation,
        )
    post_mlp_norm = _load_optional_rmsnorm(module.post_mlp_norm, weights_dict, mlp_path / "post_feedforward_layernorm")

    return load_as_at(
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


def _load_weight_matrix(
    matrix: WeightMatrix,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> WeightMatrix:
    return _load_matrix(
        matrix,
        weights_dict,
        path,
        None,
        layout=Layout.OUTPUT_INPUT,
        expected_grouped_channels=matrix.shape[-1],
        full_precision_weights=lambda: weights_dict[path / "weight"],
        implementation=implementation,
    )


def _load_input_embedding_matrix(
    matrix: WeightMatrix,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> WeightMatrix:
    return _load_matrix(
        matrix,
        weights_dict,
        path,
        None,
        layout=Layout.INPUT_OUTPUT,
        expected_grouped_channels=matrix.shape[-1],
        full_precision_weights=lambda: jnp.matrix_transpose(weights_dict[path / "weight"]),
        implementation=implementation,
    )


def load_tied_embedding(
    module: TiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> TiedEmbedding:
    embedding = _load_input_embedding_matrix(
        module.embedding,
        weights_dict,
        embedding_path,
        implementation=implementation,
    )
    return eqx.tree_at(lambda m: (m.embedding,), module, (embedding,))


def load_untied_embedding(
    module: UntiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    lm_head_path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> UntiedEmbedding:
    input_emb = _load_input_embedding_matrix(
        module.input_embedding,
        weights_dict,
        embedding_path,
        implementation=implementation,
    )
    output_emb = _load_weight_matrix(
        module.output_embedding,
        weights_dict,
        lm_head_path,
        implementation=implementation,
    )
    return eqx.tree_at(
        lambda m: (m.input_embedding, m.output_embedding),
        module,
        (input_emb, output_emb),
    )


def _get_mixer_key(
    mixer_key: dict[type[object], str],
    layer: TransformerLayer,
) -> str:
    mixer_config = layer.config.mixer_config
    match mixer_config:
        case AttentionConfig():
            return mixer_key[AttentionConfig]
        case DeltaNetConfig():
            return mixer_key[DeltaNetConfig]
        case Mamba2Config():
            return mixer_key[Mamba2Config]
        case ShortConvConfig():
            return mixer_key[ShortConvConfig]
        case _:
            raise TypeError(f"Unsupported mixer config: {type(mixer_config)}")


@dataclass(frozen=True)
class DecoderLoadLayout:
    decoder_path: ParameterPath
    embedding_path: ParameterPath
    pre_mixer_norm_key: str
    mixer_key: dict[type[object], str]
    permute_conv: bool
    pre_mlp_norm_key: str
    mlp_key: str
    up_proj_key: str
    gate_proj_key: str
    down_proj_key: str
    alternating_layers: bool
    norm_key: str
    lm_head_path: ParameterPath


def _decoder_load_layout(
    weights_dict: Mapping[str, Array],
    root_path: ParameterPath,
) -> DecoderLoadLayout:
    if root_path.endswith(".language_model"):
        decoder_path = root_path
    else:
        decoder_path = root_path / "model"

    standard_layout = DecoderLoadLayout(
        decoder_path=decoder_path,
        embedding_path=decoder_path / "embed_tokens",
        pre_mixer_norm_key="input_layernorm",
        mixer_key={AttentionConfig: "self_attn", DeltaNetConfig: "linear_attn"},
        permute_conv=False,
        pre_mlp_norm_key="post_attention_layernorm",
        mlp_key="mlp",
        up_proj_key="up_proj",
        gate_proj_key="gate_proj",
        down_proj_key="down_proj",
        alternating_layers=False,
        norm_key="norm",
        lm_head_path=root_path / "lm_head",
    )

    backbone_path = root_path / "backbone"
    if _has_prefix(weights_dict, backbone_path):
        return replace(
            standard_layout,
            decoder_path=backbone_path,
            embedding_path=backbone_path / "embedding",
            mixer_key={Mamba2Config: "mixer"},
            norm_key="final_layernorm",
        )
    if _has_prefix(weights_dict, root_path / "embedding.encoder"):
        return replace(
            standard_layout,
            embedding_path=root_path / "embedding.encoder",
            pre_mixer_norm_key="norm",
            mixer_key={Mamba2Config: "layer"},
            pre_mlp_norm_key="norm",
            mlp_key="layer",
            up_proj_key="gate_proj",
            gate_proj_key="in_proj",
            down_proj_key="out_proj",
            alternating_layers=True,
            lm_head_path=root_path / "head.linear",
        )
    if (decoder_path / "layers" / "0" / "operator_norm" / "weight") in weights_dict:
        return replace(
            standard_layout,
            pre_mixer_norm_key="operator_norm",
            mixer_key={ShortConvConfig: "conv", AttentionConfig: "self_attn"},
            permute_conv=_is_mlx(weights_dict, standard_layout.embedding_path, None),
            pre_mlp_norm_key="ffn_norm",
            mlp_key="feed_forward",
            up_proj_key="w3",
            gate_proj_key="w1",
            down_proj_key="w2",
            norm_key="embedding_norm",
        )
    return standard_layout


def load_huggingface_decoder(
    module: Decoder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Decoder:
    if _has_prefix(weights_dict, ParameterPath("model.language_model")):
        root_path = ParameterPath("model.language_model")
    elif _has_prefix(weights_dict, ParameterPath("language_model")):
        root_path = ParameterPath("language_model")
    else:
        root_path = ParameterPath()

    layout = _decoder_load_layout(weights_dict, root_path)

    if isinstance(module.embedding, TiedEmbedding):
        embedding = load_tied_embedding(
            module.embedding,
            weights_dict,
            layout.embedding_path,
            implementation=implementation,
        )
    elif isinstance(module.embedding, UntiedEmbedding):
        embedding = load_untied_embedding(
            module.embedding,
            weights_dict,
            layout.embedding_path,
            layout.lm_head_path,
            implementation=implementation,
        )
    else:
        raise TypeError(f"Unsupported embedding type: {type(module.embedding)}")

    decoder_layers = tuple(
        load_transformer_layer(
            layer,
            weights_dict,
            layout.decoder_path / "layers" / ((i * 2) if layout.alternating_layers else i),
            layout.decoder_path / "layers" / ((i * 2 + 1) if layout.alternating_layers else i),
            _get_mixer_key(layout.mixer_key, layer),
            layout.mlp_key,
            layout.pre_mixer_norm_key,
            layout.pre_mlp_norm_key,
            layout.up_proj_key,
            layout.gate_proj_key,
            layout.down_proj_key,
            layout.permute_conv,
            implementation=implementation,
        )
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, layout.decoder_path / layout.norm_key)
    return load_as_at(
        lambda m: (m.embedding, m.transformer.layers, m.transformer.output_norm),
        module,
        (embedding, decoder_layers, output_norm),
    )


def load_huggingface_classifier(
    module: Classifier,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Classifier:
    def load_linear_with_reshuffling(
        module: Linear,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> Linear:
        """Loads a linear layer and reshuffles some weights in resulting matrix to meet
        requirements of downstream 'split' in MLP layer in attention."""

        assert not module.has_biases
        weights = weights_dict[path / "weight"]
        rows, _ = weights.shape
        shuffled_weights = jnp.vstack((weights[rows // 2 :, :], weights[: rows // 2, :]))
        new_weights = load_full_precision(module.weights, shuffled_weights)
        return _update_linear(module, new_weights, None)

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
            implementation=implementation,
        )
        out_projection = load_linear(module.out_projection, weights_dict, path / "Wo", implementation=implementation)

        query_norm = _load_optional_rmsnorm(module.query_norm, weights_dict, path / "q_norm")
        key_norm = _load_optional_rmsnorm(module.key_norm, weights_dict, path / "k_norm")

        return load_as_at(
            lambda m: (m.qkv_projection, m.out_projection, m.query_norm, m.key_norm),
            module,
            (qkv_projection, out_projection, query_norm, key_norm),
        )

    def load_mlp_local(module: MLPBase, weights_dict: Mapping[str, Array], path: ParameterPath) -> MLPBase:
        assert isinstance(module, DenseMLP)
        dense_module: DenseMLP = module
        up_projection = load_linear_with_reshuffling(
            dense_module.up_projection,
            weights_dict,
            path / "Wi",
        )
        down_projection = load_linear(
            dense_module.down_projection,
            weights_dict,
            path / "Wo",
            implementation=implementation,
        )
        return load_as_at(
            _dense_mlp_projections,
            dense_module,
            (up_projection, down_projection),
        )

    def load_transformer_layer_local(
        module: TransformerLayer,
        weights_dict: Mapping[str, Array],
        path: ParameterPath,
    ) -> TransformerLayer:
        pre_attention_norm = _load_optional_rmsnorm(module.pre_mixer_norm, weights_dict, path / "attn_norm")

        assert isinstance(module.mixer, Attention)
        attention = load_attention_local(module.mixer, weights_dict, path / "attn")
        assert isinstance(module.pre_mlp_norm, Normalization)
        if module.post_mixer_norm is not None:
            post_attention_norm = load_rmsnorm(module.post_mixer_norm, weights_dict, path / "post_attention_layernorm")
            pre_mlp_norm = load_rmsnorm(module.pre_mlp_norm, weights_dict, path / "pre_feedforward_layernorm")
        else:
            post_attention_norm = None
            pre_mlp_norm = load_rmsnorm(module.pre_mlp_norm, weights_dict, path / "mlp_norm")

        mlp = load_mlp_local(module.mlp, weights_dict, path / "mlp")
        post_mlp_norm = _load_optional_rmsnorm(module.post_mlp_norm, weights_dict, path / "post_feedforward_layernorm")
        return load_as_at(
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
    embedding = load_tied_embedding(module.embedding, weights_dict, decoder_path / "embeddings" / "tok_embeddings")
    embedding_norm = load_rmsnorm(module.embedding_norm, weights_dict, base_path / "model" / "embeddings" / "norm")

    decoder_layers = tuple(
        load_transformer_layer_local(layer, weights_dict, decoder_path / "layers" / i)
        for i, layer in enumerate(module.transformer.layers)
    )
    output_norm = load_rmsnorm(module.transformer.output_norm, weights_dict, decoder_path / "final_norm")
    head_dense = load_linear(
        module.prediction_head.dense,
        weights_dict,
        head_path / "dense",
        implementation=implementation,
    )
    head_norm = load_rmsnorm(module.prediction_head.norm, weights_dict, head_path / "norm")
    head_readout = load_linear(
        module.prediction_head.readout,
        weights_dict,
        classifier_path,
        implementation=implementation,
    )
    return load_as_at(
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
