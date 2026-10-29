from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array

from lalamo.compressed import IntMatrix, IntSpec, MicrofloatSpec, MLXMatrix, MLXSpec
from lalamo.compressed.utils.packing import pack_uint_to_uint8
from lalamo.module import LalamoModule
from lalamo.modules.classifier import Classifier, ClassifierConfig
from lalamo.modules.decoder import Decoder, DecoderConfig
from lalamo.modules.embedding import TiedEmbedding, UntiedEmbedding
from lalamo.modules.linear import Linear
from lalamo.modules.mlp import DenseMLP, Gemma4Router, MixtureOfExperts, MLPBase, RoutedExperts
from lalamo.modules.normalization import Normalization
from lalamo.modules.per_layer import PerLayerInputs, Projection, Scale
from lalamo.modules.token_mixers.attention import Attention
from lalamo.modules.token_mixers.convolutions import DynamicConv, DynamicConvKernels, SeparableCausalConv
from lalamo.modules.token_mixers.deltanet import DeltaNet
from lalamo.modules.token_mixers.mamba import Mamba2
from lalamo.modules.token_mixers.short_conv import ShortConv
from lalamo.utils.parameter_path import ParameterPath
from lalamo.utils.sharding import ShardingConfig
from lalamo.utils.surgery import load_as_at
from lalamo.weight_matrix import CompressionImplementation, EmbeddingMatrix, Layout, WeightMatrix

from .binding import Binder, CheckpointView, Loader
from .common import load_full_precision

__all__ = [
    "huggingface_loader",
    "load_huggingface_classifier",
    "load_huggingface_decoder",
    "load_linear",
    "load_rmsnorm",
]


AWQ_UINT4_REVERSE_ORDER = jnp.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=jnp.int32)


def _update_linear(module: Linear, weights: WeightMatrix, biases: Array | None) -> Linear:
    return eqx.tree_at(lambda m: (m.weights, m.biases), module, (weights, biases))


def _load_mxfp4_matrix(
    template: WeightMatrix,
    blocks: Array,
    scales: Array,
    *,
    group_size: int = 32,
    implementation: CompressionImplementation,
) -> WeightMatrix:
    """Packed MXFP4 blocks with raw E8M0 scale bytes."""
    if blocks.dtype != jnp.dtype(jnp.uint8):
        raise ValueError(f"MXFP4 blocks must be U8, got {blocks.dtype}")
    if blocks.ndim != scales.ndim + 1 or blocks.shape[:-1] != scales.shape:
        raise ValueError(f"MXFP4 block/scale shape mismatch: blocks={blocks.shape}, scales={scales.shape}")
    if blocks.shape[-1] * 2 != group_size:
        raise ValueError(f"MXFP4 group size {group_size} does not match {blocks.shape[-1]} packed bytes")

    if scales.dtype == jnp.dtype(jnp.float8_e8m0fnu):
        packed_scales = jax.lax.bitcast_convert_type(scales, jnp.uint8)
    elif scales.dtype == jnp.dtype(jnp.uint8):
        packed_scales = scales
    else:
        raise ValueError(f"MXFP4 scales must be F8_E8M0 or U8, got {scales.dtype}")

    packed_weights = rearrange(blocks, "... rows groups packed -> ... rows (groups packed)")
    global_scale = jnp.ones(blocks.shape[:-3], dtype=template.dtype)
    return MicrofloatSpec(group_size=group_size).from_packed_parameters(
        packed_weights=packed_weights,
        packed_scales=packed_scales,
        global_scale=global_scale,
        dtype=template.dtype,
        implementation=implementation,
        sharding_config=template.sharding_config,
        is_sharded=template.is_sharded,
    )


def _stack_gpt_oss_gate_up(blocks: Array, scales: Array) -> tuple[Array, Array]:
    """Canonical up/gate rows with 16-value groups, without decoding MXFP4."""
    up_blocks = blocks[..., 1::2, :, :]
    gate_blocks = blocks[..., 0::2, :, :]
    blocks = jnp.concatenate((up_blocks, gate_blocks), axis=-3)
    blocks = rearrange(
        blocks,
        "... rows groups (halves packed) -> ... rows (groups halves) packed",
        halves=2,
    )

    up_scales = scales[..., 1::2, :]
    gate_scales = scales[..., 0::2, :]
    scales = jnp.concatenate((up_scales, gate_scales), axis=-2)
    scales = jnp.repeat(scales, 2, axis=-1)
    return blocks, scales


def _dense_mlp_projections(module: DenseMLP) -> tuple[Linear, Linear]:
    return module.up_projection, module.down_projection


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

    spec = MLXSpec(
        bits=bits,
        group_size=group_size,
        layout=layout,
    )
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
        return load_moe(
            module,
            weights_dict,
            path,
            up_proj_key,
            gate_proj_key,
            down_proj_key,
            implementation=implementation,
        )

    raise TypeError(f"Unsupported module type for loading: {type(module)}")


def load_moe[ExpertsT: MixtureOfExperts | RoutedExperts](
    module: ExpertsT,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    up_proj_key: str = "up_proj",
    gate_proj_key: str = "gate_proj",
    down_proj_key: str = "down_proj",
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
    router_path: ParameterPath | None = None,
) -> ExpertsT:
    num_routed = module.config.num_routed_experts
    num_shared = 0
    if isinstance(module, MixtureOfExperts):
        num_shared = module.config.num_shared_experts
    experts_path = path / "experts"
    gate_up_path = experts_path / "gate_up_proj"
    down_path = experts_path / "down_proj"
    batched_gate_up_path = _first_path(weights_dict, (gate_up_path, gate_up_path / "weight"))

    def load_projection(template: Linear, weights: Array, biases: Array | None) -> Linear:
        return _update_linear(
            template,
            load_full_precision(template.weights, weights),
            biases,
        )

    def load_dense_experts(
        template: DenseMLP,
        gate_up_weights: Array,
        gate_up_biases: Array | None,
        down_weights: Array,
        down_biases: Array | None,
    ) -> DenseMLP:
        return load_as_at(
            _dense_mlp_projections,
            template,
            (
                load_projection(template.up_projection, gate_up_weights, gate_up_biases),
                load_projection(template.down_projection, down_weights, down_biases),
            ),
        )

    def stack_experts(template: DenseMLP, expert_paths: Sequence[ParameterPath]) -> DenseMLP:
        def stack(name: str) -> Array:
            return jnp.stack([weights_dict[expert_path / name] for expert_path in expert_paths], axis=0)

        gate_up_weights = jnp.concatenate([stack(f"{up_proj_key}.weight"), stack(f"{gate_proj_key}.weight")], axis=1)
        gate_up_biases = (
            jnp.concatenate([stack(f"{up_proj_key}.bias"), stack(f"{gate_proj_key}.bias")], axis=1)
            if template.up_projection.has_biases
            else None
        )
        down_biases = stack(f"{down_proj_key}.bias") if template.down_projection.has_biases else None
        return load_dense_experts(
            template,
            gate_up_weights,
            gate_up_biases,
            stack(f"{down_proj_key}.weight"),
            down_biases,
        )

    def load_shared_experts() -> DenseMLP | None:
        if not isinstance(module, MixtureOfExperts) or module.shared_experts is None:
            return None
        if (path / "shared_expert" / "up_proj.weight") in weights_dict:
            if num_shared != 1:
                raise ValueError("Single shared expert path found but num_shared_experts != 1.")
            shared_paths = (path / "shared_expert",)
        elif (path / "shared_experts" / "0" / "up_proj.weight") in weights_dict:
            shared_paths = tuple(path / "shared_experts" / str(idx) for idx in range(num_shared))
        else:
            raise KeyError("Could not find shared expert weights in HF checkpoint.")
        return stack_experts(module.shared_experts, shared_paths)

    if (experts_path / "gate_up_proj_blocks") in weights_dict:
        gate_up_blocks = weights_dict[experts_path / "gate_up_proj_blocks"]
        gate_up_scales = weights_dict[experts_path / "gate_up_proj_scales"]
        gate_up_blocks, gate_up_scales = _stack_gpt_oss_gate_up(gate_up_blocks, gate_up_scales)
        gate_up_weights = _load_mxfp4_matrix(
            module.routed_experts.up_projection.weights,
            gate_up_blocks,
            gate_up_scales,
            group_size=16,
            implementation=implementation,
        )

        gate_up_bias = weights_dict[experts_path / "gate_up_proj_bias"]
        if gate_up_bias.ndim == 1:
            gate_up_bias = jnp.broadcast_to(
                gate_up_bias,
                (gate_up_blocks.shape[0], gate_up_bias.shape[0]),
            )
        up_bias = gate_up_bias[..., 1::2] + jnp.asarray(1.0, dtype=gate_up_bias.dtype)
        gate_bias = gate_up_bias[..., 0::2]
        gate_up_bias = jnp.concatenate((up_bias, gate_bias), axis=-1)

        up_projection = _update_linear(
            module.routed_experts.up_projection,
            gate_up_weights,
            gate_up_bias,
        )

        down_blocks = weights_dict[experts_path / "down_proj_blocks"]
        down_weights = _load_mxfp4_matrix(
            module.routed_experts.down_projection.weights,
            down_blocks,
            weights_dict[experts_path / "down_proj_scales"],
            implementation=implementation,
        )
        down_biases = weights_dict[experts_path / "down_proj_bias"]
        if down_biases.ndim == 1:
            down_biases = jnp.broadcast_to(down_biases, (down_blocks.shape[0], down_biases.shape[0]))

        down_projection = _update_linear(
            module.routed_experts.down_projection,
            down_weights,
            down_biases,
        )

        routed_experts = load_as_at(
            lambda m: (m.up_projection, m.down_projection),
            module.routed_experts,
            (up_projection, down_projection),
        )
        shared_experts = load_shared_experts()
    elif batched_gate_up_path is not None:
        suffix = batched_gate_up_path.removeprefix(gate_up_path)
        gate_up_weights = weights_dict[batched_gate_up_path]
        down_weights = weights_dict[ParameterPath(f"{down_path}{suffix}")]

        intermediate_size_2 = gate_up_weights.shape[1]
        intermediate_size = intermediate_size_2 // 2
        gate_weights = gate_up_weights[:, :intermediate_size, :]
        up_weights = gate_up_weights[:, intermediate_size:, :]
        combined_up_gate_weights = jnp.concatenate([up_weights, gate_weights], axis=1)
        routed_experts = load_dense_experts(
            module.routed_experts,
            combined_up_gate_weights,
            None,
            down_weights,
            None,
        )
        shared_experts = load_shared_experts()
    else:
        routed_experts = stack_experts(
            module.routed_experts,
            tuple(experts_path / str(idx) for idx in range(num_routed)),
        )
        shared_experts = load_shared_experts()

    if isinstance(module, RoutedExperts):
        return load_as_at(lambda model: (model.routed_experts,), module, (routed_experts,))
    assert isinstance(module, MixtureOfExperts)

    if router_path is None:
        if (path / "router.weight") in weights_dict or (path / "router.qweight") in weights_dict:
            router_path = path / "router"
        elif (path / "gate.weight") in weights_dict or (path / "gate.qweight") in weights_dict:
            router_path = path / "gate"
        else:
            router_path = path / "router"
    router = load_linear(module.router, weights_dict, router_path, implementation=implementation)

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

    expert_bias = None
    if module.expert_bias is not None:
        expert_bias = weights_dict[path / "expert_bias"]

    return replace(
        module,
        router=router,
        routed_experts=routed_experts,
        shared_experts=shared_experts,
        gate=gate,
        expert_bias=expert_bias,
    )


def load_rmsnorm(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization:
    assert module.config.has_scale and module.scales is not None, "cannot load scales into a weightless normalization"
    scales = weights_dict[path / "weight"].astype(module.scales.dtype)
    return load_as_at(lambda m: (m.scales,), module, (scales,))


def _load_optional_rmsnorm(
    module: Normalization | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
) -> Normalization | None:
    if module is None or not module.config.has_scale:
        return module
    return load_rmsnorm(module, weights_dict, path)


def _load_named_rmsnorm(
    module: Normalization | None,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    names: Sequence[str],
) -> Normalization | None:
    if module is None or not module.config.has_scale:
        return module
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


def _reorder_gated_attention_weights(
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    num_heads: int,
    head_dim: int,
) -> dict[str, Array]:
    """Split Qwen's per-head-interleaved Q/G projection into synthetic Q and G sublayers."""
    q_proj_path = path / "q_proj"
    if (q_proj_path / "qweight") in weights_dict:
        raise ValueError(
            "Packed qweight is not supported for gated attention; Q/G reordering requires "
            "format-aware output-channel unpacking. Use a full-precision or MLX checkpoint.",
        )

    q_proj_prefix = f"{q_proj_path}."
    kv_proj_prefixes = (f"{path / 'k_proj'}.", f"{path / 'v_proj'}.")
    # g_proj exists only in this normalized mapping so the generic linear loader can fuse Q/K/V/G.
    gate_path = path / "g_proj"
    reordered_weights: dict[str, Array] = {}

    for key in weights_dict:
        if key.startswith(q_proj_prefix):
            suffix = key[len(q_proj_prefix) :]
            tensor = weights_dict[key]
            q_part, gate_part = _split_q_gate_tensor(tensor, num_heads, head_dim)
            reordered_weights[key] = q_part
            reordered_weights[gate_path / suffix] = gate_part
        elif key.startswith(kv_proj_prefixes):
            reordered_weights[key] = weights_dict[key]
    return reordered_weights


def load_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Attention:
    if module.config.kv_source is not None:
        projection_sublayers = ["q_proj"]
    elif module.config.values_from_keys:
        projection_sublayers = ["q_proj", "k_proj"]
    else:
        projection_sublayers = ["q_proj", "k_proj", "v_proj"]
    projection_weights = weights_dict
    if module.config.has_gate:
        if (path / "gate_proj" / "weight") in weights_dict or (path / "gate_proj" / "qweight") in weights_dict:
            projection_sublayers = [*projection_sublayers, "gate_proj"]
        else:
            projection_weights = _reorder_gated_attention_weights(
                weights_dict,
                path,
                module.config.num_heads,
                module.config.head_dim,
            )
            projection_sublayers = [*projection_sublayers, "g_proj"]

    qkvg_projection = load_linear(
        module.qkvg_projection,
        projection_weights,
        path,
        sublayers_to_fuse=projection_sublayers,
        implementation=implementation,
    )

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
            m.qkvg_projection,
            m.out_projection,
            m.query_norm,
            m.key_norm,
            m.sinks,
        ),
        module,
        (qkvg_projection, out_projection, query_norm, key_norm, sinks),
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


def _load_output_embedding_matrix(
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


def load_input_embedding_matrix(
    matrix: EmbeddingMatrix,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> EmbeddingMatrix:
    loaded = _load_matrix(
        matrix,
        weights_dict,
        path,
        None,
        layout=Layout.INPUT_OUTPUT,
        expected_grouped_channels=matrix.shape[-1],
        full_precision_weights=lambda: jnp.matrix_transpose(weights_dict[path / "weight"]),
        implementation=implementation,
    )
    assert isinstance(loaded, EmbeddingMatrix)
    return loaded


def load_tied_embedding(
    module: TiedEmbedding,
    weights_dict: Mapping[str, Array],
    embedding_path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> TiedEmbedding:
    embedding = load_input_embedding_matrix(
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
    input_emb = load_input_embedding_matrix(
        module.input_embedding,
        weights_dict,
        embedding_path,
        implementation=implementation,
    )
    output_emb = _load_output_embedding_matrix(
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


def load_scale(
    module: Scale,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG001
) -> Scale:
    return load_as_at(lambda m: (m.scale,), module, (weights_dict[path],))


def load_projection(
    module: Projection,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Projection:
    return load_as_at(
        lambda m: (m.linear,),
        module,
        (load_linear(module.linear, weights_dict, path, implementation=implementation),),
    )


def load_dynamic_conv_kernels(
    module: DynamicConvKernels,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DynamicConvKernels:
    return load_as_at(
        lambda m: (m.projection,),
        module,
        (load_linear(module.projection, weights_dict, path / "kernel_projection", implementation=implementation),),
    )


def load_dynamic_conv(
    module: DynamicConv,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,  # noqa: ARG001
) -> DynamicConv:
    # One `base_kernel` holds the kernels of both convolutions around a sublayer, stored oldest tap last.
    base_kernels = rearrange(
        jnp.flip(weights_dict[path / "base_kernel"], axis=1),
        "sides kernel channels -> sides channels kernel",
        sides=2,
    )
    return load_as_at(lambda m: (m.conv.weights,), module, (base_kernels[module.config.side],))


def load_gemma4_router(
    module: Gemma4Router,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Gemma4Router:
    return load_as_at(
        lambda m: (m.projection, m.scale, m.expert_scales),
        module,
        (
            load_linear(module.projection, weights_dict, path / "proj", implementation=implementation),
            weights_dict[path / "scale"],
            weights_dict[path / "per_expert_scale"],
        ),
    )


def load_per_layer_inputs(
    module: PerLayerInputs,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> PerLayerInputs:
    return load_as_at(
        lambda m: (m.token_embedding, m.model_projection, m.projection_norm),
        module,
        (
            load_input_embedding_matrix(
                module.token_embedding,
                weights_dict,
                path / "embed_tokens_per_layer",
                implementation=implementation,
            ),
            load_linear(
                module.model_projection,
                weights_dict,
                path / "per_layer_model_projection",
                implementation=implementation,
            ),
            load_rmsnorm(module.projection_norm, weights_dict, path / "per_layer_projection_norm"),
        ),
    )


def _load_normalization(
    module: Normalization,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation,  # noqa: ARG001
) -> Normalization:
    return load_rmsnorm(module, weights_dict, path)


def huggingface_loader(module: LalamoModule) -> Loader:
    """The loader for a bound primitive stored in the standard Hugging Face layout."""
    match module:
        case Normalization():
            return _load_normalization
        case Attention():
            return load_attention
        case DenseMLP():
            return partial(load_mlp, up_proj_key="up_proj", gate_proj_key="gate_proj", down_proj_key="down_proj")
        case MixtureOfExperts() | RoutedExperts():
            return load_moe
        case DeltaNet():
            return partial(load_delta_net_attention, permute_conv=False)
        case Mamba2():
            return partial(load_mamba2, permute_conv=False)
        case ShortConv():
            return partial(load_short_conv, permute_conv=False)
        case TiedEmbedding():
            return load_tied_embedding
        case UntiedEmbedding():
            return partial(load_untied_embedding, lm_head_path=ParameterPath("lm_head"))
        case Gemma4Router():
            return load_gemma4_router
        case PerLayerInputs():
            return load_per_layer_inputs
        case Projection():
            return load_projection
        case DynamicConvKernels():
            return load_dynamic_conv_kernels
        case DynamicConv():
            return load_dynamic_conv
        case Scale():
            return load_scale
    raise TypeError(f"No default checkpoint loader for {type(module).__name__}; bind it with an explicit loader.")


def load_huggingface_decoder(
    module: Decoder,
    config: DecoderConfig,
    binder: Binder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Decoder:
    weights_dict = CheckpointView.of(weights_dict)
    return load_as_at(
        lambda m: (m.embedding, m.program),
        module,
        (
            binder.load(module.embedding, config.embedding_config, weights_dict, implementation=implementation),
            binder.load(module.program, config.program, weights_dict, implementation=implementation),
        ),
    )


def load_modern_bert_attention(
    module: Attention,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Attention:
    return load_as_at(
        lambda m: (m.qkvg_projection, m.out_projection),
        module,
        (
            load_linear(module.qkvg_projection, weights_dict, path / "Wqkv", implementation=implementation),
            load_linear(module.out_projection, weights_dict, path / "Wo", implementation=implementation),
        ),
    )


def load_modern_bert_mlp(
    module: DenseMLP,
    weights_dict: Mapping[str, Array],
    path: ParameterPath,
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> DenseMLP:
    # ModernBERT stores [gate; up] while the fused projection expects [up; gate].
    assert not module.up_projection.has_biases
    weights = weights_dict[path / "Wi" / "weight"]
    rows, _ = weights.shape
    shuffled_weights = jnp.vstack((weights[rows // 2 :, :], weights[: rows // 2, :]))
    return load_as_at(
        _dense_mlp_projections,
        module,
        (
            _update_linear(
                module.up_projection, load_full_precision(module.up_projection.weights, shuffled_weights), None
            ),
            load_linear(module.down_projection, weights_dict, path / "Wo", implementation=implementation),
        ),
    )


def load_huggingface_classifier(
    module: Classifier,
    config: ClassifierConfig,
    binder: Binder,
    weights_dict: Mapping[str, Array],
    *,
    implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
) -> Classifier:
    head_path = ParameterPath("head")
    return load_as_at(
        lambda m: (
            m.embedding,
            m.program,
            m.prediction_head.dense,
            m.prediction_head.norm,
            m.prediction_head.readout,
        ),
        module,
        (
            binder.load(module.embedding, config.embedding_config, weights_dict, implementation=implementation),
            binder.load(module.program, config.program, weights_dict, implementation=implementation),
            load_linear(
                module.prediction_head.dense, weights_dict, head_path / "dense", implementation=implementation
            ),
            load_rmsnorm(module.prediction_head.norm, weights_dict, head_path / "norm"),
            load_linear(
                module.prediction_head.readout,
                weights_dict,
                ParameterPath("classifier"),
                implementation=implementation,
            ),
        ),
    )
