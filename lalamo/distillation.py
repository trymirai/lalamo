import dataclasses
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.modules.common import (
    ParameterLeafInfo,
    ParameterNorm,
    combine_parameter_leaves,
    partition_parameter_leaves,
)
from lalamo.modules.decoder import Decoder
from lalamo.modules.linear import (
    GroupQuantizedLinear,
    GroupQuantizedLinearConfig,
    QLoRALinear,
    QLoRALinearConfig,
)
from lalamo.modules.normalization import Normalization
from lalamo.quantization import QuantizationMode, stochastic_quantize_weights

__all__ = [
    "DistillBatch",
    "DistillBatchMetrics",
    "DistillOptimizerState",
    "DistillParameterSummary",
    "DistillTrainConfig",
    "DistillTrainingState",
    "OptimizerGroup",
    "TraceDistillBatch",
    "apply_distill_gradients",
    "compute_distill_batch_metrics",
    "compute_distill_step_gradients",
    "compute_trace_distill_batch_metrics",
    "compute_trace_distill_step_gradients",
    "get_optimizer_group",
    "initialize_distill_training_state",
    "inject_lora_adapter_configs",
    "inject_lora_adapters",
    "is_leaf_trainable",
    "lora_trainable_filter",
    "make_trace_distill_batch",
    "materialize_trainable_module",
    "stochastically_quantize_module",
    "summarize_distill_parameters",
]


class DistillTrainConfig(eqx.Module):
    master_dtype: DTypeLike = jnp.float32
    compute_dtype: DTypeLike = jnp.bfloat16


@dataclass(frozen=True)
class DistillParameterSummary:
    total_parameters: int
    trainable_parameters: int
    total_master_bytes: int
    by_group: dict["OptimizerGroup", int]


class OptimizerGroup(StrEnum):
    FROZEN = "frozen"
    MUON = "muon"
    DEFAULT = "default"


class DistillTrainingState(eqx.Module):
    master_weights: object
    muon_weight_dimension_numbers: object


class DistillOptimizerState(eqx.Module):
    training_state: DistillTrainingState
    optimizer_state: optax.OptState


class DistillBatch(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    lengths_without_padding: Int[Array, " batch"] | None = None


class DistillBatchMetrics(eqx.Module):
    loss: Float[Array, ""]
    valid_tokens: Int[Array, ""]
    top1_matches: Int[Array, ""]


class TraceDistillBatch(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    prefix_lengths: Int[Array, " batch"]
    completion_lengths: Int[Array, " batch"]
    support_token_ids: Int[Array, "batch completion_tokens support"]
    support_logits: Float[Array, "batch completion_tokens support"]
    support_mask: Bool[Array, "batch completion_tokens support"]


def is_leaf_trainable(info: ParameterLeafInfo) -> bool:
    return info.alias_of is None and info.trainable


def get_optimizer_group(info: ParameterLeafInfo) -> OptimizerGroup:
    if not is_leaf_trainable(info):
        return OptimizerGroup.FROZEN
    if info.norm == ParameterNorm.SPECTRAL and len(info.shape) == 2:
        return OptimizerGroup.MUON
    return OptimizerGroup.DEFAULT


def lora_trainable_filter(info: ParameterLeafInfo) -> bool:
    return (
        issubclass(info.owner_type, QLoRALinear)
        and not info.quantized
        and info.norm == ParameterNorm.SPECTRAL
        and len(info.shape) == 2
    ) or issubclass(info.owner_type, Normalization)


def _inject_qlora_linear(
    linear: GroupQuantizedLinear,
    lora_rank: int,
    lora_scale: float,
    key: Key[Array, ""],
) -> QLoRALinear:
    qlora_config = _make_qlora_config(linear.config, lora_rank, lora_scale)

    leading_dims = linear.weights.shape[:-2]
    hidden_lora_rank = linear.num_outputs * lora_rank
    max_down_abs_value = 1 / math.sqrt(linear.input_dim)
    lora_down_weights = jax.random.uniform(
        key,
        (*leading_dims, linear.input_dim, hidden_lora_rank),
        minval=-max_down_abs_value,
        maxval=max_down_abs_value,
        dtype=qlora_config.activation_precision,
    )
    lora_up_weights = tuple(
        jnp.zeros(
            (*leading_dims, lora_rank, output_dim),
            dtype=qlora_config.activation_precision,
        )
        for output_dim in linear.output_dims
    )

    return QLoRALinear(
        config=qlora_config,
        output_dims=linear.output_dims,
        weights=linear.weights,
        scales=linear.scales,
        zero_points=linear.zero_points,
        biases=linear.biases,
        lora_down_weights=lora_down_weights,
        lora_up_weights=lora_up_weights,
    )


def _is_replaceable_quantized_linear(leaf: object) -> bool:
    return isinstance(leaf, GroupQuantizedLinear) and not isinstance(leaf, QLoRALinear)


def _make_qlora_config(
    config: GroupQuantizedLinearConfig,
    lora_rank: int,
    lora_scale: float,
) -> QLoRALinearConfig:
    return QLoRALinearConfig(
        **{field.name: getattr(config, field.name) for field in dataclasses.fields(config)},
        lora_rank=lora_rank,
        lora_scale=lora_scale,
    )


def _rewrite_config_tree[C](config: C, rewrite: Callable[[object], object]) -> C:
    rewritten = rewrite(config)
    if rewritten is not config:
        return rewritten
    if dataclasses.is_dataclass(config):
        return dataclasses.replace(
            config,
            **{
                field.name: _rewrite_config_tree(getattr(config, field.name), rewrite)
                for field in dataclasses.fields(config)
            },
        )
    if isinstance(config, tuple):
        return tuple(_rewrite_config_tree(item, rewrite) for item in config)
    if isinstance(config, list):
        return [_rewrite_config_tree(item, rewrite) for item in config]
    return config


def inject_lora_adapters[M: eqx.Module](
    module: M,
    lora_rank: int,
    lora_scale: float,
    key: Key[Array, ""],
) -> M:
    quantized_linears = [
        leaf
        for leaf in jtu.tree_leaves(module, is_leaf=_is_replaceable_quantized_linear)
        if _is_replaceable_quantized_linear(leaf)
    ]
    if not quantized_linears:
        return module

    flat_module, treedef = jtu.tree_flatten(module, is_leaf=_is_replaceable_quantized_linear)
    split_keys = tuple(jax.random.split(key, len(quantized_linears)))
    key_index = 0
    lora_key_leaves: list[Key[Array, ""] | None] = []
    for leaf in flat_module:
        if not _is_replaceable_quantized_linear(leaf):
            lora_key_leaves.append(None)
            continue
        lora_key_leaves.append(split_keys[key_index])
        key_index += 1
    lora_keys = jtu.tree_unflatten(
        treedef,
        lora_key_leaves,
    )
    assert key_index == len(split_keys)

    return jax.tree.map(
        lambda leaf, lora_key: leaf
        if lora_key is None
        else _inject_qlora_linear(leaf, lora_rank, lora_scale, lora_key),
        module,
        lora_keys,
        is_leaf=_is_replaceable_quantized_linear,
    )


def inject_lora_adapter_configs[C](
    config: C,
    lora_rank: int,
    lora_scale: float,
) -> C:
    return _rewrite_config_tree(
        config,
        lambda node: _make_qlora_config(node, lora_rank, lora_scale)
        if isinstance(node, GroupQuantizedLinearConfig) and not isinstance(node, QLoRALinearConfig)
        else node,
    )


def summarize_distill_parameters(
    leaves: Sequence[ParameterLeafInfo],
    config: DistillTrainConfig,
    *,
    trainable_filter: Callable[[ParameterLeafInfo], bool] | None = None,
) -> DistillParameterSummary:
    by_group: dict[OptimizerGroup, int] = defaultdict(int)
    master_dtype = jnp.dtype(config.master_dtype)

    total_parameters = 0
    trainable_parameters = 0
    total_master_bytes = 0

    for info in leaves:
        if info.alias_of is not None:
            continue

        parameter_count = math.prod(info.shape)
        total_parameters += parameter_count

        if not is_leaf_trainable(info) or (trainable_filter is not None and not trainable_filter(info)):
            by_group[OptimizerGroup.FROZEN] += parameter_count
            continue

        optimizer_group = get_optimizer_group(info)
        by_group[optimizer_group] += parameter_count

        trainable_parameters += parameter_count
        total_master_bytes += parameter_count * master_dtype.itemsize

    return DistillParameterSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        total_master_bytes=total_master_bytes,
        by_group=dict(by_group),
    )


def initialize_distill_training_state(
    module: eqx.Module,
    config: DistillTrainConfig,
    *,
    trainable_filter: Callable[[ParameterLeafInfo], bool] | None = None,
) -> DistillTrainingState:
    master_dtype = jnp.dtype(config.master_dtype)
    muon_dims = optax.contrib.MuonDimensionNumbers(reduction_axis=1, output_axis=0)

    def should_train(info: ParameterLeafInfo) -> bool:
        return is_leaf_trainable(info) and (trainable_filter is None or trainable_filter(info))

    return DistillTrainingState(
        master_weights=partition_parameter_leaves(
            module,
            should_train,
            lambda leaf, _info: _cast_array_like(leaf, master_dtype),
        ),
        muon_weight_dimension_numbers=partition_parameter_leaves(
            module,
            should_train,
            lambda _leaf, info: muon_dims if get_optimizer_group(info) == OptimizerGroup.MUON else None,
        ),
    )


def materialize_trainable_module[M: eqx.Module](
    module: M,
    master_weights: object,
    config: DistillTrainConfig,
) -> M:
    compute_dtype = jnp.dtype(config.compute_dtype)
    compute_weights = jax.tree.map(
        lambda leaf: None if leaf is None else _cast_array_like(leaf, compute_dtype),
        master_weights,
        is_leaf=lambda leaf: leaf is None,
    )
    return combine_parameter_leaves(module, compute_weights)


def stochastically_quantize_module[M: eqx.Module](
    module: M,
    quantization_mode: QuantizationMode,
    key: Key[Array, ""],
) -> M:
    quantized_weights = partition_parameter_leaves(
        module,
        lambda info: info.quantized,
        lambda leaf, _info: leaf,
    )
    quantized_leaves = [
        leaf
        for leaf in jtu.tree_leaves(
            quantized_weights,
            is_leaf=lambda leaf: leaf is None,
        )
        if leaf is not None
    ]
    if not quantized_leaves:
        return module

    flat_quantized_weights, treedef = jtu.tree_flatten(quantized_weights, is_leaf=lambda leaf: leaf is None)
    split_keys = tuple(jax.random.split(key, len(quantized_leaves)))
    key_index = 0
    quantization_key_leaves: list[Key[Array, ""] | None] = []
    for leaf in flat_quantized_weights:
        if leaf is None:
            quantization_key_leaves.append(None)
            continue
        quantization_key_leaves.append(split_keys[key_index])
        key_index += 1
    quantization_keys = jtu.tree_unflatten(
        treedef,
        quantization_key_leaves,
    )
    assert key_index == len(split_keys)
    quantized_weights = jax.tree.map(
        lambda leaf, quantization_key: None
        if leaf is None
        else leaf
        + jax.lax.stop_gradient(
            stochastic_quantize_weights(leaf, quantization_mode, quantization_key) - leaf,
        ),
        quantized_weights,
        quantization_keys,
        is_leaf=lambda leaf: leaf is None,
    )
    return combine_parameter_leaves(module, quantized_weights)


def _cast_array_like(value: Array | jax.ShapeDtypeStruct, dtype: DTypeLike) -> Array | jax.ShapeDtypeStruct:
    if eqx.is_array(value):
        return value.astype(dtype)
    return jax.ShapeDtypeStruct(value.shape, dtype)


def _token_positions(token_ids: Int[Array, "batch tokens"]) -> Int[Array, "batch tokens"]:
    _, num_tokens = token_ids.shape
    return jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), token_ids.shape)


def make_trace_distill_batch(
    traces: Sequence[LalamoCompletion],
    *,
    pad_token_id: int,
) -> TraceDistillBatch:
    if not traces:
        raise ValueError("Trace distillation batch requires at least one trace")

    prefix_lengths = jnp.array([len(trace.prefix_token_ids) for trace in traces], dtype=jnp.int32)
    completion_lengths = jnp.array([len(trace.completion_token_ids) for trace in traces], dtype=jnp.int32)

    max_sequence_tokens = max(len(trace.prefix_token_ids) + len(trace.completion_token_ids) for trace in traces)
    max_completion_tokens = max((len(trace.completion_token_ids) for trace in traces), default=0)
    max_support_tokens = max(
        (len(token_logits) for trace in traces for token_logits in trace.completion_token_logits),
        default=0,
    )

    token_ids = jnp.full((len(traces), max_sequence_tokens), pad_token_id, dtype=jnp.int32)
    support_token_ids = jnp.zeros((len(traces), max_completion_tokens, max_support_tokens), dtype=jnp.int32)
    support_logits = jnp.full(
        (len(traces), max_completion_tokens, max_support_tokens),
        -jnp.inf,
        dtype=jnp.float32,
    )
    support_mask = jnp.zeros((len(traces), max_completion_tokens, max_support_tokens), dtype=bool)

    for batch_index, trace in enumerate(traces):
        sequence_token_ids = jnp.array(
            [*trace.prefix_token_ids, *trace.completion_token_ids],
            dtype=jnp.int32,
        )
        token_ids = token_ids.at[batch_index, : sequence_token_ids.shape[0]].set(sequence_token_ids)

        for completion_index, token_logits in enumerate(trace.completion_token_logits):
            if not token_logits:
                continue

            support_items = tuple(token_logits.items())
            support_ids = jnp.array([token_id for token_id, _ in support_items], dtype=jnp.int32)
            support_values = jnp.array([logit for _, logit in support_items], dtype=jnp.float32)
            support_width = support_ids.shape[0]

            support_token_ids = support_token_ids.at[batch_index, completion_index, :support_width].set(support_ids)
            support_logits = support_logits.at[batch_index, completion_index, :support_width].set(support_values)
            support_mask = support_mask.at[batch_index, completion_index, :support_width].set(True)

    return TraceDistillBatch(
        token_ids=token_ids,
        prefix_lengths=prefix_lengths,
        completion_lengths=completion_lengths,
        support_token_ids=support_token_ids,
        support_logits=support_logits,
        support_mask=support_mask,
    )


def _decoder_logits(
    decoder: Decoder,
    batch: DistillBatch,
) -> Float[Array, "batch prediction_tokens vocabulary"]:
    token_positions = _token_positions(batch.token_ids)
    decoder_result = decoder(
        token_ids=batch.token_ids,
        token_positions=token_positions,
    )
    return decoder_result.logits[:, :-1, :]


@eqx.filter_jit
def compute_distill_batch_metrics(
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
) -> DistillBatchMetrics:
    teacher_logits = _decoder_logits(teacher, batch).astype(jnp.float32)
    student_logits = _decoder_logits(student, batch).astype(jnp.float32)

    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)

    batch_size, num_tokens = batch.token_ids.shape
    prediction_tokens = max(num_tokens - 1, 0)
    lengths = (
        batch.lengths_without_padding
        if batch.lengths_without_padding is not None
        else jnp.full(
            (batch_size,),
            num_tokens,
            dtype=jnp.int32,
        )
    )
    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    prediction_mask = (jnp.arange(prediction_tokens, dtype=jnp.int32)[None, :] < (lengths[:, None] - 1)).astype(
        token_kl.dtype,
    )
    valid_tokens = prediction_mask.sum(dtype=jnp.int32)
    loss = jnp.sum(token_kl * prediction_mask) / valid_tokens
    student_top1 = jnp.argmax(student_logits, axis=-1)
    teacher_top1 = jnp.argmax(teacher_logits, axis=-1)
    top1_matches = jnp.sum(
        jnp.logical_and(student_top1 == teacher_top1, prediction_mask.astype(bool)),
        dtype=jnp.int32,
    )
    return DistillBatchMetrics(
        loss=loss,
        valid_tokens=valid_tokens,
        top1_matches=top1_matches,
    )


@eqx.filter_jit
def compute_trace_distill_batch_metrics(
    student: Decoder,
    batch: TraceDistillBatch,
) -> DistillBatchMetrics:
    token_positions = _token_positions(batch.token_ids)
    decoder_result = student(
        token_ids=batch.token_ids,
        token_positions=token_positions,
    )
    student_logits = decoder_result.logits[:, :-1, :].astype(jnp.float32)

    _, completion_tokens, _ = batch.support_token_ids.shape
    completion_range = jnp.arange(completion_tokens, dtype=jnp.int32)[None, :]
    completion_mask = completion_range < batch.completion_lengths[:, None]

    raw_indices = batch.prefix_lengths[:, None] - 1 + completion_range
    max_indices = jnp.maximum(batch.prefix_lengths + batch.completion_lengths - 2, 0)
    prediction_indices = jnp.minimum(raw_indices, max_indices[:, None])

    batch_indices = jnp.arange(batch.token_ids.shape[0], dtype=jnp.int32)[:, None]
    student_completion_logits = student_logits[batch_indices, prediction_indices, :]
    student_support_logits = jnp.take_along_axis(
        student_completion_logits,
        batch.support_token_ids,
        axis=-1,
    )
    support_rows = batch.support_mask.any(axis=-1, keepdims=True)
    masked_teacher_logits = jnp.where(batch.support_mask, batch.support_logits, -jnp.inf)
    masked_student_logits = jnp.where(batch.support_mask, student_support_logits, -jnp.inf)
    safe_teacher_logits = jnp.where(support_rows, masked_teacher_logits, jnp.zeros_like(masked_teacher_logits))
    safe_student_logits = jnp.where(support_rows, masked_student_logits, jnp.zeros_like(masked_student_logits))

    teacher_log_probs = jax.nn.log_softmax(safe_teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(safe_student_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)
    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)

    valid_tokens = completion_mask.sum(dtype=jnp.int32)
    loss = jnp.sum(token_kl * completion_mask.astype(token_kl.dtype)) / valid_tokens
    student_top1 = jnp.argmax(safe_student_logits, axis=-1)
    teacher_top1 = jnp.argmax(safe_teacher_logits, axis=-1)
    top1_matches = jnp.sum(
        jnp.logical_and(student_top1 == teacher_top1, completion_mask),
        dtype=jnp.int32,
    )
    return DistillBatchMetrics(
        loss=loss,
        valid_tokens=valid_tokens,
        top1_matches=top1_matches,
    )


@eqx.filter_jit
def compute_distill_step_gradients(
    training_state: DistillTrainingState,
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillTrainConfig,
    *,
    quantization_key: Key[Array, ""] | None = None,
    quantization_mode: QuantizationMode | None = None,
) -> tuple[object, DistillBatchMetrics]:
    def loss_fn(master_weights: object) -> tuple[Float[Array, ""], DistillBatchMetrics]:
        materialized_student = materialize_trainable_module(student, master_weights, config)
        if quantization_key is not None and quantization_mode is not None:
            materialized_student = stochastically_quantize_module(
                materialized_student,
                quantization_mode,
                quantization_key,
            )
        metrics = compute_distill_batch_metrics(materialized_student, teacher, batch)
        return metrics.loss, metrics

    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.master_weights)
    return grads, metrics


@eqx.filter_jit
def compute_trace_distill_step_gradients(
    training_state: DistillTrainingState,
    student: Decoder,
    batch: TraceDistillBatch,
    config: DistillTrainConfig,
    *,
    quantization_key: Key[Array, ""] | None = None,
    quantization_mode: QuantizationMode | None = None,
) -> tuple[object, DistillBatchMetrics]:
    def loss_fn(master_weights: object) -> tuple[Float[Array, ""], DistillBatchMetrics]:
        materialized_student = materialize_trainable_module(student, master_weights, config)
        if quantization_key is not None and quantization_mode is not None:
            materialized_student = stochastically_quantize_module(
                materialized_student,
                quantization_mode,
                quantization_key,
            )
        metrics = compute_trace_distill_batch_metrics(materialized_student, batch)
        return metrics.loss, metrics

    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(training_state.master_weights)
    return grads, metrics


@eqx.filter_jit
def apply_distill_gradients(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    grads: object,
) -> DistillOptimizerState:
    updates, new_optimizer_state = optimizer.update(
        grads,
        optimizer_state.optimizer_state,
        optimizer_state.training_state.master_weights,
    )
    updated_master_weights = optax.apply_updates(optimizer_state.training_state.master_weights, updates)
    return replace(
        optimizer_state,
        training_state=replace(
            optimizer_state.training_state,
            master_weights=updated_master_weights,
        ),
        optimizer_state=new_optimizer_state,
    )
