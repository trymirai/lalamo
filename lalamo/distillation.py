import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.tree_util import keystr
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.modules.common import ParameterLeafInfo, iter_parameter_leaves
from lalamo.modules.decoder import Decoder
from lalamo.quantization import stochastic_quantize_weights

__all__ = [
    "DistillBatch",
    "DistillBatchMetrics",
    "DistillOptimizerState",
    "DistillParameterSummary",
    "DistillStepMetrics",
    "DistillTrainConfig",
    "DistillTrainableParameter",
    "DistillTrainingState",
    "OptimizerGroup",
    "TraceDistillBatch",
    "compute_distill_batch_metrics",
    "compute_distill_kl_loss",
    "compute_trace_distill_kl_loss",
    "distill_train_step",
    "get_muon_weight_dimension_numbers",
    "get_optimizer_group",
    "initialize_distill_optimizer_state",
    "initialize_distill_training_state",
    "is_leaf_trainable",
    "make_trace_distill_batch",
    "materialize_trainable_module",
    "stochastically_quantize_module",
    "summarize_distill_parameters",
]


@dataclass(frozen=True)
class DistillTrainConfig:
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


@dataclass(frozen=True)
class DistillTrainableParameter:
    path: str
    alias_paths: tuple[str, ...]
    optimizer_group: OptimizerGroup
    master_weight: Array | jax.ShapeDtypeStruct


@dataclass(frozen=True)
class DistillTrainingState:
    trainable_parameters: tuple[DistillTrainableParameter, ...]


@dataclass(frozen=True)
class DistillOptimizerState:
    training_state: DistillTrainingState
    optimizer_state: optax.OptState


@dataclass(frozen=True)
class DistillBatch:
    token_ids: Int[Array, "batch tokens"]
    lengths_without_padding: Int[Array, " batch"] | None = None


@dataclass(frozen=True)
class DistillStepMetrics:
    loss: Float[Array, ""]
    valid_tokens: Int[Array, ""]


@dataclass(frozen=True)
class DistillBatchMetrics:
    loss: Float[Array, ""]
    valid_tokens: Int[Array, ""]
    top1_matches: Int[Array, ""]


@dataclass(frozen=True)
class TraceDistillBatch:
    token_ids: Int[Array, "batch tokens"]
    prefix_lengths: Int[Array, " batch"]
    completion_lengths: Int[Array, " batch"]
    support_token_ids: Int[Array, "batch completion_tokens support"]
    support_logits: Float[Array, "batch completion_tokens support"]
    support_mask: Bool[Array, "batch completion_tokens support"]


def is_leaf_trainable(info: ParameterLeafInfo) -> bool:
    if info.alias_of is not None:
        return False
    return info.trainable


def get_optimizer_group(info: ParameterLeafInfo) -> OptimizerGroup:
    if not is_leaf_trainable(info):
        return OptimizerGroup.FROZEN
    if info.matrix and len(info.shape) == 2:
        return OptimizerGroup.MUON
    return OptimizerGroup.DEFAULT


def summarize_distill_parameters(
    leaves: Sequence[ParameterLeafInfo],
    config: DistillTrainConfig,
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

        optimizer_group = get_optimizer_group(info)
        by_group[optimizer_group] += parameter_count

        if optimizer_group == OptimizerGroup.FROZEN:
            continue

        trainable_parameters += parameter_count
        total_master_bytes += parameter_count * master_dtype.itemsize

    return DistillParameterSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        total_master_bytes=total_master_bytes,
        by_group=dict(by_group),
    )


def _leaf_map(module: eqx.Module) -> dict[str, object]:
    flat_with_path, _ = jtu.tree_flatten_with_path(module)
    return {keystr(path).lstrip("."): leaf for path, leaf in flat_with_path}


def initialize_distill_training_state(
    module: eqx.Module,
    config: DistillTrainConfig,
) -> DistillTrainingState:
    master_dtype = jnp.dtype(config.master_dtype)
    leaves = iter_parameter_leaves(module)
    path_to_leaf = _leaf_map(module)

    alias_paths: dict[str, list[str]] = defaultdict(list)
    for info in leaves:
        canonical_path = info.alias_of or info.path
        alias_paths[canonical_path].append(info.path)

    trainable_parameters: list[DistillTrainableParameter] = []
    for info in leaves:
        if not is_leaf_trainable(info):
            continue

        value = path_to_leaf[info.path]
        if not (eqx.is_array(value) or isinstance(value, jax.ShapeDtypeStruct)):
            raise TypeError(f"Trainable leaf {info.path} must be array-like, got {type(value)}")

        optimizer_group = get_optimizer_group(info)
        trainable_parameters.append(
            DistillTrainableParameter(
                path=info.path,
                alias_paths=tuple(alias_paths[info.path]),
                optimizer_group=optimizer_group,
                master_weight=_cast_array_like(value, master_dtype),
            ),
        )

    return DistillTrainingState(trainable_parameters=tuple(trainable_parameters))


def get_muon_weight_dimension_numbers(
    training_state: DistillTrainingState,
) -> tuple[optax.contrib.MuonDimensionNumbers | None, ...]:
    result: list[optax.contrib.MuonDimensionNumbers | None] = []

    for parameter in training_state.trainable_parameters:
        if parameter.optimizer_group != OptimizerGroup.MUON:
            result.append(None)
            continue

        ndim = len(parameter.master_weight.shape)
        if ndim != 2:
            raise ValueError(f"Muon parameters must have rank 2, got {parameter.path} with {ndim}")

        result.append(
            optax.contrib.MuonDimensionNumbers(
                reduction_axis=1,
                output_axis=0,
            ),
        )

    return tuple(result)


def _select_parameter_paths(module: eqx.Module, paths: Sequence[str]) -> list[object]:
    path_to_leaf = _leaf_map(module)
    return [path_to_leaf[path] for path in paths]


def materialize_trainable_module[M: eqx.Module](
    module: M,
    state: DistillTrainingState,
    config: DistillTrainConfig,
) -> M:
    compute_dtype = jnp.dtype(config.compute_dtype)

    replacement_paths: list[str] = []
    replacement_values: list[Array | jax.ShapeDtypeStruct] = []
    for parameter in state.trainable_parameters:
        compute_weight = _cast_array_like(parameter.master_weight, compute_dtype)
        for path in parameter.alias_paths:
            replacement_paths.append(path)
            replacement_values.append(compute_weight)

    if not replacement_paths:
        return module

    return eqx.tree_at(
        lambda tree: _select_parameter_paths(tree, replacement_paths),
        module,
        replacement_values,
        is_leaf=lambda value: value is None,
    )


def stochastically_quantize_module[M: eqx.Module](
    module: M,
    key: PRNGKeyArray,
) -> M:
    leaf_infos = iter_parameter_leaves(module)
    quantized_infos = [info for info in leaf_infos if info.alias_of is None and info.quantization_mode is not None]
    if not quantized_infos:
        return module

    alias_paths = _alias_paths_by_canonical_path(leaf_infos)
    quantization_keys = _split_quantization_keys(key, quantized_infos)
    path_to_leaf = _leaf_map(module)

    replacement_paths: list[str] = []
    replacement_values: list[Array] = []
    for info in quantized_infos:
        leaf = path_to_leaf[info.path]
        if not eqx.is_array(leaf):
            continue

        stochastic_value = stochastic_quantize_weights(leaf, info.quantization_mode, quantization_keys[info.path])
        rounded_value = leaf + jax.lax.stop_gradient(stochastic_value - leaf)
        for path in alias_paths[info.path]:
            replacement_paths.append(path)
            replacement_values.append(rounded_value)

    if not replacement_paths:
        return module

    return eqx.tree_at(
        lambda tree: _select_parameter_paths(tree, replacement_paths),
        module,
        replacement_values,
        is_leaf=lambda value: value is None,
    )


def _cast_array_like(value: Array | jax.ShapeDtypeStruct, dtype: DTypeLike) -> Array | jax.ShapeDtypeStruct:
    if eqx.is_array(value):
        return value.astype(dtype)
    return jax.ShapeDtypeStruct(value.shape, dtype)


def _alias_paths_by_canonical_path(leaves: Sequence[ParameterLeafInfo]) -> dict[str, tuple[str, ...]]:
    alias_paths: dict[str, list[str]] = defaultdict(list)
    for info in leaves:
        alias_paths[info.alias_of or info.path].append(info.path)
    return {path: tuple(paths) for path, paths in alias_paths.items()}


def _split_quantization_keys(
    quantization_key: PRNGKeyArray,
    leaves: Sequence[ParameterLeafInfo],
) -> dict[str, PRNGKeyArray]:
    if not leaves:
        return {}
    return {info.path: key for info, key in zip(leaves, jax.random.split(quantization_key, len(leaves)), strict=True)}


def _concrete_master_weights(state: DistillTrainingState) -> tuple[Array, ...]:
    result: list[Array] = []
    for parameter in state.trainable_parameters:
        if not eqx.is_array(parameter.master_weight):
            raise TypeError(f"Optimizer state requires concrete arrays, got {type(parameter.master_weight)}")
        result.append(parameter.master_weight)
    return tuple(result)


def _replace_master_weights(
    state: DistillTrainingState,
    master_weights: Sequence[Array],
) -> DistillTrainingState:
    trainable_parameters = tuple(
        replace(parameter, master_weight=master_weight)
        for parameter, master_weight in zip(state.trainable_parameters, master_weights, strict=True)
    )
    return DistillTrainingState(trainable_parameters=trainable_parameters)


def initialize_distill_optimizer_state(
    training_state: DistillTrainingState,
    optimizer: optax.GradientTransformation,
) -> DistillOptimizerState:
    return DistillOptimizerState(
        training_state=training_state,
        optimizer_state=optimizer.init(_concrete_master_weights(training_state)),
    )


def _batch_lengths(batch: DistillBatch) -> Int[Array, " batch"]:
    if batch.lengths_without_padding is not None:
        return batch.lengths_without_padding
    batch_size, num_tokens = batch.token_ids.shape
    return jnp.full((batch_size,), num_tokens, dtype=jnp.int32)


def _token_positions(batch: DistillBatch) -> Int[Array, "batch tokens"]:
    _, num_tokens = batch.token_ids.shape
    return jnp.broadcast_to(jnp.arange(num_tokens, dtype=jnp.int32), batch.token_ids.shape)


def _prediction_mask(batch: DistillBatch) -> Bool[Array, "batch prediction_tokens"]:
    _, num_tokens = batch.token_ids.shape
    prediction_tokens = max(num_tokens - 1, 0)
    lengths_without_padding = _batch_lengths(batch)
    return jnp.arange(prediction_tokens, dtype=jnp.int32)[None, :] < (lengths_without_padding[:, None] - 1)


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
    support_mask = jnp.zeros((len(traces), max_completion_tokens, max_support_tokens), dtype=jnp.bool_)

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
    token_positions = _token_positions(batch)
    decoder_result = decoder(
        token_ids=batch.token_ids,
        token_positions=token_positions,
    )
    return decoder_result.logits[:, :-1, :]


def compute_distill_kl_loss(
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
) -> DistillStepMetrics:
    metrics = compute_distill_batch_metrics(student, teacher, batch)
    return DistillStepMetrics(loss=metrics.loss, valid_tokens=metrics.valid_tokens)


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

    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    prediction_mask = _prediction_mask(batch).astype(token_kl.dtype)
    valid_tokens = prediction_mask.sum(dtype=jnp.int32)
    loss = jnp.sum(token_kl * prediction_mask) / valid_tokens
    student_top1 = jnp.argmax(student_logits, axis=-1)
    teacher_top1 = jnp.argmax(teacher_logits, axis=-1)
    top1_matches = jnp.sum((student_top1 == teacher_top1) & prediction_mask.astype(jnp.bool_), dtype=jnp.int32)
    return DistillBatchMetrics(
        loss=loss,
        valid_tokens=valid_tokens,
        top1_matches=top1_matches,
    )


def _trace_completion_mask(batch: TraceDistillBatch) -> Bool[Array, "batch completion_tokens"]:
    _, completion_tokens, _ = batch.support_token_ids.shape
    return jnp.arange(completion_tokens, dtype=jnp.int32)[None, :] < batch.completion_lengths[:, None]


def _trace_prediction_indices(batch: TraceDistillBatch) -> Int[Array, "batch completion_tokens"]:
    _, completion_tokens, _ = batch.support_token_ids.shape
    raw_indices = batch.prefix_lengths[:, None] - 1 + jnp.arange(completion_tokens, dtype=jnp.int32)[None, :]
    max_indices = jnp.maximum(batch.prefix_lengths + batch.completion_lengths - 2, 0)
    return jnp.minimum(raw_indices, max_indices[:, None])


def compute_trace_distill_kl_loss(
    student: Decoder,
    batch: TraceDistillBatch,
) -> DistillStepMetrics:
    token_positions = _token_positions(DistillBatch(token_ids=batch.token_ids))
    decoder_result = student(
        token_ids=batch.token_ids,
        token_positions=token_positions,
    )
    student_logits = decoder_result.logits[:, :-1, :].astype(jnp.float32)

    prediction_indices = _trace_prediction_indices(batch)
    batch_indices = jnp.arange(batch.token_ids.shape[0], dtype=jnp.int32)[:, None]
    student_completion_logits = student_logits[batch_indices, prediction_indices, :]
    student_support_logits = jnp.take_along_axis(
        student_completion_logits,
        batch.support_token_ids,
        axis=-1,
    )

    completion_mask = _trace_completion_mask(batch)
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
    return DistillStepMetrics(loss=loss, valid_tokens=valid_tokens)


def distill_train_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillTrainConfig,
    *,
    quantization_key: PRNGKeyArray | None = None,
) -> tuple[DistillOptimizerState, DistillStepMetrics]:
    current_master_weights = _concrete_master_weights(optimizer_state.training_state)

    def loss_fn(master_weights: tuple[Array, ...]) -> tuple[Float[Array, ""], DistillStepMetrics]:
        training_state = _replace_master_weights(optimizer_state.training_state, master_weights)
        materialized_student = materialize_trainable_module(student, training_state, config)
        if quantization_key is not None:
            materialized_student = stochastically_quantize_module(materialized_student, quantization_key)
        metrics = compute_distill_kl_loss(materialized_student, teacher, batch)
        return metrics.loss, metrics

    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        current_master_weights,
    )
    del loss

    updates, new_optimizer_state = optimizer.update(
        grads,
        optimizer_state.optimizer_state,
        current_master_weights,
    )
    updated_master_weights = optax.apply_updates(current_master_weights, updates)
    updated_training_state = _replace_master_weights(optimizer_state.training_state, updated_master_weights)
    return (
        DistillOptimizerState(
            training_state=updated_training_state,
            optimizer_state=new_optimizer_state,
        ),
        metrics,
    )
