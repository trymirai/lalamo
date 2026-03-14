import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax.tree_util import keystr
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.modules.common import ParameterLeafInfo, ParameterRole, iter_parameter_leaves
from lalamo.modules.decoder import Decoder

__all__ = [
    "DistillBatch",
    "DistillOptimizerState",
    "DistillParameterSummary",
    "DistillStepMetrics",
    "DistillTrainConfig",
    "DistillTrainableParameter",
    "DistillTrainingState",
    "compute_distill_kl_loss",
    "distill_train_step",
    "get_optimizer_group",
    "initialize_distill_optimizer_state",
    "initialize_distill_training_state",
    "is_leaf_trainable",
    "materialize_trainable_module",
    "summarize_distill_parameters",
]


@dataclass(frozen=True)
class DistillTrainConfig:
    train_bias: bool = True
    train_norm: bool = True
    train_embedding: bool = False
    train_adapter: bool = True
    train_quant_aux: bool = False
    train_base_weight: bool = False
    master_dtype: DTypeLike = jnp.float32
    compute_dtype: DTypeLike = jnp.bfloat16


@dataclass(frozen=True)
class DistillParameterSummary:
    total_parameters: int
    trainable_parameters: int
    total_master_bytes: int
    by_role: dict[str, int]
    by_group: dict[str, int]


@dataclass(frozen=True)
class DistillTrainableParameter:
    path: str
    alias_paths: tuple[str, ...]
    parameter_role: ParameterRole
    optimizer_group: str
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


def _is_embedding_leaf(info: ParameterLeafInfo) -> bool:
    return info.parameter_role in {
        ParameterRole.INPUT_EMBEDDING,
        ParameterRole.OUTPUT_EMBEDDING,
        ParameterRole.INPUT_OUTPUT_EMBEDDING,
    }


def _is_norm_leaf(info: ParameterLeafInfo) -> bool:
    return info.parameter_role in {
        ParameterRole.NORM_SCALE,
        ParameterRole.NORM_BIAS,
    }


def _is_quant_aux_leaf(info: ParameterLeafInfo) -> bool:
    return info.parameter_role in {
        ParameterRole.QUANT_SCALE,
        ParameterRole.QUANT_ZERO_POINT,
        ParameterRole.QUANT_DEQ_BIAS,
    }


def _is_adapter_leaf(info: ParameterLeafInfo) -> bool:
    return info.owner_type.__name__ == "QLoRALinear" and info.field_name in {
        "lora_down_weights",
        "lora_up_weights",
    }


def _is_bias_leaf(info: ParameterLeafInfo) -> bool:
    return info.parameter_role == ParameterRole.LINEAR_BIAS


def _is_base_weight_leaf(info: ParameterLeafInfo) -> bool:
    return info.parameter_role == ParameterRole.LINEAR_WEIGHT and not _is_adapter_leaf(info)


def is_leaf_trainable(info: ParameterLeafInfo, config: DistillTrainConfig) -> bool:  # noqa: PLR0911
    if info.alias_of is not None:
        return False
    if _is_adapter_leaf(info):
        return config.train_adapter
    if _is_quant_aux_leaf(info):
        return config.train_quant_aux
    if _is_embedding_leaf(info):
        return config.train_embedding
    if _is_norm_leaf(info):
        return config.train_norm
    if _is_bias_leaf(info):
        return config.train_bias
    if _is_base_weight_leaf(info):
        return config.train_base_weight
    return info.trainable_default


def get_optimizer_group(info: ParameterLeafInfo, config: DistillTrainConfig) -> str:  # noqa: PLR0911
    if not is_leaf_trainable(info, config):
        return "frozen"
    if _is_quant_aux_leaf(info):
        return "quant_aux"
    if _is_embedding_leaf(info):
        return "embedding"
    if _is_norm_leaf(info):
        return "norm"
    if _is_bias_leaf(info):
        return "bias"
    if len(info.shape) >= 2:
        return "muon"
    return "default"


def summarize_distill_parameters(
    leaves: Sequence[ParameterLeafInfo],
    config: DistillTrainConfig,
) -> DistillParameterSummary:
    by_role: dict[str, int] = defaultdict(int)
    by_group: dict[str, int] = defaultdict(int)
    master_dtype = jnp.dtype(config.master_dtype)

    total_parameters = 0
    trainable_parameters = 0
    total_master_bytes = 0

    for info in leaves:
        if info.alias_of is not None:
            continue

        parameter_count = math.prod(info.shape)
        total_parameters += parameter_count

        role_key = info.parameter_role.value
        by_role[role_key] += parameter_count

        optimizer_group = get_optimizer_group(info, config)
        by_group[optimizer_group] += parameter_count

        if optimizer_group == "frozen":
            continue

        trainable_parameters += parameter_count
        total_master_bytes += parameter_count * master_dtype.itemsize

    return DistillParameterSummary(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        total_master_bytes=total_master_bytes,
        by_role=dict(by_role),
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
        if not is_leaf_trainable(info, config):
            continue

        value = path_to_leaf[info.path]
        if not (eqx.is_array(value) or isinstance(value, jax.ShapeDtypeStruct)):
            raise TypeError(f"Trainable leaf {info.path} must be array-like, got {type(value)}")

        trainable_parameters.append(
            DistillTrainableParameter(
                path=info.path,
                alias_paths=tuple(alias_paths[info.path]),
                parameter_role=info.parameter_role,
                optimizer_group=get_optimizer_group(info, config),
                master_weight=_cast_array_like(value, master_dtype),
            ),
        )

    return DistillTrainingState(trainable_parameters=tuple(trainable_parameters))


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


def _cast_array_like(value: Array | jax.ShapeDtypeStruct, dtype: DTypeLike) -> Array | jax.ShapeDtypeStruct:
    if eqx.is_array(value):
        return value.astype(dtype)
    return jax.ShapeDtypeStruct(value.shape, dtype)


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
    teacher_logits = _decoder_logits(teacher, batch).astype(jnp.float32)
    student_logits = _decoder_logits(student, batch).astype(jnp.float32)

    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    teacher_probs = jnp.exp(teacher_log_probs)

    token_kl = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    prediction_mask = _prediction_mask(batch).astype(token_kl.dtype)
    valid_tokens = prediction_mask.sum(dtype=jnp.int32)
    loss = jnp.sum(token_kl * prediction_mask) / valid_tokens
    return DistillStepMetrics(loss=loss, valid_tokens=valid_tokens)


def distill_train_step(
    optimizer_state: DistillOptimizerState,
    optimizer: optax.GradientTransformation,
    student: Decoder,
    teacher: Decoder,
    batch: DistillBatch,
    config: DistillTrainConfig,
) -> tuple[DistillOptimizerState, DistillStepMetrics]:
    current_master_weights = _concrete_master_weights(optimizer_state.training_state)

    def loss_fn(master_weights: tuple[Array, ...]) -> tuple[Float[Array, ""], DistillStepMetrics]:
        training_state = _replace_master_weights(optimizer_state.training_state, master_weights)
        materialized_student = materialize_trainable_module(student, training_state, config)
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
