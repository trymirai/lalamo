import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import jax.tree_util as jtu
import jax.numpy as jnp
from jax.tree_util import keystr
from jaxtyping import Array, DTypeLike

from lalamo.modules.common import ParameterLeafInfo, ParameterRole, iter_parameter_leaves

__all__ = [
    "DistillTrainConfig",
    "DistillParameterSummary",
    "DistillTrainableParameter",
    "DistillTrainingState",
    "get_optimizer_group",
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
    master_weight: Array


@dataclass(frozen=True)
class DistillTrainingState:
    trainable_parameters: tuple[DistillTrainableParameter, ...]


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


def is_leaf_trainable(info: ParameterLeafInfo, config: DistillTrainConfig) -> bool:
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


def get_optimizer_group(info: ParameterLeafInfo, config: DistillTrainConfig) -> str:
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
    return {
        keystr(path).lstrip("."): leaf
        for path, leaf in flat_with_path
    }


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
        if not eqx.is_array(value):
            raise TypeError(f"Trainable leaf {info.path} must be an array, got {type(value)}")

        trainable_parameters.append(
            DistillTrainableParameter(
                path=info.path,
                alias_paths=tuple(alias_paths[info.path]),
                parameter_role=info.parameter_role,
                optimizer_group=get_optimizer_group(info, config),
                master_weight=value.astype(master_dtype),
            )
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
    replacement_values: list[Array] = []
    for parameter in state.trainable_parameters:
        compute_weight = parameter.master_weight.astype(compute_dtype)
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
