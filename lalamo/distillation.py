import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import DTypeLike

from lalamo.modules.common import ParameterLeafInfo, ParameterRole

__all__ = [
    "DistillTrainConfig",
    "DistillParameterSummary",
    "get_optimizer_group",
    "is_leaf_trainable",
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
