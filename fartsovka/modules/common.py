from dataclasses import dataclass
from enum import Enum
from types import UnionType

import equinox as eqx
from cattrs import Converter
from jax import numpy as jnp
from jaxtyping import DTypeLike

from fartsovka.common import ParameterDict

__all__ = [
    "AttentionType",
    "DummyUnionMember",
    "FartsovkaModule",
    "config_converter",
    "register_config_union",
]


class WeightLayout(Enum):
    INPUT_OUTPUT = "input_output"
    OUTPUT_INPUT = "output_input"

    def __str__(self) -> str:
        match self:
            case WeightLayout.INPUT_OUTPUT:
                return "(input, output)"
            case WeightLayout.OUTPUT_INPUT:
                return "(output, input)"
        raise ValueError(f"Unknown weight layout: {self}")


class AttentionType(Enum):
    GLOBAL = "global"
    SLIDING_WINDOW = "sliding_window"


class FartsovkaModule[ConfigT](eqx.Module):
    config: ConfigT = eqx.field(static=True)

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.INPUT_OUTPUT) -> ParameterDict:
        raise NotImplementedError


def _dtype_to_str(dtype: jnp.dtype) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    return str(dtype.dtype)  # type: ignore


def _str_to_dtype(dtype_str: str) -> jnp.dtype:
    return {
        "int4": jnp.int4,
        "int8": jnp.int8,
        "int16": jnp.int16,
        "int32": jnp.int32,
        "int64": jnp.int64,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
        "float64": jnp.float64,
    }[dtype_str]


config_converter = Converter()


config_converter.register_unstructure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    _dtype_to_str,
)

config_converter.register_structure_hook_func(
    lambda t: t in [jnp.dtype, DTypeLike],
    lambda s, _: _str_to_dtype(s),
)


def register_config_union(union_type: UnionType) -> None:
    union_members = union_type.__args__
    name_to_type = {m.__name__: m for m in union_members}

    def unstructure(obj: object) -> dict | None:
        if obj is None:
            return None
        return {
            "type": obj.__class__.__name__,
            **config_converter.unstructure(obj),
        }

    config_converter.register_unstructure_hook(
        union_type,
        unstructure,
    )

    config_converter.register_unstructure_hook(
        union_type | None,
        unstructure,
    )

    def structure[T](config: dict | None, _: type[T]) -> T | None:
        if config is None:
            return None
        new_config = dict(config)
        type_name = new_config.pop("type")
        target_type = name_to_type[type_name]
        return name_to_type[type_name](**config_converter.structure(new_config, target_type))

    config_converter.register_structure_hook(
        union_type,
        structure,
    )

    config_converter.register_structure_hook(
        union_type | None,
        structure,
    )


@dataclass
class DummyUnionMember:
    pass
