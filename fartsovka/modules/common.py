from dataclasses import dataclass
from types import UnionType

import equinox as eqx
from cattrs import Converter
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from fartsovka.common import DType, ParameterDict

__all__ = [
    "ModuleSample",
    "FartsovkaModule",
    "config_converter",
    "register_config_union",
]


@dataclass
class ModuleSample:
    inputs: tuple[Array, ...]
    outputs: tuple[Array, ...]

    def export(self) -> ParameterDict:
        return ParameterDict(
            inputs=self.inputs,
            outputs=self.outputs
        )


class FartsovkaModule[ConfigT](eqx.Module):
    config: ConfigT = eqx.field(static=True)

    def export_weights(self) -> ParameterDict:
        raise NotImplementedError

    def export_samples(self, suffix_length: int, key: PRNGKeyArray) -> ParameterDict:
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
    lambda t: t in [jnp.dtype, DType],
    _dtype_to_str,
)

config_converter.register_structure_hook_func(
    lambda t: t in [jnp.dtype, DType],
    lambda s, _: _str_to_dtype(s),
)


def register_config_union(union_type: UnionType) -> None:
    union_members = union_type.__args__
    name_to_type = {m.__name__: m for m in union_members}
    config_converter.register_unstructure_hook(
        union_type,
        lambda o: {
            "type": o.__class__.__name__,
            **config_converter.unstructure(o),
        },
    )

    def structure[T](config: dict, _: type[T]) -> T:
        new_config = dict(config)
        type_name = new_config.pop("type")
        target_type = name_to_type[type_name]
        return name_to_type[type_name](**config_converter.structure(new_config, target_type))

    config_converter.register_structure_hook(
        union_type,
        structure,
    )


@dataclass
class DummyUnionMember:
    pass
