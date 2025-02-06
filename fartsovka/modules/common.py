from dataclasses import dataclass
from types import UnionType

import equinox as eqx
from cattrs import Converter
from jax import numpy as jnp

from fartsovka.common import DType, ParameterDict

__all__ = [
    "FartsovkaModule",
    "DummyUnionMember",
    "config_converter",
    "register_config_union",
]


class FartsovkaModule[ConfigT](eqx.Module):
    config: ConfigT = eqx.field(static=True)

    def export_weights(self) -> ParameterDict:
        raise NotImplementedError


def _dtype_to_str(dtype: jnp.dtype) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    return str(dtype.dtype)  # type: ignore


config_converter = Converter()


config_converter.register_unstructure_hook_func(
    lambda t: t in [jnp.dtype, DType],
    _dtype_to_str,
)

config_converter.register_structure_hook_func(
    lambda t: t in [jnp.dtype, DType],
    jnp.dtype,
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
