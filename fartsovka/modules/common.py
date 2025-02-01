from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import UnionType

import equinox as eqx
from cattrs import Converter
from jax import numpy as jnp
from jaxtyping import Array

from fartsovka.common import DType, ParameterPath

type NestedParameters = Mapping[str, Array | NestedParameters] | Iterable[Array | NestedParameters]


__all__ = [
    "FartsovkaModule",
    "ParameterDict",
    "DummyUnionMember",
    "config_converter",
    "register_config_union",
]


class ParameterDict(dict[str, Array]):
    def __init__(self, **kwargs: Array | NestedParameters | Iterable[Array | NestedParameters]) -> None:
        super().__init__(self._flatten(kwargs))

    def __setitem__(
        self,
        key: str,
        value: Array | NestedParameters | Iterable[Array | NestedParameters],
    ) -> None:
        key = ParameterPath(key)

        if isinstance(value, Array):
            super().__setitem__(key, value)
            return

        for subkey, subvalue in self._flatten(value).items():
            super().__setitem__(key / subkey, subvalue)

    @classmethod
    def _flatten(cls, nested_parameters: NestedParameters) -> dict[str, Array]:
        result: dict[str, Array] = {}
        if not isinstance(nested_parameters, Mapping):
            nested_parameters = {str(i): value for i, value in enumerate(nested_parameters)}
        for key, value in nested_parameters.items():
            key_path = ParameterPath(key)
            if isinstance(value, Array):
                result[key_path] = value
            else:
                result.update({key_path / subkey: subvalue for subkey, subvalue in cls._flatten(value).items()})
        return result


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


def register_config_union(union_type: UnionType) -> None:
    config_converter.register_unstructure_hook(
        union_type,
        lambda o: {
            "type": o.__class__.__name__,
            **config_converter.unstructure(o),
        },
    )


@dataclass
class DummyUnionMember:
    pass
