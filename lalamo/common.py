from collections.abc import Iterable, Mapping

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

__all__ = [
    "DEFAULT_PRECISION",
    "ParameterDict",
    "ParameterPath",
]

DEFAULT_PRECISION: DTypeLike = jnp.bfloat16


type NestedParameters = Mapping[str, Array | NestedParameters] | Iterable[Array | NestedParameters]


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


class ParameterPath(str):
    __slots__ = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.split("."))

    def __truediv__(self, other: str | int) -> "ParameterPath":
        if not self:
            return ParameterPath(str(other))
        return ParameterPath(self + "." + str(other))
