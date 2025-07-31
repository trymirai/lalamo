from collections import defaultdict
from collections.abc import Mapping

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

__all__ = [
    "DEFAULT_PRECISION",
    "ParameterPath",
    "ParameterTree",
    "flatten_parameters",
    "unflatten_parameters",
]

DEFAULT_PRECISION: DTypeLike = jnp.bfloat16


type ParameterTree = Mapping[str, Array | ParameterTree] | list[Array | ParameterTree]


def flatten_parameters(nested_parameters: ParameterTree) -> dict[str, Array]:
    result: dict[str, Array] = {}
    if not isinstance(nested_parameters, Mapping):
        nested_parameters = {str(i): value for i, value in enumerate(nested_parameters)}
    for key, value in nested_parameters.items():
        key_path = ParameterPath(key)
        if isinstance(value, Array):
            result[key_path] = value
        else:
            result.update({key_path / subkey: subvalue for subkey, subvalue in flatten_parameters(value).items()})
    return result


def unflatten_parameters(flat_parameters: dict[str, Array]) -> ParameterTree:
    groups: dict[str, dict[str, Array] | Array] = defaultdict(dict)
    for key, value in flat_parameters.items():
        match key.split(".", maxsplit=1):
            case [head]:
                groups[head] = value
            case [head, tail]:
                groups[head][tail] = value

    unflattened_groups = {}
    for key, value in groups.items():
        if isinstance(value, Array):
            unflattened_groups[key] = value
        else:
            unflattened_groups[key] = unflatten_parameters(value)

    if any(key.isnumeric() for key in unflattened_groups):
        assert set(unflattened_groups.keys()) == set(map(str, range(len(unflattened_groups))))
        return list(unflattened_groups.values())
    return unflattened_groups


class ParameterPath(str):
    __slots__ = ()

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.split("."))

    def __truediv__(self, other: str | int) -> "ParameterPath":
        if not self:
            return ParameterPath(str(other))
        return ParameterPath(self + "." + str(other))
