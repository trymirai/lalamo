from collections.abc import Iterable, Mapping

import equinox as eqx
from jaxtyping import Array

from fartsovka.common import ParameterPath

type NestedParameters = Mapping[str, Array | NestedParameters] | Iterable[Array | NestedParameters]


__all__ = ["ParameterDict", "FartsovkaModule"]


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


class FartsovkaModule(eqx.Module):
    def export_weights(self) -> ParameterDict:
        raise NotImplementedError
