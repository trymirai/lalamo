from abc import abstractmethod
from typing import Self

import equinox as eqx
from jax.tree_util import register_pytree_node_class

from lalamo.common import ParameterTree

__all__ = ["State", "StateLayerBase"]


class StateLayerBase(eqx.Module):
    @abstractmethod
    def export(self) -> ParameterTree: ...


@register_pytree_node_class
class State(tuple[StateLayerBase, ...]):
    __slots__ = ()

    def tree_flatten(self) -> tuple[tuple[StateLayerBase, ...], None]:
        return (tuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[StateLayerBase, ...]) -> Self:  # noqa: ARG003
        return cls(children)
