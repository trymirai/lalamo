from typing import Self

import equinox as eqx
from jax.tree_util import register_pytree_node_class

__all__ = ["State", "StateLayerBase"]


class StateLayerBase(eqx.Module):
    pass


@register_pytree_node_class
class State(tuple[StateLayerBase, ...]):
    __slots__ = ()

    def tree_flatten(self) -> tuple[tuple[StateLayerBase, ...], None]:
        return (tuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[StateLayerBase, ...]) -> Self:  # noqa: ARG003
        return cls(children)
