from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Int

from lalamo.common import ParameterTree

__all__ = ["CompactableStateLayer", "State", "StateLayerBase"]


class StateLayerBase(eqx.Module):
    @abstractmethod
    def export(self) -> ParameterTree: ...


class CompactableStateLayer(StateLayerBase, ABC):
    @abstractmethod
    def prefix_lengths(self) -> Int[Array, "*batch"]: ...

    def prefix_length_for_sample(self, sample_index: int) -> int:
        prefix_lengths = self.prefix_lengths()
        if prefix_lengths.ndim == 0:
            return int(prefix_lengths)
        return int(prefix_lengths[sample_index])

    @abstractmethod
    def compact(
        self,
        cache_len: Int[Array, " batch"],
        accepted_indices: Int[Array, "batch max_slots"],
        num_accepted: Int[Array, " batch"],
        max_slots: int,
    ) -> Self: ...


@register_pytree_node_class
class State(tuple[StateLayerBase, ...]):
    __slots__ = ()

    def tree_flatten(self) -> tuple[tuple[StateLayerBase, ...], None]:
        return (tuple(self), None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[StateLayerBase, ...]) -> Self:  # noqa: ARG003
        return cls(children)
