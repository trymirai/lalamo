from abc import ABC, abstractmethod
from lalamo.speculator.proposal import TrieProposal
from lalamo.speculator.state import LMState, MemoryBuffers, RingBuffer, StateRequest

__all__ = [
    "LMState",
    "MemoryBuffers",
    "NoSpeculator",
    "RingBuffer",
    "Speculator",
    "StateRequest",
]


class Speculator(ABC):
    @property
    def state_request(self) -> StateRequest:
        return StateRequest()

    @abstractmethod
    def draft(self, state: LMState) -> TrieProposal: ...


class NoSpeculator(Speculator):
    def draft(self, state: LMState) -> TrieProposal:
        return state.create_root_proposal(budget=1)
