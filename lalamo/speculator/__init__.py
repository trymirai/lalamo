from .common import (
    EmptySpeculatorDraftState,
    EmptySpeculatorState,
    NoSpeculator,
    Speculator,
    SpeculatorDraftState,
    SpeculatorState,
)
from .proposal import AcceptedProposal, ProposalInputs, TrieProposal
from .state import LMState, MemoryBuffers, RingBuffer, StateRequest

__all__ = [
    "AcceptedProposal",
    "CollectTracesEvent",
    "EmptySpeculatorDraftState",
    "EmptySpeculatorState",
    "LMState",
    "MemoryBuffers",
    "NoSpeculator",
    "ProposalInputs",
    "RingBuffer",
    "Speculator",
    "SpeculatorDraftState",
    "SpeculatorState",
    "StateRequest",
    "TrieProposal",
    "inference_collect_traces",
]


def __getattr__(name: str) -> object:
    if name in {"CollectTracesEvent", "inference_collect_traces"}:
        from .inference import CollectTracesEvent, inference_collect_traces

        return {
            "CollectTracesEvent": CollectTracesEvent,
            "inference_collect_traces": inference_collect_traces,
        }[name]
    raise AttributeError(name)
