from .common import EmptySpeculatorState, NoSpeculator, Speculator, SpeculatorState
from .proposal import AcceptedProposal, ProposalInputs, TrieProposal
from .state import LMState, MemoryBuffers, RingBuffer, StateRequest

__all__ = [
    "AcceptedProposal",
    "CollectTracesEvent",
    "EmptySpeculatorState",
    "LMState",
    "MemoryBuffers",
    "NoSpeculator",
    "ProposalInputs",
    "RingBuffer",
    "Speculator",
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
