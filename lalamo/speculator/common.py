from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from inspect import isabstract
from pathlib import Path
from typing import Any, ClassVar

import msgpack
from jaxtyping import Array, Int, Key

from lalamo.data.completion_features import FeatureRequest
from lalamo.modules.decoder import Decoder
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.proposal import ChainProposal, Proposal
from lalamo.speculator.state import LMState, PrefillResults
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "ARTIFACT_HEADER",
    "BACKEND_ENTRY_POINT_GROUP",
    "LMState",
    "NoSpeculator",
    "PrefillResults",
    "Speculator",
    "SpeculatorBackend",
    "get_speculator_backend",
    "load_speculator",
    "read_speculator_artifact",
    "speculator_backends",
    "write_speculator_artifact",
]


BACKEND_ENTRY_POINT_GROUP = "lalamo.speculator_backends"
ARTIFACT_HEADER = "mirai.speculator"


class Speculator(ABC):
    @property
    def feature_request(self) -> FeatureRequest:
        return FeatureRequest(completions=())

    @property
    @abstractmethod
    def max_step_tokens(self) -> int: ...

    def init_state(
        self,
        prefill_results: PrefillResults,
        next_token_position: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
    ) -> LMState:
        return LMState.from_prefill(
            prefill_results,
            next_token_position,
            sampling_policy,
            gumbel_keys,
        )

    @abstractmethod
    def draft(self, state: LMState) -> Proposal: ...


class NoSpeculator(Speculator):
    @property
    def max_step_tokens(self) -> int:
        return 1

    def draft(self, state: LMState) -> ChainProposal:
        return state.create_chain_proposal()


class SpeculatorBackend[ConfigT](RegistryABC):
    name: ClassVar[str]
    config_type: ClassVar[type[Any]]

    @classmethod
    @abstractmethod
    def create_trainer(
        cls,
        config: ConfigT,
        artifact_path: Path,
        target_model: Decoder,
    ) -> Any: ...  # noqa: ANN401

    @classmethod
    @abstractmethod
    def deserialize(cls, fields: tuple[Any, ...], target_model: Decoder) -> Speculator: ...


def write_speculator_artifact(
    path: Path | str,
    backend: type[SpeculatorBackend[Any]],
    *backend_fields: Any,  # noqa: ANN401
) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as file:
        msgpack.pack((ARTIFACT_HEADER, backend.name, *backend_fields), file, use_bin_type=True)


def read_speculator_artifact(path: Path | str) -> tuple[str, tuple[Any, ...]]:
    with Path(path).open("rb") as file:
        values = msgpack.unpack(file, raw=False)

    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("Speculator artifact must be a msgpack array with at least format and speculator kind.")

    artifact_format, speculator_kind, *backend_fields = values
    if artifact_format != ARTIFACT_HEADER:
        raise ValueError(f"Unsupported speculator artifact format {artifact_format!r}.")
    if not isinstance(speculator_kind, str):
        raise TypeError("Speculator kind must be a string.")
    return speculator_kind, tuple(backend_fields)


def speculator_backends() -> dict[str, type[SpeculatorBackend[Any]]]:
    for entry_point in entry_points(group=BACKEND_ENTRY_POINT_GROUP):
        entry_point.load()

    backends: dict[str, type[SpeculatorBackend[Any]]] = {}
    for backend in SpeculatorBackend.__descendants__():
        if isabstract(backend):
            continue
        if not isinstance(backend.name, str):
            raise TypeError(f"Speculator backend {backend.__name__} must define a string name.")
        if backend.name in backends:
            raise ValueError(f"Duplicate speculator backend {backend.name!r}.")
        backends[backend.name] = backend
    return backends


def get_speculator_backend(name: str) -> type[SpeculatorBackend[Any]]:
    backends = speculator_backends()
    backend = backends.get(name)
    if backend is None:
        available = ", ".join(sorted(backends)) or "none"
        raise ValueError(f"Unknown speculator backend {name!r}. Available backends: {available}.")
    return backend


def load_speculator(path: Path | str, target_model: Decoder) -> Speculator:
    speculator_kind, backend_fields = read_speculator_artifact(path)
    backend = get_speculator_backend(speculator_kind)
    return backend.deserialize(backend_fields, target_model)
