import inspect
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Self

from jaxtyping import Array, Float
from typer import Typer

from lalamo.modules.decoder import Decoder
from lalamo.modules.token_mixers.state.common import State
from lalamo.registry_abc import RegistryABC


@dataclass(frozen=True)
class LMState:
    kv_cache: State
    layer_outputs: tuple[Float[Array, "suffix channels"], ...]
    output_norm: Float[Array, "suffix channels"] | None
    logits: Float[Array, " vocab"]
    position: int
    bonus: int


@dataclass(frozen=True)
class SamplerConfig:
    width: int = 4
    K: int = 8
    max_tokens: int = 2048
    seed: int = 42


@dataclass(frozen=True, kw_only=True)
class SpeculationStep:
    accepted: list[int]
    bonus: int


@dataclass(frozen=True, kw_only=True)
class Speculator[P](RegistryABC):
    decoder: Decoder
    config: SamplerConfig
    eos_set: frozenset[int]
    trace_layer_outputs: tuple[int, ...] | None = None
    trace_output_norm: bool = False
    prefill_hidden_range: int | None = None

    name: ClassVar[str]

    @classmethod
    def deserialize(
        cls,
        name: str,
        data: bytes,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        **extra: object,
    ) -> "Speculator":
        for subcls in cls.registered_types():
            if subcls.name == name:
                return subcls.deserialize_impl(
                    data,
                    decoder=decoder,
                    config=config,
                    eos_set=eos_set,
                    **extra,
                )
        known = ", ".join(sorted(s.name for s in cls.registered_types()))
        raise ValueError(f"Unknown speculator {name!r}. Registered: {known}")

    @classmethod
    def registered_types(cls) -> Iterator[type["Speculator"]]:
        return (subcls for subcls in cls.__descendants__() if not inspect.isabstract(subcls))

    @classmethod
    def deserialize_impl(
        cls,
        data: bytes,
        *,
        decoder: Decoder,
        config: SamplerConfig,
        eos_set: frozenset[int],
        **extra: object,
    ) -> Self:
        raise NotImplementedError(f"{cls.__name__} does not support deserialization")

    def serialize(self) -> bytes:
        raise NotImplementedError(f"{type(self).__name__} does not support serialization")

    @abstractmethod
    def draft(self, lm: LMState) -> P: ...

    @abstractmethod
    def step(self, lm: LMState) -> tuple[Self, LMState, SpeculationStep]: ...

    @abstractmethod
    def prefill(self, prompt_ids: list[int]) -> tuple[Self, LMState]: ...

    @property
    def generation_capacity(self) -> int:
        return self.config.max_tokens + self.config.K * self.config.width + 16

    @staticmethod
    def train_command(app: Typer, callbacks_type: type) -> None:
        pass
