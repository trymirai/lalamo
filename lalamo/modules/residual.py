from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.utils.registry_abc import RegistryABC

__all__ = [
    "AdditiveResidualConfig",
    "AdditiveResidualConnection",
    "AdditiveResidualHead",
    "ResidualConfig",
    "ResidualConnectionBase",
    "ResidualHeadBase",
]


@dataclass(frozen=True)
class ResidualConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init_connection(self, initializer: Initializer, model_dim: int) -> "ResidualConnectionBase": ...

    @abstractmethod
    def init_head(self, initializer: Initializer, model_dim: int) -> "ResidualHeadBase": ...


class ResidualConnectionBase[ConfigT: ResidualConfig, StreamT](LalamoModule[ConfigT]):
    @abstractmethod
    def __call__[AuxT](
        self,
        stream: StreamT,
        sublayer: Callable[
            [Float[Array, "batch suffix_tokens channels"]],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
    ) -> tuple[StreamT, AuxT]: ...


class ResidualHeadBase[ConfigT: ResidualConfig, StreamT](LalamoModule[ConfigT]):
    @abstractmethod
    def expand(self, embeddings: Float[Array, "batch suffix_tokens channels"]) -> StreamT: ...

    @abstractmethod
    def readout(self, stream: StreamT) -> Float[Array, "batch suffix_tokens channels"]: ...


@dataclass(frozen=True)
class AdditiveResidualConfig(ResidualConfig):
    def init_connection(self, initializer: Initializer, model_dim: int) -> "AdditiveResidualConnection":  # noqa: ARG002
        return AdditiveResidualConnection(config=self, sharding_config=initializer.sharding_config)

    def init_head(self, initializer: Initializer, model_dim: int) -> "AdditiveResidualHead":  # noqa: ARG002
        return AdditiveResidualHead(config=self, sharding_config=initializer.sharding_config)


class AdditiveResidualConnection(
    ResidualConnectionBase[AdditiveResidualConfig, Float[Array, "batch suffix_tokens channels"]],
):
    def __call__[AuxT](
        self,
        stream: Float[Array, "batch suffix_tokens channels"],
        sublayer: Callable[
            [Float[Array, "batch suffix_tokens channels"]],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], AuxT]:
        update, aux = sublayer(stream)
        updated_stream = stream + update
        assert updated_stream.dtype == stream.dtype
        return updated_stream, aux


class AdditiveResidualHead(ResidualHeadBase[AdditiveResidualConfig, Float[Array, "batch suffix_tokens channels"]]):
    def expand(
        self,
        embeddings: Float[Array, "batch suffix_tokens channels"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return embeddings

    def readout(
        self, stream: Float[Array, "batch suffix_tokens channels"]
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return stream
