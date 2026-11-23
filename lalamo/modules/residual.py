from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import MatmulConfig

from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped_twice

__all__ = [
    "AdditiveResidualConfig",
    "AdditiveResidualConnection",
    "AdditiveResidualHead",
    "GatedResidualConfig",
    "GatedResidualConnection",
    "GatedResidualHead",
    "GatedStreamReader",
    "ResidualConfig",
    "ResidualConnectionBase",
    "ResidualHeadBase",
    "normalize_streams",
]


def normalize_streams(
    norm: Normalization,
    streams: Float[Array, "batch suffix_tokens streams channels"],
) -> Float[Array, "batch suffix_tokens streams channels"]:
    return eqx.filter_vmap(call_vmapped_twice, in_axes=(0, 2), out_axes=2)(norm, streams)


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
            [Float[Array, "batch suffix_tokens channels"], Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[StreamT, AuxT]: ...


class ResidualHeadBase[ConfigT: ResidualConfig, StreamT](LalamoModule[ConfigT]):
    @abstractmethod
    def expand(self, embeddings: Float[Array, "batch suffix_tokens channels"]) -> StreamT: ...

    @abstractmethod
    def readout(
        self,
        stream: StreamT,
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]: ...


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
            [Float[Array, "batch suffix_tokens channels"], Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,  # noqa: ARG002
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], AuxT]:
        update, aux = sublayer(stream, keychain)
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
        self,
        stream: Float[Array, "batch suffix_tokens channels"],
        matmul_config: MatmulConfig,  # noqa: ARG002
        *,
        keychain: Keychain,  # noqa: ARG002
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return stream


@dataclass(frozen=True)
class GatedResidualConfig(ResidualConfig):
    stream_count: int
    low_rank_dim: int
    norm_config: NormalizationConfig
    linear_config: LinearConfig

    def __post_init__(self) -> None:
        if self.stream_count < 1:
            raise ValueError(f"Gated residual requires at least one stream, got {self.stream_count}.")

    def init_reader(self, initializer: Initializer, model_dim: int) -> "GatedStreamReader":
        return GatedStreamReader(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=self.norm_config.init_mixture(initializer, self.stream_count, model_dim),
            down_projection=self.linear_config.init(
                initializer,
                input_dim=self.stream_count * model_dim,
                output_dims=(self.low_rank_dim,),
                has_biases=False,
            ),
            up_projection=self.linear_config.init(
                initializer,
                input_dim=self.low_rank_dim,
                output_dims=(self.stream_count * model_dim,),
                has_biases=False,
            ),
        )

    def init_connection(self, initializer: Initializer, model_dim: int) -> "GatedResidualConnection":
        return GatedResidualConnection(
            config=self,
            sharding_config=initializer.sharding_config,
            reader=self.init_reader(initializer, model_dim),
            injection=self.linear_config.init(
                initializer,
                input_dim=self.stream_count * model_dim,
                output_dims=(self.stream_count,),
                has_biases=False,
            ),
        )

    def init_head(self, initializer: Initializer, model_dim: int) -> "GatedResidualHead":
        return GatedResidualHead(
            config=self,
            sharding_config=initializer.sharding_config,
            reader=self.init_reader(initializer, model_dim),
        )


class GatedStreamReader(LalamoModule[GatedResidualConfig]):
    norm: Normalization
    down_projection: Linear
    up_projection: Linear

    def __call__(
        self,
        stream: Float[Array, "batch suffix_tokens streams channels"],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens streams channels"], Float[Array, "batch suffix_tokens channels"]]:
        down_keychain, up_keychain = keychain.split()
        added_sharding_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized_streams = normalize_streams(self.norm, stream)
        flat_normalized_streams = rearrange(
            normalized_streams, "batch tokens streams channels -> batch tokens (streams channels)"
        )
        (low_rank_mix,) = call_vmapped_twice(
            self.down_projection,
            flat_normalized_streams,
            forward_pass_config=matmul_config,
            keychain=down_keychain,
            added_sharding_axes=added_sharding_axes,
        )
        (flat_mix_logits,) = call_vmapped_twice(
            self.up_projection,
            jax.nn.silu(low_rank_mix / self.config.stream_count),
            forward_pass_config=matmul_config,
            keychain=up_keychain,
            added_sharding_axes=added_sharding_axes,
        )
        mix_weights = rearrange(
            jax.nn.sigmoid(flat_mix_logits),
            "batch tokens (streams channels) -> batch tokens streams channels",
            streams=self.config.stream_count,
        )
        mixed_stream = jnp.mean(mix_weights * normalized_streams, axis=2)
        return normalized_streams, mixed_stream


class GatedResidualConnection(
    ResidualConnectionBase[GatedResidualConfig, Float[Array, "batch suffix_tokens streams channels"]],
):
    reader: GatedStreamReader
    injection: Linear

    def __call__[AuxT](
        self,
        stream: Float[Array, "batch suffix_tokens streams channels"],
        sublayer: Callable[
            [Float[Array, "batch suffix_tokens channels"], Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens streams channels"], AuxT]:
        reader_keychain, injection_keychain, sublayer_keychain = keychain.split(3)
        normalized_streams, mixed_stream = self.reader(stream, matmul_config, keychain=reader_keychain)
        (injection_logits,) = call_vmapped_twice(
            self.injection,
            rearrange(normalized_streams, "batch tokens streams channels -> batch tokens (streams channels)"),
            forward_pass_config=matmul_config,
            keychain=injection_keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        injection_weights = 2 * jax.nn.sigmoid(injection_logits / self.config.stream_count)
        update, aux = sublayer(mixed_stream, sublayer_keychain)
        updated_stream = stream + rearrange(
            injection_weights,
            "batch tokens streams -> batch tokens streams 1",
        ) * rearrange(update, "batch tokens channels -> batch tokens 1 channels")
        assert updated_stream.dtype == stream.dtype
        return updated_stream, aux


class GatedResidualHead(ResidualHeadBase[GatedResidualConfig, Float[Array, "batch suffix_tokens streams channels"]]):
    reader: GatedStreamReader

    def expand(
        self,
        embeddings: Float[Array, "batch suffix_tokens channels"],
    ) -> Float[Array, "batch suffix_tokens streams channels"]:
        return repeat(
            embeddings,
            "batch tokens channels -> batch tokens streams channels",
            streams=self.config.stream_count,
        )

    def readout(
        self,
        stream: Float[Array, "batch suffix_tokens streams channels"],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        _, mixed_stream = self.reader(stream, matmul_config, keychain=keychain)
        return mixed_stream
