import math
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field
from lalamo.utils.registry_abc import RegistryABC
from lalamo.weight_matrix import MatmulConfig

from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped_twice

__all__ = [
    "AdditiveResidualConfig",
    "AdditiveResidualConnection",
    "AdditiveResidualHead",
    "BlockAttentionConnection",
    "BlockAttentionHead",
    "BlockAttentionResidualConfig",
    "BlockAttentionStream",
    "ConnectionContext",
    "GatedResidualConfig",
    "GatedResidualConnection",
    "GatedResidualHead",
    "GatedStreamReader",
    "MessageGate",
    "MessageGateConfig",
    "ResidualConfig",
    "ResidualConnectionBase",
    "ResidualHeadBase",
    "SublayerRole",
    "normalize_streams",
]


type SublayerInputs = (
    Float[Array, "batch suffix_tokens channels"] | Float[Array, "batch suffix_tokens inputs channels"]
)


def normalize_streams(
    norm: Normalization,
    streams: Float[Array, "batch suffix_tokens streams channels"],
) -> Float[Array, "batch suffix_tokens streams channels"]:
    return eqx.filter_vmap(call_vmapped_twice, in_axes=(0, 2), out_axes=2)(norm, streams)


def replicate_inputs(
    inputs: Float[Array, "batch suffix_tokens channels"],
    num_inputs: int,
) -> SublayerInputs:
    if num_inputs == 1:
        return inputs
    return repeat(inputs, "batch tokens channels -> batch tokens inputs channels", inputs=num_inputs)


class SublayerRole(StrEnum):
    MIXER = "mixer"
    MLP = "mlp"


@dataclass(frozen=True)
class ConnectionContext:
    """Where a residual connection sits. `num_reads` is 1 when all `num_inputs` share one read of the stream."""

    layer_index: int
    role: SublayerRole
    num_inputs: int
    num_reads: int
    is_looped: bool


@dataclass(frozen=True)
class ResidualConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init_connection(
        self,
        initializer: Initializer,
        model_dim: int,
        context: ConnectionContext | None = None,
    ) -> "ResidualConnectionBase": ...

    @abstractmethod
    def init_head(self, initializer: Initializer, model_dim: int) -> "ResidualHeadBase": ...


class ResidualConnectionBase[ConfigT: ResidualConfig, StreamT](LalamoModule[ConfigT]):
    @abstractmethod
    def __call__[AuxT](
        self,
        stream: StreamT,
        sublayer: Callable[
            [SublayerInputs, Keychain],
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

    def begin_loop_pass(self, loop_input: StreamT, previous_pass_output: StreamT) -> StreamT:
        """The stream entering a repeated pass over looped layers."""
        raise NotImplementedError(f"{type(self).__name__} does not support looped layers.")

    def end_loop(self, stream: StreamT) -> StreamT:
        """The stream leaving the last pass over looped layers."""
        raise NotImplementedError(f"{type(self).__name__} does not support looped layers.")


@dataclass(frozen=True)
class AdditiveResidualConfig(ResidualConfig):
    def init_connection(
        self,
        initializer: Initializer,
        model_dim: int,  # noqa: ARG002
        context: ConnectionContext | None = None,
    ) -> "AdditiveResidualConnection":
        return AdditiveResidualConnection(
            config=self,
            sharding_config=initializer.sharding_config,
            num_inputs=context.num_inputs if context is not None else 1,
        )

    def init_head(self, initializer: Initializer, model_dim: int) -> "AdditiveResidualHead":  # noqa: ARG002
        return AdditiveResidualHead(config=self, sharding_config=initializer.sharding_config)


class AdditiveResidualConnection(
    ResidualConnectionBase[AdditiveResidualConfig, Float[Array, "batch suffix_tokens channels"]],
):
    num_inputs: int = field(static=True, default=1)

    def __call__[AuxT](
        self,
        stream: Float[Array, "batch suffix_tokens channels"],
        sublayer: Callable[
            [SublayerInputs, Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,  # noqa: ARG002
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], AuxT]:
        update, aux = sublayer(replicate_inputs(stream, self.num_inputs), keychain)
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

    def init_connection(
        self,
        initializer: Initializer,
        model_dim: int,
        context: ConnectionContext | None = None,
    ) -> "GatedResidualConnection":
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
            num_inputs=context.num_inputs if context is not None else 1,
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
    num_inputs: int = field(static=True, default=1)

    def __call__[AuxT](
        self,
        stream: Float[Array, "batch suffix_tokens streams channels"],
        sublayer: Callable[
            [SublayerInputs, Keychain],
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
        update, aux = sublayer(replicate_inputs(mixed_stream, self.num_inputs), sublayer_keychain)
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


def _unit_rms(
    sources: Float[Array, "*sources channels"],
    epsilon: float,
) -> Float[Array, "*sources channels"]:
    upcasted = sources.astype(jnp.float32)
    return upcasted * jax.lax.rsqrt(jnp.mean(jnp.square(upcasted), axis=-1, keepdims=True) + epsilon)


def _depth_attention(
    normalized_sources: Float[Array, "batch tokens sources channels"],
    queries: Float[Array, "queries channels"],
) -> Float[Array, "batch tokens queries channels"]:
    *_, model_dim = normalized_sources.shape
    logits = jnp.einsum(
        "btsc,qc->btqs",
        normalized_sources,
        queries.astype(jnp.float32),
    )
    weights = jax.nn.softmax(logits * model_dim**-0.5, axis=-1)
    return jnp.einsum(
        "btqs,btsc->btqc",
        weights,
        normalized_sources,
    )


class BlockAttentionStream(eqx.Module):
    """The partial sum of the current block, the inputs of the completed blocks and the messages of a loop pass."""

    partial: Float[Array, "batch tokens channels"]
    blocks: Float[Array, "batch tokens blocks channels"]
    messages: Float[Array, "batch tokens sources channels"] | None = None

    def sources(self) -> Float[Array, "batch tokens sources channels"]:
        return jnp.concatenate([self.blocks, self.partial[:, :, None]], axis=2)


@dataclass(frozen=True)
class MessageGateConfig(LalamoConfig):
    """Mixes a message into a sublayer input as `(1 - tanh(forget)) * input + tanh(gain) * message`.

    Both gates are low-rank projections of the RMS-normalized input and message.
    """

    rank: int
    linear_config: LinearConfig

    def init(self, initializer: Initializer, model_dim: int) -> "MessageGate":
        def init_gate() -> Linear:
            return self.linear_config.init(
                initializer,
                self.rank,
                (model_dim,),
                has_biases=False,
                is_sharded=False,
                dtype=jnp.float32,
            )

        return MessageGate(
            config=self,
            sharding_config=initializer.sharding_config,
            down_projection=self.linear_config.init(
                initializer,
                2 * model_dim,
                (self.rank,),
                has_biases=False,
                is_sharded=False,
            ),
            forget_projection=init_gate(),
            gain_projection=init_gate(),
        )


class MessageGate(LalamoModule[MessageGateConfig]):
    down_projection: Linear
    forget_projection: Linear
    gain_projection: Linear

    def __call__(
        self,
        inputs: Float[Array, " channels"],
        message: Float[Array, " channels"],
        *,
        epsilon: float,
        forward_pass_config: MatmulConfig,
        keychain: Keychain,
    ) -> Float[Array, " channels"]:
        down_keychain, forget_keychain, gain_keychain = keychain.split(3)
        upcasted_inputs = inputs.astype(jnp.float32)
        upcasted_message = message.astype(jnp.float32)
        joint = jnp.concatenate([_unit_rms(upcasted_inputs, epsilon), _unit_rms(upcasted_message, epsilon)])
        (low_rank,) = self.down_projection(
            joint.astype(self.down_projection.weights.dtype),
            keychain=down_keychain,
            forward_pass_config=forward_pass_config,
        )
        low_rank = low_rank.astype(jnp.float32)
        (forget,) = self.forget_projection(low_rank, keychain=forget_keychain, forward_pass_config=forward_pass_config)
        (gain,) = self.gain_projection(low_rank, keychain=gain_keychain, forward_pass_config=forward_pass_config)
        return (1 - jnp.tanh(forget)) * upcasted_inputs + jnp.tanh(gain) * upcasted_message


@dataclass(frozen=True)
class BlockAttentionResidualConfig(ResidualConfig):
    """Attention residuals over blocks of `block_size` layers.

    Every sublayer input is a softmax mix of the RMS-normalized inputs of the completed blocks and of the partial
    sum of the current block, weighted by learned queries. A sublayer adds its output to the partial sum; the mixer
    of a layer that starts a block stores the partial sum as a completed block and restarts it from its output. The
    first layer's mixer reads the embeddings directly. In repeated passes over looped layers, each input also reads
    a message from the previous pass and mixes it in through a `MessageGate`.
    """

    block_size: int
    epsilon: float
    message_gate_config: MessageGateConfig | None

    def init_connection(
        self,
        initializer: Initializer,
        model_dim: int,
        context: ConnectionContext | None = None,
    ) -> "BlockAttentionConnection":
        if context is None:
            raise ValueError("Block attention connections need to know where they sit.")
        if context.num_reads not in (1, context.num_inputs):
            raise ValueError(f"{context.num_reads} reads cannot serve {context.num_inputs} inputs.")
        reads_stream_directly = context.role == SublayerRole.MIXER and context.layer_index == 0
        if reads_stream_directly and context.num_reads != 1:
            raise ValueError("The first mixer reads the embeddings directly, so its inputs share one read.")
        query_scale = 1 / math.sqrt(model_dim)
        if context.is_looped:
            if self.message_gate_config is None:
                raise ValueError("Looped layers need a message gate config.")
            message_queries = initializer.normal(query_scale, (context.num_inputs, model_dim), dtype=jnp.float32)
            message_gates = tuple(
                self.message_gate_config.init(initializer, model_dim) for _ in range(context.num_inputs)
            )
        else:
            message_queries = None
            message_gates = None
        return BlockAttentionConnection(
            config=self,
            sharding_config=initializer.sharding_config,
            queries=(
                None
                if reads_stream_directly
                else initializer.normal(query_scale, (context.num_reads, model_dim), dtype=jnp.float32)
            ),
            message_queries=message_queries,
            message_gates=message_gates,
            num_inputs=context.num_inputs,
            starts_block=context.role == SublayerRole.MIXER and context.layer_index % self.block_size == 0,
        )

    def init_head(self, initializer: Initializer, model_dim: int) -> "BlockAttentionHead":
        return BlockAttentionHead(
            config=self,
            sharding_config=initializer.sharding_config,
            query=initializer.normal(1 / math.sqrt(model_dim), (1, model_dim), dtype=jnp.float32),
        )


class BlockAttentionConnection(ResidualConnectionBase[BlockAttentionResidualConfig, BlockAttentionStream]):
    queries: Float[Array, "reads channels"] | None
    message_queries: Float[Array, "inputs channels"] | None
    message_gates: tuple[MessageGate, ...] | None
    num_inputs: int = field(static=True)
    starts_block: bool = field(static=True)

    def _read(
        self,
        stream: BlockAttentionStream,
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> SublayerInputs:
        if self.queries is None:
            mixes = stream.partial[:, :, None].astype(jnp.float32)
        else:
            mixes = _depth_attention(_unit_rms(stream.sources(), self.config.epsilon), self.queries)
        reads = mixes.shape[2]
        if reads != self.num_inputs:
            mixes = repeat(
                mixes[:, :, 0], "batch tokens channels -> batch tokens inputs channels", inputs=self.num_inputs
            )
        if stream.messages is not None:
            if self.message_gates is None or self.message_queries is None:
                raise ValueError("This connection received loop messages but has no message gates.")
            messages = _depth_attention(_unit_rms(stream.messages, self.config.epsilon), self.message_queries)
            gate_keychains = keychain.split(self.num_inputs)
            mixes = jnp.stack(
                [
                    call_vmapped_twice(
                        partial(gate, epsilon=self.config.epsilon),
                        mixes[:, :, input_index],
                        messages[:, :, input_index],
                        forward_pass_config=matmul_config,
                        keychain=gate_keychain,
                        added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
                    )
                    for input_index, (gate, gate_keychain) in enumerate(
                        zip(self.message_gates, gate_keychains, strict=True)
                    )
                ],
                axis=2,
            )
        mixes = mixes.astype(stream.partial.dtype)
        if self.num_inputs == 1:
            return mixes[:, :, 0]
        return mixes

    def __call__[AuxT](
        self,
        stream: BlockAttentionStream,
        sublayer: Callable[
            [SublayerInputs, Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[BlockAttentionStream, AuxT]:
        read_keychain, sublayer_keychain = keychain.split()
        update, aux = sublayer(self._read(stream, matmul_config, keychain=read_keychain), sublayer_keychain)
        update = update.astype(stream.partial.dtype)
        if self.starts_block:
            updated_stream = BlockAttentionStream(
                partial=update,
                blocks=jnp.concatenate([stream.blocks, stream.partial[:, :, None]], axis=2),
                messages=stream.messages,
            )
        else:
            updated_stream = BlockAttentionStream(
                partial=stream.partial + update,
                blocks=stream.blocks,
                messages=stream.messages,
            )
        return updated_stream, aux


class BlockAttentionHead(ResidualHeadBase[BlockAttentionResidualConfig, BlockAttentionStream]):
    query: Float[Array, "1 channels"]

    def expand(self, embeddings: Float[Array, "batch suffix_tokens channels"]) -> BlockAttentionStream:
        batch, tokens, channels = embeddings.shape
        return BlockAttentionStream(
            partial=embeddings,
            blocks=jnp.zeros((batch, tokens, 0, channels), dtype=embeddings.dtype),
        )

    def readout(
        self,
        stream: BlockAttentionStream,
        matmul_config: MatmulConfig,  # noqa: ARG002
        *,
        keychain: Keychain,  # noqa: ARG002
    ) -> Float[Array, "batch suffix_tokens channels"]:
        mixed = _depth_attention(_unit_rms(stream.sources(), self.config.epsilon), self.query)
        return mixed[:, :, 0].astype(stream.partial.dtype)

    def begin_loop_pass(
        self,
        loop_input: BlockAttentionStream,
        previous_pass_output: BlockAttentionStream,
    ) -> BlockAttentionStream:
        # The previous pass first stored the loop input itself as a block; its messages come from the later ones.
        _, _, num_loop_input_blocks, _ = loop_input.blocks.shape
        return BlockAttentionStream(
            partial=loop_input.partial,
            blocks=loop_input.blocks,
            messages=jnp.concatenate(
                [
                    previous_pass_output.blocks[:, :, num_loop_input_blocks + 1 :],
                    previous_pass_output.partial[:, :, None],
                ],
                axis=2,
            ),
        )

    def end_loop(self, stream: BlockAttentionStream) -> BlockAttentionStream:
        return BlockAttentionStream(partial=stream.partial, blocks=stream.blocks)
