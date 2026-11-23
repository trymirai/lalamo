from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.weight_matrix import MatmulConfig

from .linear import Linear, LinearConfig
from .residual import ConnectionConfig, ResidualConfig, ResidualConnectionBase, ResidualHeadBase
from .token_mixer import QKVInputs
from .utils import call_vmapped_twice

__all__ = [
    "AttnResQKVReadConfig",
    "AttnResReadConfig",
    "BlockAttnResConfig",
    "BlockAttnResConnection",
    "BlockAttnResHead",
    "BlockAttnResQKVConnection",
    "BlockAttnResStream",
    "MessageGate",
    "MessageGateConfig",
    "MessageGates",
    "QKVReadMode",
    "attention_residual_read",
    "rms_normalize",
]


class BlockAttnResStream(eqx.Module):
    blocks: Float[Array, "batch suffix_tokens blocks channels"]
    partial: Float[Array, "batch suffix_tokens channels"]
    previous_pass: Float[Array, "batch suffix_tokens previous_sources channels"] | None

    @property
    def sources(self) -> Float[Array, "batch suffix_tokens sources channels"]:
        return jnp.concatenate([self.blocks, self.partial[:, :, None]], axis=2)


def rms_normalize(inputs: Float[Array, "*dims channels"], epsilon: float) -> Float[Array, "*dims channels"]:
    upcast_inputs = inputs.astype(jnp.float32)
    return upcast_inputs * jax.lax.rsqrt(jnp.mean(jnp.square(upcast_inputs), axis=-1, keepdims=True) + epsilon)


def attention_residual_read(
    scaled_queries: Float[Array, "queries channels"],
    sources: Float[Array, "batch suffix_tokens sources channels"],
    epsilon: float,
) -> Float[Array, "batch suffix_tokens queries channels"]:
    normalized_sources = rms_normalize(sources, epsilon)
    logits = einsum(
        normalized_sources,
        scaled_queries.astype(jnp.float32),
        "batch tokens sources channels, queries channels -> batch tokens queries sources",
    )
    return einsum(
        jax.nn.softmax(logits, axis=-1),
        normalized_sources,
        "batch tokens queries sources, batch tokens sources channels -> batch tokens queries channels",
    )


class QKVReadMode(StrEnum):
    PARTIAL = "partial"
    SHARED = "shared"
    SEPARATE = "separate"


@dataclass(frozen=True)
class MessageGateConfig(LalamoConfig):
    linear_config: LinearConfig
    rank: int

    def init_gates(
        self,
        initializer: Initializer,
        model_dim: int,
        num_gates: int,
        *,
        shares_hidden_projection: bool,
    ) -> "MessageGates":
        if shares_hidden_projection:
            hidden_projections = (
                self.linear_config.init(
                    initializer,
                    input_dim=model_dim,
                    output_dims=(self.rank,) * num_gates,
                    has_biases=False,
                ),
            )
        else:
            hidden_projections = tuple(
                self.linear_config.init(initializer, input_dim=model_dim, output_dims=(self.rank,), has_biases=False)
                for _ in range(num_gates)
            )
        return MessageGates(
            config=self,
            sharding_config=initializer.sharding_config,
            hidden_projections=hidden_projections,
            gates=tuple(
                MessageGate(
                    config=self,
                    sharding_config=initializer.sharding_config,
                    scaled_query=initializer.zeros((model_dim,), dtype=jnp.float32),
                    message_projection=self.linear_config.init(
                        initializer,
                        input_dim=model_dim,
                        output_dims=(self.rank,),
                        has_biases=False,
                    ),
                    forget_gain_projection=self.linear_config.init(
                        initializer,
                        input_dim=self.rank,
                        output_dims=(model_dim, model_dim),
                        has_biases=False,
                        dtype=jnp.float32,
                    ),
                )
                for _ in range(num_gates)
            ),
        )


class MessageGate(LalamoModule[MessageGateConfig]):
    scaled_query: Float[Array, " channels"]
    message_projection: Linear
    forget_gain_projection: Linear

    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        hidden_low_rank: Float[Array, "batch suffix_tokens rank"],
        previous_pass: Float[Array, "batch suffix_tokens previous_sources channels"],
        epsilon: float,
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        message_keychain, forget_gain_keychain = keychain.split()
        added_sharding_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        message = attention_residual_read(self.scaled_query[None], previous_pass, epsilon)[:, :, 0]
        (message_low_rank,) = call_vmapped_twice(
            self.message_projection,
            rms_normalize(message, epsilon),
            forward_pass_config=matmul_config,
            keychain=message_keychain,
            added_sharding_axes=added_sharding_axes,
        )
        forget_logits, gain_logits = call_vmapped_twice(
            self.forget_gain_projection,
            hidden_low_rank.astype(jnp.float32) + message_low_rank.astype(jnp.float32),
            forward_pass_config=matmul_config,
            keychain=forget_gain_keychain,
            added_sharding_axes=added_sharding_axes,
        )
        forget = jnp.tanh(forget_logits.astype(jnp.float32))
        gain = jnp.tanh(gain_logits.astype(jnp.float32))
        return (1 - forget) * inputs.astype(jnp.float32) + gain * message


class MessageGates(LalamoModule[MessageGateConfig]):
    hidden_projections: tuple[Linear, ...]
    gates: tuple[MessageGate, ...]

    def __call__(
        self,
        inputs: tuple[Float[Array, "batch suffix_tokens channels"], ...],
        previous_pass: Float[Array, "batch suffix_tokens previous_sources channels"],
        epsilon: float,
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], ...]:
        *gate_keychains, hidden_keychain = keychain.split(len(self.gates) + 1)
        added_sharding_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        hidden_low_ranks = tuple(
            low_rank
            for projection, hidden, projection_keychain in zip(
                self.hidden_projections,
                inputs,
                hidden_keychain.split(len(self.hidden_projections)),
                strict=True,
            )
            for low_rank in call_vmapped_twice(
                projection,
                rms_normalize(hidden, epsilon),
                forward_pass_config=matmul_config,
                keychain=projection_keychain,
                added_sharding_axes=added_sharding_axes,
            )
        )
        gate_inputs = inputs if len(inputs) == len(self.gates) else inputs * len(self.gates)
        return tuple(
            gate(hidden, low_rank, previous_pass, epsilon, matmul_config, keychain=gate_keychain)
            for gate, hidden, low_rank, gate_keychain in zip(
                self.gates,
                gate_inputs,
                hidden_low_ranks,
                gate_keychains,
                strict=True,
            )
        )


@dataclass(frozen=True)
class AttnResReadConfig(ConnectionConfig):
    message_gate_config: MessageGateConfig | None


@dataclass(frozen=True)
class AttnResQKVReadConfig(ConnectionConfig):
    read_mode: QKVReadMode
    starts_block: bool
    message_gate_config: MessageGateConfig | None


@dataclass(frozen=True)
class BlockAttnResConfig(ResidualConfig):
    epsilon: float

    def init_connection(
        self,
        initializer: Initializer,
        model_dim: int,
        connection_config: ConnectionConfig | None,
    ) -> "BlockAttnResConnection | BlockAttnResQKVConnection":
        match connection_config:
            case AttnResReadConfig(message_gate_config=message_gate_config):
                return BlockAttnResConnection(
                    config=self,
                    sharding_config=initializer.sharding_config,
                    scaled_query=initializer.zeros((model_dim,), dtype=jnp.float32),
                    message_gates=(
                        message_gate_config.init_gates(initializer, model_dim, 1, shares_hidden_projection=True)
                        if message_gate_config is not None
                        else None
                    ),
                )
            case AttnResQKVReadConfig(read_mode=read_mode, message_gate_config=message_gate_config):
                num_queries = {QKVReadMode.PARTIAL: 0, QKVReadMode.SHARED: 1, QKVReadMode.SEPARATE: 3}[read_mode]
                return BlockAttnResQKVConnection(
                    config=self,
                    sharding_config=initializer.sharding_config,
                    read_config=connection_config,
                    scaled_queries=(
                        initializer.zeros((num_queries, model_dim), dtype=jnp.float32) if num_queries else None
                    ),
                    message_gates=(
                        message_gate_config.init_gates(
                            initializer,
                            model_dim,
                            3,
                            shares_hidden_projection=read_mode != QKVReadMode.SEPARATE,
                        )
                        if message_gate_config is not None
                        else None
                    ),
                )
        raise ValueError(f"Block attention residuals need an AttnRes read config, got {connection_config}.")

    def init_head(self, initializer: Initializer, model_dim: int) -> "BlockAttnResHead":
        return BlockAttnResHead(
            config=self,
            sharding_config=initializer.sharding_config,
            scaled_output_query=initializer.zeros((model_dim,), dtype=jnp.float32),
        )


class BlockAttnResConnection(
    ResidualConnectionBase[BlockAttnResConfig, BlockAttnResStream, Float[Array, "batch suffix_tokens channels"]],
):
    scaled_query: Float[Array, " channels"]
    message_gates: MessageGates | None

    def __call__[AuxT](
        self,
        stream: BlockAttnResStream,
        sublayer: Callable[
            [Float[Array, "batch suffix_tokens channels"], Keychain],
            tuple[Float[Array, "batch suffix_tokens channels"], AuxT],
        ],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[BlockAttnResStream, AuxT]:
        gate_keychain, sublayer_keychain = keychain.split()
        inputs = attention_residual_read(self.scaled_query[None], stream.sources, self.config.epsilon)[:, :, 0]
        if self.message_gates is not None and stream.previous_pass is not None:
            (inputs,) = self.message_gates(
                (inputs,),
                stream.previous_pass,
                self.config.epsilon,
                matmul_config,
                keychain=gate_keychain,
            )
        update, aux = sublayer(inputs.astype(stream.partial.dtype), sublayer_keychain)
        partial = (stream.partial.astype(jnp.float32) + update.astype(jnp.float32)).astype(stream.partial.dtype)
        return eqx.tree_at(lambda updated: updated.partial, stream, partial), aux


class BlockAttnResQKVConnection(ResidualConnectionBase[BlockAttnResConfig, BlockAttnResStream, QKVInputs]):
    read_config: AttnResQKVReadConfig = eqx.field(static=True)
    scaled_queries: Float[Array, "queries channels"] | None
    message_gates: MessageGates | None

    def read(self, stream: BlockAttnResStream) -> tuple[Float[Array, "batch suffix_tokens channels"], ...]:
        if self.scaled_queries is None:
            return (stream.partial.astype(jnp.float32),)
        reads = attention_residual_read(self.scaled_queries, stream.sources, self.config.epsilon)
        return tuple(rearrange(reads, "batch tokens queries channels -> queries batch tokens channels"))

    def __call__[AuxT](
        self,
        stream: BlockAttnResStream,
        sublayer: Callable[[QKVInputs, Keychain], tuple[Float[Array, "batch suffix_tokens channels"], AuxT]],
        matmul_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[BlockAttnResStream, AuxT]:
        gate_keychain, sublayer_keychain = keychain.split()
        reads = self.read(stream)
        if self.message_gates is not None and stream.previous_pass is not None:
            reads = self.message_gates(
                reads,
                stream.previous_pass,
                self.config.epsilon,
                matmul_config,
                keychain=gate_keychain,
            )
        match tuple(read.astype(stream.partial.dtype) for read in reads):
            case (query,):
                inputs = QKVInputs(query=query, key=None, value=None)
            case (query, key, value):
                inputs = QKVInputs(query=query, key=key, value=value)
            case _:
                raise ValueError(f"Expected one or three attention inputs, got {len(reads)}.")
        update, aux = sublayer(inputs, sublayer_keychain)
        if self.read_config.starts_block:
            blocks, partial = stream.sources, update
        else:
            blocks, partial = stream.blocks, stream.partial.astype(jnp.float32) + update.astype(jnp.float32)
        return BlockAttnResStream(
            blocks=blocks,
            partial=partial.astype(stream.partial.dtype),
            previous_pass=stream.previous_pass,
        ), aux


class BlockAttnResHead(ResidualHeadBase[BlockAttnResConfig, BlockAttnResStream]):
    scaled_output_query: Float[Array, " channels"]

    def expand(self, embeddings: Float[Array, "batch suffix_tokens channels"]) -> BlockAttnResStream:
        batch_size, num_tokens, channels = embeddings.shape
        return BlockAttnResStream(
            blocks=jnp.zeros((batch_size, num_tokens, 0, channels), dtype=embeddings.dtype),
            partial=embeddings,
            previous_pass=None,
        )

    def readout(
        self,
        stream: BlockAttnResStream,
        matmul_config: MatmulConfig,  # noqa: ARG002
        *,
        keychain: Keychain,  # noqa: ARG002
    ) -> Float[Array, "batch suffix_tokens channels"]:
        outputs = attention_residual_read(self.scaled_output_query[None], stream.sources, self.config.epsilon)
        return outputs[:, :, 0].astype(stream.partial.dtype)

    def restart(self, stream: BlockAttnResStream, entry_stream: BlockAttnResStream) -> BlockAttnResStream:
        _, _, num_entry_sources, _ = entry_stream.sources.shape
        return BlockAttnResStream(
            blocks=entry_stream.blocks,
            partial=entry_stream.partial,
            previous_pass=stream.sources[:, :, num_entry_sources:],
        )
