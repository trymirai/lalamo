from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int
from tokenizers import Tokenizer

from lalamo.audio.utils import dummy_char_level_tokenizer_config
from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.raw_text_codec import RawTextCodec, RawTextCodecConfig
from lalamo.module import ForwardPassMode, Keychain, SpeculatorState
from lalamo.modules.decoder import DecoderResult
from lalamo.modules.embedding import EmbeddingBase
from lalamo.utils.sharding import ShardingConfig

__all__ = [
    "AcceptedProposal",
    "ChainProposal",
    "NoSpeculator",
    "NoSpeculatorConfig",
    "NoSpeculatorState",
    "Proposal",
    "ProposalInputs",
    "Speculator",
    "SpeculatorConfig",
    "TreeProposal",
]


class ProposalInputs(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    lengths_without_padding: Int[Array, " batch"]
    attention_parent_indices: Int[Array, "batch nodes"] | None = None
    forward_pass_mode: ForwardPassMode = eqx.field(static=True, default=ForwardPassMode.MULTI_TOKEN)


class AcceptedProposal(eqx.Module):
    token_ids: Int[Array, "batch tokens"]
    token_positions: Int[Array, "batch tokens"]
    source_indices: Int[Array, "batch tokens"]
    lengths: Int[Array, " batch"]
    accepted_node_indices: Int[Array, "batch nodes"]
    num_accepted_nodes: Int[Array, " batch"]

    def last_token_ids(
        self,
        stopped_token_ids: Int[Array, " batch"],
    ) -> Int[Array, " batch"]:
        batch_size, _ = self.token_ids.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        slots = jnp.maximum(self.lengths - 1, 0)
        lengths_sharding = jax.typeof(self.lengths).sharding
        if lengths_sharding.mesh.empty:
            token_ids = self.token_ids[batch_indices, slots]
        else:
            token_ids = self.token_ids.at[batch_indices, slots].get(out_sharding=lengths_sharding)
        stopped_token_ids = jnp.zeros_like(self.lengths, dtype=stopped_token_ids.dtype) + stopped_token_ids
        return jnp.where(self.lengths > 0, token_ids, stopped_token_ids)

    def last_token_indices(self) -> Int[Array, " batch"]:
        batch_size, _ = self.token_positions.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        slots = jnp.maximum(self.lengths - 1, 0)
        lengths_sharding = jax.typeof(self.lengths).sharding
        if lengths_sharding.mesh.empty:
            token_positions = self.token_positions[batch_indices, slots]
        else:
            token_positions = self.token_positions.at[batch_indices, slots].get(out_sharding=lengths_sharding)
        return jnp.where(self.lengths > 0, token_positions - 1, jnp.zeros_like(token_positions))

    def with_lengths(self, new_lengths: Int[Array, " batch"]) -> "AcceptedProposal":
        emitted = jnp.arange(self.token_ids.shape[1], dtype=jnp.int32)[None, :] < new_lengths[:, None]
        return AcceptedProposal(
            token_ids=jnp.where(emitted, self.token_ids, 0),
            token_positions=jnp.where(emitted, self.token_positions, 0),
            source_indices=jnp.where(emitted, self.source_indices, -1),
            lengths=new_lengths,
            accepted_node_indices=self.accepted_node_indices,
            num_accepted_nodes=self.num_accepted_nodes,
        )

    def has_eos(self, eos_token_ids: Int[Array, " eos_tokens"]) -> Bool[Array, " batch"]:
        slots = jnp.arange(self.token_ids.shape[1], dtype=jnp.int32)[None, :]
        is_eos = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        return jnp.any((slots < self.lengths[:, None]) & is_eos, axis=1)

    def trim_at_eos(self, eos_token_ids: Int[Array, " eos_tokens"]) -> "AcceptedProposal":
        slots = jnp.arange(self.token_ids.shape[1], dtype=jnp.int32)[None, :]
        is_eos = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        eos_hits = (slots < self.lengths[:, None]) & is_eos
        eos_length = (jnp.argmax(eos_hits, axis=1) + 1).astype(self.lengths.dtype)
        new_lengths = jnp.where(jnp.any(eos_hits, axis=1), jnp.minimum(self.lengths, eos_length), self.lengths)
        return self.with_lengths(new_lengths)

    def where_active(self, active_mask: Bool[Array, " batch"]) -> "AcceptedProposal":
        slots = jnp.arange(self.token_ids.shape[1], dtype=jnp.int32)[None, :]
        lengths = jnp.where(active_mask, self.lengths, 0)
        num_accepted_nodes = jnp.where(active_mask, self.num_accepted_nodes, 0)
        emitted = slots < lengths[:, None]
        return AcceptedProposal(
            token_ids=jnp.where(emitted, self.token_ids, 0),
            token_positions=jnp.where(emitted, self.token_positions, 0),
            source_indices=jnp.where(emitted, self.source_indices, -1),
            lengths=lengths,
            accepted_node_indices=jnp.where(
                slots < num_accepted_nodes[:, None],
                self.accepted_node_indices,
                -1,
            ),
            num_accepted_nodes=num_accepted_nodes,
        )

    def gather_top_k(
        self,
        node_top_k_token_ids: Int[Array, "batch nodes k"],
        node_top_k_token_logits: Float[Array, "batch nodes k"],
    ) -> tuple[Int[Array, "batch tokens k"], Float[Array, "batch tokens k"]]:
        batch_size, _, _ = node_top_k_token_ids.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        source_indices = jnp.clip(self.source_indices, 0, node_top_k_token_ids.shape[1] - 1)
        valid_sources = self.source_indices >= 0
        token_top_k_ids = node_top_k_token_ids[batch_indices, source_indices]
        token_top_k_logits = node_top_k_token_logits[batch_indices, source_indices]
        return (
            jnp.where(valid_sources[:, :, None], token_top_k_ids, 0),
            jnp.where(valid_sources[:, :, None], token_top_k_logits, 0),
        )


class Proposal(eqx.Module, ABC):
    @abstractmethod
    def forward_inputs(self) -> ProposalInputs: ...

    @abstractmethod
    def accept(self, sampled_token_ids: Int[Array, "batch nodes"]) -> AcceptedProposal: ...


class ChainProposal(Proposal):
    token_ids: Int[Array, "batch proposal_slots"]
    token_positions: Int[Array, "batch proposal_slots"]
    lengths: Int[Array, " batch"]

    @classmethod
    def empty(
        cls,
        token_positions: Int[Array, "batch proposal_slots"],
        token_dtype: DTypeLike,
    ) -> "ChainProposal":
        return cls(
            token_ids=jnp.full_like(token_positions, -1, dtype=token_dtype),
            token_positions=token_positions,
            lengths=jnp.zeros_like(token_positions[:, 0]),
        )

    def forward_inputs(self) -> ProposalInputs:
        _, num_proposal_slots = self.token_ids.shape
        if num_proposal_slots == 1:
            return ProposalInputs(
                token_ids=self.token_ids,
                token_positions=self.token_positions,
                lengths_without_padding=self.lengths,
                forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            )
        parent_indices = jnp.broadcast_to(
            jnp.arange(num_proposal_slots, dtype=jnp.int32) - 1,
            self.token_ids.shape,
        )
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            lengths_without_padding=self.lengths,
            attention_parent_indices=parent_indices,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )

    def accept(self, sampled_token_ids: Int[Array, "batch nodes"]) -> AcceptedProposal:
        _, nodes = self.token_ids.shape
        slots = jnp.arange(nodes, dtype=self.lengths.dtype)[None, :]
        matches = self.token_ids[:, 1:] == sampled_token_ids[:, :-1]
        num_emitted = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1).astype(self.lengths.dtype), axis=1) + 1
        num_accepted_nodes = jnp.where(self.lengths > 0, num_emitted, 0)

        emitted = slots < num_emitted[:, None]
        node_indices = slots.astype(jnp.int32)
        return AcceptedProposal(
            token_ids=jnp.where(emitted, sampled_token_ids, 0),
            token_positions=jnp.where(emitted, self.token_positions + 1, 0),
            source_indices=jnp.where(emitted, node_indices, -1),
            lengths=num_emitted,
            accepted_node_indices=jnp.where(slots < num_accepted_nodes[:, None], node_indices, -1),
            num_accepted_nodes=num_accepted_nodes,
        )


def walk_accepted_path(
    token_ids: Int[Array, " nodes"],
    parent_indices: Int[Array, " nodes"],
    sampled_token_ids: Int[Array, " nodes"],
    length: Int[Array, ""],
) -> tuple[Int[Array, " nodes"], Int[Array, ""]]:
    (num_nodes,) = token_ids.shape
    node_ids = jnp.arange(num_nodes, dtype=jnp.int32)
    valid = node_ids < length

    def step(
        carry: tuple[Int[Array, ""], Bool[Array, ""]],
        _: None,
    ) -> tuple[tuple[Int[Array, ""], Bool[Array, ""]], Int[Array, ""]]:
        current, done = carry
        matches = valid & (parent_indices == current) & (token_ids == sampled_token_ids[current]) & ~done
        has_match = jnp.any(matches)
        matched_node = jnp.argmax(matches).astype(jnp.int32)
        next_node = jnp.where(has_match, matched_node, -1)
        return (jnp.where(has_match, matched_node, current), done | ~has_match), next_node

    _, next_nodes = jax.lax.scan(
        step,
        (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.bool)),
        length=num_nodes - 1,
    )
    path = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), next_nodes])
    path_length = 1 + jnp.sum(next_nodes >= 0, dtype=jnp.int32)
    path = jnp.where(node_ids < path_length, path, -1)
    return path, path_length


class TreeProposal(Proposal):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    parent_indices: Int[Array, "batch nodes"]
    lengths: Int[Array, " batch"]

    @classmethod
    def empty(
        cls,
        token_positions: Int[Array, "batch nodes"],
        token_dtype: DTypeLike,
        parent_indices: Int[Array, "batch nodes"],
    ) -> "TreeProposal":
        return cls(
            token_ids=jnp.full_like(token_positions, -1, dtype=token_dtype),
            token_positions=token_positions,
            parent_indices=parent_indices,
            lengths=jnp.zeros_like(token_positions[:, 0]),
        )

    def forward_inputs(self) -> ProposalInputs:
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            lengths_without_padding=self.lengths,
            attention_parent_indices=self.parent_indices,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )

    def accept(self, sampled_token_ids: Int[Array, "batch nodes"]) -> AcceptedProposal:
        batch_size, num_nodes = self.token_ids.shape
        path, num_emitted = jax.vmap(walk_accepted_path)(
            self.token_ids,
            self.parent_indices,
            sampled_token_ids,
            self.lengths,
        )
        num_accepted_nodes = jnp.where(self.lengths > 0, num_emitted, 0)

        slots = jnp.arange(num_nodes, dtype=jnp.int32)[None, :]
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        emitted = slots < num_emitted[:, None]
        path_nodes = jnp.maximum(path, 0)
        return AcceptedProposal(
            token_ids=jnp.where(emitted, sampled_token_ids[batch_indices, path_nodes], 0),
            token_positions=jnp.where(emitted, self.token_positions[batch_indices, path_nodes] + 1, 0),
            source_indices=jnp.where(emitted, path_nodes, -1),
            lengths=num_emitted,
            accepted_node_indices=jnp.where(slots < num_accepted_nodes[:, None], path, -1),
            num_accepted_nodes=num_accepted_nodes,
        )


@dataclass(frozen=True)
class SpeculatorConfig(ModelConfig[RawTextCodecConfig]):
    @abstractmethod
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "Speculator": ...


class Speculator[SpeculatorStateT: SpeculatorState, ConfigT: SpeculatorConfig](
    Model[RawTextCodecConfig, ConfigT, RawTextCodec],
    ABC,
):
    token_codec: RawTextCodec

    requires_activation_trace: ClassVar[bool] = True

    @property
    def max_proposal_tokens(self) -> int:
        return 1

    def empty_proposal(
        self,
        token_positions: Int[Array, "batch nodes"],
        token_dtype: DTypeLike,
    ) -> Proposal:
        return ChainProposal.empty(token_positions, token_dtype)

    @abstractmethod
    def init_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> SpeculatorStateT: ...

    def prefill_chunk(
        self,
        state: SpeculatorStateT,
        decoder_result: DecoderResult,
        chunk_lengths: Int[Array, " batch"],
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT:
        _ = (decoder_result, chunk_lengths, keychain)
        return state

    def update_state(
        self,
        state: SpeculatorStateT,
        decoder_result: DecoderResult,
        accepted: AcceptedProposal,
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT:
        _ = (decoder_result, accepted, keychain)
        return state

    @abstractmethod
    def draft(
        self,
        state: SpeculatorStateT,
        last_token_ids: Int[Array, " batch"],
        last_token_indices: Int[Array, " batch"],
        target_embedding: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> Proposal: ...


class NoSpeculatorState(SpeculatorState):
    pass


@cache
def _dummy_token_codec() -> RawTextCodec:
    tokenizer = Tokenizer.from_str(dummy_char_level_tokenizer_config())
    return RawTextCodecConfig().init(tokenizer)


@dataclass(frozen=True)
class NoSpeculatorConfig(SpeculatorConfig):
    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "NoSpeculator":
        return NoSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
        )


class NoSpeculator(Speculator[NoSpeculatorState, NoSpeculatorConfig]):
    requires_activation_trace: ClassVar[bool] = False

    @classmethod
    def build(cls, sharding_config: ShardingConfig) -> "Speculator":
        token_codec = _dummy_token_codec()
        return cls(
            config=NoSpeculatorConfig(token_codec_config=token_codec.config),
            sharding_config=sharding_config,
            token_codec=token_codec,
        )

    def init_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> NoSpeculatorState:
        _ = (batch_size, context_capacity, dtype)
        return NoSpeculatorState()

    def draft(
        self,
        state: NoSpeculatorState,
        last_token_ids: Int[Array, " batch"],
        last_token_indices: Int[Array, " batch"],
        target_embedding: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> ChainProposal:
        _ = (state, keychain, target_embedding)
        return ChainProposal(
            token_ids=last_token_ids[:, None],
            token_positions=last_token_indices[:, None] + 1,
            lengths=jnp.ones_like(last_token_indices),
        )
