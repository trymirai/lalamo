from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import equinox as eqx
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
from lalamo.modules.token_mixer import State
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
        token_ids = self.token_ids[batch_indices, slots]
        return jnp.where(self.lengths > 0, token_ids, stopped_token_ids)

    def last_token_indices(self) -> Int[Array, " batch"]:
        batch_size, _ = self.token_positions.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        slots = jnp.maximum(self.lengths - 1, 0)
        token_positions = self.token_positions[batch_indices, slots]
        return jnp.where(self.lengths > 0, token_positions - 1, 0)

    def has_eos(self, eos_token_ids: Int[Array, " eos_tokens"]) -> Bool[Array, " batch"]:
        slots = jnp.arange(self.token_ids.shape[1], dtype=self.lengths.dtype)[None, :]
        valid = slots < self.lengths[:, None]
        eos_hits = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        return jnp.any(valid & eos_hits, axis=1)

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
    def accept(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal: ...

    @abstractmethod
    def rollback_state(
        self,
        state: State,
        committed_length: Int[Array, " batch"],
        accepted: AcceptedProposal,
    ) -> State: ...


class ChainProposal(Proposal):
    token_ids: Int[Array, "batch proposal_slots"]
    token_positions: Int[Array, "batch proposal_slots"]
    lengths: Int[Array, " batch"]

    def forward_inputs(self) -> ProposalInputs:
        _, num_proposal_slots = self.token_ids.shape
        forward_pass_mode = ForwardPassMode.MULTI_TOKEN
        if num_proposal_slots == 1:
            forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            lengths_without_padding=self.lengths,
            forward_pass_mode=forward_pass_mode,
        )

    def accept(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal:
        batch_size, num_proposal_slots = self.token_ids.shape
        proposal_slots = jnp.arange(num_proposal_slots, dtype=self.lengths.dtype)[None, :]
        draft_slots = jnp.arange(num_proposal_slots - 1, dtype=self.lengths.dtype)[None, :]
        draft_valid = draft_slots < jnp.maximum(self.lengths[:, None] - 1, 0)
        draft_matches = self.token_ids[:, 1:] == sampled_token_ids[:, :-1]
        accepted_draft_mask = jnp.cumprod((draft_matches & draft_valid).astype(jnp.int32), axis=1).astype(bool)
        num_accepted_drafts = jnp.sum(accepted_draft_mask.astype(self.lengths.dtype), axis=1)
        num_accepted_nodes = jnp.where(self.lengths > 0, num_accepted_drafts + 1, 0)
        if active_mask is not None:
            num_accepted_nodes = jnp.where(active_mask, num_accepted_nodes, 0)
        accepted_node_mask = proposal_slots < num_accepted_nodes[:, None]

        output_slots = proposal_slots
        draft_output_mask = output_slots < num_accepted_drafts[:, None]
        draft_token_ids = jnp.pad(self.token_ids[:, 1:], ((0, 0), (0, 1)))
        draft_token_positions = jnp.pad(self.token_positions[:, 1:], ((0, 0), (0, 1)))
        draft_eos_hits = draft_output_mask & jnp.any(
            draft_token_ids[:, :, None] == eos_token_ids[None, None, :],
            axis=-1,
        )
        draft_eos_counts = jnp.cumsum(draft_eos_hits.astype(jnp.int32), axis=1)
        after_draft_eos = (draft_eos_counts - draft_eos_hits.astype(jnp.int32)) > 0
        draft_output_mask = draft_output_mask & jnp.logical_not(after_draft_eos)
        has_draft_eos = jnp.any(draft_eos_hits, axis=1)

        bonus_source_indices = jnp.minimum(num_accepted_drafts, num_proposal_slots - 1).astype(jnp.int32)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        bonus_token_ids = sampled_token_ids[batch_indices, bonus_source_indices]
        bonus_token_positions = self.token_positions[batch_indices, bonus_source_indices] + 1
        bonus_mask = output_slots == num_accepted_drafts[:, None]
        bonus_mask = bonus_mask & jnp.logical_not(has_draft_eos[:, None])

        token_ids = jnp.where(draft_output_mask, draft_token_ids, 0)
        token_ids = jnp.where(bonus_mask, bonus_token_ids[:, None], token_ids)
        token_positions = jnp.where(draft_output_mask, draft_token_positions, 0)
        token_positions = jnp.where(bonus_mask, bonus_token_positions[:, None], token_positions)
        source_indices = jnp.where(draft_output_mask, output_slots.astype(jnp.int32), -1)
        source_indices = jnp.where(bonus_mask, bonus_source_indices[:, None], source_indices)

        valid_without_capacity = draft_output_mask | bonus_mask
        eos_hits = jnp.any(token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        eos_counts = jnp.cumsum((valid_without_capacity & eos_hits).astype(jnp.int32), axis=1)
        before_or_at_first_eos = (eos_counts - (valid_without_capacity & eos_hits).astype(jnp.int32)) == 0
        valid = valid_without_capacity & before_or_at_first_eos & (output_slots < remaining_lengths[:, None])
        if active_mask is not None:
            valid = valid & active_mask[:, None]

        return AcceptedProposal(
            token_ids=jnp.where(valid, token_ids, 0),
            token_positions=jnp.where(valid, token_positions, 0),
            source_indices=jnp.where(valid, source_indices, -1),
            lengths=jnp.sum(valid.astype(self.lengths.dtype), axis=1),
            accepted_node_indices=jnp.where(
                accepted_node_mask,
                jnp.broadcast_to(output_slots.astype(jnp.int32), self.token_ids.shape),
                -1,
            ),
            num_accepted_nodes=num_accepted_nodes,
        )

    def rollback_state(
        self,
        state: State,
        committed_length: Int[Array, " batch"],
        accepted: AcceptedProposal,
    ) -> State:
        return state.truncate(committed_length + accepted.num_accepted_nodes)


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
        target_lm_head: EmbeddingBase,
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
        target_lm_head: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> ChainProposal:
        _ = (state, keychain, target_embedding, target_lm_head)
        return ChainProposal(
            token_ids=last_token_ids[:, None],
            token_positions=last_token_indices[:, None] + 1,
            lengths=jnp.ones_like(last_token_indices),
        )
