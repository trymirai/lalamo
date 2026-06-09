from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp

from lalamo.module import ForwardPassMode, Keychain
from lalamo.utils.sharding import ShardingConfig

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, DTypeLike, Int

    from lalamo.modules.decoder import DecoderResult
    from lalamo.modules.embedding import EmbeddingBase

__all__ = [
    "AcceptedProposal",
    "ChainProposal",
    "CommittedTokens",
    "NoSpeculator",
    "Proposal",
    "ProposalInputs",
    "Speculator",
    "SpeculatorState",
    "import_speculator",
]

type SpeculatorState = eqx.Module | None


class ProposalInputs(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    lengths_without_padding: Int[Array, " batch"]
    forward_pass_mode: ForwardPassMode = eqx.field(static=True)
    attention_parent_indices: Int[Array, "batch nodes"] | None = None


class AcceptedProposal(eqx.Module):
    token_ids: Int[Array, "batch accepted_slots"]
    token_positions: Int[Array, "batch accepted_slots"]
    indices: Int[Array, "batch accepted_slots"]
    lengths: Int[Array, " batch"]
    updates_speculator: Bool[Array, ""]

    def num_tokens(self) -> Int[Array, " batch"]:
        return self.lengths


class CommittedTokens(eqx.Module):
    token_ids: Int[Array, "batch commit_slots"]
    source_indices: Int[Array, "batch commit_slots"]
    lengths: Int[Array, " batch"]

    def has_eos(self, eos_token_ids: Int[Array, " eos_tokens"]) -> Bool[Array, " batch"]:
        slots = jnp.arange(self.token_ids.shape[1], dtype=self.lengths.dtype)[None, :]
        valid = slots < self.lengths[:, None]
        eos_hits = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        return jnp.any(valid & eos_hits, axis=1)


@runtime_checkable
class Proposal(Protocol):
    def forward_inputs(
        self,
    ) -> ProposalInputs: ...

    def accept(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal: ...

    def committed_tokens(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        accepted: AcceptedProposal,
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> CommittedTokens: ...


class ChainProposal(eqx.Module):
    token_ids: Int[Array, "batch proposal_slots"]
    token_positions: Int[Array, "batch proposal_slots"]
    lengths: Int[Array, " batch"]
    updates_speculator: Bool[Array, ""]

    def forward_inputs(
        self,
    ) -> ProposalInputs:
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
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal:
        _, num_proposal_slots = self.token_ids.shape
        proposal_slots = jnp.arange(num_proposal_slots, dtype=self.lengths.dtype)[None, :]
        draft_slots = jnp.arange(num_proposal_slots - 1, dtype=self.lengths.dtype)[None, :]
        draft_valid = draft_slots < jnp.maximum(self.lengths[:, None] - 1, 0)
        draft_matches = self.token_ids[:, 1:] == sampled_token_ids[:, :-1]
        accepted_draft_mask = jnp.cumprod((draft_matches & draft_valid).astype(jnp.int32), axis=1).astype(jnp.bool_)
        num_accepted_drafts = jnp.sum(accepted_draft_mask.astype(self.lengths.dtype), axis=1)
        accepted_lengths = jnp.where(self.lengths > 0, num_accepted_drafts + 1, 0)
        if active_mask is not None:
            accepted_lengths = jnp.where(active_mask, accepted_lengths, 0)
        accepted_mask = proposal_slots < accepted_lengths[:, None]
        return AcceptedProposal(
            token_ids=jnp.where(accepted_mask, self.token_ids, 0),
            token_positions=jnp.where(accepted_mask, self.token_positions, 0),
            indices=jnp.where(
                accepted_mask,
                jnp.broadcast_to(proposal_slots.astype(jnp.int32), self.token_ids.shape),
                -1,
            ),
            lengths=accepted_lengths,
            updates_speculator=self.updates_speculator,
        )

    def committed_tokens(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        accepted: AcceptedProposal,
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> CommittedTokens:
        batch_size, num_proposal_slots = self.token_ids.shape
        commit_slots = jnp.arange(num_proposal_slots, dtype=self.lengths.dtype)[None, :]
        num_committed_drafts = jnp.maximum(accepted.lengths - 1, 0)
        draft_output_mask = commit_slots < num_committed_drafts[:, None]
        draft_token_ids = jnp.pad(self.token_ids[:, 1:], ((0, 0), (0, 1)))
        draft_eos_hits = draft_output_mask & jnp.any(draft_token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        draft_eos_counts = jnp.cumsum(draft_eos_hits.astype(jnp.int32), axis=1)
        after_draft_eos = (draft_eos_counts - draft_eos_hits.astype(jnp.int32)) > 0
        draft_output_mask = draft_output_mask & jnp.logical_not(after_draft_eos)
        has_draft_eos = jnp.any(draft_eos_hits, axis=1)

        bonus_source_indices = jnp.minimum(num_committed_drafts, num_proposal_slots - 1).astype(jnp.int32)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
        bonus_token_ids = sampled_token_ids[batch_indices, bonus_source_indices]
        bonus_slots = num_committed_drafts[:, None]
        bonus_mask = commit_slots == bonus_slots
        bonus_mask = bonus_mask & jnp.logical_not(has_draft_eos[:, None])

        token_ids = jnp.where(draft_output_mask, draft_token_ids, 0)
        token_ids = jnp.where(bonus_mask, bonus_token_ids[:, None], token_ids)
        source_indices = jnp.where(draft_output_mask, commit_slots.astype(jnp.int32), -1)
        source_indices = jnp.where(bonus_mask, bonus_source_indices[:, None], source_indices)

        eos_hits = jnp.any(token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        valid_without_capacity = draft_output_mask | bonus_mask
        eos_counts = jnp.cumsum((valid_without_capacity & eos_hits).astype(jnp.int32), axis=1)
        before_or_at_first_eos = (eos_counts - (valid_without_capacity & eos_hits).astype(jnp.int32)) == 0
        valid_without_capacity = valid_without_capacity & before_or_at_first_eos
        capacity_mask = commit_slots < remaining_lengths[:, None]
        valid = valid_without_capacity & capacity_mask
        if active_mask is not None:
            valid = valid & active_mask[:, None]
        return CommittedTokens(
            token_ids=jnp.where(valid, token_ids, 0),
            source_indices=jnp.where(valid, source_indices, -1),
            lengths=jnp.sum(valid.astype(self.lengths.dtype), axis=1),
        )


class Speculator[SpeculatorStateT: SpeculatorState](eqx.Module, ABC):
    @property
    def requires_activation_trace(self) -> bool:
        return True

    @property
    def max_proposal_tokens(self) -> int:
        return 1

    def prefill_chunk(
        self,
        state: SpeculatorStateT | None,
        decoder_result: DecoderResult,
        chunk_lengths: Int[Array, " batch"],
        context_capacity: int,
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT | None:
        pass

    def update_state(
        self,
        state: SpeculatorStateT,
        decoder_result: DecoderResult,
        accepted: AcceptedProposal,
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT | None:
        pass

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


class NoSpeculator(Speculator[None]):
    @property
    def requires_activation_trace(self) -> bool:
        return False

    def draft(
        self,
        state: None,
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
            updates_speculator=jnp.asarray(True),
        )


def import_speculator(
    speculator_dir: Path | str,
    *,
    sharding_config: ShardingConfig,
    dtype: DTypeLike | None = None,
) -> Speculator:
    from .dflash import DFlashSpeculator  # noqa: PLC0415

    speculator_dir = Path(speculator_dir)
    if not speculator_dir.exists():
        raise FileNotFoundError(f"Speculator directory does not exist: {speculator_dir}")
    if not speculator_dir.is_dir():
        raise ValueError(f"Speculator path is not a directory: {speculator_dir}")
    return DFlashSpeculator.load(
        speculator_dir,
        sharding_config=sharding_config,
        dtype=dtype,
    )
