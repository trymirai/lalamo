from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
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
    token_ids: Int[Array, "batch proposal_tokens"]
    indices: Int[Array, "batch proposal_tokens"]
    mask: Bool[Array, "batch proposal_tokens"]

    def append_to(
        self,
        output_token_ids: Int[Array, "batch response_tokens"],
        output_lengths: Int[Array, " batch"],
    ) -> Int[Array, "batch response_tokens"]:
        batch_size, num_proposal_tokens = self.token_ids.shape
        proposal_slots = jnp.arange(num_proposal_tokens, dtype=output_lengths.dtype)[None, :]
        output_positions = output_lengths[:, None] + proposal_slots
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        return output_token_ids.at[batch_indices, output_positions].set(
            jnp.where(self.mask, self.token_ids, 0),
            mode="drop",
        )

    def append_values(
        self,
        output_values: Array,
        output_lengths: Int[Array, " batch"],
        values: Array,
    ) -> Array:
        batch_size, num_proposal_tokens = self.token_ids.shape
        proposal_slots = jnp.arange(num_proposal_tokens, dtype=output_lengths.dtype)[None, :]
        output_positions = output_lengths[:, None] + proposal_slots
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        mask_shape = (*self.mask.shape, *((1,) * (values.ndim - self.mask.ndim)))
        return output_values.at[batch_indices, output_positions].set(
            jnp.where(jnp.reshape(self.mask, mask_shape), values, 0),
            mode="drop",
        )

    def iter_accepted_token_ids(self) -> Iterable[Int[Array, ""]]:
        for token_id, is_accepted in zip(self.token_ids[0], self.mask[0], strict=True):
            if bool(is_accepted.item()):
                yield token_id

    def num_tokens(self) -> Int[Array, " batch"]:
        return jnp.sum(self.mask.astype(jnp.int32), axis=1)

    def last_indices(self) -> Int[Array, " batch"]:
        return jnp.maximum(self.num_tokens() - 1, 0)

    def has_eos(self, eos_token_ids: Int[Array, " eos_tokens"]) -> Bool[Array, " batch"]:
        eos_hits = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        return jnp.any(self.mask & eos_hits, axis=1)


@runtime_checkable
class Proposal(Protocol):
    def forward_inputs(
        self,
        last_token_indices: Int[Array, " batch"],
    ) -> ProposalInputs: ...

    def verify(
        self,
        sampled_token_ids: Int[Array, "batch proposal_tokens"],
    ) -> Int[Array, "batch max_slots"]: ...

    def accept(
        self,
        sampled_token_ids: Int[Array, "batch proposal_tokens"],
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal: ...


@dataclass(frozen=True)
class ChainProposal(Proposal):
    token_ids: Int[Array, "batch proposal_tokens"]

    def forward_inputs(
        self,
        last_token_indices: Int[Array, " batch"],
    ) -> ProposalInputs:
        batch_size, num_proposal_tokens = self.token_ids.shape
        token_offsets = jnp.arange(num_proposal_tokens, dtype=last_token_indices.dtype)[None, :] + 1
        forward_pass_mode = ForwardPassMode.MULTI_TOKEN
        if num_proposal_tokens == 1:
            forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=last_token_indices[:, None] + token_offsets,
            lengths_without_padding=jnp.full(
                (batch_size,),
                num_proposal_tokens,
                dtype=last_token_indices.dtype,
            ),
            forward_pass_mode=forward_pass_mode,
        )

    def verify(
        self,
        sampled_token_ids: Int[Array, "batch proposal_tokens"],
    ) -> Int[Array, "batch max_slots"]:
        accepted_mask = jnp.cumprod(self.token_ids == sampled_token_ids, axis=1).astype(jnp.bool_)
        _, num_proposal_tokens = self.token_ids.shape
        proposal_indices = jnp.broadcast_to(
            jnp.arange(num_proposal_tokens, dtype=jnp.int32)[None, :],
            self.token_ids.shape,
        )
        return jnp.where(accepted_mask, proposal_indices, -1)

    def accept(
        self,
        sampled_token_ids: Int[Array, "batch proposal_tokens"],
        remaining_lengths: Int[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
        active_mask: Bool[Array, " batch"] | None = None,
    ) -> AcceptedProposal:
        accepted_indices = self.verify(sampled_token_ids)
        accepted_mask = accepted_indices >= 0
        _, num_proposal_tokens = self.token_ids.shape
        proposal_slots = jnp.arange(num_proposal_tokens, dtype=remaining_lengths.dtype)[None, :]
        output_capacity_mask = proposal_slots < remaining_lengths[:, None]
        eos_hits = jnp.any(self.token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1)
        accepted_eos_hits = accepted_mask & eos_hits
        accepted_eos_counts = jnp.cumsum(accepted_eos_hits.astype(jnp.int32), axis=1)
        previous_eos_hits = (accepted_eos_counts - accepted_eos_hits.astype(jnp.int32)) > 0
        commit_mask = accepted_mask & output_capacity_mask & jnp.logical_not(previous_eos_hits)
        if active_mask is not None:
            commit_mask = commit_mask & active_mask[:, None]
        return AcceptedProposal(
            token_ids=self.token_ids,
            indices=jnp.where(commit_mask, accepted_indices, -1),
            mask=commit_mask,
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
        _ = (state, last_token_indices, keychain, target_embedding, target_lm_head)
        return ChainProposal(token_ids=last_token_ids[:, None])


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
