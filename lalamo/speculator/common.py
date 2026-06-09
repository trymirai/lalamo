from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import equinox as eqx
import jax.numpy as jnp

from lalamo.module import ForwardPassMode, Keychain
from lalamo.utils.sharding import ShardingConfig

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Int

    from lalamo.models.language_model import PrefillResults
    from lalamo.modules.decoder import DecoderActivationTrace
    from lalamo.modules.embedding import EmbeddingBase

__all__ = [
    "ChainProposal",
    "NoSpeculator",
    "Proposal",
    "ProposalInputs",
    "Speculator",
    "import_speculator",
]


class ProposalInputs(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    lengths_without_padding: Int[Array, " batch"]
    forward_pass_mode: ForwardPassMode = eqx.field(static=True)
    attention_parent_indices: Int[Array, "batch nodes"] | None = None


class Proposal(Protocol):
    def forward_inputs(
        self,
        last_token_indices: Int[Array, " batch"],
    ) -> ProposalInputs: ...

    def verify(
        self,
        sampled_token_ids: Int[Array, "batch proposal_tokens"],
    ) -> Int[Array, "batch max_slots"]: ...


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
        _, num_proposal_tokens = self.token_ids.shape
        proposal_indices = jnp.broadcast_to(
            jnp.arange(num_proposal_tokens, dtype=jnp.int32)[None, :],
            self.token_ids.shape,
        )
        if sampled_token_ids.shape[1] == 1:
            return proposal_indices

        draft_matches = self.token_ids[:, 1:] == sampled_token_ids[:, :-1]
        draft_accepted_mask = jnp.cumprod(draft_matches, axis=1).astype(jnp.bool_)
        root_mask = jnp.ones_like(self.token_ids[:, :1], dtype=jnp.bool_)
        accepted_mask = jnp.concatenate((root_mask, draft_accepted_mask), axis=1)
        return jnp.where(accepted_mask, proposal_indices, -1)


class Speculator[SpeculatorStateT](eqx.Module, ABC):
    @property
    def requires_activation_trace(self) -> bool:
        return True

    def init_state(
        self,
        prefill_results: PrefillResults,
        activation_trace: DecoderActivationTrace,
        context_capacity: int,
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT: ...

    def update_state(
        self,
        state: SpeculatorStateT,
        activation_trace: DecoderActivationTrace,
        num_tokens_to_append: Int[Array, " batch"],
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT: ...

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
