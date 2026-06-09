from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import equinox as eqx
import jax.numpy as jnp

from lalamo.module import Keychain
from lalamo.utils.sharding import ShardingConfig

if TYPE_CHECKING:
    from jaxtyping import Array, Bool, DTypeLike, Float, Int

    from lalamo.models.language_model import PrefillResults
    from lalamo.modules.decoder import DecoderActivationTrace
    from lalamo.modules.embedding import EmbeddingBase

__all__ = [
    "ChainProposal",
    "ChainVerification",
    "Proposal",
    "Speculator",
    "import_speculator",
]


class ChainVerification(NamedTuple):
    accepted_mask: Bool[Array, "batch proposal_tokens"]
    num_accepted_tokens: Int[Array, " batch"]


class Proposal(Protocol):
    def verify(
        self,
        target_token_ids: Int[Array, "batch proposal_tokens"],
    ) -> ChainVerification: ...


@dataclass(frozen=True)
class ChainProposal(Proposal):
    token_ids: Int[Array, "batch proposal_tokens"]
    logits: Float[Array, "batch proposal_tokens vocabulary"]

    def verify(
        self,
        target_token_ids: Int[Array, "batch proposal_tokens"],
    ) -> ChainVerification:
        token_matches = self.token_ids == target_token_ids
        accepted_mask = jnp.cumprod(token_matches, axis=1).astype(jnp.bool_)
        return ChainVerification(
            accepted_mask=accepted_mask,
            num_accepted_tokens=jnp.sum(accepted_mask, axis=1),
        )


class Speculator[SpeculatorStateT: eqx.Module](eqx.Module, ABC):
    @property
    def requires_activation_trace(self) -> bool:
        return True

    @abstractmethod
    def init_state(
        self,
        prefill_results: PrefillResults,
        activation_trace: DecoderActivationTrace,
        context_capacity: int,
        *,
        keychain: Keychain,
    ) -> SpeculatorStateT: ...

    @abstractmethod
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
