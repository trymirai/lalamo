# ruff: noqa: TC002, ANN401
from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from lalamo.module import Keychain
from lalamo.modules import Decoder
from lalamo.speculator.common import SpeculatorBackend, write_speculator_artifact
from lalamo.speculator.proposal import TrieProposal
from lalamo.speculator.state import LMState

from .dflash import (
    DEFAULT_CONTEXT_CAPACITY,
    DFlashDraftModel,
    DFlashLMState,
    DFlashSpeculator,
    draft_block_from_cache_batched,
    load_from_hf,
)

__all__ = [
    "DDTreeBackend",
    "DDTreeConfig",
    "DDTreeSpeculator",
    "build_ddtree_proposal",
    "ddtree_nodes_from_logits",
    "write_ddtree_artifact",
]


@dataclass(frozen=True)
class DDTreeConfig:
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY
    tree_budget: int = 128
    keychain_seed: int = 0


@partial(
    jtu.register_dataclass,
    data_fields=["model", "embedding", "noise_template"],
    meta_fields=["context_capacity", "tree_budget", "keychain_seed"],
)
@dataclass(frozen=True, kw_only=True)
class DDTreeSpeculator(DFlashSpeculator):
    @classmethod
    def create(
        cls,
        model: DFlashDraftModel,
        target_model: Decoder,
        context_capacity: int = DEFAULT_CONTEXT_CAPACITY,
        tree_budget: int | None = None,
        keychain_seed: int = 0,
    ) -> Self:
        resolved_tree_budget = 128 if tree_budget is None else int(tree_budget)
        if resolved_tree_budget < 1:
            raise ValueError("tree_budget must be positive")
        base = DFlashSpeculator.create(
            model,
            target_model,
            context_capacity=context_capacity,
            tree_budget=resolved_tree_budget,
            keychain_seed=keychain_seed,
        )
        return cls(
            model=base.model,
            embedding=base.embedding,
            noise_template=base.noise_template,
            context_capacity=base.context_capacity,
            tree_budget=resolved_tree_budget,
            keychain_seed=base.keychain_seed,
        )

    @property
    def max_step_tokens(self) -> int:
        return min(self.tree_budget, max(self.model.config.block_size - 1, 0)) + 1

    def draft(self, state: LMState) -> TrieProposal:
        if not isinstance(state, DFlashLMState):
            raise TypeError(f"DDTree requires DFlashLMState, got {type(state).__name__}")
        batch_size = state.root_bonus_id.shape[0]
        draft = draft_block_from_cache_batched(
            self.model,
            self.embedding,
            self.noise_template,
            state.draft_kv_state,
            state.root_bonus_id,
            keychain=Keychain.init(self.keychain_seed, (batch_size,)),
        )
        return build_ddtree_proposal(state, draft.draft_logits.astype(jnp.float32), self.tree_budget)


def build_ddtree_proposal(
    state: DFlashLMState,
    draft_logits: Float[Array, "batch depth vocabulary"],
    node_budget: int,
) -> TrieProposal:
    node_budget = max(int(node_budget), 1)
    proposal = state.create_root_proposal(budget=node_budget + 1)
    token_ids, parent_indices, depths, node_mask = ddtree_nodes_from_logits(
        draft_logits,
        node_budget,
    )
    return proposal.add_nodes(
        token_ids,
        parent_indices,
        depths,
        node_mask,
        min(node_budget, draft_logits.shape[1]),
    )


def ddtree_nodes_from_logits(
    logits: Float[Array, "batch depth vocabulary"],
    node_budget: int,
) -> tuple[
    Int[Array, "batch budget"],
    Int[Array, "batch budget"],
    Int[Array, "batch budget"],
    Bool[Array, "batch budget"],
]:
    if node_budget < 1:
        raise ValueError("node_budget must be positive")
    batch_size, _depth, vocabulary_size = logits.shape
    top_k = min(node_budget, vocabulary_size)
    top_log_probs, top_token_ids = jax.lax.top_k(jax.nn.log_softmax(logits, axis=-1), top_k)
    output_shape = jax.ShapeDtypeStruct((batch_size, node_budget), jnp.int32)
    mask_shape = jax.ShapeDtypeStruct((batch_size, node_budget), jnp.bool)
    return jax.pure_callback(
        lambda token_ids, log_probs: ddtree_nodes_from_topk_numpy(token_ids, log_probs, node_budget),
        (output_shape, output_shape, output_shape, mask_shape),
        top_token_ids,
        top_log_probs,
        vmap_method="sequential",
    )


def ddtree_nodes_from_topk_numpy(
    top_token_ids: Any,
    top_log_probs: Any,
    node_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_token_ids = np.asarray(top_token_ids)
    top_log_probs = np.asarray(top_log_probs)
    batch_size, _depth, _top_k = top_token_ids.shape
    token_ids = np.zeros((batch_size, node_budget), dtype=np.int32)
    parent_indices = np.zeros((batch_size, node_budget), dtype=np.int32)
    node_depths = np.zeros((batch_size, node_budget), dtype=np.int32)
    node_mask = np.zeros((batch_size, node_budget), dtype=np.bool_)
    for batch_index in range(batch_size):
        row_token_ids, row_parent_indices, row_depths, row_mask = ddtree_row_from_topk_numpy(
            top_token_ids[batch_index],
            top_log_probs[batch_index],
            node_budget,
        )
        token_ids[batch_index] = row_token_ids
        parent_indices[batch_index] = row_parent_indices
        node_depths[batch_index] = row_depths
        node_mask[batch_index] = row_mask
    return token_ids, parent_indices, node_depths, node_mask


def ddtree_row_from_topk_numpy(
    top_token_ids: np.ndarray,
    top_log_probs: np.ndarray,
    node_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth, top_k = top_token_ids.shape
    token_ids = np.zeros((node_budget,), dtype=np.int32)
    parent_indices = np.zeros((node_budget,), dtype=np.int32)
    node_depths = np.zeros((node_budget,), dtype=np.int32)
    node_mask = np.zeros((node_budget,), dtype=np.bool_)
    if depth == 0 or top_k == 0:
        return token_ids, parent_indices, node_depths, node_mask

    heap: list[tuple[float, int, int, int]] = [(-float(top_log_probs[0, 0]), 1, 0, 0)]
    node_count = 0
    while heap and node_count < node_budget:
        negative_score, node_depth, parent_index, rank = heapq.heappop(heap)
        score = -negative_score
        depth_index = node_depth - 1
        node_index = node_count + 1
        token_ids[node_count] = np.int32(top_token_ids[depth_index, rank])
        parent_indices[node_count] = np.int32(parent_index)
        node_depths[node_count] = np.int32(node_depth)
        node_mask[node_count] = True
        node_count += 1

        sibling_rank = rank + 1
        if sibling_rank < top_k:
            sibling_score = score - float(top_log_probs[depth_index, rank]) + float(
                top_log_probs[depth_index, sibling_rank],
            )
            heapq.heappush(heap, (-sibling_score, node_depth, parent_index, sibling_rank))

        child_depth = node_depth + 1
        if child_depth <= depth:
            child_score = score + float(top_log_probs[child_depth - 1, 0])
            heapq.heappush(heap, (-child_score, child_depth, node_index, 0))

    return token_ids, parent_indices, node_depths, node_mask


class DDTreeBackend(SpeculatorBackend[DDTreeConfig]):
    name: ClassVar[str] = "ddtree"
    config_type: ClassVar[type[Any]] = DDTreeConfig

    @classmethod
    def create_trainer(
        cls,
        config: DDTreeConfig,
        artifact_path: Path,
        target_model: Decoder,
    ) -> None:
        del config, artifact_path, target_model
        raise RuntimeError("DDTree uses pretrained HuggingFace DFlash checkpoints.")

    @classmethod
    def deserialize(
        cls,
        fields: tuple[Any, ...],
        target_model: Decoder,
    ) -> DDTreeSpeculator:
        if len(fields) not in (1, 2, 3, 4):
            raise ValueError(
                "ddtree artifact must contain repo_or_path, optional context_capacity, "
                "optional tree_budget, and optional keychain_seed",
            )
        repo_or_path = fields[0]
        if isinstance(repo_or_path, bytes):
            repo_or_path = repo_or_path.decode()
        if not isinstance(repo_or_path, str):
            raise TypeError("ddtree repo_or_path must be a string")

        context_capacity = DEFAULT_CONTEXT_CAPACITY
        tree_budget = 128
        keychain_seed = 0
        if len(fields) >= 2:
            context_capacity = int(fields[1])
        if len(fields) >= 3:
            tree_budget = 128 if fields[2] is None else int(fields[2])
        if len(fields) >= 4:
            keychain_seed = int(fields[3])

        _, model = load_from_hf(repo_or_path)
        return DDTreeSpeculator.create(
            model,
            target_model,
            context_capacity=context_capacity,
            tree_budget=tree_budget,
            keychain_seed=keychain_seed,
        )


def write_ddtree_artifact(
    path: Path | str,
    repo_or_path: str | Path,
    context_capacity: int = DEFAULT_CONTEXT_CAPACITY,
    tree_budget: int = 128,
    keychain_seed: int = 0,
) -> None:
    write_speculator_artifact(
        path,
        DDTreeBackend,
        str(repo_or_path),
        int(context_capacity),
        int(tree_budget),
        int(keychain_seed),
    )


BACKEND = DDTreeBackend
