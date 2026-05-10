from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.token_mixers.state.common import State
from lalamo.speculator.sampler import GumbelSampler

__all__ = ["AcceptedProposal", "ProposalNode", "TrieProposal"]


@dataclass(frozen=True)
class AcceptedProposal:
    token_ids: tuple[int, ...]
    node_indices: tuple[int, ...]
    compact_indices: Int[Array, " max_slots"]
    num_compact_indices: int
    terminal_node_index: int
    bonus_token_id: int


@dataclass(frozen=True)
class ProposalNode:
    token_id: int
    parent_index: int
    gumbel_position: int
    depth: int


@dataclass(frozen=True)
class TrieProposal:
    nodes: tuple[ProposalNode, ...]

    @staticmethod
    def create(root: int, root_sample_position: int) -> TrieProposal:
        return TrieProposal((ProposalNode(root, -1, root_sample_position, 0),))

    def add_node(self, parent_index: int, token_id: int) -> tuple[TrieProposal, int]:
        node_index = len(self.nodes)
        parent = self.nodes[parent_index]
        node = ProposalNode(token_id, parent_index, parent.gumbel_position + 1, parent.depth + 1)
        return TrieProposal((*self.nodes, node)), node_index

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(node.token_id for node in self.nodes)

    @property
    def parent_indices(self) -> tuple[int, ...]:
        return tuple(node.parent_index for node in self.nodes)

    @property
    def gumbel_positions(self) -> tuple[int, ...]:
        return tuple(node.gumbel_position for node in self.nodes)

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(node.depth for node in self.nodes)

    def forward(
        self,
        decoder: Decoder,
        kv_cache: State,
        next_token_position: int,
        sampler: GumbelSampler,
        return_activation_trace: bool,
    ) -> tuple[DecoderResult, tuple[int, ...]]:
        token_ids = jnp.array([self.token_ids], dtype=jnp.int32)
        token_positions = jnp.array(
            [[next_token_position + depth for depth in self.depths]],
            dtype=jnp.int32,
        )
        parent_indices = jnp.array([self.parent_indices], dtype=jnp.int32)
        decoder_result = decoder(
            token_ids,
            token_positions,
            kv_cache,
            return_updated_state=True,
            return_activation_trace=return_activation_trace,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            attention_parent_indices=parent_indices,
        )
        return decoder_result, self.sample(decoder_result.logits[0], sampler)

    def sample(self, logits: Float[Array, "nodes vocab"], sampler: GumbelSampler) -> tuple[int, ...]:
        return tuple(
            sampler.sample_logits(logit, node.gumbel_position) for logit, node in zip(logits, self.nodes, strict=True)
        )

    def verify(self, sampled_token_ids: tuple[int, ...]) -> AcceptedProposal:
        child_index_by_parent_and_token = {
            (node.parent_index, node.token_id): node_index for node_index, node in enumerate(self.nodes)
        }
        path: list[int] = []
        node_index = 0

        for _ in self.nodes[1:]:
            sampled_token_id = sampled_token_ids[node_index]
            child_index = child_index_by_parent_and_token.get((node_index, sampled_token_id))
            if child_index is None:
                break

            path.append(child_index)
            node_index = child_index

        node_indices = tuple(path)
        token_ids = tuple(self.nodes[index].token_id for index in node_indices)
        accepted_indices = (0, *node_indices)
        compact_indices = jnp.array(
            accepted_indices + (0,) * (len(self.nodes) - len(accepted_indices)),
            dtype=jnp.int32,
        )
        return AcceptedProposal(
            token_ids=token_ids,
            node_indices=node_indices,
            compact_indices=compact_indices,
            num_compact_indices=len(accepted_indices),
            terminal_node_index=node_index,
            bonus_token_id=sampled_token_ids[node_index],
        )
