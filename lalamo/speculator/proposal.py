from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderResult
from lalamo.modules.token_mixers.state.common import State
from lalamo.speculator.sampler import GumbelSampler

__all__ = ["AcceptedProposal", "TrieProposal"]


@dataclass(frozen=True)
class AcceptedProposal:
    token_ids: Int[Array, "batch max_slots"]
    node_indices: Int[Array, "batch max_slots"]
    compact_indices: Int[Array, "batch max_slots"]
    num_compact_indices: Int[Array, " batch"]
    terminal_node_indices: Int[Array, " batch"]
    bonus_token_ids: Int[Array, " batch"]


@dataclass(frozen=True)
class TrieProposal:
    token_ids: Int[Array, "batch nodes"]
    parent_indices: Int[Array, "batch nodes"]
    depths: Int[Array, "batch nodes"]
    gumbel_positions: Int[Array, "batch nodes"]
    node_mask: Bool[Array, "batch nodes"]
    num_nodes: int

    @staticmethod
    def create(
        root_ids: Int[Array, " batch"],
        root_sample_positions: Int[Array, " batch"],
        budget: int = 128,
    ) -> TrieProposal:
        (batch_size,) = root_ids.shape
        token_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        parent_indices = jnp.full((batch_size, budget), -1, dtype=jnp.int32)
        depths = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        gumbel_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        return TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            gumbel_positions=gumbel_positions.at[:, 0].set(root_sample_positions),
            node_mask=node_mask.at[:, 0].set(True),
            num_nodes=1,
        )

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def budget(self) -> int:
        return self.token_ids.shape[1]

    def add_nodes(
        self,
        batch_indices: Int[Array, " active"],
        parent_indices: Int[Array, " active"],
        token_ids: Int[Array, " active"],
    ) -> tuple[TrieProposal, int]:
        node_index = self.num_nodes
        return (
            TrieProposal(
                token_ids=self.token_ids.at[batch_indices, node_index].set(token_ids),
                parent_indices=self.parent_indices.at[batch_indices, node_index].set(parent_indices),
                depths=self.depths.at[batch_indices, node_index].set(
                    self.depths[batch_indices, parent_indices] + 1,
                ),
                gumbel_positions=self.gumbel_positions.at[batch_indices, node_index].set(
                    self.gumbel_positions[batch_indices, parent_indices] + 1,
                ),
                node_mask=self.node_mask.at[batch_indices, node_index].set(True),
                num_nodes=node_index + 1,
            ),
            node_index,
        )

    def forward(
        self,
        decoder: Decoder,
        kv_cache: State,
        next_token_positions: Int[Array, " batch"],
        sampler: GumbelSampler,
        return_activation_trace: bool,
    ) -> tuple[DecoderResult, Int[Array, "batch nodes"]]:
        token_positions = next_token_positions[:, None] + self.depths
        decoder_result = decoder(
            self.token_ids,
            token_positions,
            kv_cache,
            return_updated_state=True,
            return_activation_trace=return_activation_trace,
            lengths_without_padding=jnp.full((self.batch_size,), self.num_nodes, dtype=jnp.int32),
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            attention_parent_indices=jnp.where(self.node_mask, self.parent_indices, -1),
        )
        return decoder_result, self.sample(decoder_result.logits, sampler)

    def sample(self, logits: Float[Array, "batch nodes vocab"], sampler: GumbelSampler) -> Int[Array, "batch nodes"]:
        sampled_token_ids = sampler.sample_logits(
            logits.reshape((-1, logits.shape[-1])),
            self.gumbel_positions.reshape(-1),
        )
        return jnp.where(self.node_mask, sampled_token_ids.reshape(self.gumbel_positions.shape), 0)

    def verify(self, sampled_token_ids: Int[Array, "batch nodes"]) -> AcceptedProposal:
        token_ids = []
        node_indices = []
        compact_indices = []
        num_compact_indices = []
        terminal_node_indices = []
        bonus_token_ids = []

        for batch_index in range(self.batch_size):
            child_index_by_parent_and_token = {}
            for child_index in range(1, self.num_nodes):
                if bool(self.node_mask[batch_index, child_index]):
                    parent_index = int(self.parent_indices[batch_index, child_index])
                    token_id = int(self.token_ids[batch_index, child_index])
                    child_index_by_parent_and_token[(parent_index, token_id)] = child_index
            path: list[int] = []
            node_index = 0

            for _ in range(self.num_nodes - 1):
                sampled_token_id = int(sampled_token_ids[batch_index, node_index])
                child_index = child_index_by_parent_and_token.get((node_index, sampled_token_id))
                if child_index is None:
                    break

                path.append(child_index)
                node_index = child_index

            accepted_token_ids = [int(self.token_ids[batch_index, index]) for index in path]
            accepted_indices = [0, *path]
            token_ids.append(accepted_token_ids + [0] * (self.budget - len(accepted_token_ids)))
            node_indices.append(path + [0] * (self.budget - len(path)))
            compact_indices.append(accepted_indices + [0] * (self.budget - len(accepted_indices)))
            num_compact_indices.append(len(accepted_indices))
            terminal_node_indices.append(node_index)
            bonus_token_ids.append(int(sampled_token_ids[batch_index, node_index]))

        return AcceptedProposal(
            token_ids=jnp.array(token_ids, dtype=jnp.int32),
            node_indices=jnp.array(node_indices, dtype=jnp.int32),
            compact_indices=jnp.array(compact_indices, dtype=jnp.int32),
            num_compact_indices=jnp.array(num_compact_indices, dtype=jnp.int32),
            terminal_node_indices=jnp.array(terminal_node_indices, dtype=jnp.int32),
            bonus_token_ids=jnp.array(bonus_token_ids, dtype=jnp.int32),
        )
