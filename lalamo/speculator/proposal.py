from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.modules.common import ForwardPassMode
from lalamo.sampling import SamplingPolicy

__all__ = ["AcceptedProposal", "ProposalInputs", "TrieProposal"]


class AcceptedProposal(eqx.Module):
    accepted_token_ids: Int[Array, "batch max_slots"]
    node_indices: Int[Array, "batch max_slots"]
    compact_indices: Int[Array, "batch max_slots"]
    num_compact_indices: Int[Array, " batch"]
    terminal_node_indices: Int[Array, " batch"]
    bonus_token_ids: Int[Array, " batch"]

    def accepted_token_logits(
        self,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        root_sample_logits: Float[Array, "batch vocabulary"],
    ) -> Float[Array, "batch max_slots vocabulary"]:
        batch_size, _ = self.compact_indices.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        parent_logit_indices = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=jnp.int32), self.compact_indices[:, :-1]],
            axis=1,
        )
        token_logits = processed_tree_logits[batch_indices, parent_logit_indices]
        return token_logits.at[:, 0].set(root_sample_logits)

    def truncate(
        self,
        current_output_lengths: Int[Array, " batch"],
        max_output_length: int,
        done: Bool[Array, " batch"],
        eos_token_ids: Int[Array, " eos_tokens"],
    ) -> tuple["AcceptedProposal", Bool[Array, "batch max_slots"]]:
        slots = jnp.arange(self.accepted_token_ids.shape[1], dtype=jnp.int32)[None, :]
        valid = jnp.logical_and(
            slots < self.num_compact_indices[:, None],
            slots < (max_output_length - current_output_lengths)[:, None],
        )
        valid = jnp.logical_and(valid, jnp.logical_not(done)[:, None])
        if eos_token_ids.shape[0] > 0:
            eos_hits = jnp.logical_and(
                valid,
                jnp.any(self.accepted_token_ids[:, :, None] == eos_token_ids[None, None, :], axis=-1),
            )
        else:
            eos_hits = jnp.zeros_like(valid)
        prior_eos_count = jnp.cumsum(eos_hits.astype(jnp.int32), axis=1) - eos_hits.astype(jnp.int32)
        mask = jnp.logical_and(valid, prior_eos_count == 0)
        num_compact_indices = jnp.sum(mask, axis=1).astype(jnp.int32)
        return eqx.tree_at(lambda proposal: proposal.num_compact_indices, self, num_compact_indices), mask


class ProposalInputs(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    token_positions: Int[Array, "batch nodes"]
    lengths_without_padding: Int[Array, " batch"]
    forward_pass_mode: ForwardPassMode = eqx.field(static=True)
    attention_parent_indices: Int[Array, "batch nodes"] | None = None


class TrieProposal(eqx.Module):
    token_ids: Int[Array, "batch nodes"]
    parent_indices: Int[Array, "batch nodes"]
    depths: Int[Array, "batch nodes"]
    sample_positions: Int[Array, "batch nodes"]
    node_mask: Bool[Array, "batch nodes"]
    num_nodes: int = eqx.field(static=True)

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
        sample_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        return TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            sample_positions=sample_positions.at[:, 0].set(root_sample_positions),
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
                sample_positions=self.sample_positions.at[batch_indices, node_index].set(
                    self.sample_positions[batch_indices, parent_indices] + 1,
                ),
                node_mask=self.node_mask.at[batch_indices, node_index].set(True),
                num_nodes=node_index + 1,
            ),
            node_index,
        )

    def sample(
        self,
        logits: Float[Array, "batch nodes vocabulary"],
        sampling_policy: SamplingPolicy,
        output_lengths: Int[Array, " batch"],
        per_position_keys: Key[Array, "batch positions"],
    ) -> tuple[Float[Array, "batch nodes vocabulary"], Int[Array, "batch nodes"]]:
        processed_logits = jax.vmap(
            lambda policy, row_logits, token_ids, parent_indices, node_mask: policy.process_tree_logits(
                row_logits.astype(jnp.float32),
                token_ids,
                parent_indices,
                node_mask,
            ),
        )(
            sampling_policy,
            logits,
            self.token_ids,
            self.parent_indices,
            self.node_mask,
        )
        sample_positions = output_lengths[:, None] + self.depths + 1
        batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)[:, None]
        safe_positions = jnp.clip(sample_positions, 0, per_position_keys.shape[1] - 1)
        sample_keys = per_position_keys[batch_indices, safe_positions]
        token_ids = jax.vmap(
            lambda row_keys, row_logits: jax.vmap(
                lambda key, row_logit: jax.random.categorical(key, row_logit),
            )(row_keys, row_logits),
        )(sample_keys, processed_logits).astype(jnp.int32)
        return processed_logits, jnp.where(self.node_mask, token_ids, -1)

    def forward_inputs(
        self,
        next_token_positions: Int[Array, " batch"],
    ) -> ProposalInputs:
        token_positions = next_token_positions[:, None] + self.depths
        forward_pass_mode = ForwardPassMode.MULTI_TOKEN
        if self.batch_size == 1 and self.num_nodes == 1:
            forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
        attention_parent_indices = None
        if self.num_nodes > 1:
            attention_parent_indices = jnp.where(self.node_mask, self.parent_indices, -1)
        return ProposalInputs(
            token_ids=self.token_ids,
            token_positions=token_positions,
            lengths_without_padding=jnp.full((self.batch_size,), self.num_nodes, dtype=jnp.int32),
            forward_pass_mode=forward_pass_mode,
            attention_parent_indices=attention_parent_indices,
        )

    def verify(self, sampled_token_ids: Int[Array, "batch nodes"]) -> AcceptedProposal:
        batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)
        if self.num_nodes == 1:
            zeros = jnp.zeros((self.batch_size, 1), dtype=jnp.int32)
            return AcceptedProposal(
                accepted_token_ids=self.token_ids[:, :1],
                node_indices=zeros,
                compact_indices=zeros,
                num_compact_indices=jnp.ones((self.batch_size,), dtype=jnp.int32),
                terminal_node_indices=jnp.zeros((self.batch_size,), dtype=jnp.int32),
                bonus_token_ids=sampled_token_ids[:, 0],
            )

        candidate_node_indices = jnp.arange(self.budget, dtype=jnp.int32)[None, :]

        def scan_step(
            carry: tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            _: None,
        ) -> tuple[
            tuple[Int[Array, " batch"], Bool[Array, " batch"]],
            tuple[Int[Array, " batch"], Bool[Array, " batch"]],
        ]:
            terminal_node_indices, alive = carry
            sampled_at_terminal = sampled_token_ids[batch_indices, terminal_node_indices]
            child_mask = jnp.logical_and(
                self.node_mask,
                jnp.logical_and(
                    candidate_node_indices > 0,
                    jnp.logical_and(
                        self.parent_indices == terminal_node_indices[:, None],
                        self.token_ids == sampled_at_terminal[:, None],
                    ),
                ),
            )
            accepted = jnp.logical_and(alive, jnp.any(child_mask, axis=1))
            child_indices = jnp.argmax(child_mask, axis=1).astype(jnp.int32)
            next_terminal_node_indices = jnp.where(accepted, child_indices, terminal_node_indices)
            return (next_terminal_node_indices, accepted), (child_indices, accepted)

        (terminal_node_indices, _), (path_node_indices, path_mask) = jax.lax.scan(
            scan_step,
            (jnp.zeros((self.batch_size,), dtype=jnp.int32), jnp.ones((self.batch_size,), dtype=jnp.bool)),
            xs=None,
            length=self.budget - 1,
        )

        path_node_indices = path_node_indices.T
        path_mask = path_mask.T
        node_indices = jnp.concatenate(
            [
                jnp.where(path_mask, path_node_indices, 0),
                jnp.zeros((self.batch_size, 1), dtype=jnp.int32),
            ],
            axis=1,
        )
        compact_indices = jnp.concatenate(
            [jnp.zeros((self.batch_size, 1), dtype=jnp.int32), node_indices[:, :-1]],
            axis=1,
        )
        num_compact_indices = jnp.sum(path_mask, axis=1).astype(jnp.int32) + 1
        slots = jnp.arange(self.budget, dtype=jnp.int32)[None, :]
        accepted_token_ids = jnp.where(
            slots < num_compact_indices[:, None],
            jnp.take_along_axis(self.token_ids, compact_indices, axis=1),
            -1,
        )
        bonus_token_ids = sampled_token_ids[batch_indices, terminal_node_indices]

        return AcceptedProposal(
            accepted_token_ids=accepted_token_ids,
            node_indices=node_indices,
            compact_indices=compact_indices,
            num_compact_indices=num_compact_indices,
            terminal_node_indices=terminal_node_indices,
            bonus_token_ids=bonus_token_ids,
        )
