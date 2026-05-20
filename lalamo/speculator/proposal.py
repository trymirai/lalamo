import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Key, Shaped

from lalamo.module import ForwardPassMode
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy

__all__ = [
    "AcceptedProposal",
    "ProposalInputs",
    "TrieProposal",
]


class AcceptedProposal(eqx.Module):
    accepted_token_ids: Int[Array, "batch max_slots"]
    node_indices: Int[Array, "batch max_slots"]
    compact_indices: Int[Array, "batch max_slots"]
    num_compact_indices: Int[Array, " batch"]
    terminal_node_indices: Int[Array, " batch"]
    bonus_token_ids: Int[Array, " batch"]
    next_sampling_policy: SamplingPolicy

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

    def compact(
        self,
        values: Shaped[Array, "batch nodes *features"],
    ) -> Shaped[Array, "batch max_slots *features"]:
        batch_size, max_slots = self.compact_indices.shape
        slots = jnp.arange(max_slots, dtype=jnp.int32)[None, :]
        valid = slots < self.num_compact_indices[:, None]
        node_indices = jnp.where(valid, self.compact_indices, 0)
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        compacted = values[batch_indices, node_indices]
        broadcast_valid = valid.reshape(valid.shape + (1,) * (values.ndim - 2))
        return jnp.where(broadcast_valid, compacted, 0)

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
    gumbel_positions: Int[Array, "batch nodes"]
    gumbel_node_ids: Int[Array, "batch nodes"]
    node_mask: Bool[Array, "batch nodes"]
    sampling_policy: SamplingPolicy
    gumbel_keys: Key[Array, " batch"]
    vocabulary_size: int = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)

    @staticmethod
    def create(
        root_ids: Int[Array, " batch"],
        root_gumbel_positions: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
        vocabulary_size: int,
        budget: int = 128,
    ) -> "TrieProposal":
        (batch_size,) = root_ids.shape
        token_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        parent_indices = jnp.full((batch_size, budget), -1, dtype=jnp.int32)
        depths = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        gumbel_positions = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        gumbel_node_ids = jnp.zeros((batch_size, budget), dtype=jnp.int32)
        node_mask = jnp.zeros((batch_size, budget), dtype=jnp.bool)
        return TrieProposal(
            token_ids=token_ids.at[:, 0].set(root_ids),
            parent_indices=parent_indices,
            depths=depths,
            gumbel_positions=gumbel_positions.at[:, 0].set(root_gumbel_positions),
            gumbel_node_ids=gumbel_node_ids,
            node_mask=node_mask.at[:, 0].set(True),
            sampling_policy=sampling_policy,
            gumbel_keys=gumbel_keys,
            vocabulary_size=vocabulary_size,
            num_nodes=1,
            max_depth=0,
        )

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def budget(self) -> int:
        return self.token_ids.shape[1]

    def add_nodes(
        self,
        token_ids: Int[Array, "batch nodes"],
        parent_indices: Int[Array, "batch nodes"],
        depths: Int[Array, "batch nodes"],
        node_mask: Bool[Array, "batch nodes"],
        max_depth: int,
    ) -> "TrieProposal":
        batch_size, num_new_nodes = token_ids.shape
        if batch_size != self.batch_size:
            raise ValueError("node batch size must match proposal batch size.")
        if parent_indices.shape != (batch_size, num_new_nodes):
            raise ValueError("parent_indices must match token_ids shape.")
        if depths.shape != (batch_size, num_new_nodes):
            raise ValueError("depths must match token_ids shape.")
        if node_mask.shape != (batch_size, num_new_nodes):
            raise ValueError("node_mask must match token_ids shape.")

        remaining_slots = max(self.budget - self.num_nodes, 0)
        retained_nodes = min(num_new_nodes, remaining_slots)
        slot_mask = jnp.arange(num_new_nodes, dtype=jnp.int32) < remaining_slots
        mask = node_mask & slot_mask[None, :]
        node_indices = jnp.arange(self.num_nodes, self.num_nodes + num_new_nodes, dtype=jnp.int32)[None, :]
        node_indices = jnp.broadcast_to(node_indices, (batch_size, num_new_nodes))
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        root_gumbel_positions = self.gumbel_positions[:, :1]
        gumbel_positions = root_gumbel_positions + depths

        def mask_values(values: Int[Array, "batch nodes"], empty_value: int) -> Int[Array, "batch nodes"]:
            return jnp.where(mask, values, empty_value)

        def write_slots(
            values: Int[Array, "batch budget"] | Bool[Array, "batch budget"],
            updates: Int[Array, "batch nodes"] | Bool[Array, "batch nodes"],
        ) -> Int[Array, "batch budget"] | Bool[Array, "batch budget"]:
            return values.at[batch_indices, node_indices].set(updates, mode="drop")

        return TrieProposal(
            token_ids=write_slots(self.token_ids, mask_values(token_ids, 0)),
            parent_indices=write_slots(self.parent_indices, mask_values(parent_indices, -1)),
            depths=write_slots(self.depths, mask_values(depths, 0)),
            gumbel_positions=write_slots(self.gumbel_positions, mask_values(gumbel_positions, 0)),
            gumbel_node_ids=write_slots(self.gumbel_node_ids, mask_values(node_indices, 0)),
            node_mask=write_slots(self.node_mask, mask),
            sampling_policy=self.sampling_policy,
            gumbel_keys=self.gumbel_keys,
            vocabulary_size=self.vocabulary_size,
            num_nodes=min(self.num_nodes + num_new_nodes, self.budget),
            max_depth=max(self.max_depth, min(max_depth, self.max_depth + retained_nodes)),
        )

    def sample_chain_nodes(
        self,
        logits: Float[Array, "batch depth vocabulary"],
    ) -> tuple[
        Int[Array, "batch depth"],
        Int[Array, "batch depth"],
        Int[Array, "batch depth"],
        Bool[Array, "batch depth"],
    ]:
        self.validate_sampling_policy()
        batch_size, depth, vocabulary_size = logits.shape
        if batch_size != self.batch_size:
            raise ValueError("logits batch size must match proposal batch size.")
        if vocabulary_size != self.vocabulary_size:
            raise ValueError("logits vocabulary dimension must match proposal vocabulary size.")
        if depth < 1:
            raise ValueError("chain depth must be positive.")

        offsets = jnp.arange(depth, dtype=jnp.int32)
        parent_indices = jnp.broadcast_to(self.num_nodes + offsets[None, :] - 1, (batch_size, depth))
        parent_indices = parent_indices.at[:, 0].set(0)
        depths = jnp.broadcast_to(offsets[None, :] + 1, (batch_size, depth))
        node_mask = jnp.ones((batch_size, depth), dtype=jnp.bool)
        parent_gumbel_positions = self.gumbel_positions[:, :1] + offsets[None, :]
        parent_gumbel_node_ids = jnp.broadcast_to(self.num_nodes + offsets[None, :] - 1, (batch_size, depth))
        parent_gumbel_node_ids = parent_gumbel_node_ids.at[:, 0].set(0)

        def sample_with_policy(
            policy: SamplingPolicy,
            row_logits: Float[Array, "depth vocabulary"],
            key: Key[Array, ""],
            positions: Int[Array, " depth"],
            node_ids: Int[Array, " depth"],
        ) -> Int[Array, " depth"]:
            def sample_one(
                active_logits: Float[Array, " vocabulary"],
                position: Int[Array, ""],
                node_id: Int[Array, ""],
            ) -> Int[Array, ""]:
                sample_key = self.fold_gumbel_key(key, position, node_id)
                processed_logits = policy.process_logits(active_logits.astype(jnp.float32))
                if policy.temperature is None:
                    sample_logits = processed_logits + jax.random.gumbel(
                        sample_key,
                        processed_logits.shape,
                        dtype=processed_logits.dtype,
                    )
                else:
                    sample_logits = jax.lax.cond(
                        policy.temperature < 1e-5,
                        lambda _: processed_logits,
                        lambda _: processed_logits
                        + jax.random.gumbel(sample_key, processed_logits.shape, dtype=processed_logits.dtype),
                        operand=None,
                    )
                return jnp.argmax(sample_logits).astype(jnp.int32)

            return jax.vmap(sample_one)(row_logits, positions, node_ids)

        token_ids = jax.vmap(sample_with_policy)(
            self.sampling_policy,
            logits.astype(jnp.float32),
            self.gumbel_keys,
            parent_gumbel_positions,
            parent_gumbel_node_ids,
        )
        return token_ids, parent_indices, depths, node_mask

    def validate_sampling_policy(self) -> None:
        if self.sampling_policy.has_unsupported_tree_verification_processors:
            raise ValueError(
                "Speculative decoding only supports temperature, top_k, top_p, and min_p sampling processors.",
            )

    def fold_gumbel_key(
        self,
        key: Key[Array, ""],
        position: Int[Array, ""],
        node_id: Int[Array, ""],
    ) -> Key[Array, ""]:
        key = jax.random.fold_in(key, position.astype(jnp.int32))
        return jax.random.fold_in(key, node_id.astype(jnp.int32))

    def sample(
        self,
        logits: Float[Array, "batch nodes vocabulary"],
    ) -> tuple[Float[Array, "batch nodes vocabulary"], Int[Array, "batch nodes"], SamplingPolicy]:
        if logits.shape != (self.batch_size, self.num_nodes, self.vocabulary_size):
            raise ValueError("logits shape must match proposal batch, node, and vocabulary dimensions.")

        if self.num_nodes == 1:
            root_logits = logits.reshape(self.batch_size, self.vocabulary_size)
            processed_logits = call_vmapped(
                lambda policy, logits: policy.process_logits(logits),
                self.sampling_policy,
                root_logits.astype(jnp.float32),
            )
            sample_keys = jax.vmap(jax.random.fold_in)(
                self.gumbel_keys,
                self.gumbel_positions[:, 0],
            )
            token_ids = jax.vmap(jax.random.categorical)(sample_keys, processed_logits).astype(jnp.int32)
            next_sampling_policy = call_vmapped(
                lambda policy, token_id: policy.with_next_token_count(token_id),
                self.sampling_policy,
                token_ids,
            )
            return processed_logits[:, None], token_ids[:, None], next_sampling_policy

        self.validate_sampling_policy()

        def sample_with_policy(
            policy: SamplingPolicy,
            node_logits: Float[Array, " vocabulary"],
            key: Key[Array, ""],
            position: Int[Array, ""],
            node_id: Int[Array, ""],
            active_mask: Bool[Array, ""],
        ) -> tuple[Float[Array, " vocabulary"], Int[Array, ""]]:
            sample_key = self.fold_gumbel_key(key, position, node_id)
            safe_logits = jnp.where(active_mask, node_logits, jnp.zeros_like(node_logits))
            processed_logits = policy.process_logits(safe_logits.astype(jnp.float32))

            if policy.temperature is None:
                sample_logits = processed_logits + jax.random.gumbel(
                    sample_key,
                    processed_logits.shape,
                    dtype=processed_logits.dtype,
                )
            else:
                sample_logits = jax.lax.cond(
                    policy.temperature < 1e-5,
                    lambda _: processed_logits,
                    lambda _: processed_logits
                    + jax.random.gumbel(sample_key, processed_logits.shape, dtype=processed_logits.dtype),
                    operand=None,
                )
            token_id = jnp.argmax(sample_logits).astype(jnp.int32)
            return (
                jnp.where(active_mask, processed_logits, jnp.zeros_like(processed_logits)),
                jnp.where(active_mask, token_id, -1),
            )

        processed_logits, token_ids = jax.vmap(
            jax.vmap(sample_with_policy, in_axes=(None, 0, None, 0, 0, 0)),
            in_axes=(0, 0, 0, 0, 0, 0),
        )(
            self.sampling_policy,
            logits,
            self.gumbel_keys,
            self.gumbel_positions[:, : self.num_nodes],
            self.gumbel_node_ids[:, : self.num_nodes],
            self.node_mask[:, : self.num_nodes],
        )
        return processed_logits, token_ids, self.sampling_policy

    def forward_inputs(
        self,
        next_token_positions: Int[Array, " batch"],
    ) -> ProposalInputs:
        token_ids = self.token_ids[:, : self.num_nodes]
        depths = self.depths[:, : self.num_nodes]
        token_positions = next_token_positions[:, None] + depths
        forward_pass_mode = ForwardPassMode.MULTI_TOKEN
        if self.num_nodes == 1:
            forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
        attention_parent_indices = None
        if self.num_nodes > 1:
            attention_parent_indices = jnp.where(
                self.node_mask[:, : self.num_nodes],
                self.parent_indices[:, : self.num_nodes],
                -1,
            )
        return ProposalInputs(
            token_ids=token_ids,
            token_positions=token_positions,
            lengths_without_padding=jnp.full((self.batch_size,), self.num_nodes, dtype=jnp.int32),
            forward_pass_mode=forward_pass_mode,
            attention_parent_indices=attention_parent_indices,
        )

    def verify(
        self,
        sampled_token_ids: Int[Array, "batch nodes"],
        next_sampling_policies: SamplingPolicy,
    ) -> AcceptedProposal:
        if self.num_nodes == 1:
            compact_indices = jnp.zeros((self.batch_size, 1), dtype=jnp.int32)
            terminal_node_indices = jnp.zeros((self.batch_size,), dtype=jnp.int32)
            return AcceptedProposal(
                accepted_token_ids=self.token_ids[:, :1],
                node_indices=compact_indices,
                compact_indices=compact_indices,
                num_compact_indices=jnp.ones((self.batch_size,), dtype=jnp.int32),
                terminal_node_indices=terminal_node_indices,
                bonus_token_ids=sampled_token_ids[:, 0],
                next_sampling_policy=next_sampling_policies,
            )

        node_mask = self.node_mask[:, : self.num_nodes]
        parent_indices = self.parent_indices[:, : self.num_nodes]
        token_ids = self.token_ids[:, : self.num_nodes]

        path_shape = jax.ShapeDtypeStruct((self.batch_size, self.max_depth), jnp.int32)
        mask_shape = jax.ShapeDtypeStruct((self.batch_size, self.max_depth), jnp.bool)
        terminal_shape = jax.ShapeDtypeStruct((self.batch_size,), jnp.int32)
        path_node_indices, path_mask, terminal_node_indices = jax.pure_callback(
            lambda row_token_ids, row_parent_indices, row_node_mask, row_sampled_token_ids: (
                TrieProposal.accept_batch_numpy(
                    row_token_ids,
                    row_parent_indices,
                    row_node_mask,
                    row_sampled_token_ids,
                    self.max_depth,
                )
            ),
            (path_shape, mask_shape, terminal_shape),
            token_ids,
            parent_indices,
            node_mask,
            sampled_token_ids[:, : self.num_nodes],
            vmap_method="sequential",
        )
        raw_node_indices = jnp.concatenate(
            [
                jnp.where(path_mask, path_node_indices, 0),
                jnp.zeros((self.batch_size, 1), dtype=jnp.int32),
            ],
            axis=1,
        )
        raw_compact_indices = jnp.concatenate(
            [jnp.zeros((self.batch_size, 1), dtype=jnp.int32), raw_node_indices[:, :-1]],
            axis=1,
        )
        num_compact_indices = jnp.sum(path_mask, axis=1).astype(jnp.int32) + 1
        slots = jnp.arange(self.max_depth + 1, dtype=jnp.int32)[None, :]
        accepted_token_ids = jnp.where(
            slots < num_compact_indices[:, None],
            jnp.take_along_axis(self.token_ids, raw_compact_indices, axis=1),
            -1,
        )
        node_indices = jnp.concatenate(
            [raw_compact_indices[:, 1:], jnp.zeros((self.batch_size, 1), dtype=jnp.int32)],
            axis=1,
        )
        batch_indices = jnp.arange(self.batch_size, dtype=jnp.int32)
        bonus_token_ids = sampled_token_ids[batch_indices, terminal_node_indices]
        return AcceptedProposal(
            accepted_token_ids=accepted_token_ids,
            node_indices=node_indices,
            compact_indices=raw_compact_indices,
            num_compact_indices=num_compact_indices,
            terminal_node_indices=terminal_node_indices,
            bonus_token_ids=bonus_token_ids,
            next_sampling_policy=next_sampling_policies,
        )

    @staticmethod
    def accept_batch_numpy(
        token_ids: Int[np.ndarray, "batch nodes"],
        parent_indices: Int[np.ndarray, "batch nodes"],
        node_mask: Bool[np.ndarray, "batch nodes"],
        sampled_token_ids: Int[np.ndarray, "batch nodes"],
        max_depth: int,
    ) -> tuple[Int[np.ndarray, "batch depth"], Bool[np.ndarray, "batch depth"], Int[np.ndarray, " batch"]]:
        token_ids = np.asarray(token_ids)
        parent_indices = np.asarray(parent_indices)
        node_mask = np.asarray(node_mask)
        sampled_token_ids = np.asarray(sampled_token_ids)
        batch_size, num_nodes = token_ids.shape
        path_node_indices = np.zeros((batch_size, max_depth), dtype=np.int32)
        path_mask = np.zeros((batch_size, max_depth), dtype=np.bool_)
        terminal_node_indices = np.zeros((batch_size,), dtype=np.int32)
        for batch_index in range(batch_size):
            children_map: dict[int, dict[int, int]] = {}
            for node_index in range(num_nodes):
                if not bool(node_mask[batch_index, node_index]):
                    continue
                parent_index = int(parent_indices[batch_index, node_index])
                if parent_index < 0:
                    continue
                token_id = int(token_ids[batch_index, node_index])
                children = children_map.setdefault(parent_index, {})
                if token_id not in children:
                    children[token_id] = node_index

            terminal_node_index = 0
            for depth in range(max_depth):
                sampled_token_id = int(sampled_token_ids[batch_index, terminal_node_index])
                children = children_map.get(terminal_node_index)
                if children is None or sampled_token_id not in children:
                    break
                terminal_node_index = children[sampled_token_id]
                path_node_indices[batch_index, depth] = terminal_node_index
                path_mask[batch_index, depth] = True

            terminal_node_indices[batch_index] = terminal_node_index
        return path_node_indices, path_mask, terminal_node_indices
