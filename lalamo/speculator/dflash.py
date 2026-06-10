from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.dflash import DFlashDraftConfig, DFlashDraftModel, DFlashDraftState
from lalamo.modules.embedding import EmbeddingBase
from lalamo.modules.utils import call_vmapped_twice

from .common import AcceptedProposal, ChainProposal, Speculator, SpeculatorConfig

__all__ = [
    "DFlashSpeculator",
    "DFlashSpeculatorConfig",
]


@dataclass(frozen=True)
class DFlashSpeculatorConfig(SpeculatorConfig):
    draft_config: DFlashDraftConfig

    def init(self, initializer: Initializer) -> "DFlashSpeculator":
        return DFlashSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            draft_model=self.draft_config.init(initializer),
        )


class DFlashSpeculator(Speculator[DFlashDraftState, DFlashSpeculatorConfig]):
    draft_model: DFlashDraftModel

    @property
    def max_proposal_tokens(self) -> int:
        return self.draft_model.config.block_size

    def extract_target_features(
        self,
        activation_trace: DecoderActivationTrace,
    ) -> Float[Array, "batch tokens target_channels"]:
        return jnp.concatenate(
            tuple(
                activation_trace.layer_results[layer_id].outputs
                for layer_id in self.draft_model.config.target_layer_ids
            ),
            axis=-1,
        )

    def init_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> DFlashDraftState:
        return self.draft_model.empty_state(batch_size, context_capacity, dtype)

    def prefill_chunk(
        self,
        state: DFlashDraftState,
        decoder_result: DecoderResult,
        chunk_lengths: Int[Array, " batch"],
        *,
        keychain: Keychain,
    ) -> DFlashDraftState:
        activation_trace = decoder_result.activation_trace
        if activation_trace is None:
            raise ValueError("DFlashSpeculator requires decoder activation traces.")
        target_features = self.extract_target_features(activation_trace)
        return self.draft_model.append_state(
            state,
            target_features,
            activation_trace.token_positions,
            chunk_lengths,
            keychain=keychain,
        )

    def update_state(
        self,
        state: DFlashDraftState,
        decoder_result: DecoderResult,
        accepted: AcceptedProposal,
        *,
        keychain: Keychain,
    ) -> DFlashDraftState:
        activation_trace = decoder_result.activation_trace
        if activation_trace is None:
            raise ValueError("DFlashSpeculator requires decoder activation traces.")
        target_features = self.extract_target_features(activation_trace)
        batch_size, num_accepted_slots = accepted.accepted_node_indices.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        accepted_slots = jnp.arange(num_accepted_slots, dtype=accepted.num_accepted_nodes.dtype)[None, :]
        valid = accepted_slots < accepted.num_accepted_nodes[:, None]
        source_indices = jnp.clip(accepted.accepted_node_indices, 0, target_features.shape[1] - 1)
        selected_target_features = target_features[batch_indices, source_indices, :]
        selected_token_positions = activation_trace.token_positions[batch_indices, source_indices]
        return self.draft_model.append_state(
            state,
            jnp.where(valid[:, :, None], selected_target_features, 0),
            jnp.where(valid, selected_token_positions, 0),
            accepted.num_accepted_nodes,
            keychain=keychain,
        )

    def draft(
        self,
        state: DFlashDraftState,
        last_token_ids: Int[Array, " batch"],
        last_token_indices: Int[Array, " batch"],
        target_embedding: EmbeddingBase,
        target_lm_head: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> ChainProposal:
        block_size = self.draft_model.config.block_size

        (batch_size,) = last_token_ids.shape
        embedding_keychain, draft_keychain, readout_keychain = keychain.split(3)
        num_draft_tokens = block_size - 1
        mask_token_ids = jnp.full(
            (batch_size, num_draft_tokens),
            self.draft_model.config.mask_token_id,
            dtype=last_token_ids.dtype,
        )
        block_token_ids = jnp.concatenate((last_token_ids[:, None], mask_token_ids), axis=1)
        noise_embeddings = target_embedding.embed(block_token_ids, keychain=embedding_keychain)

        draft_offsets = jnp.arange(block_size, dtype=last_token_indices.dtype)[None, :] + 1
        draft_positions = last_token_indices[:, None] + draft_offsets
        first_layer_state, *_ = state.layer_states
        _, context_capacity, _, _ = first_layer_state.keys.shape
        context_slots = jnp.arange(context_capacity, dtype=state.context_lengths.dtype)[None, :]
        context_positions = (
            last_token_indices[:, None]
            - state.context_lengths[:, None]
            + 1
            + context_slots.astype(last_token_indices.dtype)
        )
        token_positions = jnp.concatenate((context_positions, draft_positions), axis=1)
        context_mask = context_slots < state.context_lengths[:, None]
        draft_mask = jnp.ones((batch_size, block_size), dtype=jnp.bool_)
        key_mask = jnp.concatenate((context_mask, draft_mask), axis=1)
        attention_mask = jnp.broadcast_to(
            key_mask[:, None, :],
            (batch_size, block_size, context_capacity + block_size),
        )
        draft_hidden_states = self.draft_model(
            noise_embeddings,
            state,
            token_positions,
            attention_mask,
            keychain=draft_keychain,
        )
        batch_axis = self.draft_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
        draft_logits = call_vmapped_twice(
            target_lm_head.readout,
            draft_hidden_states[:, 1:, :],
            keychain=readout_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        draft_token_ids = jnp.argmax(draft_logits, axis=-1).astype(last_token_ids.dtype)
        return ChainProposal(
            token_ids=jnp.concatenate((last_token_ids[:, None], draft_token_ids), axis=1),
            token_positions=draft_positions,
            lengths=jnp.full((batch_size,), block_size, dtype=last_token_indices.dtype),
        )
