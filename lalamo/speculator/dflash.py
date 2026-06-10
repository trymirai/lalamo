from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import jax.numpy as jnp

from lalamo.exportable import Exportable, ExportResults
from lalamo.initializer import EmptyInitializer
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.dflash import DFlashDraftConfig, DFlashDraftModel, DFlashDraftState
from lalamo.modules.utils import call_vmapped_twice
from lalamo.safetensors import safe_read, safe_write
from lalamo.utils.json import JSON
from lalamo.utils.sharding import ShardingConfig

from .common import AcceptedProposal, ChainProposal, Speculator

if TYPE_CHECKING:
    from jaxtyping import Array, DTypeLike, Float, Int

    from lalamo.modules.embedding import EmbeddingBase

__all__ = [
    "DFlashSpeculator",
]


class DFlashSpeculator(Speculator[DFlashDraftState]):
    draft_model: DFlashDraftModel

    @property
    def max_proposal_tokens(self) -> int:
        return self.draft_model.config.block_size

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        exported_model = Exportable.export(self.draft_model)
        metadata = None
        if exported_model.metadata:
            metadata = {key: json.dumps(value) for key, value in exported_model.metadata.items()}

        with (directory / "model.safetensors").open("wb") as weights_file:
            safe_write(
                weights_file,
                exported_model.arrays,
                metadata=metadata,
            )

        with (directory / "config.json").open("w") as config_file:
            json.dump(self.draft_model.config.to_json(), config_file, indent=4)

    @classmethod
    def load(
        cls,
        directory: Path | str,
        sharding_config: ShardingConfig,
        dtype: DTypeLike | None = None,
    ) -> Self:
        directory = Path(directory)
        with (directory / "config.json").open() as config_file:
            config = DFlashDraftConfig.from_json(json.load(config_file))

        with (directory / "model.safetensors").open("rb") as weights_file:
            metadata, arrays = safe_read(weights_file)
            decoded_metadata: dict[str, JSON] = {}
            if metadata is not None:
                decoded_metadata = {key: cast("JSON", json.loads(value)) for key, value in metadata.items()}
            template = config.init(EmptyInitializer(dtype, sharding_config))
            result = Exportable.load_exported(
                template,
                ExportResults(
                    arrays=arrays,
                    metadata=decoded_metadata,
                ),
            )

        assert isinstance(result, DFlashDraftModel)
        return cls(draft_model=result)

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
