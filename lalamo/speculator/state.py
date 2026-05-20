from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.data.completion_features import FeatureRequest
from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.token_mixer import State
from lalamo.modules.token_mixers.kv_cache import StaticKVCacheLayer, compact_state_layers
from lalamo.modules.utils import call_vmapped
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal

__all__ = [
    "LMState",
    "PrefillResults",
]


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    input_token_ids: Int[Array, "batch tokens"]
    input_lengths: Int[Array, " batch"]
    activation_trace: DecoderActivationTrace | None = None


class LMState(eqx.Module):
    kv_cache: State
    next_token_position: Int[Array, " batch"]
    root_bonus_id: Int[Array, " batch"]
    root_sample_logits: Float[Array, "batch vocabulary"]
    sampling_policy: SamplingPolicy
    gumbel_keys: Key[Array, " batch"]
    output_lengths: Int[Array, " batch"]
    stop_flags: Bool[Array, " batch"]

    @classmethod
    def from_prefill(
        cls,
        prefill_results: PrefillResults,
        next_token_position: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        gumbel_keys: Key[Array, " batch"],
    ) -> "LMState":
        root_sample_logits, root_bonus_id, next_sampling_policy = call_vmapped(
            lambda policy, key, position, logits: policy.sample(
                logits,
                jax.random.fold_in(key, position.astype(jnp.int32)),
            ),
            sampling_policy,
            gumbel_keys,
            next_token_position,
            prefill_results.last_token_logits,
        )
        return cls(
            kv_cache=prefill_results.state,
            next_token_position=next_token_position,
            root_bonus_id=root_bonus_id,
            root_sample_logits=root_sample_logits,
            sampling_policy=next_sampling_policy,
            gumbel_keys=gumbel_keys,
            output_lengths=jnp.zeros(next_token_position.shape, dtype=jnp.int32),
            stop_flags=jnp.zeros(next_token_position.shape, dtype=jnp.bool),
        )

    def create_root_proposal(self, budget: int = 128) -> TrieProposal:
        return TrieProposal.create(
            root_ids=self.root_bonus_id,
            root_gumbel_positions=self.next_token_position + 1,
            sampling_policy=self.sampling_policy,
            gumbel_keys=self.gumbel_keys,
            vocabulary_size=self.root_sample_logits.shape[-1],
            budget=budget,
        )

    def update_status(
        self,
        kv_cache: State,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        accepted: AcceptedProposal,
        accepted_token_ids: Int[Array, "batch max_slots"],
        accepted_output_features: Float[Array, "batch max_slots channels"] | None,
        accepted_layer_outputs: tuple[Float[Array, "batch max_slots channels"], ...],
        stop_flags: Bool[Array, " batch"],
    ) -> "LMState":
        del accepted_token_ids
        del accepted_output_features
        del accepted_layer_outputs
        root_sample_logits = jnp.take_along_axis(
            processed_tree_logits,
            accepted.terminal_node_indices[:, None, None],
            axis=1,
        )[:, 0]
        return eqx.tree_at(
            lambda state: (
                state.kv_cache,
                state.next_token_position,
                state.root_bonus_id,
                state.root_sample_logits,
                state.sampling_policy,
                state.output_lengths,
                state.stop_flags,
            ),
            self,
            (
                kv_cache,
                self.next_token_position + accepted.num_compact_indices,
                accepted.bonus_token_ids,
                root_sample_logits,
                accepted.next_sampling_policy,
                self.output_lengths + accepted.num_compact_indices,
                stop_flags,
            ),
            is_leaf=lambda value: value is None,
        )

    def commit(
        self,
        decoder_result: DecoderResult,
        processed_tree_logits: Float[Array, "batch nodes vocabulary"],
        accepted: AcceptedProposal,
        stop_flags: Bool[Array, " batch"],
        feature_request: FeatureRequest,
    ) -> "LMState":
        updated_state = decoder_result.updated_state
        assert updated_state is not None, "updated_state should not be None"
        accepted_output_features = None
        accepted_layer_outputs = ()
        if feature_request.output_features or feature_request.layer_indices:
            activation_trace = decoder_result.activation_trace
            if activation_trace is None:
                raise ValueError("requested decoder features require activation trace")
            if feature_request.output_features:
                accepted_output_features = accepted.compact(activation_trace.output_norm)
            accepted_layer_outputs = tuple(
                accepted.compact(activation_trace.layer_results[layer_index].outputs)
                for layer_index in feature_request.layer_indices
            )
        if accepted.compact_indices.shape[1] == 1:
            return self.update_status(
                updated_state,
                processed_tree_logits,
                accepted,
                accepted.accepted_token_ids,
                accepted_output_features,
                accepted_layer_outputs,
                stop_flags,
            )

        first_layer = self.kv_cache[0]
        if not isinstance(first_layer, StaticKVCacheLayer):
            raise TypeError(
                f"tree proposal compaction requires StaticKVCacheLayer, got {type(first_layer).__name__}",
            )
        kv_cache = compact_state_layers(
            updated_state,
            cache_len=first_layer.current_length,
            accepted_indices=accepted.compact_indices,
            num_accepted=accepted.num_compact_indices,
            max_slots=accepted.compact_indices.shape[1],
        )
        return self.update_status(
            kv_cache,
            processed_tree_logits,
            accepted,
            accepted.accepted_token_ids,
            accepted_output_features,
            accepted_layer_outputs,
            stop_flags,
        )
