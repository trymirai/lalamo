import jax
import jax.numpy as jnp

from lalamo.data.completion_features import FeatureRequest
from lalamo.modules.decoder import DecoderResult
from lalamo.modules.token_mixer import State
from lalamo.sampling import SamplingPolicy
from lalamo.speculator.common import Speculator
from lalamo.speculator.proposal import AcceptedProposal, TrieProposal
from lalamo.speculator.state import LMState, PrefillResults


class HookedLMState(LMState):
    hook_value: jax.Array


class InitStateSpeculator(Speculator):
    @property
    def max_step_tokens(self) -> int:
        return 1

    def init_state(
        self,
        prefill_results: PrefillResults,
        next_token_position: jax.Array,
        sampling_policy: SamplingPolicy,
        gumbel_keys: jax.Array,
    ) -> HookedLMState:
        state = LMState.from_prefill(
            prefill_results,
            next_token_position,
            sampling_policy,
            gumbel_keys,
        )
        return HookedLMState(
            kv_cache=state.kv_cache,
            next_token_position=state.next_token_position,
            root_bonus_id=state.root_bonus_id,
            root_sample_logits=state.root_sample_logits,
            sampling_policy=state.sampling_policy,
            gumbel_keys=state.gumbel_keys,
            output_lengths=state.output_lengths,
            stop_flags=state.stop_flags,
            hook_value=jnp.asarray(7, dtype=jnp.int32),
        )

    def draft(self, state: LMState) -> TrieProposal:
        del state
        raise AssertionError("draft is not used in this test")


def test_speculator_init_state_can_return_lmstate_subclass() -> None:
    speculator = InitStateSpeculator()
    prefill_results = PrefillResults(
        last_token_logits=jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32),
        last_token_indices=jnp.asarray([0], dtype=jnp.int32),
        state=State(()),
        input_token_ids=jnp.asarray([[1]], dtype=jnp.int32),
        input_lengths=jnp.asarray([1], dtype=jnp.int32),
    )

    state = speculator.init_state(
        prefill_results,
        jnp.asarray([5], dtype=jnp.int32),
        SamplingPolicy.init().broadcast(1),
        jax.random.split(jax.random.key(0), 1),
    )

    assert isinstance(state, HookedLMState)
    assert jnp.array_equal(state.hook_value, jnp.asarray(7, dtype=jnp.int32))


def test_lmstate_commit_preserves_subclass_fields() -> None:
    state = HookedLMState(
        kv_cache=State(()),
        next_token_position=jnp.asarray([5], dtype=jnp.int32),
        root_bonus_id=jnp.asarray([1], dtype=jnp.int32),
        root_sample_logits=jnp.zeros((1, 4), dtype=jnp.float32),
        sampling_policy=SamplingPolicy.init().broadcast(1),
        gumbel_keys=jax.random.split(jax.random.key(1), 1),
        output_lengths=jnp.asarray([0], dtype=jnp.int32),
        stop_flags=jnp.asarray([False], dtype=jnp.bool),
        hook_value=jnp.asarray(11, dtype=jnp.int32),
    )
    accepted = AcceptedProposal(
        accepted_token_ids=jnp.asarray([[2]], dtype=jnp.int32),
        node_indices=jnp.asarray([[0]], dtype=jnp.int32),
        compact_indices=jnp.asarray([[0]], dtype=jnp.int32),
        num_compact_indices=jnp.asarray([1], dtype=jnp.int32),
        terminal_node_indices=jnp.asarray([0], dtype=jnp.int32),
        bonus_token_ids=jnp.asarray([3], dtype=jnp.int32),
        next_sampling_policy=SamplingPolicy.init().broadcast(1),
    )
    decoder_result = DecoderResult(
        logits=jnp.asarray([[[0.0, 1.0, 2.0, 3.0]]], dtype=jnp.float32),
        updated_state=State(()),
    )

    next_state = state.commit(
        decoder_result,
        decoder_result.logits,
        accepted,
        jnp.asarray([False], dtype=jnp.bool),
        FeatureRequest(completions=()),
    )

    assert isinstance(next_state, HookedLMState)
    assert jnp.array_equal(next_state.hook_value, jnp.asarray(11, dtype=jnp.int32))
    assert jnp.array_equal(next_state.root_bonus_id, jnp.asarray([3], dtype=jnp.int32))
    assert jnp.array_equal(next_state.next_token_position, jnp.asarray([6], dtype=jnp.int32))
