from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING
from jax import pure_callback
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from lalamo.message_processor import AssistantMessage, Message, MessageProcessor
from lalamo.modules import (
    Decoder,
    DecoderConfig,
    DecoderForwardPassConfig,
    ForwardPassMode,
    LalamoModule,
    State,
)
from lalamo.sampling import SamplingPolicy, make_policy

from .common import TextModel, TextModelConfig

from typing import Any

__all__ = [
    "ForwardPassConfig",
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = [512 * 2**i for i in range(10)]


type ForwardPassConfig = DecoderForwardPassConfig

class JaxNGramSpeculator(eqx.Module):
    candidates: Int[Array, "table_size k"]
    probs: Float[Array, "table_size k"]
    
    # Метаданные
    table_size: int = eqx.field(static=True)
    ngram_n: int = eqx.field(static=True)
    ngram_candidates: int = eqx.field(static=True)
    ngram_pad: int = eqx.field(static=True)

    @classmethod
    def from_python(cls, py_spec: Any) -> "JaxNGramSpeculator":
        """Create JAX-version of NgramSpeculator"""
        flat_keys = jnp.array(py_spec.ngram_keys, dtype=jnp.int32)
        candidates = flat_keys.reshape((py_spec.hashtable_size, py_spec.ngram_k))
        
        flat_probs = jnp.array(py_spec.ngram_values, dtype=jnp.float32)
        probs = flat_probs.reshape((py_spec.hashtable_size, py_spec.ngram_k))
        
        return cls(
            candidates=candidates,
            probs=probs,
            table_size=py_spec.hashtable_size,
            ngram_n=py_spec.ngram_n,
            ngram_candidates=py_spec.ngram_k,
            ngram_pad=py_spec.ngram_pad
        )


class PrefillResults(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State


class DecodingState(NamedTuple):
    last_token_logits: Float[Array, "batch vocabulary"]
    last_token_indices: Int[Array, " batch"]
    state: State
    stop_flags: Bool[Array, " batch"]


class GenerationStepResults(NamedTuple):
    token_ids: Int[Array, " batch"]
    top_k_token_ids: Int[Array, " batch k"] | None
    top_k_token_logits: Float[Array, " batch k"] | None


class GenerationResults(NamedTuple):
    token_ids: Int[Array, "batch response_tokens"]
    top_k_token_ids: Int[Array, "batch response_tokens k"] | None
    top_k_token_logits: Float[Array, "batch response_tokens k"] | None


@dataclass(frozen=True)
class GenerationConfig:
    stop_token_ids: tuple[int, ...] = tuple()
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    banned_tokens: tuple[int, ...] | None = None

    def default_policy(self) -> SamplingPolicy:
        return make_policy(self.temperature, self.top_k, self.top_p, self.min_p, self.banned_tokens)


@dataclass(frozen=True)
class LanguageModelConfig(TextModelConfig[DecoderConfig]):
    generation_config: GenerationConfig

    def init(
        self,
        model: LalamoModule,
        message_processor: MessageProcessor,
    ) -> "LanguageModel":
        assert isinstance(model, Decoder)
        return LanguageModel(self, model, message_processor)

    @classmethod
    def load_model(cls, path: Path | str) -> "LanguageModel":
        result = super().load_model(path)
        assert isinstance(result, LanguageModel)
        return result


class LanguageModel(TextModel[LanguageModelConfig, Decoder]):
    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.config.generation_config.stop_token_ids

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    @eqx.filter_jit
    def _prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        state_capacity: int | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape
        token_positions = jnp.repeat(jnp.arange(sequence_length, dtype=jnp.int32)[None, ...], batch_size, axis=0)
        if state_capacity is not None:
            state = self.model.init_static_state(batch_size, state_capacity)
        else:
            state = None

        decoder_outputs = self.model(
            token_ids,
            token_positions,
            state,
            return_updated_state=True,
            lengths_without_padding=lengths_without_padding,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=forward_pass_config,
        )

        if lengths_without_padding is not None:
            last_logits_indices = lengths_without_padding - 1
        else:
            last_logits_indices = jnp.array([sequence_length - 1] * batch_size, dtype=jnp.int32)

        last_token_logits = vmap(lambda logits, index: logits[index])(decoder_outputs.logits, last_logits_indices)

        assert decoder_outputs.updated_state is not None
        return PrefillResults(
            last_token_logits=last_token_logits,
            last_token_indices=last_logits_indices,
            state=decoder_outputs.updated_state,
        )

    @eqx.filter_jit
    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        prompt_lengths_without_padding: Int[Array, " batch"] | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        num_top_logits_to_return: int | None = None,
        *,
        key: PRNGKeyArray | None = None,
        speculator: JaxNGramSpeculator | None = None,
    ) -> GenerationResults:
        
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)
        if key is None:
            key = jax.random.PRNGKey(0)

        batch_size, sequence_length = prompt_token_ids.shape

        if speculator is not None:
            spec_buffer = speculator.ngram_candidates + 10
        else:
            spec_buffer = 0

        prefill_results = self._prefill(
            prompt_token_ids,
            prompt_lengths_without_padding,
            state_capacity=sequence_length + max_output_length + spec_buffer,
            forward_pass_config=forward_pass_config,
        )

        if speculator is not None:
            K = speculator.ngram_candidates
            
            out_buffer = jnp.zeros((1, sequence_length + max_output_length + spec_buffer), dtype=jnp.int32)
            out_buffer = out_buffer.at[:, :sequence_length].set(prompt_token_ids)

            init_val = (
                jnp.int32(sequence_length),
                prefill_results.state,
                key,
                jnp.bool_(False),
                out_buffer
            )

            def get_ngram_preds(history_tokens):
                def _cpu_hash(toks):
                    from lalamo.speculator.ngram import seqhash
                    return seqhash(toks, speculator.table_size)
                
                ctx_len = speculator.ngram_n - 1
                padding = jnp.full((ctx_len,), speculator.ngram_pad, dtype=jnp.int32)
                full_seq = jnp.concatenate([padding, history_tokens])
                processed_ctx = full_seq[-ctx_len:]

                h = pure_callback(_cpu_hash, jax.ShapeDtypeStruct((), jnp.int32), processed_ctx)
                
                return speculator.candidates[h], speculator.probs[h]

            def spec_step(carry):
                pos, kv, rng, finished, buf = carry
                rng, step_rng = jax.random.split(rng)
                
                noise = jax.random.gumbel(step_rng, (K + 1, self.model.config.vocab_size))

                def draft_scan_body(c, i):
                    curr_window, _ = c
                    cands, probs = get_ngram_preds(curr_window)
                    

                    cand_noise = noise[i, cands]
                    scores = jnp.log(probs + 1e-10) + cand_noise
                    
                    next_token = cands[jnp.argmax(scores)]
                    
                    next_window = jnp.roll(curr_window, -1).at[-1].set(next_token)
                    return (next_window, None), next_token

                ctx_start = pos - (speculator.ngram_n - 1)
                start_window = jax.lax.dynamic_slice(buf[0], (ctx_start,), (speculator.ngram_n - 1,))
                
                _, draft_tokens = jax.lax.scan(draft_scan_body, (start_window, None), jnp.arange(K))


                last_confirmed = buf[0, pos - 1]
                model_input = jnp.concatenate([last_confirmed[None], draft_tokens])
                model_pos = jnp.arange(K + 1) + (pos - 1)

                decoder_out = self.model(
                    model_input[None, :], model_pos[None, :], kv,
                    return_updated_state=True, forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
                    forward_pass_config=forward_pass_config
                )
                
                target_logits = decoder_out.logits[0]
                if sampling_policy is not None:
                    target_logits = vmap(sampling_policy.process_logits)(target_logits)


                target_choices = jnp.argmax(target_logits + noise, axis=-1)
                

                matches = (draft_tokens == target_choices[:-1])
                
                valid_mask = jnp.cumprod(matches)
                num_accepted = jnp.sum(valid_mask)
                valid_mask = jnp.cumprod(matches)
                num_accepted = jnp.sum(valid_mask)

                jax.debug.print("🚀 Accepted: {n} tokens", n=num_accepted)
                

                tokens_to_write = target_choices
                count_to_write = num_accepted + 1
                

                write_mask = jnp.arange(K + 1) < count_to_write
                old_vals = jax.lax.dynamic_slice(buf[0], (pos,), (K + 1,))
                safe_vals = jnp.where(write_mask, tokens_to_write, old_vals)
                new_buf = jax.lax.dynamic_update_slice(buf, safe_vals[None, :], (0, pos))
                new_pos = pos + count_to_write
                
                just_written = jnp.where(write_mask, tokens_to_write, -1)
                is_eos = jnp.any(jnp.isin(just_written, eos_token_ids))
                is_full = new_pos >= (sequence_length + max_output_length)

                return new_pos, decoder_out.updated_state, rng, finished | is_eos | is_full, new_buf

            final_pos, _, _, _, final_buf = jax.lax.while_loop(
                lambda s: ~s[3],
                spec_step,
                init_val
            )
            
            return GenerationResults(final_buf[:, sequence_length:], None, None)

        else:
            initial_state = DecodingState(
                prefill_results.last_token_logits,
                prefill_results.last_token_indices,
                prefill_results.state,
                jnp.zeros(batch_size, dtype=jnp.bool),
            )

            if key is None:
                key = jax.random.PRNGKey(0)
            keys = jax.random.split(key, num=max_output_length)

            def loop_iteration(
                state: DecodingState,
                key: PRNGKeyArray,
            ) -> tuple[DecodingState, GenerationStepResults]:
                def sample_and_update() -> tuple[DecodingState, GenerationStepResults]:
                    upcasted_logits = state.last_token_logits.astype(jnp.float32)
                    processed_logits = vmap(sampling_policy.process_logits)(upcasted_logits)
                    next_token_ids = jax.random.categorical(key, processed_logits)
                    next_token_ids = jnp.where(state.stop_flags, jnp.zeros(batch_size, dtype=jnp.int32), next_token_ids)
                    if num_top_logits_to_return is not None:
                        next_top_k_token_logits, next_top_k_token_ids = jax.lax.top_k(
                            processed_logits,
                            num_top_logits_to_return,
                        )
                    else:
                        next_top_k_token_ids = None
                        next_top_k_token_logits = None
                    next_token_indices = state.last_token_indices + 1

                    stop_flags = state.stop_flags | jnp.any(next_token_ids[:, None] == eos_token_ids[None, :], axis=-1)

                    if batch_size == 1:
                        forward_pass_mode = ForwardPassMode.SINGLE_TOKEN
                    else:
                        forward_pass_mode = ForwardPassMode.MULTI_TOKEN

                    decoder_outputs = self.model(
                        next_token_ids[:, None],
                        next_token_indices[:, None],
                        state.state,
                        return_updated_state=True,
                        forward_pass_mode=forward_pass_mode,
                        forward_pass_config=forward_pass_config,
                    )
                    assert decoder_outputs.updated_state is not None, "updated_state should not be None"
                    new_state = DecodingState(
                        decoder_outputs.logits.squeeze(1),
                        next_token_indices,
                        decoder_outputs.updated_state,
                        stop_flags,
                    )
                    return new_state, GenerationStepResults(next_token_ids, next_top_k_token_ids, next_top_k_token_logits)

                def pad_and_repeat_state() -> tuple[DecodingState, GenerationStepResults]:
                    (batch_size,) = state.stop_flags.shape
                    pad_token = jnp.zeros(batch_size, dtype=jnp.int32)
                    if num_top_logits_to_return is not None:
                        top_k_token_ids = jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.int32)
                        top_k_token_logits = jnp.zeros((batch_size, num_top_logits_to_return), dtype=jnp.float32)
                    else:
                        top_k_token_ids = None
                        top_k_token_logits = None
                    return state, GenerationStepResults(pad_token, top_k_token_ids, top_k_token_logits)

                return jax.lax.cond(jnp.all(state.stop_flags), pad_and_repeat_state, sample_and_update)

            _, generated = jax.lax.scan(loop_iteration, initial_state, keys)

            token_ids = rearrange(generated.token_ids, "iteration batch -> batch iteration")

            if num_top_logits_to_return is not None:
                top_k_token_ids = rearrange(generated.top_k_token_ids, "iteration batch k -> batch iteration k")
                top_k_token_logits = rearrange(generated.top_k_token_logits, "iteration batch k -> batch iteration k")
            else:
                top_k_token_ids = None
                top_k_token_logits = None

            return GenerationResults(token_ids, top_k_token_ids, top_k_token_logits)

    def reply(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> AssistantMessage:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)[None, :]
        response_ids = self.generate_tokens(
            token_ids,
            sampling_policy,
            forward_pass_config=forward_pass_config,
            key=key,
        ).token_ids.squeeze(0)
        response_text = self.message_processor.detokenize(response_ids.tolist())
        return self.message_processor.parse_response(response_text)

    def stream_reply_text(
        self,
        messages: Iterable[Message],
        sampling_policy: SamplingPolicy | None = None,
        max_output_length: int = 8192,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[str]:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)
        for token_id in self.stream_tokens(
            token_ids,
            sampling_policy,
            max_output_length,
            forward_pass_config=forward_pass_config,
            key=key,
        ):
            yield self.message_processor.detokenize([token_id.item()])

    def stream_tokens(
        self,
        prompt_token_ids: Int[Array, " prompt_tokens"],
        sampling_policy: SamplingPolicy | None = None,
        max_output_length: int = 8192,
        eos_token_ids: Int[Array, " eos_tokens"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Iterable[Int[Array, ""]]:
        if sampling_policy is None:
            sampling_policy = self.default_sampling_policy()
        if eos_token_ids is None:
            eos_token_ids = jnp.array(self.stop_token_ids, dtype=jnp.int32)

        (input_length,) = prompt_token_ids.shape

        padded_input_length = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= input_length)
        padded_token_ids = jnp.zeros((padded_input_length,), dtype=jnp.int32)
        padded_token_ids = padded_token_ids.at[:input_length].set(prompt_token_ids)

        prefill_results = self._prefill(
            padded_token_ids[None, :],
            jnp.array([input_length], dtype=jnp.int32),
            padded_input_length + max_output_length,
            forward_pass_config=forward_pass_config,
        )

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=max_output_length)

        state = DecodingState(
            prefill_results.last_token_logits,
            prefill_results.last_token_indices,
            prefill_results.state,
            jnp.array([0], dtype=jnp.bool),
        )

        for iter_key in keys:
            upcasted_logits = state.last_token_logits.astype(jnp.float32)
            processed_logits = sampling_policy.process_logits(upcasted_logits.squeeze(0))
            next_token_id = jax.random.categorical(iter_key, processed_logits)

            yield next_token_id

            if jnp.any(next_token_id == eos_token_ids):
                return

            next_token_indices = state.last_token_indices + 1
            decoder_outputs = self.model(
                next_token_id.reshape(1, 1),
                next_token_indices.reshape(1, 1),
                state.state,
                return_updated_state=True,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_state is not None, "updated_state should not be None"
            state = DecodingState(
                decoder_outputs.logits.squeeze(1),
                next_token_indices,
                decoder_outputs.updated_state,
                state.stop_flags,
            )
