from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING
from jax import pure_callback
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat
from jax import vmap
from jaxtyping import Array, Bool, Float, Int, Key, PRNGKeyArray

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

from .common import BatchSizeInfo, BatchSizesComputedEvent, InferenceConfig, TextModel, TextModelConfig
from .compile_helpers import compile_generate_tokens
from .lm_helpers import (
    decrease_batchsize_on_oom,
    estimate_batchsizes_from_vram,
    merge_small_buckets,
    pad_keys_to_size,
    pad_sequences,
)

from typing import Any

__all__ = [
    "ForwardPassConfig",
    "GenerationConfig",
    "LanguageModel",
    "LanguageModelConfig",
]


_COMPILED_PROMPT_LENGTHS = [256 * 2**i for i in range(12)]


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


class Chunk(eqx.Module):
    tokens: Int[Array, "num_chunks batch chunk_size"]
    indices: Int[Array, "num_chunks batch chunk_size"]
    sequence_ends: Int[Array, "num_chunks batch"]
    is_last_token_inside: Bool[Array, "num_chunks batch"]


class LanguageModel(TextModel[LanguageModelConfig, Decoder]):
    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return self.config.generation_config.stop_token_ids

    def default_sampling_policy(self) -> SamplingPolicy:
        return self.config.generation_config.default_policy()

    @eqx.filter_jit
    def _make_chunks(
        self,
        token_ids: Int[Array, "batch tokens"],
        lengths_without_padding: Int[Array, " batch"] | None,
        chunk_size: int,
    ) -> Chunk:
        batch_size, sequence_length = token_ids.shape
        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        # If all sequences fit in a single chunk, use sequence_length as the chunk size
        chunk_size = min(chunk_size, sequence_length)

        n_chunks = (sequence_length + chunk_size - 1) // chunk_size
        padded_length = n_chunks * chunk_size

        token_ids = jnp.pad(token_ids, [(0, 0), (0, padded_length - sequence_length)])

        # Reshape tokens to (num_chunks, batch, chunk_size)
        tokens = rearrange(
            token_ids,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )

        # Create position indices (num_chunks, batch, chunk_size)
        indices = jnp.arange(padded_length, dtype=jnp.int32)
        indices = repeat(indices, "token_idx -> batch token_idx", batch=batch_size)
        indices = rearrange(
            indices,
            "batch (num_chunks chunk_size) -> num_chunks batch chunk_size",
            chunk_size=chunk_size,
        )

        # sequence_ends: for each chunk, how many valid tokens per batch item
        chunk_starts = jnp.arange(n_chunks, dtype=jnp.int32) * chunk_size
        sequence_ends = jnp.clip(
            lengths_without_padding[None, :] - chunk_starts[:, None],
            0,
            chunk_size,
        )

        # last_token_inside: whether the last valid token (at index length-1) is in this chunk
        last_token_idx = lengths_without_padding - 1
        chunk_ends = chunk_starts + chunk_size
        is_last_token_inside = (last_token_idx[None, :] >= chunk_starts[:, None]) & (
            last_token_idx[None, :] < chunk_ends[:, None]
        )

        return Chunk(
            tokens=tokens,
            indices=indices,
            sequence_ends=sequence_ends,
            is_last_token_inside=is_last_token_inside,
        )

    @eqx.filter_jit
    def _prefill(
        self,
        token_ids: Int[Array, "batch tokens"],
        state_capacity: int,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: ForwardPassConfig | None = None,
        chunk_size: int = 512,  # vllm default
    ) -> PrefillResults:
        batch_size, sequence_length = token_ids.shape

        if lengths_without_padding is None:
            lengths_without_padding = jnp.full((batch_size,), sequence_length, dtype=jnp.int32)

        chunks = self._make_chunks(token_ids, lengths_without_padding, chunk_size)

        num_chunks, _, chunk_size = chunks.tokens.shape
        state_capacity = max(state_capacity, num_chunks * chunk_size)

        state = self.model.init_static_state(batch_size, state_capacity)
        logits_like = jnp.zeros((batch_size, self.model.vocab_size), dtype=jnp.float32)

        def apply_chunk(state_and_logits: tuple, chunk: Chunk) -> tuple:
            state, prev_logits = state_and_logits
            decoder_outputs = self.model(
                chunk.tokens,
                chunk.indices,
                state,
                return_updated_state=True,
                lengths_without_padding=chunk.sequence_ends,
                forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
                forward_pass_config=forward_pass_config,
            )
            assert decoder_outputs.updated_state is not None

            chunk_logits = decoder_outputs.logits[jnp.arange(batch_size), chunk.sequence_ends - 1, :]
            new_logits = jnp.where(chunk.is_last_token_inside[:, None], chunk_logits, prev_logits)

            return (decoder_outputs.updated_state, new_logits), None

        (final_state, final_logits), _ = jax.lax.scan(apply_chunk, (state, logits_like), chunks)

        return PrefillResults(
            last_token_logits=final_logits,
            last_token_indices=jnp.maximum(lengths_without_padding - 1, 0),
            state=final_state,
        )

    def generate_tokens(
        self,
        prompt_token_ids: Int[Array, "batch prompt_tokens"],
        generation_config: GenerationConfig | None = None,
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

    def _generate_tokens_batch(
        self,
        batch: tuple[list[int], ...],
        batch_keys: tuple[Key[Array, ""], ...],
        *,
        generation_config: GenerationConfig | None,
        inference_config: InferenceConfig,
        forward_pass_config: ForwardPassConfig | None,
    ) -> Iterator[GenerationResults]:
        assert inference_config.batch_size is not None
        batch_size = inference_config.batch_size

        padded_token_ids = pad_sequences(batch, (batch_size, inference_config.padded_length), dtype=jnp.int32)

        lengths = jnp.array([len(tokens) for tokens in batch], dtype=jnp.int32)
        padded_lengths = jnp.pad(lengths, (0, batch_size - len(batch)))

        padded_keys = pad_keys_to_size(batch_keys, batch_size)

        generate_tokens_fn = compile_generate_tokens(
            self,
            generation_config,
            inference_config,
            forward_pass_config=forward_pass_config,
        )
        results = generate_tokens_fn(
            self,
            prompt_token_ids=padded_token_ids,
            prompt_lengths_without_padding=padded_lengths,
            keys=padded_keys,
        )
        for i in range(len(batch)):
            yield GenerationResults(
                token_ids=results.token_ids[i],
                top_k_token_ids=results.top_k_token_ids[i] if results.top_k_token_ids is not None else None,
                top_k_token_logits=results.top_k_token_logits[i] if results.top_k_token_logits is not None else None,
            )

    def generate_tokens_many(
        self,
        tokenized: Iterable[list[int]],
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
    ) -> Iterator[GenerationResults]:
        tokenized = list(tokenized)  # load eagerly to RAM

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(tokenized))

        if len(keys) != len(tokenized):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        def process_batches(batch_size: int) -> Iterator[tuple[int, GenerationResults]]:
            new_inference_config = replace(inference_config, batch_size=batch_size)

            for batch_items in batched(zip(tokenized, keys, strict=True), batch_size):
                real_batch, batch_keys = zip(*batch_items, strict=True)
                yield from self._generate_tokens_batch(
                    real_batch,
                    batch_keys,
                    generation_config=generation_config,
                    inference_config=new_inference_config,
                    forward_pass_config=forward_pass_config,
                )

        assert inference_config.batch_size is not None
        yield from decrease_batchsize_on_oom(
            process_batches,
            starting_batch_size=inference_config.batch_size,
        )

    def estimate_memory_consumption(
        self,
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
    ) -> int:
        memory_analysis = compile_generate_tokens(
            self,
            generation_config=generation_config,
            inference_config=inference_config,
            forward_pass_config=forward_pass_config,
        ).memory_analysis()

        assert hasattr(memory_analysis, "argument_size_in_bytes")
        assert hasattr(memory_analysis, "output_size_in_bytes")
        assert hasattr(memory_analysis, "temp_size_in_bytes")

        return (
            memory_analysis.argument_size_in_bytes
            + memory_analysis.output_size_in_bytes
            + memory_analysis.temp_size_in_bytes
        )

    def _trim_at_eos(self, token_ids: list[int]) -> list[int]:
        if not self.stop_token_ids:
            return token_ids
        stop_set = set(self.stop_token_ids)
        end = next((i for i, token_id in enumerate(token_ids) if token_id in stop_set), len(token_ids))
        return token_ids[: end + 1]

    def reply(
        self,
        messages: Iterable[Message],
        generation_config: GenerationConfig | None = None,
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        key: PRNGKeyArray | None = None,
    ) -> AssistantMessage:
        formatted_messages = self.message_processor.render_request(messages)
        token_ids = jnp.array(self.message_processor.tokenize_text(formatted_messages), dtype=jnp.int32)[None, :]
        response_ids = self.generate_tokens(
            token_ids,
            generation_config,
            forward_pass_config=forward_pass_config,
            keys=key[None, ...] if key is not None else None,
        ).token_ids.squeeze(0)
        trimmed_ids = self._trim_at_eos(response_ids.tolist())
        response_text = self.message_processor.detokenize(trimmed_ids)
        return self.message_processor.parse_response(response_text)

    def reply_many(
        self,
        messages: Iterable[Iterable[Message]],
        generation_config: GenerationConfig | None = None,
        inference_config: InferenceConfig = InferenceConfig(),  # noqa: B008
        *,
        forward_pass_config: ForwardPassConfig | None = None,
        keys: Key[Array, " num_sequences"] | None = None,
        vram_bytes: int | None = None,
        batch_sizes_callback: Callable[[BatchSizesComputedEvent], None] | None = None,
    ) -> Iterator[tuple[int, AssistantMessage]]:
        messages = list(messages)  # eagerly load the dataset into RAM

        if keys is None:
            keys = jax.random.split(jax.random.key(0), num=len(messages))

        if len(keys) != len(messages):
            raise ValueError(
                f"Length of 'keys' should be equal to the number of sequences passed or None; got {len(keys)}",
            )

        if vram_bytes is not None and inference_config.batch_size is not None:
            raise ValueError("You have to specify only one of batch_size and vram_gb, not both.")

        if vram_bytes is None and inference_config.batch_size is None:
            raise ValueError("You have to specify either batch_size or vram_gb, but you provided neither.")

        tokenized: list[list[int]] = self.message_processor.tokenize_requests(messages)

        buckets: dict[int, list[tuple[int, list[int]]]] = {}
        max_prompt_length = max(_COMPILED_PROMPT_LENGTHS)
        for idx, sequence in enumerate(tokenized):
            assert len(sequence) <= max_prompt_length, (
                f"Sequence length {len(sequence)} exceeds largest bucket {max_prompt_length}"
            )
            # we choose the smallest size from precomputed ones that is longer or equal to the current sequence
            padded_len = min(length for length in _COMPILED_PROMPT_LENGTHS if length >= len(sequence))
            buckets.setdefault(padded_len, []).append((idx, sequence))
        sorted_lengths = sorted(buckets.keys())

        if inference_config.batch_size is not None:
            batch_size_per_bucket = dict.fromkeys(sorted_lengths, inference_config.batch_size)
        else:
            batch_size_per_bucket = estimate_batchsizes_from_vram(
                lambda config: self.estimate_memory_consumption(inference_config=config),
                sorted_lengths,
                vram_bytes,  # type: ignore
                inference_config,
            )

        buckets = merge_small_buckets(buckets, batch_size_per_bucket, min_batches=2)
        assert sum(len(bucket) for bucket in buckets.values()) == len(tokenized)

        if batch_sizes_callback is not None:
            batch_sizes = tuple(
                BatchSizeInfo(
                    prefix_length=padded_length,
                    num_elements=len(buckets[padded_length]),
                    batch_size=batch_size_per_bucket.get(padded_length, 1),
                )
                for padded_length in sorted(buckets.keys())
            )
            batch_sizes_callback(BatchSizesComputedEvent(batch_sizes=batch_sizes))

        # Process longest sequences first so batchsize=1 OOM happens as early as possible, if it does happen
        for padded_length in sorted(buckets.keys(), reverse=True):
            sequence_ids, sequence_tokenized = zip(*buckets[padded_length], strict=True)
            sequence_ids = list(sequence_ids)
            batch_size = batch_size_per_bucket[padded_length]

            bucket_inference_config = replace(inference_config, batch_size=batch_size, padded_length=padded_length)

            all_results = self.generate_tokens_many(
                sequence_tokenized,
                generation_config=generation_config,
                inference_config=bucket_inference_config,
                forward_pass_config=forward_pass_config,
                keys=keys[np.array(sequence_ids)],
            )

            for idx, result in zip(sequence_ids, all_results, strict=True):
                trimmed_ids = self._trim_at_eos(result.token_ids.tolist())
                response = self.message_processor.parse_tokenized_response(trimmed_ids)
                yield (idx, response)

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
            padded_input_length + max_output_length,
            lengths_without_padding=jnp.array([input_length], dtype=jnp.int32),
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
