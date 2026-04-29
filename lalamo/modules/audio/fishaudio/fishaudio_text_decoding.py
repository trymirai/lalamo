from dataclasses import dataclass
from typing import Any

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, ShardingAxis
from lalamo.modules.audio.fishaudio.fishaudio_common import (
    default_fishaudio_sampling_policy,
)
from lalamo.modules.audio.fishaudio.fishaudio_consts import REPEAT_WINDOW_SIZE, SHORT_LOGITS_SIZE
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfig
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.token_mixer import State
from lalamo.modules.transformer import Transformer, TransformerConfig, TransformerForwardPassConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.sampling import SamplingPolicy


@dataclass
class FishAudioTextDecoderResult:
    token_codes: Float[Array, "batch codes"]
    hidden_states: Array | None
    state: State | None


@dataclass(frozen=True)
class FishAudioTextDecoderConfig(TTSTextDecoderConfig):
    slow_embeddings_config: TiedEmbeddingConfig
    slow_model_config: TransformerConfig
    slow_readout_config: LinearConfig

    fast_embeddings_config: TiedEmbeddingConfig
    fast_model_config: TransformerConfig
    fast_readout_config: LinearConfig

    codebook_embeddings_config: TiedEmbeddingConfig
    fast_model_projection_config: LinearConfig | None

    semantic_token_begin_id: int
    semantic_token_end_id: int
    im_end_token_id: int
    codebook_size: int
    vocab_size: int
    slow_model_dim: int
    fast_model_dim: int
    num_codebooks: int
    max_seq_len: int

    scale_codebook_embeddings: bool

    short_logits_size: int = SHORT_LOGITS_SIZE
    repeat_window_size: int = REPEAT_WINDOW_SIZE

    def init(self, initializer: Initializer) -> "FishAudioTextDecoder":
        embeddings_slow = self.slow_embeddings_config.init(
            initializer,
            model_dim=self.slow_model_dim,
            vocab_size=self.vocab_size,
        )
        embeddings_fast = self.fast_embeddings_config.init(
            initializer,
            model_dim=self.fast_model_dim,
            vocab_size=self.codebook_size,
        )
        codebook_embeddings = self.codebook_embeddings_config.init(
            initializer,
            model_dim=self.slow_model_dim,
            vocab_size=self.codebook_size * self.num_codebooks,
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.init(
                initializer,
                input_dim=self.slow_model_dim,
                output_dims=(self.fast_model_dim,),
                has_biases=False,
            )
        else:
            fast_model_projection = None

        assert isinstance(embeddings_slow, TiedEmbedding)
        assert isinstance(embeddings_fast, TiedEmbedding)
        assert isinstance(codebook_embeddings, TiedEmbedding)

        return FishAudioTextDecoder(
            config=self,
            embeddings_slow=embeddings_slow,
            transformer_slow=self.slow_model_config.init(initializer),
            readout_slow=self.slow_readout_config.init(
                initializer,
                input_dim=self.slow_model_dim,
                output_dims=(self.vocab_size,),
                has_biases=False,
            ),
            embeddings_fast=embeddings_fast,
            transformer_fast=self.fast_model_config.init(initializer),
            readout_fast=self.fast_readout_config.init(
                initializer,
                input_dim=self.fast_model_dim,
                output_dims=(self.codebook_size,),
                has_biases=False,
            ),
            codebook_embeddings=codebook_embeddings,
            fast_model_projection=fast_model_projection,
        )


class FishAudioTextDecoder(TTSTextDecoder[FishAudioTextDecoderConfig]):
    embeddings_slow: TiedEmbedding
    transformer_slow: Transformer
    readout_slow: Linear

    embeddings_fast: TiedEmbedding
    transformer_fast: Transformer
    readout_fast: Linear

    codebook_embeddings: TiedEmbedding
    fast_model_projection: Linear | None

    def embed_slow_model(self) -> Array:
        return jnp.zeros((1, 2, 3))

    def __call__(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_policy: SamplingPolicy,
        *,
        keychain: Keychain,
        input_pos: Int[Array, "batch tokens"] | None = None,
        state: State | None = None,
    ) -> FishAudioTextDecoderResult:
        batch_size, seq_length = text_tokens.shape
        if input_pos is None:
            input_pos = jnp.arange(seq_length)[None, :]

        text_and_codebooks = jnp.zeros(
            (batch_size, self.config.num_codebooks + 1, seq_length),
            dtype=text_tokens.dtype,
        )
        # NOTE: the rest of codebook lines should be filled in case audio prompt is used, but
        # ignore it for now
        text_and_codebooks = text_and_codebooks.at[:, 0, :].set(text_tokens)

        embed_keychain, decode_keychain = keychain.split()
        embeddings = self.embed(text_and_codebooks, keychain=embed_keychain)

        codes, updated_state = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=state,
            input_pos=input_pos,
            sampling_policy=sampling_policy,
            previous_tokens=None,
            keychain=decode_keychain,
        )

        return FishAudioTextDecoderResult(token_codes=codes, hidden_states=None, state=updated_state)

    def embed(
        self,
        inp: Int[Array, "batch codebooks tokens"],
        apply_codebook_embeddings: bool = False,
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens embedding"]:
        """
        apply_codebook_embeddings argument should be set to 'True' if audio-prompt is used. In this
        case we expect codebook lines [1:-1] to be filled with something meaningful
        """

        vq_masks = (inp[:, 0] >= self.config.semantic_token_begin_id) & (
            inp[:, 0] <= self.config.semantic_token_end_id
        )
        slow_embed_keychain, codebook_embed_keychain = keychain.split()
        embeddings = call_vmapped_twice(
            self.embeddings_slow.embed,
            inp[:, 0],
            keychain=slow_embed_keychain,
            added_sharding_axes=(ShardingAxis.DATA, None),
        )

        if apply_codebook_embeddings or jnp.any(vq_masks):
            _, _, seq_length = inp.shape
            codebook_offsets = (jnp.arange(self.config.num_codebooks) * self.config.codebook_size).reshape(-1, 1)
            codebook_offsets = jnp.tile(codebook_offsets, (1, seq_length))

            def embed_codebook_batch(
                codebook_ids: Int[Array, "codebooks tokens"],
                *,
                keychain: Keychain,
            ) -> Float[Array, "codebooks tokens embedding"]:
                return call_vmapped_twice(
                    self.codebook_embeddings.embed,
                    codebook_ids,
                    keychain=keychain,
                )

            codebook_embeds = call_vmapped(
                embed_codebook_batch,
                inp[:, 1:, :] + codebook_offsets,
                keychain=codebook_embed_keychain,
                added_sharding_axis=ShardingAxis.DATA,
            )

            vq_embeds_sum = codebook_embeds.sum(axis=1)
            vq_embeds_sum = vq_embeds_sum.at[~vq_masks].set(0)
            embeddings = embeddings + vq_embeds_sum

        if self.config.scale_codebook_embeddings:
            # Expand vq_masks to match x's shape
            vq_masks_expanded = jnp.expand_dims(vq_masks, axis=-1)
            vq_masks_expanded = jnp.broadcast_to(vq_masks_expanded, embeddings.shape)
            embeddings = jnp.where(vq_masks_expanded, embeddings / jnp.sqrt(self.config.num_codebooks + 1), embeddings)
            assert isinstance(embeddings, Array)

        return embeddings

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_policy: SamplingPolicy | None = None,
        *,
        keychain: Keychain,
    ) -> Int[Array, "num_codebooks tokens"]:
        """
        Generate semantic tokens for a full utterance given text tokens in an autoregressive
        generation loop. Processing text tokens through the slow transformer and generating
        codebook tokens until the end token is reached or max sequence length is exceeded.
        Returns:
            Generated codebook tokens for DAC codec
        """

        batch_size, prompt_length = text_tokens.shape
        assert batch_size == 1, "Only batch_size=1 is supported"

        codebook_dim = 1 + self.config.num_codebooks
        max_seq_len = self.config.max_seq_len

        if prompt_length >= max_seq_len:
            raise ValueError(f"Input sequence length {prompt_length} exceeds max_seq_len {max_seq_len}")

        if sampling_policy is None:
            sampling_policy = default_fishaudio_sampling_policy()

        max_new_tokens = max_seq_len - prompt_length

        # Prepare prompt: text tokens in first row
        # Rest of codebook rows are zeros until we start using audio-embeddings for explicit style
        prompt = jnp.zeros((batch_size, codebook_dim, prompt_length), dtype=text_tokens.dtype)
        prompt = prompt.at[:, 0, :].set(text_tokens)

        # Initialize sequence buffer to store generated tokens
        seq = jnp.zeros((codebook_dim, max_seq_len), dtype=jnp.int32)
        seq = seq.at[:, :prompt_length].set(prompt[0])

        # Track previous tokens for repetition penalty (windowed)
        previous_tokens = jnp.zeros((codebook_dim, max_seq_len), dtype=jnp.int32)

        input_pos = jnp.arange(prompt_length)[None, :]
        embed_keychain, decode_keychain = keychain.split()
        first_decode_keychain, loop_keychain = decode_keychain.split()
        embeddings = self.embed(prompt, keychain=embed_keychain)
        first_codes, state_slow = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=None,
            input_pos=input_pos,
            sampling_policy=sampling_policy,
            keychain=first_decode_keychain,
            previous_tokens=None,
        )

        seq = seq.at[:, prompt_length].set(first_codes[0])
        previous_tokens = previous_tokens.at[:, 0].set(first_codes[0])

        if first_codes[0, 0] == self.config.im_end_token_id:
            return seq[1:, prompt_length : prompt_length + 1]

        cur_token = first_codes
        generated_count = 1
        for i in range(1, max_new_tokens):
            cur_token_expanded = cur_token.reshape(batch_size, codebook_dim, 1)

            # Get windowed previous tokens for repetition penalty
            win_size = self.config.repeat_window_size
            if i < win_size:
                window = previous_tokens[:, :win_size]
            else:
                window = previous_tokens[:, i - win_size : i]

            loop_keychain, decode_keychain = loop_keychain.split()
            embeddings = self.embed(cur_token_expanded, keychain=embed_keychain)

            input_pos = jnp.array([[prompt_length + i - 1]])

            next_codes, state_slow = decode_next_token(
                model=self,
                x=embeddings,
                state_slow=state_slow,
                input_pos=input_pos,
                sampling_policy=sampling_policy,
                keychain=decode_keychain,
                previous_tokens=window,
            )

            seq = seq.at[:, prompt_length + i].set(next_codes[0])
            previous_tokens = previous_tokens.at[:, i].set(next_codes[0])
            generated_count += 1

            if next_codes[0, 0] == self.config.im_end_token_id:
                break

            cur_token = next_codes

        # Extract codebook codes (exclude text token row and prompt, exclude last token which is end token)
        codes = seq[1:, prompt_length : prompt_length + generated_count - 1]
        assert jnp.all(codes >= 0), "Negative code found"

        return codes


def decode_next_token(
    model: FishAudioTextDecoder,
    x: Array,
    state_slow: State | None,
    input_pos: Array,
    sampling_policy: SamplingPolicy,
    keychain: Keychain,
    previous_tokens: Array | None = None,  # noqa: ARG001, reserved for future when repetition penalty is done
) -> tuple[Int[Array, "batch codes"], State | None]:
    batch_size = x.shape[0]
    assert batch_size == 1, "Batching not supported yet"
    (
        slow_transformer_keychain,
        fast_projection_keychain,
        slow_readout_keychain,
        fast_transformer_keychain,
        fast_readout_keychain,
        fast_embed_keychain,
    ) = keychain.split(6)
    slow_sampling_keychain, loop_sampling_keychain = keychain.split()

    slow_model_result = model.transformer_slow(
        inner_features=x,
        token_positions=input_pos,
        state=state_slow,
        return_updated_state=True,
        return_layer_results=True,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        forward_pass_config=TransformerForwardPassConfig(),
        keychain=slow_transformer_keychain,
    )
    assert slow_model_result.layer_results is not None
    hidden_states = slow_model_result.layer_results[-1].outputs[:, -1:]
    if model.fast_model_projection is not None:
        (hidden_states,) = call_vmapped(
            model.fast_model_projection,
            hidden_states,
            keychain=fast_projection_keychain,
        )

    (logits,) = call_vmapped_twice(
        model.readout_slow,
        slow_model_result.outputs,
        keychain=slow_readout_keychain,
    )

    n_codes = model.config.num_codebooks + 1

    codebooks = jnp.zeros((batch_size, n_codes), dtype=jnp.int32)
    first_code = call_vmapped(
        sampling_policy,
        logits[:, -1, :],
        keychain=slow_sampling_keychain,
    )
    codebooks = codebooks.at[0, 0].set(first_code[0])
    first_fast_code = jnp.array([codebooks[0, 0] - model.config.semantic_token_begin_id])
    first_fast_code = first_fast_code.at[first_fast_code < 0].set(0)
    codebooks = codebooks.at[0, 1].set(first_fast_code[0])

    state_fast = model.transformer_fast.init_static_state(batch_size, n_codes, hidden_states.dtype)

    input_pos_fast = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    fast_first_result = model.transformer_fast(
        inner_features=hidden_states,
        token_positions=input_pos_fast,
        state=state_fast,
        return_updated_state=True,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        forward_pass_config=TransformerForwardPassConfig(),
        keychain=fast_transformer_keychain,
    )
    state_fast = fast_first_result.updated_state

    embedded_logits = call_vmapped(
        model.embeddings_fast.embed,
        first_fast_code,
        keychain=fast_embed_keychain,
        added_sharding_axis=ShardingAxis.DATA,
    )
    loop_sampling_keys = jax.random.split(loop_sampling_keychain.vmapped_keys, max(n_codes - 2, 0))

    def loop_iteration(
        iteration_state: tuple[State | None, Array, Any],
        loop_input: tuple[jnp.int32, Array],
    ) -> tuple[tuple[State | None, Array, Any], None]:
        index, sample_vmapped_key = loop_input
        transformer_state, logits, codebooks = iteration_state
        logits = logits.reshape(logits.shape[0], 1, -1)
        input_pos_fast = jnp.array([index - 1])[None, :]
        fast_result = model.transformer_fast(
            inner_features=logits,
            token_positions=input_pos_fast,
            state=transformer_state,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            forward_pass_config=TransformerForwardPassConfig(),
            keychain=fast_transformer_keychain,
        )
        (fast_logits,) = call_vmapped_twice(
            model.readout_fast,
            fast_result.outputs,
            keychain=fast_readout_keychain,
        )
        new_state = fast_result.updated_state

        short_logits = fast_logits[:, :, : model.config.short_logits_size]
        sample_vmapped_keys = jax.random.split(sample_vmapped_key, short_logits.shape[0])
        code = call_vmapped(
            sampling_policy,
            short_logits[:, -1, :],
            keychain=Keychain(vmapped_keys=sample_vmapped_keys, batch_key=keychain.batch_key),
        )

        new_logits = call_vmapped(
            model.embeddings_fast.embed,
            code,
            keychain=fast_embed_keychain,
            added_sharding_axis=ShardingAxis.DATA,
        )
        codebooks = codebooks.at[0, index].set(code[0])
        return (new_state, new_logits, codebooks), None

    scan_result, _ = jax.lax.scan(
        loop_iteration,
        (state_fast, embedded_logits, codebooks),
        (jnp.arange(2, n_codes, dtype=jnp.int32), loop_sampling_keys),
    )
    _, _, codebooks = scan_result

    return codebooks, slow_model_result.updated_state
