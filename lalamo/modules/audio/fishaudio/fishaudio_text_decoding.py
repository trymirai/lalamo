from dataclasses import dataclass, replace
from typing import NamedTuple, Self

import jax
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int, Key, PRNGKeyArray

from lalamo.common import ParameterTree, require_mapping, require_tree
from lalamo.modules.activations import Identity
from lalamo.modules.audio.fishaudio.fishaudio_common import (
    default_fishaudio_sampling_policy,
)
from lalamo.modules.audio.fishaudio.fishaudio_consts import REPEAT_WINDOW_SIZE, SHORT_LOGITS_SIZE
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.token_mixers.state.common import State
from lalamo.modules.transformer import Transformer, TransformerConfig
from lalamo.modules.utils import vmap_twice
from lalamo.sampling import SamplingPolicy


@dataclass
class FishAudioTextDecoderResult:
    token_codes: Float[Array, "batch codes"]
    hidden_states: Array | None
    state: State | None


class DecodeNextTokenResult(NamedTuple):
    codes: Int[Array, "batch codebooks"]
    slow_state: State
    sampling_policies: SamplingPolicy


class DecodeUtteranceLoopState(NamedTuple):
    slow_state: State
    current_codes: Int[Array, " codebooks"]
    sampling_policies: SamplingPolicy
    key: Key[Array, ""]
    generated_count: Int[Array, ""]
    generated_codes: Int[Array, "tokens codebooks"]


def _init_fishaudio_sampling_policies(
    sampling_policy: SamplingPolicy,
    text_tokens: Int[Array, "batch tokens"],
    codebook_dim: int,
) -> SamplingPolicy:
    _, prompt_length = text_tokens.shape
    prompt_tokens_by_codebook = jnp.zeros((codebook_dim, prompt_length), dtype=text_tokens.dtype)
    prompt_tokens_by_codebook = prompt_tokens_by_codebook.at[0].set(text_tokens[0])
    prompt_lengths_by_codebook = jnp.zeros(codebook_dim, dtype=jnp.int32).at[0].set(prompt_length)
    return vmap(sampling_policy.init)(prompt_tokens_by_codebook, prompt_lengths_by_codebook)


@dataclass(frozen=True)
class FishAudioTextDecoderConfig(TTSTextDecoderConfigBase):
    slow_embeddings_config: TiedEmbeddingConfig
    slow_model_config: TransformerConfig
    slow_readout_config: FullPrecisionLinearConfig

    fast_embeddings_config: TiedEmbeddingConfig
    fast_model_config: TransformerConfig
    fast_readout_config: FullPrecisionLinearConfig

    codebook_embeddings_config: TiedEmbeddingConfig
    fast_model_projection_config: FullPrecisionLinearConfig | None

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

    precision: DTypeLike

    short_logits_size: int = SHORT_LOGITS_SIZE
    repeat_window_size: int = REPEAT_WINDOW_SIZE

    def empty(self) -> "FishAudioTextDecoder":
        embeddings_slow = self.slow_embeddings_config.empty(self.vocab_size, self.slow_model_dim)
        embeddings_fast = self.fast_embeddings_config.empty(self.codebook_size, self.fast_model_dim)
        codebook_embeddings = self.codebook_embeddings_config.empty(
            self.codebook_size * self.num_codebooks,
            self.slow_model_dim,
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.empty(
                input_dim=self.slow_model_dim,
                output_dims=(self.fast_model_dim,),
                has_biases=False,
            )
        else:
            fast_model_projection = Identity()
        assert isinstance(embeddings_slow, TiedEmbedding)
        assert isinstance(embeddings_fast, TiedEmbedding)
        assert isinstance(codebook_embeddings, TiedEmbedding)
        return FishAudioTextDecoder(
            self,
            embeddings_slow=embeddings_slow,
            transformer_slow=self.slow_model_config.empty(),
            readout_slow=self.slow_readout_config.empty(
                input_dim=self.slow_model_dim,
                output_dims=(self.vocab_size,),
                has_biases=False,
            ),
            embeddings_fast=embeddings_fast,
            transformer_fast=self.fast_model_config.empty(),
            readout_fast=self.fast_readout_config.empty(
                input_dim=self.fast_model_dim,
                output_dims=(self.codebook_size,),
                has_biases=False,
            ),
            codebook_embeddings=codebook_embeddings,
            fast_model_projection=fast_model_projection,
        )

    def random_init(self, *, key: PRNGKeyArray) -> "FishAudioTextDecoder":
        (
            key_emb_slow,
            key_transformer_slow,
            key_readout_slow,
            key_emb_fast,
            key_transformer_fast,
            key_readout_fast,
            key_emb_codebook,
            key_fast_proj,
        ) = jax.random.split(key, 8)

        embeddings_slow = self.slow_embeddings_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.slow_model_dim,
            key=key_emb_slow,
        )
        embeddings_fast = self.fast_embeddings_config.random_init(
            vocab_size=self.codebook_size,
            model_dim=self.fast_model_dim,
            key=key_emb_fast,
        )
        codebook_embeddings = self.codebook_embeddings_config.random_init(
            vocab_size=self.codebook_size * self.num_codebooks,
            model_dim=self.slow_model_dim,
            key=key_emb_codebook,
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.random_init(
                input_dim=self.slow_model_dim,
                output_dims=(self.fast_model_dim,),
                has_biases=False,
                key=key_fast_proj,
            )
        else:
            fast_model_projection = Identity()

        assert isinstance(embeddings_slow, TiedEmbedding)
        assert isinstance(embeddings_fast, TiedEmbedding)
        assert isinstance(codebook_embeddings, TiedEmbedding)

        return FishAudioTextDecoder(
            self,
            embeddings_slow=embeddings_slow,
            transformer_slow=self.slow_model_config.random_init(key=key_transformer_slow),
            readout_slow=self.slow_readout_config.random_init(
                input_dim=self.slow_model_dim,
                output_dims=(self.vocab_size,),
                has_biases=False,
                key=key_readout_slow,
            ),
            embeddings_fast=embeddings_fast,
            transformer_fast=self.fast_model_config.random_init(key=key_transformer_fast),
            readout_fast=self.fast_readout_config.random_init(
                input_dim=self.fast_model_dim,
                output_dims=(self.codebook_size,),
                has_biases=False,
                key=key_readout_fast,
            ),
            codebook_embeddings=codebook_embeddings,
            fast_model_projection=fast_model_projection,
        )


class FishAudioTextDecoder(TTSTextDecoder[FishAudioTextDecoderConfig]):
    embeddings_slow: TiedEmbedding
    transformer_slow: Transformer
    readout_slow: FullPrecisionLinear

    embeddings_fast: TiedEmbedding
    transformer_fast: Transformer
    readout_fast: FullPrecisionLinear

    codebook_embeddings: TiedEmbedding
    fast_model_projection: FullPrecisionLinear | Identity

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def export_weights(self) -> ParameterTree:
        if isinstance(self.fast_model_projection, Identity):
            fast_model_proj_weighs = {}
        else:
            fast_model_proj_weighs = self.fast_model_projection.export_weights()

        return {
            "embeddings_slow": self.embeddings_slow.export_weights(),
            "embeddings_fast": self.embeddings_fast.export_weights(),
            "transformer_slow": self.transformer_slow.export_weights(),
            "transformer_fast": self.transformer_fast.export_weights(),
            "readout_slow": self.readout_slow.export_weights(),
            "readout_fast": self.readout_fast.export_weights(),
            "codebook_embeddings": self.codebook_embeddings.export_weights(),
            "fast_model_projection": fast_model_proj_weighs,
        }

    def import_weights(self, weights: ParameterTree) -> Self:
        weights = require_mapping(weights)
        return replace(
            self,
            embeddings_slow=self.embeddings_slow.import_weights(require_tree(weights["embeddings_slow"])),
            embeddings_fast=self.embeddings_fast.import_weights(require_tree(weights["embeddings_fast"])),
            transformer_slow=self.transformer_slow.import_weights(require_tree(weights["transformer_slow"])),
            transformer_fast=self.transformer_fast.import_weights(require_tree(weights["transformer_fast"])),
            readout_slow=self.readout_slow.import_weights(require_tree(weights["readout_slow"])),
            readout_fast=self.readout_fast.import_weights(require_tree(weights["readout_fast"])),
            codebook_embeddings=self.codebook_embeddings.import_weights(require_tree(weights["codebook_embeddings"])),
            fast_model_projection=self.fast_model_projection.import_weights(
                require_tree(weights["fast_model_projection"]),
            )
            if isinstance(self.fast_model_projection, FullPrecisionLinear)
            else Identity(),
        )

    @property
    def semantic_begin_id(self) -> int:
        return self.config.semantic_token_begin_id

    @property
    def semantic_end_id(self) -> int:
        return self.config.semantic_token_end_id

    @property
    def num_codebooks(self) -> int:
        return self.config.num_codebooks

    def embed_slow_model(self) -> Array:
        return jnp.zeros((1, 2, 3))

    def __call__(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_policy: SamplingPolicy,
        key: PRNGKeyArray,
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

        embeddings = self.embed(text_and_codebooks)
        sampling_policies = _init_fishaudio_sampling_policies(
            sampling_policy,
            text_tokens,
            self.config.num_codebooks + 1,
        )

        decoding_result = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=state,
            input_pos=input_pos,
            sampling_policies=sampling_policies,
            key=key,
        )

        return FishAudioTextDecoderResult(
            token_codes=decoding_result.codes,
            hidden_states=None,
            state=decoding_result.slow_state,
        )

    def embed(
        self,
        inp: Int[Array, "batch codebooks tokens"],
    ) -> Float[Array, "batch tokens embedding"]:
        vq_masks = (inp[:, 0] >= self.semantic_begin_id) & (inp[:, 0] <= self.semantic_end_id)
        embeddings = self.embeddings_slow.embed(inp[:, 0])

        _, _, seq_length = inp.shape
        codebook_offsets = (jnp.arange(self.config.num_codebooks) * self.config.codebook_size).reshape(-1, 1)
        codebook_offsets = jnp.tile(codebook_offsets, (1, seq_length))
        codebook_embeds = vmap(self.codebook_embeddings.embed)(inp[:, 1:, :] + codebook_offsets)
        vq_embeds_sum = codebook_embeds.sum(axis=1)
        vq_embeds_sum = jnp.where(vq_masks[..., None], vq_embeds_sum, 0)
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
        key: PRNGKeyArray | None = None,
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
        if key is None:
            key = jax.random.PRNGKey(123)

        max_new_tokens = max_seq_len - prompt_length
        im_end_id = self.config.im_end_token_id

        # Prepare prompt: text tokens in first row
        # Rest of codebook rows are zeros until we start using audio-embeddings for explicit style
        prompt = jnp.zeros((batch_size, codebook_dim, prompt_length), dtype=text_tokens.dtype)
        prompt = prompt.at[:, 0, :].set(text_tokens)

        initial_slow_state = self.transformer_slow.init_static_state(batch_size, max_seq_len)

        input_pos = jnp.arange(prompt_length)[None, :]
        embeddings = self.embed(prompt)
        sampling_policies = _init_fishaudio_sampling_policies(sampling_policy, text_tokens, codebook_dim)

        first_token_result = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=initial_slow_state,
            input_pos=input_pos,
            sampling_policies=sampling_policies,
            key=key,
        )
        first_codes = first_token_result.codes

        if first_codes[0, 0] == im_end_id:
            return first_codes[0, 1:][:, None]

        max_remaining = max_new_tokens - 1
        out_buf = jnp.zeros((max_remaining, codebook_dim), dtype=jnp.int32)
        first_codes_unbatched = first_codes[0]

        def cond_fn(loop_state: DecodeUtteranceLoopState) -> Bool[Array, ""]:
            return (loop_state.generated_count < max_remaining) & (loop_state.current_codes[0] != im_end_id)

        def body_fn(loop_state: DecodeUtteranceLoopState) -> DecodeUtteranceLoopState:
            new_key, subkey = jax.random.split(loop_state.key)
            embeddings = self.embed(loop_state.current_codes[None, :, None])
            input_pos = (prompt_length + loop_state.generated_count)[None, None]
            next_token_result = decode_next_token(
                model=self,
                x=embeddings,
                state_slow=loop_state.slow_state,
                input_pos=input_pos,
                sampling_policies=loop_state.sampling_policies,
                key=subkey,
            )
            next_codes = next_token_result.codes[0]
            return DecodeUtteranceLoopState(
                slow_state=next_token_result.slow_state,
                current_codes=next_codes,
                sampling_policies=next_token_result.sampling_policies,
                key=new_key,
                generated_count=loop_state.generated_count + 1,
                generated_codes=loop_state.generated_codes.at[loop_state.generated_count].set(next_codes),
            )

        initial_loop_state = DecodeUtteranceLoopState(
            slow_state=first_token_result.slow_state,
            current_codes=first_codes_unbatched,
            sampling_policies=first_token_result.sampling_policies,
            key=key,
            generated_count=jnp.int32(0),
            generated_codes=out_buf,
        )
        final_loop_state = jax.lax.while_loop(cond_fn, body_fn, initial_loop_state)

        # If the loop exited because the freshly-generated token was im_end, that token sits at
        # final_buf[final_i - 1]; strip it so we only return the audio codebooks.
        ended_with_im_end = bool(final_loop_state.current_codes[0] == im_end_id)
        valid_count = int(final_loop_state.generated_count) - (1 if ended_with_im_end else 0)

        all_tokens = jnp.concatenate(
            [first_codes_unbatched[None, :], final_loop_state.generated_codes[:valid_count]],
            axis=0,
        )
        codes = all_tokens[:, 1:].T
        assert jnp.all(codes >= 0), "Negative code found"
        return codes


def decode_next_token(
    model: FishAudioTextDecoder,
    x: Array,
    state_slow: State | None,
    input_pos: Array,
    sampling_policies: SamplingPolicy,
    key: Key[Array, ""],
) -> DecodeNextTokenResult:
    batch_size = x.shape[0]
    assert batch_size == 1, "Batching not supported yet"

    slow_model_result = model.transformer_slow(
        inner_features=x,
        token_positions=input_pos,
        state=state_slow,
        return_updated_state=True,
        return_layer_results=True,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        forward_pass_config=None,
    )
    assert slow_model_result.layer_results is not None
    hidden_states = slow_model_result.layer_results[-1].outputs[:, -1:]
    (hidden_states,) = vmap(model.fast_model_projection)(hidden_states)
    hidden_states = hidden_states.reshape(hidden_states.shape[0], 1, -1)

    (logits,) = vmap_twice(model.readout_slow)(slow_model_result.outputs)

    n_codes = model.num_codebooks + 1

    codebooks = jnp.zeros((batch_size, n_codes), dtype=jnp.int32)
    slow_sampling_policy = jax.tree.map(lambda leaf: leaf[0], sampling_policies)
    semantic_token = slow_sampling_policy(logits[0, -1, :], key=key)
    codebooks = codebooks.at[0, 0].set(semantic_token)
    first_fast_code = jnp.array([codebooks[0, 0] - model.semantic_begin_id])
    first_fast_code = jnp.where(first_fast_code < 0, 0, first_fast_code)
    codebooks = codebooks.at[0, 1].set(first_fast_code[0])

    state_fast = model.transformer_fast.init_static_state(batch_size, n_codes)

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
        forward_pass_config=None,
    )
    state_fast = fast_first_result.updated_state
    assert state_fast is not None

    embedded_logits = model.embeddings_fast.embed(first_fast_code)

    def loop_iteration(
        iteration_state: tuple[State, Array, Int[Array, "batch codebooks"]],
        index: Int[Array, ""],
    ) -> tuple[tuple[State, Array, Int[Array, "batch codebooks"]], None]:
        transformer_state, fast_embedding, codebooks = iteration_state
        fast_embedding = fast_embedding.reshape(fast_embedding.shape[0], 1, -1)
        input_pos_fast = jnp.array([index - 1])[None, :]
        fast_result = model.transformer_fast(
            inner_features=fast_embedding,
            token_positions=input_pos_fast,
            state=transformer_state,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            forward_pass_config=None,
        )
        (fast_logits,) = vmap_twice(model.readout_fast)(fast_result.outputs)
        new_state = fast_result.updated_state
        assert new_state is not None

        short_logits = fast_logits[:, :, : model.config.short_logits_size]
        fast_sampling_policy = jax.tree.map(lambda leaf: leaf[index], sampling_policies)
        code = fast_sampling_policy(short_logits[0, -1, :], key=key)

        new_embedding = model.embeddings_fast.embed(code[None])
        codebooks = codebooks.at[0, index].set(code)
        return (new_state, new_embedding, codebooks), None

    scan_result, _ = jax.lax.scan(
        loop_iteration,
        (state_fast, embedded_logits, codebooks),
        jnp.arange(2, n_codes, dtype=jnp.int32),
    )
    _, _, codebooks = scan_result
    updated_sampling_policies = vmap(lambda policy, code: policy.update(code))(sampling_policies, codebooks[0])

    assert slow_model_result.updated_state is not None
    return DecodeNextTokenResult(
        codes=codebooks,
        slow_state=slow_model_result.updated_state,
        sampling_policies=updated_sampling_policies,
    )
