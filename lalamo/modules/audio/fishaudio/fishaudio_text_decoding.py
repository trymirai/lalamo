from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.activations import Identity
from lalamo.modules.audio.fishaudio.fishaudio_common import DEFAULT_FISH_AUDIO_SAMPLING_POLICY, fishaudio_logger
from lalamo.modules.audio.text_decoder import TTSTextDecoder
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


@dataclass(frozen=True)
class FishAudioTextDecoderConfig:
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

    # NOTE: magic constants from FishAudio code
    short_logits_size: int = 1024
    repeat_window_size: int = 16

    def empty(self) -> "FishAudioTextDecoder":
        embeddings_slow = self.slow_embeddings_config.empty(self.vocab_size, self.slow_model_dim)
        embeddings_fast = self.fast_embeddings_config.empty(self.codebook_size, self.fast_model_dim)
        codebook_embeddings = self.codebook_embeddings_config.empty(
            self.codebook_size * self.num_codebooks, self.slow_model_dim
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.empty(
                self.slow_model_dim, (self.fast_model_dim,), False
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
                input_dim=self.fast_model_dim, output_dims=(self.codebook_size,), has_biases=False
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
            vocab_size=self.vocab_size, model_dim=self.slow_model_dim, key=key_emb_slow
        )
        embeddings_fast = self.fast_embeddings_config.random_init(
            vocab_size=self.codebook_size, model_dim=self.fast_model_dim, key=key_emb_fast
        )
        codebook_embeddings = self.codebook_embeddings_config.random_init(
            vocab_size=self.codebook_size * self.num_codebooks, model_dim=self.slow_model_dim, key=key_emb_codebook
        )
        if self.fast_model_projection_config is not None:
            fast_model_projection = self.fast_model_projection_config.random_init(
                input_dim=self.slow_model_dim, output_dims=(self.fast_model_dim,), has_biases=False, key=key_fast_proj
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
                input_dim=self.slow_model_dim, output_dims=(self.vocab_size,), has_biases=False, key=key_readout_slow
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
        assert isinstance(weights, Mapping)
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
                require_tree(weights["fast_model_projection"])
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
            (batch_size, self.config.num_codebooks + 1, seq_length), dtype=text_tokens.dtype
        )
        # NOTE: the rest of codebook lines should be filled in case audio promt is used, but
        # ignore it for now
        text_and_codebooks = text_and_codebooks.at[:, 0, :].set(text_tokens)

        embeddings = self.embed(text_and_codebooks)
        codes, updated_state = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=state,
            input_pos=input_pos,
            sampling_policy=sampling_policy,
            previous_tokens=None,
            key=key,
        )
        return FishAudioTextDecoderResult(token_codes=codes, hidden_states=None, state=updated_state)

    def embed(
        self, inp: Int[Array, "batch codebooks tokens"], apply_codebook_embeddings: bool = False
    ) -> Float[Array, "batch tokens embedding"]:
        """
        apply_codebook_embeddings argumet should be set to 'True' if audio-prompt is used. In this
        case we expect codebook lines [1:-1] to be filled with something meaningful
        """

        vq_masks = (inp[:, 0] >= self.semantic_begin_id) & (inp[:, 0] <= self.semantic_end_id)
        embeddings = self.embeddings_slow.embed(inp[:, 0])

        if apply_codebook_embeddings or jnp.any(vq_masks):
            _, _, seq_length = inp.shape
            codebook_offsets = (jnp.arange(self.config.num_codebooks) * self.config.codebook_size).reshape(-1, 1)
            codebook_offsets = jnp.tile(codebook_offsets, (1, seq_length))
            codebook_embeds = vmap(self.codebook_embeddings.embed)(inp[:, 1:, :] + codebook_offsets)

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
            sampling_policy = DEFAULT_FISH_AUDIO_SAMPLING_POLICY
        if key is None:
            key = jax.random.PRNGKey(123)

        max_new_tokens = max_seq_len - prompt_length

        # Prepare prompt: text tokens in first row, zeros for codebook rows
        prompt = jnp.zeros((batch_size, codebook_dim, prompt_length), dtype=text_tokens.dtype)
        prompt = prompt.at[:, 0, :].set(text_tokens)

        # Initialize sequence buffer to store generated tokens
        seq = jnp.zeros((codebook_dim, max_seq_len), dtype=jnp.int32)
        seq = seq.at[:, :prompt_length].set(prompt[0])

        # Track previous tokens for repetition penalty (windowed)
        previous_tokens = jnp.zeros((codebook_dim, max_seq_len), dtype=jnp.int32)

        # Embed and generate first token
        input_pos = jnp.arange(prompt_length)[None, :]
        embeddings = self.embed(prompt)

        first_codes, state_slow = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=None,
            input_pos=input_pos,
            sampling_policy=sampling_policy,
            key=key,
            previous_tokens=None,
        )

        fishaudio_logger.debug(f"{0} : code={first_codes[0]}")

        seq = seq.at[:, prompt_length].set(first_codes[0])
        previous_tokens = previous_tokens.at[:, 0].set(first_codes[0])

        # Check for early termination
        if first_codes[0, 0] == self.config.im_end_token_id:
            codes = seq[1:, prompt_length : prompt_length + 1]
            return codes

        # Generate remaining tokens
        cur_token = first_codes
        generated_count = 1

        for i in range(1, max_new_tokens):
            # Prepare current token for embedding
            cur_token_expanded = cur_token.reshape(batch_size, codebook_dim, 1)

            # Get windowed previous tokens for repetition penalty
            win_size = self.config.repeat_window_size
            if i < win_size:
                window = previous_tokens[:, :win_size]
            else:
                window = previous_tokens[:, i - win_size : i]

            embeddings = self.embed(cur_token_expanded)

            input_pos = jnp.array([[prompt_length + i - 1]])

            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None

            next_codes, state_slow = decode_next_token(
                model=self,
                x=embeddings,
                state_slow=state_slow,
                input_pos=input_pos,
                sampling_policy=sampling_policy,
                key=subkey,
                previous_tokens=window,
            )

            seq = seq.at[:, prompt_length + i].set(next_codes[0])
            previous_tokens = previous_tokens.at[:, i].set(next_codes[0])
            generated_count += 1

            fishaudio_logger.debug(f"{i} : code={next_codes[0]}")

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
    key: PRNGKeyArray,
    previous_tokens: Array | None = None,  # noqa: ARG001, reserved for future when repetition penalty is done
) -> tuple[Int[Array, "batch codes"], State | None]:
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

    codebooks = [vmap(lambda x: sampling_policy(x, key=key))(logits[:, -1, :])]

    batch_size, *_ = x.shape
    input_pos_fast = jnp.zeros((batch_size, 1), dtype=jnp.int32)
    fast_first_result = model.transformer_fast(
        inner_features=hidden_states,
        token_positions=input_pos_fast,
        state=None,
        return_updated_state=True,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
        forward_pass_config=None,
    )
    state_fast = fast_first_result.updated_state
    first_code = codebooks[0] - model.semantic_begin_id
    first_code = first_code.at[first_code < 0].set(0)
    codebooks.append(first_code)

    hidden_states = model.embeddings_fast.embed(first_code)

    for codebook_idx in range(1, model.num_codebooks):
        hidden_states = hidden_states.reshape(hidden_states.shape[0], 1, -1)
        input_pos_fast = jnp.array([codebook_idx])[None, :]
        fast_result = model.transformer_fast(
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
        (fast_logits,) = vmap_twice(model.readout_fast)(fast_result.outputs)
        state_fast = fast_result.updated_state

        short_logits = fast_logits[:, :, : model.config.short_logits_size]

        code = vmap(lambda x: sampling_policy(x, key=key))(short_logits[:, -1, :])

        hidden_states = model.embeddings_fast.embed(code)
        codebooks.append(code)

    codebooks = jnp.stack(codebooks, axis=1)

    return codebooks, slow_model_result.updated_state
