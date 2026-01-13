import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax
import torch
from fish_speech.models.text2semantic.llama import BaseModelArgs, DualARModelArgs, DualARTransformer, NaiveModelArgs
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer
from jax import numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray
from tokenizers import Tokenizer
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

from lalamo.common import ParameterPath, ParameterTree
from lalamo.model_import.loaders.fishaudio_loaders import load_fish_audio_text_decoding_modules
from lalamo.model_import.loaders.huggingface import load_linear, load_tied_embedding
from lalamo.modules import (
    AttentionConfig,
    DenseMLPConfig,
    ForwardPassMode,
    FullPrecisionLinear,
    FullPrecisionLinearConfig,
    Identity,
    NormalizationConfig,
    SiLU,
    State,
    TiedEmbedding,
    TiedEmbeddingConfig,
    Transformer,
    TransformerConfig,
    TransformerLayerConfig,
    UpcastMode,
)
from lalamo.modules.audio.text_decoder import TTSTextDecoder
from lalamo.modules.torch_interop import torch_to_jax
from lalamo.modules.utils import vmap_twice
from lalamo.utils import MapDictValues

from .fishaudio_common import cast_if_float
from .fishaudio_modules import RoPEConfigFishAudio
from .fishaudio_sampling import FishAudioSamplingParams, sample
from .fishaudio_thin_wrapper import FishAudioTextDecoderConfig_Foreign


def load_tokenizer_from_fish_audio(path_to_chkpt: str) -> Tokenizer:
    output_temp_dir = tempfile.mkdtemp()
    try:
        fishspeech_tokenizer = FishTokenizer.from_pretrained(path_to_chkpt)

        convert_tiktoken_to_fast(fishspeech_tokenizer.tkt_model, output_temp_dir)
        tokenizer = Tokenizer.from_file(output_temp_dir + "/tokenizer.json")
        return tokenizer
    finally:
        shutil.rmtree(output_temp_dir)


class ConfigMapping:
    @staticmethod
    def extract_fast_transformer_params(fish_transformer_config: DualARModelArgs) -> BaseModelArgs:
        return BaseModelArgs(
            model_type=fish_transformer_config.model_type,
            vocab_size=fish_transformer_config.vocab_size,
            n_layer=fish_transformer_config.n_fast_layer,
            n_head=fish_transformer_config.fast_n_head,
            dim=fish_transformer_config.fast_dim,
            intermediate_size=fish_transformer_config.fast_intermediate_size,
            n_local_heads=fish_transformer_config.fast_n_local_heads,
            head_dim=fish_transformer_config.fast_head_dim,
            rope_base=fish_transformer_config.rope_base,
            norm_eps=fish_transformer_config.norm_eps,
            max_seq_len=fish_transformer_config.max_seq_len,
            dropout=fish_transformer_config.dropout,
            tie_word_embeddings=fish_transformer_config.tie_word_embeddings,
            attention_qkv_bias=fish_transformer_config.fast_attention_qkv_bias,
            attention_o_bias=fish_transformer_config.fast_attention_o_bias,
            attention_qk_norm=fish_transformer_config.fast_attention_qk_norm,
            codebook_size=fish_transformer_config.codebook_size,
            num_codebooks=fish_transformer_config.num_codebooks,
        )

    @staticmethod
    def lalamo_transformer_cfg_from_fish_text_decoder_cfg(
        config: BaseModelArgs, precision: DTypeLike
    ) -> tuple[TransformerConfig, FullPrecisionLinearConfig]:
        global_rope_config = RoPEConfigFishAudio(
            precision=precision,
            base=config.rope_base,
            max_sequence_length=config.max_seq_len,
        )
        local_rope_config = None

        norm_config = NormalizationConfig(
            scale_precision=precision,
            accumulation_precision=precision,
            epsilon=config.norm_eps,
            scale_offset=None,
            upcast_mode=UpcastMode.ONLY_NORMALIZATION,
            subtract_mean=False,
        )

        qkv_projection_config = FullPrecisionLinearConfig(precision=precision)
        out_projection_config = FullPrecisionLinearConfig(precision=precision)
        mixer_config = AttentionConfig(
            qkv_projection_config=qkv_projection_config,
            out_projection_config=out_projection_config,
            query_norm_config=norm_config if config.attention_qk_norm else None,
            key_norm_config=norm_config if config.attention_qk_norm else None,
            num_heads=config.n_head,
            num_groups=config.n_local_heads,
            head_dim=config.head_dim,
            is_causal=True,
            scale=None,
            sliding_window_size=None,
            logit_soft_cap=None,
            has_sinks=False,
            has_qkv_biases=False,
            has_out_biases=False,
        )

        mlp_linear_config = FullPrecisionLinearConfig(precision=precision)
        mlp_use_up_biases = False
        mlp_use_down_biases = False
        mlp_config = DenseMLPConfig(
            linear_config=mlp_linear_config,
            activation=SiLU(),
            has_up_biases=mlp_use_up_biases,
            has_down_biases=mlp_use_down_biases,
            gate_clipping=None,
            up_clipping=None,
        )

        pre_mixer_norm_config = norm_config
        post_mixer_norm_config = None
        pre_mlp_norm_config = norm_config
        post_mlp_norm_config = None

        layer_config = TransformerLayerConfig(
            pre_mixer_norm_config=pre_mixer_norm_config,
            mixer_config=mixer_config,
            post_mixer_norm_config=post_mixer_norm_config,
            pre_mlp_norm_config=pre_mlp_norm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=post_mlp_norm_config,
        )
        model_dim = config.dim
        hidden_dim = config.intermediate_size
        context_length = config.max_seq_len

        transformer_cfg = TransformerConfig(
            global_rope_config=global_rope_config,
            local_rope_config=local_rope_config,
            layer_configs=tuple([layer_config] * config.n_layer),
            output_norm_config=norm_config,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )
        linear_out_cfg = FullPrecisionLinearConfig(precision=precision)

        return (transformer_cfg, linear_out_cfg)


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

    @classmethod
    def from_fish_audio_config(
        cls,
        fish_audio_cfg: DualARModelArgs,
        tokenizer: FishTokenizer,
        precision: DTypeLike,
    ) -> "FishAudioTextDecoderConfig":
        slow_transformer_cfg, slow_readout_cfg = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(
            fish_audio_cfg, precision
        )
        fast_fish_cfg = ConfigMapping.extract_fast_transformer_params(fish_audio_cfg)
        fast_transformer_cfg, fast_readout_cfg = ConfigMapping.lalamo_transformer_cfg_from_fish_text_decoder_cfg(
            fast_fish_cfg, precision
        )

        slow_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
        fast_embedding_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)

        codebook_embeddings_cfg = TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision)
        if fish_audio_cfg.dim == fish_audio_cfg.fast_dim:
            fast_model_projection_config = None
        else:
            fast_model_projection_config = FullPrecisionLinearConfig(precision)

        assert fish_audio_cfg.fast_dim is not None
        return FishAudioTextDecoderConfig(
            slow_embeddings_config=slow_embedding_cfg,
            slow_model_config=slow_transformer_cfg,
            slow_readout_config=slow_readout_cfg,
            fast_embeddings_config=fast_embedding_cfg,
            fast_model_config=fast_transformer_cfg,
            fast_readout_config=fast_readout_cfg,
            codebook_embeddings_config=codebook_embeddings_cfg,
            fast_model_projection_config=fast_model_projection_config,
            semantic_token_begin_id=tokenizer.semantic_begin_id,
            semantic_token_end_id=tokenizer.semantic_end_id,
            im_end_token_id=tokenizer.get_token_id(IM_END_TOKEN),
            codebook_size=fish_audio_cfg.codebook_size,
            precision=precision,
            vocab_size=fish_audio_cfg.vocab_size,
            slow_model_dim=fish_audio_cfg.dim,
            fast_model_dim=fish_audio_cfg.fast_dim,
            num_codebooks=fish_audio_cfg.num_codebooks,
            max_seq_len=fish_audio_cfg.max_seq_len,
            scale_codebook_embeddings=fish_audio_cfg.scale_codebook_embeddings,
        )

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

    @classmethod
    def from_foreign_model(
        cls, fish_model_or_path: Path | DualARTransformer, precision: DTypeLike
    ) -> "FishAudioTextDecoder":
        if isinstance(fish_model_or_path, Path):
            fish_tokenizer = FishTokenizer.from_pretrained(str(fish_model_or_path))
            fish_model_cfg: DualARModelArgs | NaiveModelArgs = BaseModelArgs.from_pretrained(
                str(fish_model_or_path / "config.json")
            )
            assert isinstance(fish_model_cfg, DualARModelArgs)

            fish_model = FishAudioTextDecoderConfig_Foreign._load_fish_model(fish_model_or_path, fish_model_cfg)
            fish_model_dict = fish_model.state_dict()
        else:
            fish_tokenizer = fish_model_or_path.tokenizer
            fish_model_cfg = fish_model_or_path.config
            fish_model_dict = fish_model_or_path.state_dict()

        assert isinstance(fish_model_cfg, DualARModelArgs)
        config = cls.from_fish_audio_config(fish_model_cfg, fish_tokenizer, precision)
        weights_mapping = dict(MapDictValues(lambda v: cast_if_float(torch_to_jax(v), precision), fish_model_dict))

        transformer_slow, readout_slow = load_fish_audio_text_decoding_modules(
            config.slow_model_config.empty(),
            config.slow_readout_config.empty(
                input_dim=config.slow_model_dim,
                output_dims=(config.vocab_size,),
                has_biases=False,
            ),
            weights_mapping,
            fast=False,
        )
        transformer_fast, readout_fast = load_fish_audio_text_decoding_modules(
            config.fast_model_config.empty(),
            config.fast_readout_config.empty(
                input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
            ),
            weights_mapping,
            fast=True,
        )
        embeddings_slow = load_tied_embedding(
            config.slow_embeddings_config.empty(config.vocab_size, config.slow_model_dim),
            weights_mapping,
            ParameterPath("embeddings"),
        )
        embeddings_fast = load_tied_embedding(
            config.fast_embeddings_config.empty(config.codebook_size, config.fast_model_dim),
            weights_mapping,
            ParameterPath("fast_embeddings"),
        )

        codebook_embeddings = load_tied_embedding(
            config.codebook_embeddings_config.empty(
                config.codebook_size * config.num_codebooks, config.slow_model_dim
            ),
            weights_mapping,
            ParameterPath("codebook_embeddings"),
        )

        if config.fast_model_projection_config is not None:
            fast_model_projection = load_linear(
                config.fast_model_projection_config.empty(
                    input_dim=config.fast_model_dim, output_dims=(config.codebook_size,), has_biases=False
                ),
                weights_mapping,
                ParameterPath("fast_project_in"),
            )
            assert isinstance(fast_model_projection, FullPrecisionLinear)
        else:
            fast_model_projection = Identity()

        return FishAudioTextDecoder(
            config=config,
            embeddings_slow=embeddings_slow,
            transformer_slow=transformer_slow,
            readout_slow=readout_slow,
            embeddings_fast=embeddings_fast,
            transformer_fast=transformer_fast,
            readout_fast=readout_fast,
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
        # TODO(peter.glushkov): implement me
        return self

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
        input_pos: Int[Array, "batch tokens"] | None = None,
        state: State | None = None,
        sampling_params: FishAudioSamplingParams | None = None,
        key: PRNGKeyArray | None = None,
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

        if sampling_params is None:
            sampling_params = FishAudioSamplingParams(
                temperature=0.808, top_p=0.808, repetition_penalty=1.1016, argmax_decoding=True
            )
        embeddings = self.embed(text_and_codebooks)
        codes, updated_state = decode_next_token(
            model=self,
            x=embeddings,
            state_slow=state,
            input_pos=input_pos,
            sampling_params=sampling_params,
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
        sampling_params: FishAudioSamplingParams | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, "num_codebooks tokens"]:
        """
        Generate semantic tokens for a full utterance given text tokens.

        This function implements the autoregressive generation loop, processing text tokens
        through the slow transformer and generating codebook tokens until the end token
        is reached or max sequence length is exceeded.

        Args:
            text_tokens: Input text tokens with shape (batch, tokens). Currently only batch=1 is supported.
            sampling_params: Sampling parameters including temperature, top_p, and argmax_decoding flag.
            key: Optional PRNG key for sampling. Required if argmax_decoding is False.

        Returns:
            Generated codebook tokens with shape (num_codebooks, generated_tokens).
        """
        assert sampling_params is not None, "sampling_params must always be specified for FishAudioTextDecoder"
        assert sampling_params.argmax_decoding or key is not None, "PRNG key required for non-argmax decoding"

        batch_size, prompt_length = text_tokens.shape
        assert batch_size == 1, "Only batch_size=1 is supported"

        codebook_dim = 1 + self.config.num_codebooks
        max_seq_len = self.config.max_seq_len

        if prompt_length >= max_seq_len:
            raise ValueError(f"Input sequence length {prompt_length} exceeds max_seq_len {max_seq_len}")

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
            sampling_params=sampling_params,
            previous_tokens=None,
            key=key,
        )

        # TODO: remove after debugging is done
        print(f"{0} : code={first_codes[0]}")

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
            # print(f" ### MY_DBG: decoding token {i}")

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
                sampling_params=sampling_params,
                previous_tokens=window,
                key=subkey,
            )

            seq = seq.at[:, prompt_length + i].set(next_codes[0])
            previous_tokens = previous_tokens.at[:, i].set(next_codes[0])
            generated_count += 1

            # TODO: remove after debugging is done
            print(f"{i} : code={next_codes[0]}")

            if next_codes[0, 0] == self.config.im_end_token_id:
                break

            cur_token = next_codes

        # Extract codebook codes (exclude text token row and prompt, exclude last token which is end token)
        codes = seq[1:, prompt_length : prompt_length + generated_count - 1]
        assert jnp.all(codes >= 0), "Negative code found"

        return codes


@torch.no_grad
def decode_next_token(
    model: FishAudioTextDecoder,
    x: Array,
    state_slow: State | None,
    input_pos: Array,
    sampling_params: FishAudioSamplingParams,
    previous_tokens: Array | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[Int[Array, "batch codes"], State | None]:
    assert sampling_params.argmax_decoding or key is not None

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

    if sampling_params.argmax_decoding:
        codebooks = [logits[:, -1].argmax(axis=-1)]
    else:
        codebooks = [
            sample(
                logits,
                key,
                sampling_params,
                previous_tokens=(previous_tokens[:, 0] if previous_tokens is not None else None),
            )[0].reshape(1)  # NOTE: reshaping to ensure its tensor, not scalar
        ]

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

        if sampling_params.argmax_decoding:
            code = short_logits[:, -1].argmax(axis=-1)
        else:
            code = sample(
                short_logits,
                key,
                sampling_params,
                previous_tokens=(previous_tokens[codebook_idx + 1] if previous_tokens is not None else None),
            )[0].reshape(1)  # NOTE: reshaping to ensure its tensor, not scalar

        hidden_states = model.embeddings_fast.embed(code)
        codebooks.append(code)

    codebooks = jnp.stack(codebooks, axis=1)

    return codebooks, slow_model_result.updated_state
