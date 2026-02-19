from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.activations import SiLU
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.mlp import DenseMLPConfig
from lalamo.modules.normalization import NormalizationConfig, UpcastMode
from lalamo.modules.rope import UnscaledRoPEConfig
from lalamo.modules.token_mixers import AttentionConfig
from lalamo.modules.transformer import Transformer, TransformerConfig
from lalamo.modules.transformer_layer import TransformerLayerConfig
from lalamo.sampling import SamplingPolicy, make_policy

__all__ = [
    "Qwen3TTSTextDecoder",
    "Qwen3TTSTextDecoderConfig",
    "default_qwen3_tts_text_decoder_config",
]


def default_qwen3_tts_text_sampling_policy() -> SamplingPolicy:
    return make_policy(temperature=0.9, top_p=1.0)


def _embed_tokens(
    embedding: TiedEmbedding,
    token_ids: Int[Array, "batch tokens"],
) -> Float[Array, "batch tokens channels"]:
    return vmap(embedding.embed)(token_ids)


def _apply_linear_ntc(
    linear: FullPrecisionLinear,
    x: Float[Array, "batch tokens channels"],
) -> Float[Array, "batch tokens channels"]:
    (y,) = vmap(vmap(linear))(x)
    return y


def _run_transformer(
    transformer: Transformer,
    x: Float[Array, "batch tokens channels"],
) -> Float[Array, "batch tokens channels"]:
    batch_size, seq_length, _ = x.shape
    token_positions = jnp.broadcast_to(jnp.arange(seq_length, dtype=jnp.int32)[None, :], (batch_size, seq_length))
    result = transformer(
        inner_features=x,
        token_positions=token_positions,
        state=None,
        return_updated_state=False,
        return_layer_results=False,
        return_positional_embeddings=False,
        lengths_without_padding=None,
        forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        forward_pass_config=None,
    )
    return result.outputs


def _sample_token_ids(
    logits: Float[Array, "batch vocabulary"],
    sampling_policy: SamplingPolicy,
    key: PRNGKeyArray,
) -> Int[Array, " batch"]:
    logits = logits.astype(jnp.float32)
    processed_logits = vmap(sampling_policy.process_logits)(logits)
    sample_keys = jax.random.split(key, logits.shape[0])
    return jax.vmap(lambda k, row: jax.random.categorical(k, row))(sample_keys, processed_logits).astype(jnp.int32)


def _build_transformer_config(
    *,
    precision: DTypeLike,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
    rms_norm_eps: float,
    attention_bias: bool,
    sliding_window_sizes: tuple[int | None, ...],
) -> TransformerConfig:
    linear_config = FullPrecisionLinearConfig(precision=precision)
    norm_config = NormalizationConfig(
        scale_precision=precision,
        accumulation_precision=precision,
        epsilon=rms_norm_eps,
        scale_offset=None,
        upcast_mode=UpcastMode.ONLY_NORMALIZATION,
        subtract_mean=False,
    )
    mlp_config = DenseMLPConfig(
        linear_config=linear_config,
        activation=SiLU(),
        has_up_biases=False,
        has_down_biases=False,
        gate_clipping=None,
        up_clipping=None,
    )

    if len(sliding_window_sizes) != num_hidden_layers:
        raise ValueError(
            f"Expected {num_hidden_layers} sliding-window entries, got {len(sliding_window_sizes)}",
        )

    layer_configs = tuple(
        TransformerLayerConfig(
            pre_mixer_norm_config=norm_config,
            mixer_config=AttentionConfig(
                qkv_projection_config=linear_config,
                out_projection_config=linear_config,
                query_norm_config=norm_config,
                key_norm_config=norm_config,
                num_heads=num_attention_heads,
                num_groups=num_key_value_heads,
                head_dim=head_dim,
                is_causal=True,
                scale=None,
                sliding_window_size=window_size,
                logit_soft_cap=None,
                has_sinks=False,
                has_qkv_biases=attention_bias,
                has_out_biases=attention_bias,
            ),
            post_mixer_norm_config=None,
            pre_mlp_norm_config=norm_config,
            mlp_config=mlp_config,
            post_mlp_norm_config=None,
        )
        for window_size in sliding_window_sizes
    )

    return TransformerConfig(
        global_rope_config=UnscaledRoPEConfig(
            precision=precision,
            base=rope_theta,
            max_sequence_length=max_position_embeddings,
        ),
        local_rope_config=None,
        layer_configs=layer_configs,
        output_norm_config=norm_config,
        model_dim=hidden_size,
        hidden_dim=intermediate_size,
        context_length=max_position_embeddings,
    )


def default_qwen3_tts_text_decoder_config(
    *,
    precision: DTypeLike,
    talker_vocab_size: int,
    text_vocab_size: int,
    talker_hidden_size: int,
    text_hidden_size: int,
    talker_intermediate_size: int,
    talker_num_hidden_layers: int,
    talker_num_attention_heads: int,
    talker_num_key_value_heads: int,
    talker_head_dim: int,
    talker_max_position_embeddings: int,
    talker_rope_theta: float,
    talker_rms_norm_eps: float,
    talker_attention_bias: bool,
    talker_sliding_window_sizes: tuple[int | None, ...],
    predictor_hidden_size: int,
    predictor_intermediate_size: int,
    predictor_num_hidden_layers: int,
    predictor_num_attention_heads: int,
    predictor_num_key_value_heads: int,
    predictor_head_dim: int,
    predictor_max_position_embeddings: int,
    predictor_rope_theta: float,
    predictor_rms_norm_eps: float,
    predictor_attention_bias: bool,
    predictor_sliding_window_sizes: tuple[int | None, ...],
    predictor_vocab_size: int,
    num_code_groups: int,
    max_new_tokens: int,
    codec_bos_id: int,
    codec_eos_token_id: int,
    codec_pad_id: int,
    codec_think_id: int,
    codec_nothing_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    tts_bos_token_id: int,
    tts_eos_token_id: int,
    tts_pad_token_id: int,
    im_start_token_id: int | None = None,
    assistant_token_id: int | None = None,
    im_end_token_id: int | None = None,
) -> "Qwen3TTSTextDecoderConfig":
    linear_config = FullPrecisionLinearConfig(precision=precision)
    talker_transformer_config = _build_transformer_config(
        precision=precision,
        hidden_size=talker_hidden_size,
        intermediate_size=talker_intermediate_size,
        num_hidden_layers=talker_num_hidden_layers,
        num_attention_heads=talker_num_attention_heads,
        num_key_value_heads=talker_num_key_value_heads,
        head_dim=talker_head_dim,
        max_position_embeddings=talker_max_position_embeddings,
        rope_theta=talker_rope_theta,
        rms_norm_eps=talker_rms_norm_eps,
        attention_bias=talker_attention_bias,
        sliding_window_sizes=talker_sliding_window_sizes,
    )
    predictor_transformer_config = _build_transformer_config(
        precision=precision,
        hidden_size=predictor_hidden_size,
        intermediate_size=predictor_intermediate_size,
        num_hidden_layers=predictor_num_hidden_layers,
        num_attention_heads=predictor_num_attention_heads,
        num_key_value_heads=predictor_num_key_value_heads,
        head_dim=predictor_head_dim,
        max_position_embeddings=predictor_max_position_embeddings,
        rope_theta=predictor_rope_theta,
        rms_norm_eps=predictor_rms_norm_eps,
        attention_bias=predictor_attention_bias,
        sliding_window_sizes=predictor_sliding_window_sizes,
    )

    return Qwen3TTSTextDecoderConfig(
        precision=precision,
        codec_embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision),
        text_embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision),
        predictor_embedding_config=TiedEmbeddingConfig(input_scale=None, logit_soft_cap=None, precision=precision),
        linear_config=linear_config,
        talker_transformer_config=talker_transformer_config,
        predictor_transformer_config=predictor_transformer_config,
        talker_vocab_size=talker_vocab_size,
        text_vocab_size=text_vocab_size,
        talker_hidden_size=talker_hidden_size,
        text_hidden_size=text_hidden_size,
        predictor_hidden_size=predictor_hidden_size,
        predictor_vocab_size=predictor_vocab_size,
        num_code_groups=num_code_groups,
        max_new_tokens=max_new_tokens,
        codec_bos_id=codec_bos_id,
        codec_eos_token_id=codec_eos_token_id,
        codec_pad_id=codec_pad_id,
        codec_think_id=codec_think_id,
        codec_nothing_id=codec_nothing_id,
        codec_think_bos_id=codec_think_bos_id,
        codec_think_eos_id=codec_think_eos_id,
        tts_bos_token_id=tts_bos_token_id,
        tts_eos_token_id=tts_eos_token_id,
        tts_pad_token_id=tts_pad_token_id,
        im_start_token_id=im_start_token_id,
        assistant_token_id=assistant_token_id,
        im_end_token_id=im_end_token_id,
    )


@dataclass(frozen=True)
class Qwen3TTSTextDecoderConfig(TTSTextDecoderConfigBase):
    precision: DTypeLike
    codec_embedding_config: TiedEmbeddingConfig
    text_embedding_config: TiedEmbeddingConfig
    predictor_embedding_config: TiedEmbeddingConfig
    linear_config: FullPrecisionLinearConfig
    talker_transformer_config: TransformerConfig
    predictor_transformer_config: TransformerConfig

    talker_vocab_size: int
    text_vocab_size: int
    talker_hidden_size: int
    text_hidden_size: int
    predictor_hidden_size: int
    predictor_vocab_size: int
    num_code_groups: int
    max_new_tokens: int

    codec_bos_id: int
    codec_eos_token_id: int
    codec_pad_id: int
    codec_think_id: int
    codec_nothing_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int
    im_start_token_id: int | None = None
    assistant_token_id: int | None = None
    im_end_token_id: int | None = None

    def empty(self) -> "Qwen3TTSTextDecoder":
        codec_embedding = self.codec_embedding_config.empty(
            vocab_size=self.talker_vocab_size,
            model_dim=self.talker_hidden_size,
        )
        text_embedding = self.text_embedding_config.empty(
            vocab_size=self.text_vocab_size,
            model_dim=self.text_hidden_size,
        )
        predictor_embeddings = tuple(
            self.predictor_embedding_config.empty(
                vocab_size=self.predictor_vocab_size,
                model_dim=self.talker_hidden_size,
            )
            for _ in range(self.num_code_groups - 1)
        )
        predictor_heads = tuple(
            self.linear_config.empty(
                input_dim=self.predictor_hidden_size,
                output_dims=(self.predictor_vocab_size,),
                has_biases=False,
            )
            for _ in range(self.num_code_groups - 1)
        )

        if self.predictor_hidden_size == self.talker_hidden_size:
            talker_to_predictor_projection = None
        else:
            talker_to_predictor_projection = self.linear_config.empty(
                input_dim=self.talker_hidden_size,
                output_dims=(self.predictor_hidden_size,),
                has_biases=True,
            )

        return Qwen3TTSTextDecoder(
            config=self,
            codec_embedding=codec_embedding,
            text_embedding=text_embedding,
            text_projection_fc1=self.linear_config.empty(
                input_dim=self.text_hidden_size,
                output_dims=(self.text_hidden_size,),
                has_biases=True,
            ),
            text_projection_fc2=self.linear_config.empty(
                input_dim=self.text_hidden_size,
                output_dims=(self.talker_hidden_size,),
                has_biases=True,
            ),
            talker_transformer=self.talker_transformer_config.empty(),
            codec_head=self.linear_config.empty(
                input_dim=self.talker_hidden_size,
                output_dims=(self.talker_vocab_size,),
                has_biases=False,
            ),
            predictor_transformer=self.predictor_transformer_config.empty(),
            predictor_embeddings=predictor_embeddings,
            predictor_heads=predictor_heads,
            talker_to_predictor_projection=talker_to_predictor_projection,
        )

    def random_init(self, *, key: PRNGKeyArray) -> "Qwen3TTSTextDecoder":
        (
            key_codec_embedding,
            key_text_embedding,
            key_text_proj1,
            key_text_proj2,
            key_talker_transformer,
            key_codec_head,
            key_predictor_transformer,
            key_predictor_embeddings,
            key_predictor_heads,
            key_predictor_projection,
        ) = jax.random.split(key, 10)

        predictor_embedding_keys = jax.random.split(key_predictor_embeddings, self.num_code_groups - 1)
        predictor_head_keys = jax.random.split(key_predictor_heads, self.num_code_groups - 1)

        if self.predictor_hidden_size == self.talker_hidden_size:
            talker_to_predictor_projection = None
        else:
            talker_to_predictor_projection = self.linear_config.random_init(
                input_dim=self.talker_hidden_size,
                output_dims=(self.predictor_hidden_size,),
                has_biases=True,
                key=key_predictor_projection,
            )

        return Qwen3TTSTextDecoder(
            config=self,
            codec_embedding=self.codec_embedding_config.random_init(
                vocab_size=self.talker_vocab_size,
                model_dim=self.talker_hidden_size,
                key=key_codec_embedding,
            ),
            text_embedding=self.text_embedding_config.random_init(
                vocab_size=self.text_vocab_size,
                model_dim=self.text_hidden_size,
                key=key_text_embedding,
            ),
            text_projection_fc1=self.linear_config.random_init(
                input_dim=self.text_hidden_size,
                output_dims=(self.text_hidden_size,),
                has_biases=True,
                key=key_text_proj1,
            ),
            text_projection_fc2=self.linear_config.random_init(
                input_dim=self.text_hidden_size,
                output_dims=(self.talker_hidden_size,),
                has_biases=True,
                key=key_text_proj2,
            ),
            talker_transformer=self.talker_transformer_config.random_init(key=key_talker_transformer),
            codec_head=self.linear_config.random_init(
                input_dim=self.talker_hidden_size,
                output_dims=(self.talker_vocab_size,),
                has_biases=False,
                key=key_codec_head,
            ),
            predictor_transformer=self.predictor_transformer_config.random_init(key=key_predictor_transformer),
            predictor_embeddings=tuple(
                self.predictor_embedding_config.random_init(
                    vocab_size=self.predictor_vocab_size,
                    model_dim=self.talker_hidden_size,
                    key=embedding_key,
                )
                for embedding_key in predictor_embedding_keys
            ),
            predictor_heads=tuple(
                self.linear_config.random_init(
                    input_dim=self.predictor_hidden_size,
                    output_dims=(self.predictor_vocab_size,),
                    has_biases=False,
                    key=head_key,
                )
                for head_key in predictor_head_keys
            ),
            talker_to_predictor_projection=talker_to_predictor_projection,
        )


class Qwen3TTSTextDecoder(TTSTextDecoder[Qwen3TTSTextDecoderConfig]):
    codec_embedding: TiedEmbedding
    text_embedding: TiedEmbedding
    text_projection_fc1: FullPrecisionLinear
    text_projection_fc2: FullPrecisionLinear
    talker_transformer: Transformer
    codec_head: FullPrecisionLinear
    predictor_transformer: Transformer
    predictor_embeddings: tuple[TiedEmbedding, ...]
    predictor_heads: tuple[FullPrecisionLinear, ...]
    talker_to_predictor_projection: FullPrecisionLinear | None

    @property
    def activation_precision(self) -> DTypeLike:
        return self.config.precision

    def _project_text_embeddings(
        self,
        text_tokens: Int[Array, "batch tokens"],
    ) -> Float[Array, "batch tokens channels"]:
        x = _embed_tokens(self.text_embedding, text_tokens)
        x = _apply_linear_ntc(self.text_projection_fc1, x)
        x = jax.nn.silu(x)
        x = _apply_linear_ntc(self.text_projection_fc2, x)
        return x

    def _apply_codec_head(
        self,
        hidden_states: Float[Array, "batch tokens channels"],
    ) -> Float[Array, "batch tokens vocabulary"]:
        (logits,) = vmap(vmap(self.codec_head))(hidden_states)
        return logits

    def _to_predictor_space(
        self,
        x: Float[Array, "batch tokens talker_hidden"],
    ) -> Float[Array, "batch tokens predictor_hidden"]:
        if self.talker_to_predictor_projection is None:
            return x
        return _apply_linear_ntc(self.talker_to_predictor_projection, x)

    def _build_talker_prompt(
        self,
        text_tokens: Int[Array, "batch tokens"],
    ) -> tuple[
        Float[Array, "batch prompt_tokens channels"],
        Float[Array, "batch trailing_tokens channels"],
        Float[Array, "batch 1 channels"],
    ]:
        text_hidden = self._project_text_embeddings(text_tokens)
        _, text_length, _ = text_hidden.shape

        special_text_tokens = jnp.asarray(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            dtype=jnp.int32,
        )
        special_hidden = self._project_text_embeddings(special_text_tokens)
        tts_bos_embed = special_hidden[:, 0:1, :]
        tts_eos_embed = special_hidden[:, 1:2, :]
        tts_pad_embed = special_hidden[:, 2:3, :]

        codec_prefill_ids = jnp.asarray(
            [
                [
                    self.config.codec_nothing_id,
                    self.config.codec_think_bos_id,
                    self.config.codec_think_eos_id,
                    self.config.codec_pad_id,
                    self.config.codec_bos_id,
                ]
            ],
            dtype=jnp.int32,
        )
        codec_prefill_embed = _embed_tokens(self.codec_embedding, codec_prefill_ids)

        role_length = min(3, text_length)
        role_hidden = text_hidden[:, :role_length, :]

        # Qwen3-TTS chat formatting appends "<|im_end|>\\n<|im_start|>assistant\\n" (5 tokens).
        # When that exact suffix is present, strip it from the text content slice used for speech generation.
        content_end = text_length
        if (
            self.config.im_start_token_id is not None
            and self.config.assistant_token_id is not None
            and self.config.im_end_token_id is not None
            and text_length >= 8
        ):
            im_end_idx = text_tokens[0, text_length - 5]
            im_start_idx = text_tokens[0, text_length - 3]
            assistant_idx = text_tokens[0, text_length - 2]
            if (
                int(im_end_idx) == self.config.im_end_token_id
                and int(im_start_idx) == self.config.im_start_token_id
                and int(assistant_idx) == self.config.assistant_token_id
            ):
                content_end = text_length - 5

        content_hidden = text_hidden[:, role_length:content_end, :]
        content_length = int(content_hidden.shape[1])
        first_text_hidden = content_hidden[:, :1, :] if content_length > 0 else tts_pad_embed
        trailing_text_hidden_core = (
            content_hidden[:, 1:, :]
            if content_length > 1
            else jnp.zeros((text_hidden.shape[0], 0, text_hidden.shape[-1]), dtype=text_hidden.dtype)
        )

        codec_prefill_body = codec_prefill_embed[:, :-1, :]
        body_length = int(codec_prefill_body.shape[1])
        if body_length <= 0:
            codec_prompt = jnp.zeros((text_hidden.shape[0], 0, text_hidden.shape[-1]), dtype=text_hidden.dtype)
        else:
            left_pad_count = body_length - 1
            codec_bias = jnp.concatenate(
                [jnp.repeat(tts_pad_embed, left_pad_count, axis=1), tts_bos_embed],
                axis=1,
            )
            codec_prompt = codec_bias + codec_prefill_body

        first_codec_plus_text = first_text_hidden + codec_prefill_embed[:, -1:, :]
        prompt = jnp.concatenate([role_hidden, codec_prompt, first_codec_plus_text], axis=1)
        trailing_text_hidden = jnp.concatenate([trailing_text_hidden_core, tts_eos_embed], axis=1)
        return prompt, trailing_text_hidden, tts_pad_embed

    def _decode_code_groups(
        self,
        last_hidden: Float[Array, "batch 1 channels"],
        first_codec_id: Int[Array, " batch"],
        sampling_policy: SamplingPolicy,
        key: PRNGKeyArray,
    ) -> tuple[
        Int[Array, "batch codebooks"],
        Float[Array, "batch 1 channels"],
    ]:
        first_codec_embedding = _embed_tokens(self.codec_embedding, first_codec_id[:, None])
        predictor_inputs = jnp.concatenate([last_hidden, first_codec_embedding], axis=1)

        all_codec_ids = [first_codec_id]
        all_codec_embeddings = [first_codec_embedding]

        for idx, (predictor_embedding, predictor_head) in enumerate(
            zip(self.predictor_embeddings, self.predictor_heads, strict=True),
        ):
            predictor_hidden = _run_transformer(self.predictor_transformer, self._to_predictor_space(predictor_inputs))
            (step_logits,) = vmap(vmap(predictor_head))(predictor_hidden[:, -1:, :])

            key, step_key = jax.random.split(key)
            next_codec_id = _sample_token_ids(step_logits[:, 0, :], sampling_policy, step_key)
            next_codec_embedding = _embed_tokens(predictor_embedding, next_codec_id[:, None])

            all_codec_ids.append(next_codec_id)
            all_codec_embeddings.append(next_codec_embedding)

            if idx + 1 < len(self.predictor_embeddings):
                predictor_inputs = jnp.concatenate([predictor_inputs, next_codec_embedding], axis=1)

        codec_ids = jnp.stack(all_codec_ids, axis=1)
        codec_hidden = jnp.sum(jnp.concatenate(all_codec_embeddings, axis=1), axis=1, keepdims=True)
        return codec_ids, codec_hidden

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_policy: SamplingPolicy | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, "codebooks tokens"]:
        if text_tokens.ndim != 2:
            raise ValueError(f"text_tokens must be rank 2, got {text_tokens.shape}")
        batch_size, _ = text_tokens.shape
        if batch_size != 1:
            raise ValueError("Qwen3-TTS decoder currently supports batch_size=1 only.")

        if sampling_policy is None:
            sampling_policy = default_qwen3_tts_text_sampling_policy()
        if key is None:
            key = jax.random.PRNGKey(123)

        talker_prompt, trailing_text_hidden, tts_pad_embed = self._build_talker_prompt(text_tokens)
        generated_step_codes: list[Array] = []

        max_new_tokens = min(
            self.config.max_new_tokens,
            self.config.talker_transformer_config.context_length - int(talker_prompt.shape[1]),
        )
        max_new_tokens = max(max_new_tokens, 0)

        talker_inputs = talker_prompt
        for step in range(max_new_tokens):
            talker_hidden = _run_transformer(self.talker_transformer, talker_inputs)
            last_hidden = talker_hidden[:, -1:, :]
            codec_logits = self._apply_codec_head(last_hidden)[:, 0, :]

            key, codec_key = jax.random.split(key)
            first_codec_id = _sample_token_ids(codec_logits, sampling_policy, codec_key)

            if int(first_codec_id[0]) == self.config.codec_eos_token_id:
                break

            key, predictor_key = jax.random.split(key)
            step_codec_ids, step_codec_hidden = self._decode_code_groups(
                last_hidden,
                first_codec_id,
                sampling_policy,
                predictor_key,
            )
            generated_step_codes.append(step_codec_ids[0])

            if step < trailing_text_hidden.shape[1]:
                next_talker_input = step_codec_hidden + trailing_text_hidden[:, step : step + 1, :]
            else:
                next_talker_input = step_codec_hidden + tts_pad_embed
            talker_inputs = jnp.concatenate([talker_inputs, next_talker_input], axis=1)

        if not generated_step_codes:
            return jnp.zeros((self.config.num_code_groups, 0), dtype=jnp.int32)

        return jnp.stack(generated_step_codes, axis=1).astype(jnp.int32)

    def export_weights(self) -> ParameterTree[Array]:
        if self.talker_to_predictor_projection is None:
            projection_weights: ParameterTree[Array] = {}
        else:
            projection_weights = self.talker_to_predictor_projection.export_weights()
        return {
            "codec_embedding": self.codec_embedding.export_weights(),
            "text_embedding": self.text_embedding.export_weights(),
            "text_projection_fc1": self.text_projection_fc1.export_weights(),
            "text_projection_fc2": self.text_projection_fc2.export_weights(),
            "talker_transformer": self.talker_transformer.export_weights(),
            "codec_head": self.codec_head.export_weights(),
            "predictor_transformer": self.predictor_transformer.export_weights(),
            "predictor_embeddings": [embedding.export_weights() for embedding in self.predictor_embeddings],
            "predictor_heads": [head.export_weights() for head in self.predictor_heads],
            "talker_to_predictor_projection": projection_weights,
        }

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        predictor_embedding_weights = weights["predictor_embeddings"]
        predictor_head_weights = weights["predictor_heads"]
        assert isinstance(predictor_embedding_weights, Sequence)
        assert isinstance(predictor_head_weights, Sequence)

        if self.talker_to_predictor_projection is None:
            projection = None
        else:
            projection = self.talker_to_predictor_projection.import_weights(
                require_tree(weights["talker_to_predictor_projection"]),
            )

        return replace(
            self,
            codec_embedding=self.codec_embedding.import_weights(require_tree(weights["codec_embedding"])),
            text_embedding=self.text_embedding.import_weights(require_tree(weights["text_embedding"])),
            text_projection_fc1=self.text_projection_fc1.import_weights(require_tree(weights["text_projection_fc1"])),
            text_projection_fc2=self.text_projection_fc2.import_weights(require_tree(weights["text_projection_fc2"])),
            talker_transformer=self.talker_transformer.import_weights(require_tree(weights["talker_transformer"])),
            codec_head=self.codec_head.import_weights(require_tree(weights["codec_head"])),
            predictor_transformer=self.predictor_transformer.import_weights(
                require_tree(weights["predictor_transformer"])
            ),
            predictor_embeddings=tuple(
                embedding.import_weights(require_tree(embedding_weights))
                for embedding, embedding_weights in zip(
                    self.predictor_embeddings,
                    predictor_embedding_weights,
                    strict=True,
                )
            ),
            predictor_heads=tuple(
                head.import_weights(require_tree(head_weights))
                for head, head_weights in zip(self.predictor_heads, predictor_head_weights, strict=True)
            ),
            talker_to_predictor_projection=projection,
        )
