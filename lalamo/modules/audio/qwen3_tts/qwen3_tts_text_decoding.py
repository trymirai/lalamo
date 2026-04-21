from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, Key

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.text_decoder import (
    CodebookCodes,
    TTSDecodingContext,
    TTSTextDecoder,
    TTSTextDecoderConfigBase,
)
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.embedding import TiedEmbedding, TiedEmbeddingConfig
from lalamo.modules.linear import FullPrecisionLinear, FullPrecisionLinearConfig
from lalamo.modules.transformer import Transformer, TransformerConfig
from lalamo.modules.utils import vmap_twice
from lalamo.sampling import SamplingPolicy, make_policy

__all__ = [
    "Qwen3TTSTextDecoder",
    "Qwen3TTSTextDecoderConfig",
]


@dataclass(frozen=True)
class TalkerInputs:
    prefill: Float[Array, "batch_size prefill_size num_channels"]
    text_continuation: Float[Array, "batch_size tail_size num_channels"]
    text_pad: Float[Array, "batch_size 1 num_channels"]


def _sample_token_ids(
    logits: Float[Array, "batch_size vocab_size"],
    sampling_policy: SamplingPolicy,
    key: Key[Array, ""],
) -> Int[Array, " batch_size"]:
    batch_size, _ = logits.shape
    processed_logits = vmap(sampling_policy.process_logits)(logits)
    sample_keys = jax.random.split(key, batch_size)
    return jax.vmap(lambda k, row: jax.random.categorical(k, row))(sample_keys, processed_logits).astype(jnp.int32)


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
    num_semantic: int
    max_new_tokens: int

    codec_bos_id: int
    codec_eos_token_id: int
    codec_pad_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int
    speaker_id: Mapping[str, int]
    language_id: Mapping[str, int]

    @classmethod
    def format_instruction(cls, style: str) -> str:
        return f"<|im_start|>user\n{style}\n"

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

    def random_init(self, *, key: Key[Array, ""]) -> "Qwen3TTSTextDecoder":
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
        text_tokens: Int[Array, "batch_size num_tokens"],
    ) -> Float[Array, "batch_size num_tokens num_channels"]:
        x = vmap(self.text_embedding.embed)(text_tokens)
        (x,) = vmap_twice(self.text_projection_fc1)(x)
        x = jax.nn.silu(x)
        (x,) = vmap_twice(self.text_projection_fc2)(x)
        return x

    def _to_predictor_space(
        self,
        x: Float[Array, "batch_size num_tokens talker_hidden_size"],
    ) -> Float[Array, "batch_size num_tokens predictor_hidden_size"]:
        if self.talker_to_predictor_projection is None:
            return x
        (y,) = vmap_twice(self.talker_to_predictor_projection)(x)
        return y

    def _embed_special_text_tokens(
        self,
    ) -> tuple[
        Float[Array, "batch_size 1 num_channels"],
        Float[Array, "batch_size 1 num_channels"],
        Float[Array, "batch_size 1 num_channels"],
    ]:
        ids = jnp.asarray(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            dtype=jnp.int32,
        )
        hidden = self._project_text_embeddings(ids)
        bos, eos, pad = jnp.split(hidden, 3, axis=1)
        return bos, eos, pad

    def _build_codec_prefix_ids(self, context: TTSDecodingContext) -> tuple[int, ...]:
        speaker_codec_id = self.config.speaker_id.get(context.speaker) if context.speaker is not None else None
        language_codec_id = (
            self.config.language_id.get(context.language.lower()) if context.language is not None else None
        )
        if language_codec_id is None:
            prefix = (self.config.codec_nothink_id, self.config.codec_think_bos_id, self.config.codec_think_eos_id)
        else:
            prefix = (
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                language_codec_id,
                self.config.codec_think_eos_id,
            )
        if speaker_codec_id is not None:
            prefix = (*prefix, speaker_codec_id)
        return (*prefix, self.config.codec_pad_id, self.config.codec_bos_id)

    def _prepare_talker_inputs(
        self,
        text_tokens: Int[Array, "batch_size num_tokens"],
        codec_prefix_ids: tuple[int, ...],
        instruction_tokens: Int[Array, "batch_size num_tokens"] | None,
    ) -> TalkerInputs:
        text_hidden = self._project_text_embeddings(text_tokens)
        batch_size, text_length, hidden_dim = text_hidden.shape

        tts_bos, tts_eos, tts_pad = self._embed_special_text_tokens()
        codec_prefix = vmap(self.codec_embedding.embed)(jnp.asarray([codec_prefix_ids], dtype=jnp.int32))

        role_length = min(3, text_length)
        role_hidden = text_hidden[:, :role_length, :]
        content_hidden = text_hidden[:, role_length:, :]
        _, content_length, _ = content_hidden.shape
        first_content = content_hidden[:, :1, :] if content_length > 0 else tts_pad
        trailing_content = (
            content_hidden[:, 1:, :]
            if content_length > 1
            else jnp.zeros((batch_size, 0, hidden_dim), dtype=text_hidden.dtype)
        )

        codec_prefix_body = codec_prefix[:, :-1, :]
        left_pad_count = int(codec_prefix_body.shape[1]) - 1
        text_bias = jnp.concatenate([jnp.repeat(tts_pad, left_pad_count, axis=1), tts_bos], axis=1)
        fused_prefix = text_bias + codec_prefix_body
        first_position = first_content + codec_prefix[:, -1:, :]

        prompt_parts = (role_hidden, fused_prefix, first_position)
        if instruction_tokens is not None:
            prompt_parts = (self._project_text_embeddings(instruction_tokens), *prompt_parts)
        prefill = jnp.concatenate(prompt_parts, axis=1)
        text_continuation = jnp.concatenate([trailing_content, tts_eos], axis=1)
        return TalkerInputs(prefill=prefill, text_continuation=text_continuation, text_pad=tts_pad)

    def _prefill_talker(
        self,
        prefill: Float[Array, "batch_size num_tokens num_channels"],
    ) -> tuple[Float[Array, "batch_size num_tokens num_channels"], object]:
        batch_size, length, _ = prefill.shape
        positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32)[None, :], (batch_size, length))
        result = self.talker_transformer(
            inner_features=prefill,
            token_positions=positions,
            state=None,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        return result.outputs, result.updated_state

    def _step_talker(
        self,
        step_input: Float[Array, "batch_size 1 num_channels"],
        state: object,
        position: int,
    ) -> tuple[Float[Array, "batch_size 1 num_channels"], object]:
        result = self.talker_transformer(
            inner_features=step_input,
            token_positions=jnp.array([[position]], dtype=jnp.int32),
            state=state,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            forward_pass_config=None,
        )
        return result.outputs, result.updated_state

    def _generate_codes(
        self,
        inputs: TalkerInputs,
        sampling_policy: SamplingPolicy,
        key: Key[Array, ""],
    ) -> list[Int[Array, " num_codebooks"]]:
        _, prefill_length, _ = inputs.prefill.shape
        max_new_tokens = max(
            0,
            min(
                self.config.max_new_tokens,
                self.config.talker_transformer_config.context_length - prefill_length,
            ),
        )
        talker_hidden, talker_state = self._prefill_talker(inputs.prefill)
        tail_length = int(inputs.text_continuation.shape[1])

        codes: list[Int[Array, " num_codebooks"]] = []
        for step in range(max_new_tokens):
            last_hidden = talker_hidden[:, -1:, :]
            (codec_logits,) = vmap_twice(self.codec_head)(last_hidden)
            codec_logits = jnp.squeeze(codec_logits, axis=1)

            key, codec_key = jax.random.split(key)
            first_codec_id = _sample_token_ids(codec_logits, sampling_policy, codec_key)
            if int(first_codec_id[0]) == self.config.codec_eos_token_id:
                break

            key, predictor_key = jax.random.split(key)
            step_codes, step_codec_hidden = self._decode_code_groups(
                last_hidden,
                first_codec_id,
                sampling_policy,
                predictor_key,
            )
            codes.append(step_codes[0])

            text_slice = inputs.text_continuation[:, step : step + 1, :] if step < tail_length else inputs.text_pad
            talker_hidden, talker_state = self._step_talker(
                step_codec_hidden + text_slice,
                talker_state,
                prefill_length + step,
            )
        return codes

    def _decode_code_groups(
        self,
        last_hidden: Float[Array, "batch_size 1 num_channels"],
        first_codec_id: Int[Array, " batch_size"],
        sampling_policy: SamplingPolicy,
        key: Key[Array, ""],
    ) -> tuple[
        Int[Array, "batch_size num_codebooks"],
        Float[Array, "batch_size 1 num_channels"],
    ]:
        first_codec_embedding = vmap(self.codec_embedding.embed)(first_codec_id[:, None])
        predictor_inputs = jnp.concatenate([last_hidden, first_codec_embedding], axis=1)

        all_codec_ids = [first_codec_id]
        all_codec_embeddings = [first_codec_embedding]

        batch_size = predictor_inputs.shape[0]
        initial_length = int(predictor_inputs.shape[1])

        predictor_state = self.predictor_transformer.init_static_state(batch_size, self.config.num_code_groups)
        initial_positions = jnp.broadcast_to(
            jnp.arange(initial_length, dtype=jnp.int32)[None, :],
            (batch_size, initial_length),
        )
        predictor_result = self.predictor_transformer(
            inner_features=self._to_predictor_space(predictor_inputs),
            token_positions=initial_positions,
            state=predictor_state,
            return_updated_state=True,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
            forward_pass_config=None,
        )
        predictor_hidden = predictor_result.outputs
        predictor_state = predictor_result.updated_state

        for idx, (predictor_embedding, predictor_head) in enumerate(
            zip(self.predictor_embeddings, self.predictor_heads, strict=True),
        ):
            (step_logits,) = vmap_twice(predictor_head)(predictor_hidden[:, -1:, :])

            key, step_key = jax.random.split(key)
            next_codec_id = _sample_token_ids(jnp.squeeze(step_logits, axis=1), sampling_policy, step_key)
            next_codec_embedding = vmap(predictor_embedding.embed)(next_codec_id[:, None])

            all_codec_ids.append(next_codec_id)
            all_codec_embeddings.append(next_codec_embedding)

            if idx + 1 < len(self.predictor_embeddings):
                step_pos = jnp.array([[initial_length + idx]], dtype=jnp.int32)
                predictor_result = self.predictor_transformer(
                    inner_features=self._to_predictor_space(next_codec_embedding),
                    token_positions=step_pos,
                    state=predictor_state,
                    return_updated_state=True,
                    return_layer_results=False,
                    return_positional_embeddings=False,
                    lengths_without_padding=None,
                    forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
                    forward_pass_config=None,
                )
                predictor_hidden = predictor_result.outputs
                predictor_state = predictor_result.updated_state

        codec_ids = jnp.stack(all_codec_ids, axis=1)
        codec_hidden = jnp.sum(jnp.concatenate(all_codec_embeddings, axis=1), axis=1, keepdims=True)
        return codec_ids, codec_hidden

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch_size num_tokens"],
        *,
        context: TTSDecodingContext,
        sampling_policy: SamplingPolicy | None = None,
        key: Key[Array, ""],
    ) -> CodebookCodes:
        sampling_policy = sampling_policy or make_policy(temperature=0.9, top_p=1.0, top_k=50)
        codec_prefix_ids = self._build_codec_prefix_ids(context)
        inputs = self._prepare_talker_inputs(text_tokens, codec_prefix_ids, context.instruction_tokens)
        step_codes = self._generate_codes(inputs, sampling_policy, key)
        codes = jnp.stack(step_codes, axis=1).astype(jnp.int32)[None, :, :]
        num_semantic = self.config.num_semantic
        return CodebookCodes(semantic=codes[:, :num_semantic, :], acoustic=codes[:, num_semantic:, :])

    def export_weights(self) -> ParameterTree[Array]:
        weights = {
            "codec_embedding": self.codec_embedding.export_weights(),
            "text_embedding": self.text_embedding.export_weights(),
            "text_projection_fc1": self.text_projection_fc1.export_weights(),
            "text_projection_fc2": self.text_projection_fc2.export_weights(),
            "talker_transformer": self.talker_transformer.export_weights(),
            "codec_head": self.codec_head.export_weights(),
            "predictor_transformer": self.predictor_transformer.export_weights(),
            "predictor_embeddings": [embedding.export_weights() for embedding in self.predictor_embeddings],
            "predictor_heads": [head.export_weights() for head in self.predictor_heads],
        }
        if self.talker_to_predictor_projection is not None:
            weights["talker_to_predictor_projection"] = self.talker_to_predictor_projection.export_weights()

        return weights

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
                require_tree(weights["predictor_transformer"]),
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
