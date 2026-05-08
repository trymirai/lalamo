from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.audio.text_decoder import TTSTextDecoder, TTSTextDecoderConfigBase
from lalamo.modules.common import ForwardPassMode
from lalamo.modules.decoder import Decoder, DecoderConfig
from lalamo.sampling import CompositePolicy, SamplingPolicy, TemperaturePolicy, TopKPolicy


def default_neutts_sampling_policy() -> SamplingPolicy:
    return CompositePolicy((TemperaturePolicy(1.0), TopKPolicy(50)))


@dataclass(frozen=True)
class NeuTTSTextDecoderConfig(TTSTextDecoderConfigBase):
    decoder_config: DecoderConfig
    speech_generation_end_token_id: int
    max_context_length: int
    language_code: str
    min_new_tokens: int = 50

    def empty(self) -> "NeuTTSTextDecoder":
        return NeuTTSTextDecoder(
            config=self,
            decoder=self.decoder_config.empty(),
        )

    def random_init(self, *, key: PRNGKeyArray) -> "NeuTTSTextDecoder":
        return NeuTTSTextDecoder(
            config=self,
            decoder=self.decoder_config.random_init(key=key),
        )


class NeuTTSTextDecoder(TTSTextDecoder[NeuTTSTextDecoderConfig]):
    decoder: Decoder

    @property
    def activation_precision(self) -> DTypeLike:
        return self.decoder.activation_precision

    def export_weights(self) -> ParameterTree[Array]:
        return {"decoder": self.decoder.export_weights()}

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        if not isinstance(weights, Mapping):
            raise TypeError("NeuTTSTextDecoder weights must be a mapping.")
        return replace(
            self,
            decoder=self.decoder.import_weights(require_tree(weights["decoder"])),
        )

    def decode_utterance(
        self,
        text_tokens: Int[Array, "batch tokens"],
        sampling_policy: SamplingPolicy | None = None,
        key: PRNGKeyArray | None = None,
    ) -> Int[Array, " generated_tokens"]:
        batch_size, prompt_length = text_tokens.shape
        if batch_size != 1:
            raise ValueError("NeuTTS currently supports batch_size=1.")
        if prompt_length >= self.config.max_context_length:
            raise ValueError(
                f"NeuTTS prompt length {prompt_length} exceeds max context {self.config.max_context_length}.",
            )

        current_key = jax.random.PRNGKey(0) if key is None else key
        current_policy = sampling_policy if sampling_policy is not None else default_neutts_sampling_policy()
        current_policy = current_policy.init(text_tokens[0], jnp.asarray(prompt_length, dtype=jnp.int32))

        state = self.decoder.init_static_state(batch_size, self.config.max_context_length)
        token_positions = jnp.arange(prompt_length, dtype=jnp.int32)[None, :]
        prefill_result = self.decoder(
            text_tokens,
            token_positions,
            state=state,
            return_updated_state=True,
            lengths_without_padding=jnp.asarray([prompt_length], dtype=jnp.int32),
            forward_pass_mode=ForwardPassMode.MULTI_TOKEN,
        )
        assert prefill_result.updated_state is not None

        current_state = prefill_result.updated_state
        current_logits = prefill_result.logits[:, -1, :].astype(jnp.float32)
        generated_tokens: list[Array] = []
        max_new_tokens = self.config.max_context_length - prompt_length
        token_position = prompt_length

        for generation_step in range(max_new_tokens):
            current_key, sample_key = jax.random.split(current_key)
            processed_logits = current_policy.process_logits(current_logits[0])
            if generation_step + 1 < self.config.min_new_tokens:
                processed_logits = processed_logits.at[self.config.speech_generation_end_token_id].set(-jnp.inf)
            next_token = jax.random.categorical(sample_key, processed_logits).astype(jnp.int32)
            generated_tokens.append(next_token)
            current_policy = current_policy.update(next_token)

            if (
                generation_step + 1 >= self.config.min_new_tokens
                and int(next_token.item()) == self.config.speech_generation_end_token_id
            ):
                break

            decode_result = self.decoder(
                next_token.reshape(1, 1),
                jnp.asarray([[token_position]], dtype=jnp.int32),
                state=current_state,
                return_updated_state=True,
                forward_pass_mode=ForwardPassMode.SINGLE_TOKEN,
            )
            assert decode_result.updated_state is not None
            current_state = decode_result.updated_state
            current_logits = decode_result.logits[:, -1, :].astype(jnp.float32)
            token_position += 1

        return jnp.asarray(generated_tokens, dtype=jnp.int32)
