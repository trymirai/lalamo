from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain, LalamoConfig
from lalamo.modules.decoder import Decoder, DecoderConfig, DecoderForwardPassConfig
from lalamo.modules.embedding import UntiedEmbedding

__all__ = [
    "DecisionConfig",
    "DecisionModel",
    "DecisionModelConfig",
]


@dataclass(frozen=True)
class DecisionConfig(LalamoConfig):
    codes: tuple[str, ...]
    temperature: float


@dataclass(frozen=True)
class DecisionModelConfig(ModelConfig[ChatCodecConfig]):
    decoder_config: DecoderConfig
    generation_config: GenerationConfig
    decision: DecisionConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "DecisionModel":
        decoder = self.decoder_config.init(initializer)
        assert isinstance(decoder.embedding, UntiedEmbedding)
        decoder = replace(
            decoder,
            embedding=replace(
                decoder.embedding,
                output_embedding=initializer.weight_matrix(
                    len(self.decision.codes),
                    self.decoder_config.transformer_config.model_dim,
                ),
            ),
        )
        return DecisionModel(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            decoder=decoder,
        )


class DecisionModel(Model[DecisionModelConfig, ChatCodec]):
    decoder: Decoder

    def probabilities(
        self,
        token_ids: Int[Array, " tokens"],
        num_options: int,
        *,
        keychain: Keychain,
    ) -> Float[Array, " options"]:
        assert token_ids.ndim == 1 and token_ids.size > 0
        assert 1 <= num_options <= len(self.config.decision.codes)
        assert self.config.decision.temperature > 0
        result = self.decoder(
            token_ids[None],
            jnp.arange(token_ids.size, dtype=jnp.int32)[None],
            forward_pass_config=DecoderForwardPassConfig.for_inference(),
            keychain=keychain,
        )
        logits = result.logits[0, -1, :num_options].astype(jnp.float32)
        return jax.nn.softmax(logits / self.config.decision.temperature)
