import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Float, Int
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.model import Model, ModelConfig
from lalamo.models.chat_codec import ChatCodec, ChatCodecConfig
from lalamo.models.language_model import GenerationConfig
from lalamo.module import Keychain, LalamoConfig
from lalamo.modules.decoder import Decoder, DecoderConfig, DecoderForwardPassConfig
from lalamo.modules.embedding import EmbeddingForwardPassConfig, UntiedEmbedding, UntiedEmbeddingConfig

__all__ = [
    "DecisionConfig",
    "DecisionModel",
    "DecisionModelConfig",
]


@dataclass(frozen=True)
class DecisionConfig(LalamoConfig):
    codes: tuple[str, ...]
    temperature: float

    def __post_init__(self) -> None:
        if not self.codes or len(set(self.codes)) != len(self.codes):
            raise ValueError("Decision codes must be nonempty and unique")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Decision temperature must be finite and positive")


@dataclass(frozen=True)
class DecisionModelConfig(ModelConfig[ChatCodecConfig]):
    decoder_config: DecoderConfig
    generation_config: GenerationConfig
    decision: DecisionConfig

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "DecisionModel":
        config = self.decoder_config
        embedding_config = config.embedding_config
        if not isinstance(embedding_config, UntiedEmbeddingConfig) or config.ple_model_config is not None:
            raise ValueError("Decision models require untied embeddings without per-layer embeddings")
        model_dim = config.transformer_config.model_dim
        embedding = UntiedEmbedding(
            config=embedding_config,
            sharding_config=initializer.sharding_config,
            input_embedding=initializer.embedding_matrix(config.vocab_size, model_dim),
            output_embedding=initializer.weight_matrix(len(self.decision.codes), model_dim),
            index_permutation=embedding_config.index_permutation_template(initializer, config.vocab_size),
        )
        decoder = Decoder(
            config=config,
            sharding_config=initializer.sharding_config,
            embedding=embedding,
            transformer=config.transformer_config.init(initializer),
            embedding_norm=(
                config.embedding_norm_config.init(initializer, model_dim)
                if config.embedding_norm_config is not None
                else None
            ),
            per_layer_embedding=None,
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
        option_count: int,
        *,
        keychain: Keychain,
    ) -> Float[Array, " options"]:
        if token_ids.ndim != 1 or token_ids.size == 0:
            raise ValueError("Decision input must be a nonempty sequence of token IDs")
        if not 1 <= option_count <= len(self.config.decision.codes):
            raise ValueError("Option count must be between one and the number of decision codes")
        if any(
            layer.rope_config is not None and token_ids.size > layer.rope_config.max_sequence_length
            for layer in self.config.decoder_config.transformer_config.layer_configs
        ):
            raise ValueError("Decision input exceeds the model context length")
        embedding_key, transformer_key, readout_key = keychain.split(3)
        forward = DecoderForwardPassConfig.for_inference()
        features = self.decoder.embedding.embed(
            token_ids[None], keychain=embedding_key, forward_pass_config=forward.embedding_forward_pass_config
        )
        if self.decoder.embedding_norm is not None:
            features = jax.vmap(jax.vmap(self.decoder.embedding_norm))(features)
        # Decision readouts normalize the final hidden row and multiply the small head in FP32.
        transformer = self.decoder.transformer
        result = replace(transformer, output_norm=None)(
            features,
            jnp.arange(token_ids.size, dtype=jnp.int32)[None],
            state=None,
            return_updated_state=False,
            return_layer_results=False,
            return_positional_embeddings=False,
            lengths_without_padding=None,
            forward_pass_config=forward.transformer_forward_pass_config,
            keychain=transformer_key,
        )
        hidden = result.outputs[0, -1].astype(jnp.float32)
        if transformer.output_norm is not None:
            hidden = transformer.output_norm(hidden)
        logits = (
            self.decoder.embedding.readout(
                hidden,
                keychain=readout_key,
                forward_pass_config=EmbeddingForwardPassConfig.for_inference(DotAlgorithmPreset.F32_F32_F32),
            )[:option_count]
            / self.config.decision.temperature
        )
        return jax.nn.softmax(logits)
