from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.speculator import DraftAdapter, DraftAdapterConfig

__all__ = [
    "DSpark",
    "DSparkConfig",
    "MarkovHeadType",
]


class MarkovHeadType(StrEnum):
    RNN_HIDDEN = "rnn_hidden"


@dataclass(frozen=True)
class DSparkConfig(DraftAdapterConfig):
    model_dim: int
    vocab_size: int
    num_context_taps: int
    markov_rank: int
    markov_head_type: MarkovHeadType
    projection_config: LinearConfig
    norm_config: NormalizationConfig
    logit_soft_cap: float | None

    def validate_draft(
        self,
        *,
        model_dim: int,
        block_size: int,  # noqa: ARG002
        vocab_size: int,
        num_context_taps: int,
    ) -> None:
        if (self.model_dim, self.vocab_size, self.num_context_taps) != (model_dim, vocab_size, num_context_taps):
            raise ValueError(
                f"DSpark expects model_dim={self.model_dim}, vocab_size={self.vocab_size} and"
                f" num_context_taps={self.num_context_taps}, but the draft has {model_dim}, {vocab_size} and"
                f" {num_context_taps}.",
            )

    def init(self, initializer: Initializer) -> "DSpark":
        return DSpark(
            config=self,
            sharding_config=initializer.sharding_config,
            context_tap_weights=initializer.zeros((self.num_context_taps, self.model_dim), dtype=jnp.float32),
            context_norm=self.norm_config.init(initializer, self.model_dim),
            token_embeddings=initializer.zeros((self.vocab_size, self.markov_rank)),
            recurrence_projection=self.projection_config.init(
                initializer,
                2 * self.markov_rank + 2 * self.model_dim,
                (self.markov_rank,) * 3,
                has_biases=True,
            ),
            output_projection=self.projection_config.init(
                initializer,
                self.markov_rank,
                (self.model_dim,),
                has_biases=True,
            ),
        )


class DSpark(DraftAdapter[DSparkConfig]):
    context_tap_weights: Float[Array, "taps channels"]
    context_norm: Normalization
    token_embeddings: Float[Array, "vocab rank"]
    recurrence_projection: Linear
    output_projection: Linear
