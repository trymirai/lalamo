from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig

__all__ = [
    "DSparkHead",
    "DSparkHeadConfig",
    "MarkovHeadType",
]


class MarkovHeadType(StrEnum):
    RNN_HIDDEN = "rnn_hidden"


@dataclass(frozen=True)
class DSparkHeadConfig(LalamoConfig):
    markov_rank: int
    markov_head_type: MarkovHeadType
    projection_config: LinearConfig
    norm_config: NormalizationConfig
    logit_soft_cap: float | None

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        vocab_size: int,
        num_context_taps: int,
    ) -> "DSparkHead":
        return DSparkHead(
            config=self,
            sharding_config=initializer.sharding_config,
            context_tap_weights=initializer.zeros((num_context_taps, model_dim), dtype=jnp.float32),
            context_norm=self.norm_config.init(initializer, model_dim),
            token_embeddings=initializer.zeros((vocab_size, self.markov_rank)),
            recurrence_projection=self.projection_config.init(
                initializer,
                2 * self.markov_rank + 2 * model_dim,
                (self.markov_rank,) * 3,
                has_biases=True,
            ),
            output_projection=self.projection_config.init(
                initializer,
                self.markov_rank,
                (model_dim,),
                has_biases=True,
            ),
        )


class DSparkHead(LalamoModule[DSparkHeadConfig]):
    context_tap_weights: Float[Array, "taps channels"]
    context_norm: Normalization
    token_embeddings: Float[Array, "vocab rank"]
    recurrence_projection: Linear
    output_projection: Linear
