from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.residual import AdditiveResidualConfig
from lalamo.modules.rope import RoPEConfig
from lalamo.modules.speculator import Speculator, SpeculatorConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.transformer_layer import TransformerLayer, TransformerLayerConfig

__all__ = [
    "DSparkLayer",
    "DSparkLayerConfig",
    "DSparkSpeculator",
    "DSparkSpeculatorConfig",
    "MarkovHead",
    "MarkovHeadConfig",
    "TapMix",
    "TapMixConfig",
]


@dataclass(frozen=True)
class TapMixConfig(LalamoConfig):
    """A softmax-weighted sum of the target's hidden-state taps, RMS-normalized."""

    num_taps: int
    norm_config: NormalizationConfig

    def init(self, initializer: Initializer, model_dim: int) -> "TapMix":
        return TapMix(
            config=self,
            sharding_config=initializer.sharding_config,
            mix_logits=initializer.zeros((self.num_taps,), dtype=jnp.float32),
            norm=self.norm_config.init(initializer, model_dim),
        )


class TapMix(LalamoModule[TapMixConfig]):
    # The weights are softmax(mix_logits).
    mix_logits: Float[Array, " taps"]
    norm: Normalization


@dataclass(frozen=True)
class DSparkLayerConfig(LalamoConfig):
    """A draft layer, whose attention also reads keys and values projected from a mix of the target's taps.

    The context projection yields the keys, then the values, of each context row.
    """

    transformer_layer_config: TransformerLayerConfig
    context_mix_config: TapMixConfig
    context_kv_projection_config: LinearConfig
    context_key_norm_config: NormalizationConfig

    @property
    def attention_config(self) -> AttentionConfig:
        attention_config = self.transformer_layer_config.mixer.body_config
        if not isinstance(attention_config, AttentionConfig):
            raise TypeError(f"DSpark draft layers use attention, got {type(attention_config).__name__}.")
        return attention_config

    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "DSparkLayer":
        attention_config = self.attention_config
        kv_dim = attention_config.num_groups * attention_config.head_dim
        return DSparkLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            transformer_layer=self.transformer_layer_config.init(
                initializer,
                AdditiveResidualConfig(),
                model_dim,
                hidden_dim,
            ),
            context_mix=self.context_mix_config.init(initializer, model_dim),
            context_kv_projection=self.context_kv_projection_config.init(
                initializer,
                model_dim,
                (kv_dim, kv_dim),
                has_biases=False,
            ),
            context_key_norm=self.context_key_norm_config.init(initializer, attention_config.head_dim),
        )


class DSparkLayer(LalamoModule[DSparkLayerConfig]):
    transformer_layer: TransformerLayer
    context_mix: TapMix
    context_kv_projection: Linear
    context_key_norm: Normalization


@dataclass(frozen=True)
class MarkovHeadConfig(LalamoConfig):
    """A recurrent head that adds a vocabulary bias to each proposal position's logits, left to right.

    Per position, with state s, the previous token's embedding e, the position's draft hidden state h and the mix of
    the newest committed taps f: [gate, candidate, latent] = recurrence([s, e, h, f]),
    s' = sigmoid(gate) * s + (1 - sigmoid(gate)) * tanh(candidate), and the bias is the target's LM head applied to
    output(tanh(latent)). The logits are soft-capped after the bias is added.
    """

    rank: int
    vocab_size: int
    context_mix_config: TapMixConfig
    recurrence_config: LinearConfig
    output_config: LinearConfig
    logit_soft_cap: float | None

    def init(self, initializer: Initializer, model_dim: int) -> "MarkovHead":
        return MarkovHead(
            config=self,
            sharding_config=initializer.sharding_config,
            context_mix=self.context_mix_config.init(initializer, model_dim),
            token_embeddings=initializer.normal(1.0, (self.vocab_size, self.rank)),
            recurrence=self.recurrence_config.init(
                initializer,
                2 * self.rank + 2 * model_dim,
                (3 * self.rank,),
                has_biases=True,
            ),
            output=self.output_config.init(initializer, self.rank, (model_dim,), has_biases=True),
        )


class MarkovHead(LalamoModule[MarkovHeadConfig]):
    context_mix: TapMix
    token_embeddings: Float[Array, "vocabulary rank"]
    recurrence: Linear
    output: Linear


@dataclass(frozen=True)
class DSparkSpeculatorConfig(SpeculatorConfig):
    """A block drafter: the anchor token and `block_size - 1` mask positions attend to each other and to the newest
    `context_window` committed context rows, and the mask positions propose through the target's LM head and the
    Markov head.

    `target_feature_indices` are target layer visits whose incoming hidden states are tapped; the index equal to the
    number of visits is the final normalized hidden state.
    """

    model_dim: int
    hidden_dim: int
    block_size: int
    target_feature_indices: tuple[int, ...]
    context_window: int
    rope_config: RoPEConfig
    normalizes_anchor_embedding: bool
    layer_configs: tuple[DSparkLayerConfig, ...]
    output_norm_config: NormalizationConfig
    markov_head_config: MarkovHeadConfig

    def __post_init__(self) -> None:
        num_taps = len(self.target_feature_indices)
        mixes = [layer_config.context_mix_config for layer_config in self.layer_configs]
        mixes.append(self.markov_head_config.context_mix_config)
        if any(mix.num_taps != num_taps for mix in mixes):
            raise ValueError(f"Every DSpark tap mix must mix the {num_taps} target taps.")

    def init(self, initializer: Initializer) -> "DSparkSpeculator":
        return DSparkSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            mask_embedding=initializer.zeros((self.model_dim,)),
            layers=tuple(
                layer_config.init(initializer, self.model_dim, self.hidden_dim) for layer_config in self.layer_configs
            ),
            output_norm=self.output_norm_config.init(initializer, self.model_dim),
            markov_head=self.markov_head_config.init(initializer, self.model_dim),
        )


class DSparkSpeculator(Speculator[DSparkSpeculatorConfig]):
    mask_embedding: Float[Array, " channels"]
    layers: tuple[DSparkLayer, ...]
    output_norm: Normalization
    markov_head: MarkovHead
