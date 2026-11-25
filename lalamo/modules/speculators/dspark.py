from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.utils import apply_soft_capping
from lalamo.weight_matrix import MatmulConfig

__all__ = [
    "DSparkHead",
    "DSparkHeadConfig",
    "MarkovHeadType",
    "mix_taps",
]


def mix_taps(
    tap_weights: Float[Array, "taps channels"],
    norm: Normalization,
    taps: Float[Array, "taps channels"],
) -> Float[Array, " channels"]:
    """The weighted sum of one row's target taps, RMS-normalized: norm(sum over taps of weights * tap).

    The weights are per channel because the loader folds the scales of the target's output normalization into
    the weight of the final tap; the sum runs in float32 and the result is float32.
    """
    mixed = einsum(
        tap_weights.astype(jnp.float32),
        taps.astype(jnp.float32),
        "taps channels, taps channels -> channels",
    )
    return norm(mixed)


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
    """A recurrent head that corrects the logits of a draft block position by position.

    Along a path of proposed tokens the head keeps a state s, zero at the anchor. At a node whose token is `e`
    (the anchor itself for the first proposal position) and whose children sit at a position with draft hidden
    state h, with f the tap mix of the newest committed row:

        [gate, candidate, latent] = recurrence([s, embedding(e), h, f])
        s' = sigmoid(gate) * s + (1 - sigmoid(gate)) * tanh(candidate)
        correction = output(tanh(latent))

    and the children's logits are soft_cap(unembedding @ (h + correction)): the draft's base logits and the
    head's bias go through the same target LM head, so they add up before it. Every method works on one node and
    can be mapped over the nodes of a tree; s' belongs to the children of the node.
    """

    context_tap_weights: Float[Array, "taps channels"]
    context_norm: Normalization
    token_embeddings: Float[Array, "vocab rank"]
    recurrence_projection: Linear
    output_projection: Linear

    def mix_context(self, taps: Float[Array, "taps channels"]) -> Float[Array, " channels"]:
        """f of the recurrence from the target taps of the newest committed row (the row before the anchor)."""
        return mix_taps(self.context_tap_weights, self.context_norm, taps)

    def initial_state(self, dtype: DTypeLike = jnp.float32) -> Float[Array, " rank"]:
        return jnp.zeros((self.config.markov_rank,), dtype=dtype)

    def step(
        self,
        state: Float[Array, " rank"],
        previous_token: Int[Array, ""] | int,
        hidden: Float[Array, " channels"],
        context: Float[Array, " channels"],
        forward_pass_config: MatmulConfig = MatmulConfig(),
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, " channels"], Float[Array, " rank"]]:
        """(correction to add to `hidden` before the LM head, state of the node's children).

        `previous_token` is the token of the node, `hidden` the draft hidden state of the position its children
        occupy, `context` the result of `mix_context`. The result is float32.
        """
        recurrence_keychain, output_keychain = keychain.split()
        inputs = jnp.concatenate(
            [
                state.astype(jnp.float32),
                self.token_embeddings[previous_token].astype(jnp.float32),
                hidden.astype(jnp.float32),
                context.astype(jnp.float32),
            ],
        )
        gate, candidate, latent = (
            part.astype(jnp.float32)
            for part in self.recurrence_projection(
                inputs,
                keychain=recurrence_keychain,
                forward_pass_config=forward_pass_config,
            )
        )
        keep = jax.nn.sigmoid(gate)
        next_state = keep * state.astype(jnp.float32) + (1.0 - keep) * jnp.tanh(candidate)
        (correction,) = self.output_projection(
            jnp.tanh(latent),
            keychain=output_keychain,
            forward_pass_config=forward_pass_config,
        )
        return correction.astype(jnp.float32), next_state

    def logits(
        self,
        hidden: Float[Array, " channels"],
        correction: Float[Array, " channels"],
        unembedding: Float[Array, "tokens channels"],
    ) -> Float[Array, " tokens"]:
        """Logits of the tokens whose rows of the target's LM head are given: soft_cap(unembedding @ (h + c)).

        With all rows this is the target's own readout of `hidden + correction`; with the rows of a candidate
        pool it is the same logits restricted to the pool. Rows past the unpadded vocabulary are padding and
        must be left out by the caller.
        """
        logits = einsum(
            unembedding.astype(jnp.float32),
            hidden.astype(jnp.float32) + correction.astype(jnp.float32),
            "tokens channels, channels -> tokens",
        )
        if self.config.logit_soft_cap is None:
            return logits
        return apply_soft_capping(logits, self.config.logit_soft_cap)
