from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig

__all__ = [
    "Weaver",
    "WeaverBlock",
    "WeaverConfig",
    "WeaverPrefix",
    "apply_linear",
    "apply_norm",
]


def apply_linear(
    linear: Linear,
    inputs: Float[Array, "*rows in_channels"],
    keychain: Keychain,
) -> Float[Array, "*rows out_channels"]:
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    flat_outputs = jax.vmap(lambda row: linear(row, keychain=keychain)[0])(flat_inputs)
    return flat_outputs.reshape(*inputs.shape[:-1], flat_outputs.shape[-1])


def apply_norm(
    norm: Normalization,
    inputs: Float[Array, "*rows channels"],
) -> Float[Array, "*rows channels"]:
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    flat_outputs = jax.vmap(norm)(flat_inputs)
    return flat_outputs.reshape(inputs.shape)


class WeaverPrefix(eqx.Module):
    keys: Float[Array, "layers batch prefix heads head_dim"]
    values: Float[Array, "layers batch prefix heads head_dim"]


@dataclass(frozen=True)
class WeaverConfig(LalamoConfig):
    d_model: int
    d_embed: int
    d_rank: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    k: int
    candidate_pool_size: int
    linear_config: LinearConfig
    norm_config: NormalizationConfig

    @property
    def head_dim(self) -> int:
        return self.d_rank // self.num_heads

    def init(self, initializer: Initializer) -> "Weaver":
        if self.d_rank % self.num_heads != 0:
            raise ValueError("d_rank must be divisible by num_heads")

        def linear(input_dim: int, output_dim: int, *, has_biases: bool) -> Linear:
            return self.linear_config.init(initializer, input_dim, (output_dim,), has_biases=has_biases)

        blocks = tuple(
            WeaverBlock(
                config=self,
                sharding_config=initializer.sharding_config,
                norm_attn=self.norm_config.init(initializer, self.d_rank),
                q_proj=linear(self.d_rank, self.d_rank, has_biases=False),
                k_proj=linear(self.d_rank, self.d_rank, has_biases=False),
                v_proj=linear(self.d_rank, self.d_rank, has_biases=False),
                o_proj=linear(self.d_rank, self.d_rank, has_biases=False),
                norm_mlp=self.norm_config.init(initializer, self.d_rank),
                fc1=linear(self.d_rank, self.mlp_dim, has_biases=True),
                fc2=linear(self.mlp_dim, self.d_rank, has_biases=True),
            )
            for _ in range(self.num_layers)
        )
        return Weaver(
            config=self,
            sharding_config=initializer.sharding_config,
            embed_norm=self.norm_config.init(initializer, self.d_embed),
            output_norm=self.norm_config.init(initializer, self.d_model),
            token_in=linear(self.d_embed, self.d_rank, has_biases=True),
            blocks=blocks,
            out_norm=self.norm_config.init(initializer, self.d_rank),
            proposal_in=linear(self.d_model, self.d_rank, has_biases=True),
            lm_head_query_in=linear(self.d_rank, self.d_model, has_biases=False),
            pos_emb=initializer.normal(0.02, (self.k, self.d_rank), dtype=jnp.float32),
        )


class WeaverBlock(LalamoModule[WeaverConfig]):
    norm_attn: Normalization
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    norm_mlp: Normalization
    fc1: Linear
    fc2: Linear

    def mlp(
        self,
        x: Float[Array, "*rows d_rank"],
        keychain: Keychain,
    ) -> Float[Array, "*rows d_rank"]:
        hidden = jax.nn.gelu(apply_linear(self.fc1, apply_norm(self.norm_mlp, x), keychain))
        return apply_linear(self.fc2, hidden, keychain)

    def project_qkv(
        self,
        x: Float[Array, "*rows d_rank"],
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "*rows heads head_dim"],
        Float[Array, "*rows heads head_dim"],
        Float[Array, "*rows heads head_dim"],
    ]:
        h = apply_norm(self.norm_attn, x)
        head_shape = (*x.shape[:-1], self.config.num_heads, self.config.head_dim)
        q = apply_linear(self.q_proj, h, keychain).reshape(head_shape)
        k = apply_linear(self.k_proj, h, keychain).reshape(head_shape)
        v = apply_linear(self.v_proj, h, keychain).reshape(head_shape)
        return q, k, v

    def prefix_forward(
        self,
        x: Float[Array, "batch prefix d_rank"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "batch prefix d_rank"],
        Float[Array, "batch prefix heads head_dim"],
        Float[Array, "batch prefix heads head_dim"],
    ]:
        batch_size, prefix, _ = x.shape
        q, k, v = self.project_qkv(x, keychain)
        scores = jnp.einsum("bshd,bthd->bhst", q, k) * (self.config.head_dim**-0.5)
        causal = jnp.tril(jnp.ones((prefix, prefix), dtype=jnp.bool))
        scores = jnp.where(causal[None, None], scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("bhst,bthd->bshd", attention, v).reshape(batch_size, prefix, self.config.d_rank)
        x = x + apply_linear(self.o_proj, y, keychain)
        return x + self.mlp(x, keychain), k, v

    def node_step(
        self,
        x: Float[Array, "rows d_rank"],
        prefix_keys: Float[Array, "rows prefix heads head_dim"],
        prefix_values: Float[Array, "rows prefix heads head_dim"],
        ancestor_keys: Float[Array, "rows depth heads head_dim"],
        ancestor_values: Float[Array, "rows depth heads head_dim"],
        ancestor_mask: Bool[Array, "rows depth"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows d_rank"],
        Float[Array, "rows heads head_dim"],
        Float[Array, "rows heads head_dim"],
    ]:
        rows, prefix, _, _ = prefix_keys.shape
        q, k, v = self.project_qkv(x, keychain)
        keys = jnp.concatenate([prefix_keys, ancestor_keys, k[:, None]], axis=1)
        values = jnp.concatenate([prefix_values, ancestor_values, v[:, None]], axis=1)
        mask = jnp.concatenate(
            [
                jnp.ones((rows, prefix), dtype=jnp.bool),
                ancestor_mask,
                jnp.ones((rows, 1), dtype=jnp.bool),
            ],
            axis=1,
        )
        scores = jnp.einsum("rhd,rthd->rht", q, keys) * (self.config.head_dim**-0.5)
        scores = jnp.where(mask[:, None], scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1)
        y = jnp.einsum("rht,rthd->rhd", attention, values).reshape(rows, self.config.d_rank)
        x = x + apply_linear(self.o_proj, y, keychain)
        return x + self.mlp(x, keychain), k, v


class Weaver(LalamoModule[WeaverConfig]):
    embed_norm: Normalization
    output_norm: Normalization
    token_in: Linear
    blocks: tuple[WeaverBlock, ...]
    out_norm: Normalization
    proposal_in: Linear
    lm_head_query_in: Linear
    pos_emb: Float[Array, "k d_rank"]

    def token_project(
        self,
        token_ids: Int[Array, " rows"],
        embed_w: Float[Array, "vocab embed"],
        keychain: Keychain,
    ) -> Float[Array, "rows d_rank"]:
        embeds = embed_w[jnp.maximum(token_ids, 0)].astype(jnp.float32)
        return apply_linear(self.token_in, apply_norm(self.embed_norm, embeds), keychain)

    def prompt_prefix(
        self,
        output_norm_features: Float[Array, "batch hidden"],
        proposal_features: Float[Array, "batch depth hidden"],
        *,
        keychain: Keychain,
    ) -> WeaverPrefix:
        depth = proposal_features.shape[1]
        output_token = apply_linear(
            self.proposal_in,
            apply_norm(self.output_norm, output_norm_features.astype(jnp.float32)),
            keychain,
        )[:, None, :]
        proposal_tokens = apply_linear(
            self.proposal_in,
            apply_norm(self.output_norm, proposal_features.astype(jnp.float32)),
            keychain,
        )
        proposal_tokens = proposal_tokens + self.pos_emb[jnp.arange(depth, dtype=jnp.int32)][None]
        x = jnp.concatenate([output_token, proposal_tokens], axis=1)
        key_layers = []
        value_layers = []
        for block in self.blocks:
            x, layer_keys, layer_values = block.prefix_forward(x, keychain=keychain)
            key_layers.append(layer_keys)
            value_layers.append(layer_values)
        return WeaverPrefix(keys=jnp.stack(key_layers), values=jnp.stack(value_layers))

    def step(
        self,
        lm_head: Float[Array, "vocab hidden"],
        embed_w: Float[Array, "vocab embed"],
        token_ids: Int[Array, " rows"],
        candidate_ids: Int[Array, "rows candidates"],
        candidate_scores: Float[Array, "rows candidates"],
        prefix_keys: Float[Array, "layers rows prefix heads head_dim"],
        prefix_values: Float[Array, "layers rows prefix heads head_dim"],
        positions: Int[Array, " rows"],
        ancestor_keys: Float[Array, "layers rows depth heads head_dim"],
        ancestor_values: Float[Array, "layers rows depth heads head_dim"],
        ancestor_mask: Bool[Array, "rows depth"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows candidates"],
        Float[Array, "layers rows heads head_dim"],
        Float[Array, "layers rows heads head_dim"],
    ]:
        x = self.token_project(token_ids, embed_w, keychain)
        x = x + self.pos_emb[jnp.clip(positions, 0, self.config.k - 1)]
        key_layers = []
        value_layers = []
        for layer_index, block in enumerate(self.blocks):
            x, layer_keys, layer_values = block.node_step(
                x,
                prefix_keys[layer_index],
                prefix_values[layer_index],
                ancestor_keys[layer_index],
                ancestor_values[layer_index],
                ancestor_mask,
                keychain=keychain,
            )
            key_layers.append(layer_keys)
            value_layers.append(layer_values)
        query = apply_linear(self.lm_head_query_in, apply_norm(self.out_norm, x), keychain)
        candidate_weights = lm_head[jnp.maximum(candidate_ids, 0)].astype(jnp.bfloat16)
        residual = jnp.einsum("rh,rch->rc", query.astype(jnp.bfloat16), candidate_weights).astype(jnp.float32)
        valid = (candidate_ids >= 0) & jnp.isfinite(candidate_scores)
        logits = candidate_scores.astype(jnp.float32) + jnp.where(valid, residual, 0.0)
        return logits, jnp.stack(key_layers), jnp.stack(value_layers)
