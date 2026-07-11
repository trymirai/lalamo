from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.utils.sharding import lookup_sharded_indices

__all__ = [
    "Weaver",
    "WeaverBlock",
    "WeaverConfig",
    "WeaverPrefix",
]


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
        x: Float[Array, "rows steps d_rank"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows steps d_rank"]:
        batch_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized = call_vmapped_twice(self.norm_mlp, x)
        (up,) = call_vmapped_twice(self.fc1, normalized, keychain=keychain, added_sharding_axes=batch_axes)
        (down,) = call_vmapped_twice(self.fc2, jax.nn.gelu(up), keychain=keychain, added_sharding_axes=batch_axes)
        return down

    def project_qkv(
        self,
        x: Float[Array, "rows steps d_rank"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows steps heads head_dim"],
        Float[Array, "rows steps heads head_dim"],
        Float[Array, "rows steps heads head_dim"],
    ]:
        batch_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized = call_vmapped_twice(self.norm_attn, x)
        (queries,) = call_vmapped_twice(self.q_proj, normalized, keychain=keychain, added_sharding_axes=batch_axes)
        (keys,) = call_vmapped_twice(self.k_proj, normalized, keychain=keychain, added_sharding_axes=batch_axes)
        (values,) = call_vmapped_twice(self.v_proj, normalized, keychain=keychain, added_sharding_axes=batch_axes)
        split = "rows steps (heads channels) -> rows steps heads channels"
        return (
            rearrange(queries, split, heads=self.config.num_heads),
            rearrange(keys, split, heads=self.config.num_heads),
            rearrange(values, split, heads=self.config.num_heads),
        )

    def attend(
        self,
        x: Float[Array, "rows steps d_rank"],
        queries: Float[Array, "rows steps heads head_dim"],
        keys: Float[Array, "rows context heads head_dim"],
        values: Float[Array, "rows context heads head_dim"],
        mask: Bool[Array, "rows steps context"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows steps d_rank"]:
        scores = jnp.einsum("bshd,bthd->bhst", queries, keys) * (self.config.head_dim**-0.5)
        scores = jnp.where(rearrange(mask, "rows steps context -> rows 1 steps context"), scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1)
        outputs = jnp.einsum("bhst,bthd->bshd", attention, values)
        outputs = rearrange(outputs, "rows steps heads channels -> rows steps (heads channels)")
        (projected,) = call_vmapped_twice(
            self.o_proj,
            outputs,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        x = x + projected
        return x + self.mlp(x, keychain=keychain)

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
        queries, keys, values = self.project_qkv(x, keychain=keychain)
        causal = jnp.tril(jnp.ones((prefix, prefix), dtype=jnp.bool))
        mask = jnp.broadcast_to(causal[None], (batch_size, prefix, prefix))
        return self.attend(x, queries, keys, values, mask, keychain=keychain), keys, values

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
        _, prefix, _, _ = prefix_keys.shape
        inputs = x[:, None]
        queries, own_keys, own_values = self.project_qkv(inputs, keychain=keychain)
        keys = jnp.concatenate([prefix_keys, ancestor_keys, own_keys], axis=1)
        values = jnp.concatenate([prefix_values, ancestor_values, own_values], axis=1)
        mask = jnp.pad(ancestor_mask, ((0, 0), (prefix, 1)), constant_values=True)[:, None]
        outputs = self.attend(inputs, queries, keys, values, mask, keychain=keychain)
        return outputs[:, 0], own_keys[:, 0], own_values[:, 0]


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
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows d_rank"]:
        embeddings = lookup_sharded_indices(embed_w, jnp.maximum(token_ids, 0)).astype(jnp.float32)
        (projected,) = call_vmapped(
            self.token_in,
            call_vmapped(self.embed_norm, embeddings),
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return projected

    def prompt_prefix(
        self,
        output_norm_features: Float[Array, "batch hidden"],
        proposal_features: Float[Array, "batch depth hidden"],
        *,
        keychain: Keychain,
    ) -> WeaverPrefix:
        depth = proposal_features.shape[1]
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        normalized_output = call_vmapped(self.output_norm, output_norm_features.astype(jnp.float32))
        (output_token,) = call_vmapped(
            self.proposal_in,
            normalized_output,
            keychain=keychain,
            added_sharding_axis=batch_axis,
        )
        output_token = output_token[:, None, :]
        normalized_proposals = call_vmapped_twice(self.output_norm, proposal_features.astype(jnp.float32))
        (proposal_tokens,) = call_vmapped_twice(
            self.proposal_in,
            normalized_proposals,
            keychain=keychain,
            added_sharding_axes=(batch_axis, None),
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
        x = self.token_project(token_ids, embed_w, keychain=keychain)
        x = x + lookup_sharded_indices(self.pos_emb, jnp.clip(positions, 0, self.config.k - 1))
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
        (query,) = call_vmapped(
            self.lm_head_query_in,
            call_vmapped(self.out_norm, x),
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        candidate_weights = lookup_sharded_indices(lm_head, jnp.maximum(candidate_ids, 0)).astype(jnp.bfloat16)
        residual = jnp.einsum("rh,rch->rc", query.astype(jnp.bfloat16), candidate_weights).astype(jnp.float32)
        valid = (candidate_ids >= 0) & jnp.isfinite(candidate_scores)
        logits = jnp.where(valid, candidate_scores.astype(jnp.float32) + residual, -jnp.inf)
        return logits, jnp.stack(key_layers), jnp.stack(value_layers)
