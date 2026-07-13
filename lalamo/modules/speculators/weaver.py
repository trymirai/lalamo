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
    keys: Float[Array, "layers batch prefix heads head_channels"]
    values: Float[Array, "layers batch prefix heads head_channels"]


@dataclass(frozen=True)
class WeaverConfig(LalamoConfig):
    model_dim: int
    target_model_dim: int
    target_embedding_dim: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    max_depth: int
    candidate_pool_size: int
    linear_config: LinearConfig
    norm_config: NormalizationConfig

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    def init(self, initializer: Initializer) -> "Weaver":
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        def linear(input_dim: int, output_dim: int, *, has_biases: bool) -> Linear:
            return self.linear_config.init(initializer, input_dim, (output_dim,), has_biases=has_biases)

        blocks = tuple(
            WeaverBlock(
                config=self,
                sharding_config=initializer.sharding_config,
                pre_attention_norm=self.norm_config.init(initializer, self.model_dim),
                qkv_projection=self.linear_config.init(
                    initializer,
                    self.model_dim,
                    (self.model_dim, self.model_dim, self.model_dim),
                    has_biases=False,
                ),
                out_projection=linear(self.model_dim, self.model_dim, has_biases=False),
                pre_mlp_norm=self.norm_config.init(initializer, self.model_dim),
                up_projection=linear(self.model_dim, self.hidden_dim, has_biases=True),
                down_projection=linear(self.hidden_dim, self.model_dim, has_biases=True),
            )
            for _ in range(self.num_layers)
        )
        return Weaver(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding_norm=self.norm_config.init(initializer, self.target_embedding_dim),
            hidden_state_norm=self.norm_config.init(initializer, self.target_model_dim),
            embedding_projection=linear(self.target_embedding_dim, self.model_dim, has_biases=True),
            blocks=blocks,
            output_norm=self.norm_config.init(initializer, self.model_dim),
            hidden_state_projection=linear(self.target_model_dim, self.model_dim, has_biases=True),
            query_projection=linear(self.model_dim, self.target_model_dim, has_biases=False),
            position_embeddings=initializer.normal(0.02, (self.max_depth, self.model_dim), dtype=jnp.float32),
        )


class WeaverBlock(LalamoModule[WeaverConfig]):
    pre_attention_norm: Normalization
    qkv_projection: Linear
    out_projection: Linear
    pre_mlp_norm: Normalization
    up_projection: Linear
    down_projection: Linear

    def mlp(
        self,
        x: Float[Array, "rows steps channels"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows steps channels"]:
        batch_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized = call_vmapped_twice(self.pre_mlp_norm, x)
        (up,) = call_vmapped_twice(self.up_projection, normalized, keychain=keychain, added_sharding_axes=batch_axes)
        (down,) = call_vmapped_twice(
            self.down_projection,
            jax.nn.gelu(up),
            keychain=keychain,
            added_sharding_axes=batch_axes,
        )
        return down

    def project_qkv(
        self,
        x: Float[Array, "rows steps channels"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows steps heads head_channels"],
        Float[Array, "rows steps heads head_channels"],
        Float[Array, "rows steps heads head_channels"],
    ]:
        batch_axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized = call_vmapped_twice(self.pre_attention_norm, x)
        queries, keys, values = call_vmapped_twice(
            self.qkv_projection,
            normalized,
            keychain=keychain,
            added_sharding_axes=batch_axes,
        )
        split = "rows steps (heads head_channels) -> rows steps heads head_channels"
        return (
            rearrange(queries, split, heads=self.config.num_heads),
            rearrange(keys, split, heads=self.config.num_heads),
            rearrange(values, split, heads=self.config.num_heads),
        )

    def attend(
        self,
        x: Float[Array, "rows steps channels"],
        queries: Float[Array, "rows steps heads head_channels"],
        keys: Float[Array, "rows context heads head_channels"],
        values: Float[Array, "rows context heads head_channels"],
        mask: Bool[Array, "rows steps context"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows steps channels"]:
        scores = jnp.einsum("bshd,bthd->bhst", queries, keys) * (self.config.head_dim**-0.5)
        scores = jnp.where(rearrange(mask, "rows steps context -> rows 1 steps context"), scores, -jnp.inf)
        attention = jax.nn.softmax(scores, axis=-1)
        outputs = jnp.einsum("bhst,bthd->bshd", attention, values)
        outputs = rearrange(outputs, "rows steps heads head_channels -> rows steps (heads head_channels)")
        (projected,) = call_vmapped_twice(
            self.out_projection,
            outputs,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        x = x + projected
        return x + self.mlp(x, keychain=keychain)

    def prefix_forward(
        self,
        x: Float[Array, "batch prefix channels"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "batch prefix channels"],
        Float[Array, "batch prefix heads head_channels"],
        Float[Array, "batch prefix heads head_channels"],
    ]:
        batch_size, prefix, _ = x.shape
        queries, keys, values = self.project_qkv(x, keychain=keychain)
        causal = jnp.tril(jnp.ones((prefix, prefix), dtype=jnp.bool))
        mask = jnp.broadcast_to(causal[None], (batch_size, prefix, prefix))
        return self.attend(x, queries, keys, values, mask, keychain=keychain), keys, values

    def node_step(
        self,
        x: Float[Array, "rows channels"],
        prefix_keys: Float[Array, "rows prefix heads head_channels"],
        prefix_values: Float[Array, "rows prefix heads head_channels"],
        ancestor_keys: Float[Array, "rows depth heads head_channels"],
        ancestor_values: Float[Array, "rows depth heads head_channels"],
        ancestor_mask: Bool[Array, "rows depth"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows channels"],
        Float[Array, "rows heads head_channels"],
        Float[Array, "rows heads head_channels"],
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
    embedding_norm: Normalization
    hidden_state_norm: Normalization
    embedding_projection: Linear
    blocks: tuple[WeaverBlock, ...]
    output_norm: Normalization
    hidden_state_projection: Linear
    query_projection: Linear
    position_embeddings: Float[Array, "max_depth channels"]

    def token_project(
        self,
        token_ids: Int[Array, " rows"],
        embedding_weights: Float[Array, "vocab embedding_channels"],
        *,
        keychain: Keychain,
    ) -> Float[Array, "rows channels"]:
        embeddings = lookup_sharded_indices(embedding_weights, jnp.maximum(token_ids, 0)).astype(jnp.float32)
        (projected,) = call_vmapped(
            self.embedding_projection,
            call_vmapped(self.embedding_norm, embeddings),
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        return projected

    def prompt_prefix(
        self,
        last_token_features: Float[Array, "batch target_channels"],
        proposal_features: Float[Array, "batch depth target_channels"],
        *,
        keychain: Keychain,
    ) -> WeaverPrefix:
        depth = proposal_features.shape[1]
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        normalized_last_token = call_vmapped(self.hidden_state_norm, last_token_features.astype(jnp.float32))
        (last_token,) = call_vmapped(
            self.hidden_state_projection,
            normalized_last_token,
            keychain=keychain,
            added_sharding_axis=batch_axis,
        )
        last_token = last_token[:, None, :]
        normalized_proposals = call_vmapped_twice(self.hidden_state_norm, proposal_features.astype(jnp.float32))
        (proposal_tokens,) = call_vmapped_twice(
            self.hidden_state_projection,
            normalized_proposals,
            keychain=keychain,
            added_sharding_axes=(batch_axis, None),
        )
        proposal_tokens = proposal_tokens + self.position_embeddings[jnp.arange(depth, dtype=jnp.int32)][None]
        x = jnp.concatenate([last_token, proposal_tokens], axis=1)
        key_layers = []
        value_layers = []
        for block in self.blocks:
            x, layer_keys, layer_values = block.prefix_forward(x, keychain=keychain)
            key_layers.append(layer_keys)
            value_layers.append(layer_values)
        return WeaverPrefix(keys=jnp.stack(key_layers), values=jnp.stack(value_layers))

    def step(
        self,
        lm_head: Float[Array, "vocab target_channels"],
        embedding_weights: Float[Array, "vocab embedding_channels"],
        token_ids: Int[Array, " rows"],
        candidate_ids: Int[Array, "rows candidates"],
        candidate_scores: Float[Array, "rows candidates"],
        prefix_keys: Float[Array, "layers rows prefix heads head_channels"],
        prefix_values: Float[Array, "layers rows prefix heads head_channels"],
        positions: Int[Array, " rows"],
        ancestor_keys: Float[Array, "layers rows depth heads head_channels"],
        ancestor_values: Float[Array, "layers rows depth heads head_channels"],
        ancestor_mask: Bool[Array, "rows depth"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows candidates"],
        Float[Array, "layers rows heads head_channels"],
        Float[Array, "layers rows heads head_channels"],
    ]:
        x = self.token_project(token_ids, embedding_weights, keychain=keychain)
        x = x + lookup_sharded_indices(self.position_embeddings, jnp.clip(positions, 0, self.config.max_depth - 1))
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
            self.query_projection,
            call_vmapped(self.output_norm, x),
            keychain=keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        candidate_weights = lookup_sharded_indices(lm_head, jnp.maximum(candidate_ids, 0)).astype(jnp.bfloat16)
        residual = jnp.einsum("rh,rch->rc", query.astype(jnp.bfloat16), candidate_weights).astype(jnp.float32)
        valid = (candidate_ids >= 0) & jnp.isfinite(candidate_scores)
        logits = jnp.where(valid, candidate_scores.astype(jnp.float32) + residual, -jnp.inf)
        return logits, jnp.stack(key_layers), jnp.stack(value_layers)
