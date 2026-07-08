from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int
from tokenizers import Tokenizer

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, SpeculatorState
from lalamo.modules import Linear, LinearConfig, Normalization, NormalizationConfig
from lalamo.modules.decoder import DecoderActivationTrace, DecoderResult
from lalamo.modules.dflash import DFlashDraftConfig, DFlashDraftModel, DFlashDraftState
from lalamo.modules.embedding import EmbeddingBase
from lalamo.modules.utils import call_vmapped_twice

from .common import AcceptedProposal, Proposal, Speculator, SpeculatorConfig, TreeProposal

__all__ = [
    "Weaver",
    "WeaverBlock",
    "WeaverConfig",
    "WeaverDraftState",
    "WeaverPrefix",
    "WeaverSpeculator",
    "WeaverSpeculatorConfig",
    "build_weaver_tree",
]

EXPAND_WIDTH = 8
BATCH_EXPAND_BUDGET_UNIT = 16


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


def _small_top_k(values: Float[Array, "... slots"], k: int) -> tuple[Array, Array]:
    # k sequential argmax passes: for small k this is a handful of fused
    # reductions instead of a full radix sort of every row.
    remaining = values
    picked_values = []
    picked_indices = []
    slots = jnp.arange(values.shape[-1], dtype=jnp.int32)
    for _ in range(k):
        index = jnp.argmax(remaining, axis=-1).astype(jnp.int32)
        value = jnp.take_along_axis(remaining, index[..., None], axis=-1)[..., 0]
        picked_values.append(value)
        picked_indices.append(index)
        remaining = jnp.where(slots == index[..., None], -jnp.inf, remaining)
    return jnp.stack(picked_values, axis=-1), jnp.stack(picked_indices, axis=-1)


def _vocab_top_k(logits: Float[Array, "... vocab"], k: int) -> tuple[Array, Array]:
    # Near-exact top-k over the vocabulary: top-2 per contiguous block (two argmax
    # sweeps), then an exact top-k over the 2*k survivors. Misses an entry only if
    # a single block holds three of the true top-k. Orders of magnitude cheaper
    # than the segmented radix sort a full top_k lowers to.
    *lead, vocab = logits.shape
    padded = -(-vocab // k) * k
    block = padded // k
    padded_logits = jnp.pad(logits, (*[(0, 0)] * len(lead), (0, padded - vocab)), constant_values=-jnp.inf)
    blocks = padded_logits.reshape(*lead, k, block)
    offsets = (jnp.arange(k, dtype=jnp.int32) * block)[(None,) * len(lead)]
    first_index = jnp.argmax(blocks, axis=-1).astype(jnp.int32)
    first_value = jnp.take_along_axis(blocks, first_index[..., None], axis=-1)[..., 0]
    masked = jnp.where(jnp.arange(block, dtype=jnp.int32) == first_index[..., None], -jnp.inf, blocks)
    second_index = jnp.argmax(masked, axis=-1).astype(jnp.int32)
    second_value = jnp.take_along_axis(masked, second_index[..., None], axis=-1)[..., 0]
    survivor_values = jnp.concatenate([first_value, second_value], axis=-1)
    survivor_indices = jnp.concatenate([first_index + offsets, second_index + offsets], axis=-1)
    top_values, top_ranks = jax.lax.top_k(survivor_values, k)
    return top_values, jnp.take_along_axis(survivor_indices, top_ranks, axis=-1)


def build_weaver_tree(
    weaver: Weaver,
    lm_head: Float[Array, "vocab hidden"],
    embed_w: Float[Array, "vocab embed"],
    root_ids: Int[Array, " batch"],
    candidate_ids: Int[Array, "batch depth candidates"],
    candidate_scores: Float[Array, "batch depth candidates"],
    prefix: WeaverPrefix,
    tree_budget: int,
    keychain: Keychain,
) -> tuple[
    Int[Array, "batch budget"],
    Int[Array, "batch budget"],
    Int[Array, "batch budget"],
    Float[Array, "batch budget"],
    Bool[Array, "batch budget"],
]:
    batch_size, depth, pool_size = candidate_ids.shape
    node_budget = int(tree_budget)
    num_slots = node_budget + 1
    expand_width = min(EXPAND_WIDTH, pool_size)
    batch_expand_width = min(-(-node_budget // BATCH_EXPAND_BUDGET_UNIT), node_budget)
    num_expand_steps = -(-node_budget // batch_expand_width)
    frontier_slots = (node_budget + 1) * expand_width
    child_offsets = jnp.arange(expand_width, dtype=jnp.int32)
    batch_indices = jnp.arange(batch_size, dtype=jnp.int32)
    row_batch_indices = jnp.repeat(batch_indices, batch_expand_width)
    num_layers = weaver.config.num_layers
    num_heads = weaver.config.num_heads
    head_dim = weaver.config.head_dim

    output_tokens = jnp.zeros((batch_size, node_budget), dtype=jnp.int32)
    output_parents = jnp.zeros((batch_size, node_budget), dtype=jnp.int32)
    output_depths = jnp.zeros((batch_size, node_budget), dtype=jnp.int32)
    output_logprobs = jnp.zeros((batch_size, node_budget), dtype=jnp.float32)
    output_mask = jnp.zeros((batch_size, node_budget), dtype=jnp.bool)

    node_keys = jnp.zeros((num_layers, batch_size, num_slots, num_heads, head_dim), dtype=jnp.float32)
    node_values = jnp.zeros_like(node_keys)
    slot_ancestors = jnp.full((batch_size, num_slots, depth), -1, dtype=jnp.int32)
    slot_ancestors = slot_ancestors.at[:, 0, 0].set(0)

    frontier_tokens = jnp.zeros((batch_size, frontier_slots), dtype=jnp.int32)
    frontier_parents = jnp.zeros((batch_size, frontier_slots), dtype=jnp.int32)
    frontier_depths = jnp.zeros((batch_size, frontier_slots), dtype=jnp.int32)
    frontier_scores = jnp.full((batch_size, frontier_slots), -jnp.inf, dtype=jnp.float32)
    frontier_logprobs = jnp.zeros((batch_size, frontier_slots), dtype=jnp.float32)
    frontier_active = jnp.zeros((batch_size, frontier_slots), dtype=jnp.bool)

    def expand_nodes(
        token: Int[Array, " rows"],
        prefix_score: Float[Array, " rows"],
        node_depth: Int[Array, " rows"],
        ancestors: Int[Array, "rows depth"],
        current_node_keys: Float[Array, "layers batch slots heads head_dim"],
        current_node_values: Float[Array, "layers batch slots heads head_dim"],
        active: Bool[Array, " rows"],
        row_batches: Int[Array, " rows"],
    ) -> tuple[
        Int[Array, "rows expand"],
        Float[Array, "rows expand"],
        Float[Array, "rows expand"],
        Bool[Array, "rows expand"],
        Float[Array, "layers rows heads head_dim"],
        Float[Array, "layers rows heads head_dim"],
    ]:
        depth_index = jnp.minimum(node_depth, depth - 1)
        row_candidate_ids = candidate_ids[row_batches, depth_index]
        row_candidate_scores = candidate_scores[row_batches, depth_index]
        ancestor_slots = jnp.clip(ancestors, 0, num_slots - 1)
        ancestor_keys = current_node_keys[:, row_batches[:, None], ancestor_slots]
        ancestor_values = current_node_values[:, row_batches[:, None], ancestor_slots]
        ancestor_mask = (ancestors >= 0) & (jnp.arange(depth, dtype=jnp.int32)[None, :] < node_depth[:, None])
        logits, own_keys, own_values = weaver.step(
            lm_head,
            embed_w,
            jnp.where(active, token, 0).astype(jnp.int32),
            row_candidate_ids,
            row_candidate_scores,
            prefix.keys[:, row_batches],
            prefix.values[:, row_batches],
            depth_index.astype(jnp.int32),
            ancestor_keys,
            ancestor_values,
            ancestor_mask,
            keychain=keychain,
        )
        child_logits, child_ranks = _small_top_k(logits.astype(jnp.float32), expand_width)
        child_log_probs = child_logits - jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1, keepdims=True)
        child_tokens = jnp.take_along_axis(row_candidate_ids, child_ranks, axis=1)
        child_scores = prefix_score[:, None] + child_log_probs
        child_valid = (
            active[:, None] & (node_depth[:, None] < depth) & (child_tokens >= 0) & jnp.isfinite(child_logits)
        )
        return child_tokens, child_scores, child_log_probs, child_valid, own_keys, own_values

    (
        root_child_tokens,
        root_child_scores,
        root_child_logprobs,
        root_child_valid,
        root_keys,
        root_values,
    ) = expand_nodes(
        root_ids.astype(jnp.int32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.int32),
        slot_ancestors[:, 0],
        node_keys,
        node_values,
        jnp.ones((batch_size,), dtype=jnp.bool),
        batch_indices,
    )
    root_child_depths = jnp.ones((batch_size, expand_width), dtype=jnp.int32)
    frontier_tokens = frontier_tokens.at[:, child_offsets].set(jnp.where(root_child_valid, root_child_tokens, 0))
    frontier_parents = frontier_parents.at[:, child_offsets].set(0)
    frontier_depths = frontier_depths.at[:, child_offsets].set(jnp.where(root_child_valid, root_child_depths, 0))
    frontier_scores = frontier_scores.at[:, child_offsets].set(
        jnp.where(root_child_valid, root_child_scores, -jnp.inf),
    )
    frontier_logprobs = frontier_logprobs.at[:, child_offsets].set(
        jnp.where(root_child_valid, root_child_logprobs, 0.0),
    )
    frontier_active = frontier_active.at[:, child_offsets].set(root_child_valid)
    node_keys = node_keys.at[:, :, 0].set(root_keys)
    node_values = node_values.at[:, :, 0].set(root_values)

    def scan_step(
        carry: tuple[Array, ...],
        step_index: Int[Array, ""],
    ) -> tuple[tuple[Array, ...], None]:
        (
            out_tokens,
            out_parents,
            out_depths,
            out_logprobs,
            out_mask,
            keys,
            values,
            ancestors,
            f_tokens,
            f_parents,
            f_depths,
            f_scores,
            f_logprobs,
            f_active,
        ) = carry
        node_offsets = step_index * batch_expand_width + jnp.arange(batch_expand_width, dtype=jnp.int32)
        slot_offsets = node_offsets + 1
        _, frontier_indices = _small_top_k(
            jnp.where(f_active, f_scores, -jnp.inf),
            batch_expand_width,
        )
        frontier_indices = frontier_indices.astype(jnp.int32)
        in_budget = node_offsets[None, :] < node_budget
        valid = f_active[batch_indices[:, None], frontier_indices] & in_budget
        token = f_tokens[batch_indices[:, None], frontier_indices]
        parent = f_parents[batch_indices[:, None], frontier_indices]
        node_depth = f_depths[batch_indices[:, None], frontier_indices]
        node_score = f_scores[batch_indices[:, None], frontier_indices]
        node_logprob = f_logprobs[batch_indices[:, None], frontier_indices]

        out_tokens = out_tokens.at[:, node_offsets].set(jnp.where(valid, token, 0), mode="drop")
        out_parents = out_parents.at[:, node_offsets].set(jnp.where(valid, parent, 0), mode="drop")
        out_depths = out_depths.at[:, node_offsets].set(jnp.where(valid, node_depth, 0), mode="drop")
        out_logprobs = out_logprobs.at[:, node_offsets].set(jnp.where(valid, node_logprob, 0.0), mode="drop")
        out_mask = out_mask.at[:, node_offsets].set(valid, mode="drop")
        f_active = f_active.at[batch_indices[:, None], frontier_indices].set(
            f_active[batch_indices[:, None], frontier_indices] & ~in_budget,
        )

        parent_ancestors = ancestors[batch_indices[:, None], parent]
        depth_slots = jnp.arange(depth, dtype=jnp.int32)[None, None, :]
        node_ancestors = jnp.where(
            depth_slots == node_depth[:, :, None],
            slot_offsets[None, :, None],
            parent_ancestors,
        )
        ancestors = ancestors.at[batch_indices[:, None], slot_offsets[None, :]].set(node_ancestors, mode="drop")

        rows = batch_size * batch_expand_width
        child_tokens, child_scores, child_logprobs, child_valid, own_keys, own_values = expand_nodes(
            token.reshape(rows),
            node_score.reshape(rows),
            node_depth.reshape(rows),
            node_ancestors.reshape(rows, depth),
            keys,
            values,
            valid.reshape(rows),
            row_batch_indices,
        )
        own_keys = own_keys.reshape(num_layers, batch_size, batch_expand_width, num_heads, head_dim)
        own_values = own_values.reshape(num_layers, batch_size, batch_expand_width, num_heads, head_dim)
        keys = keys.at[:, :, slot_offsets].set(own_keys, mode="drop")
        values = values.at[:, :, slot_offsets].set(own_values, mode="drop")

        child_tokens = child_tokens.reshape(batch_size, batch_expand_width, expand_width)
        child_scores = child_scores.reshape(batch_size, batch_expand_width, expand_width)
        child_logprobs = child_logprobs.reshape(batch_size, batch_expand_width, expand_width)
        child_valid = child_valid.reshape(batch_size, batch_expand_width, expand_width)
        child_depths = node_depth[:, :, None] + 1
        child_valid = child_valid & (child_depths <= depth)
        frontier_write_indices = (slot_offsets[None, :, None] * expand_width + child_offsets[None, None, :]).reshape(
            1,
            batch_expand_width * expand_width,
        )
        frontier_write_indices = jnp.broadcast_to(
            frontier_write_indices,
            (batch_size, batch_expand_width * expand_width),
        )
        flat_children = batch_expand_width * expand_width
        f_tokens = f_tokens.at[batch_indices[:, None], frontier_write_indices].set(
            jnp.where(child_valid, child_tokens, 0).reshape(batch_size, flat_children),
            mode="drop",
        )
        f_parents = f_parents.at[batch_indices[:, None], frontier_write_indices].set(
            jnp.broadcast_to(slot_offsets[None, :, None], child_valid.shape).reshape(batch_size, flat_children),
            mode="drop",
        )
        f_depths = f_depths.at[batch_indices[:, None], frontier_write_indices].set(
            jnp.where(child_valid, jnp.broadcast_to(child_depths, child_valid.shape), 0).reshape(
                batch_size,
                flat_children,
            ),
            mode="drop",
        )
        f_scores = f_scores.at[batch_indices[:, None], frontier_write_indices].set(
            jnp.where(child_valid, child_scores, -jnp.inf).reshape(batch_size, flat_children),
            mode="drop",
        )
        f_logprobs = f_logprobs.at[batch_indices[:, None], frontier_write_indices].set(
            jnp.where(child_valid, child_logprobs, 0.0).reshape(batch_size, flat_children),
            mode="drop",
        )
        f_active = f_active.at[batch_indices[:, None], frontier_write_indices].set(
            child_valid.reshape(batch_size, flat_children),
            mode="drop",
        )
        return (
            out_tokens,
            out_parents,
            out_depths,
            out_logprobs,
            out_mask,
            keys,
            values,
            ancestors,
            f_tokens,
            f_parents,
            f_depths,
            f_scores,
            f_logprobs,
            f_active,
        ), None

    initial = (
        output_tokens,
        output_parents,
        output_depths,
        output_logprobs,
        output_mask,
        node_keys,
        node_values,
        slot_ancestors,
        frontier_tokens,
        frontier_parents,
        frontier_depths,
        frontier_scores,
        frontier_logprobs,
        frontier_active,
    )
    (output_tokens, output_parents, output_depths, output_logprobs, output_mask, *_), _ = jax.lax.scan(
        scan_step,
        initial,
        jnp.arange(num_expand_steps, dtype=jnp.int32),
    )
    return output_tokens, output_parents, output_depths, output_logprobs, output_mask


class WeaverDraftState(SpeculatorState):
    draft_state: DFlashDraftState
    root_output_norm: Float[Array, "batch hidden"]


@dataclass(frozen=True)
class WeaverSpeculatorConfig(SpeculatorConfig):
    draft_config: DFlashDraftConfig
    weaver_config: WeaverConfig
    tree_budget: int

    def init(self, tokenizer: Tokenizer, initializer: Initializer) -> "WeaverSpeculator":
        return WeaverSpeculator(
            config=self,
            sharding_config=initializer.sharding_config,
            token_codec=self.token_codec_config.init(tokenizer),
            draft_model=self.draft_config.init(initializer),
            weaver=self.weaver_config.init(initializer),
        )


class WeaverSpeculator(Speculator[WeaverDraftState, WeaverSpeculatorConfig]):
    draft_model: DFlashDraftModel
    weaver: Weaver

    @property
    def max_proposal_tokens(self) -> int:
        return self.config.tree_budget + 1

    @property
    def tree_depth(self) -> int:
        return min(self.draft_model.config.block_size - 1, self.weaver.config.k)

    @property
    def trace_layer_ids(self) -> tuple[int, ...] | None:
        return self.draft_model.config.target_layer_ids

    def target_matrices(
        self,
        target_embedding: EmbeddingBase,
    ) -> tuple[Float[Array, "vocab hidden"], Float[Array, "vocab embed"]]:
        lm_head = target_embedding.readout_matrix.to_full_precision().decompress()
        if lm_head.shape[-1] != self.weaver.config.d_model:
            lm_head = lm_head.swapaxes(-1, -2)
        embed_w = target_embedding.embedding_matrix.to_full_precision().decompress()
        if embed_w.shape[-1] != self.weaver.config.d_embed:
            embed_w = embed_w.swapaxes(-1, -2)
        return lm_head.astype(jnp.bfloat16), embed_w.astype(jnp.bfloat16)

    def extract_target_features(
        self,
        activation_trace: DecoderActivationTrace,
    ) -> Float[Array, "batch tokens target_channels"]:
        selected_outputs = []
        for layer_id in self.draft_model.config.target_layer_ids:
            layer_result = activation_trace.layer_results[layer_id]
            if layer_result is None:
                raise ValueError(f"Activation trace is missing target layer {layer_id}.")
            selected_outputs.append(layer_result.outputs)
        return jnp.concatenate(tuple(selected_outputs), axis=-1)

    def init_state(
        self,
        batch_size: int,
        context_capacity: int,
        dtype: DTypeLike,
    ) -> WeaverDraftState:
        return WeaverDraftState(
            draft_state=self.draft_model.empty_state(batch_size, context_capacity, dtype),
            root_output_norm=jnp.zeros((batch_size, self.weaver.config.d_model), dtype=jnp.float32),
        )

    def prefill_chunk(
        self,
        state: WeaverDraftState,
        decoder_result: DecoderResult,
        chunk_lengths: Int[Array, " batch"],
        *,
        keychain: Keychain,
    ) -> WeaverDraftState:
        activation_trace = decoder_result.activation_trace
        if activation_trace is None:
            raise ValueError("WeaverSpeculator requires decoder activation traces.")
        target_features = self.extract_target_features(activation_trace)
        draft_state = self.draft_model.append_state(
            state.draft_state,
            target_features,
            activation_trace.token_positions,
            chunk_lengths,
            keychain=keychain,
        )
        batch_indices = jnp.arange(chunk_lengths.shape[0], dtype=jnp.int32)
        last_slots = jnp.maximum(chunk_lengths - 1, 0)
        chunk_output_norm = activation_trace.output_norm[batch_indices, last_slots].astype(jnp.float32)
        root_output_norm = jnp.where(
            (chunk_lengths > 0)[:, None],
            chunk_output_norm,
            state.root_output_norm,
        )
        return WeaverDraftState(draft_state=draft_state, root_output_norm=root_output_norm)

    def update_state(
        self,
        state: WeaverDraftState,
        decoder_result: DecoderResult,
        accepted: AcceptedProposal,
        *,
        keychain: Keychain,
    ) -> WeaverDraftState:
        activation_trace = decoder_result.activation_trace
        if activation_trace is None:
            raise ValueError("WeaverSpeculator requires decoder activation traces.")
        target_features = self.extract_target_features(activation_trace)
        batch_size, num_accepted_slots = accepted.accepted_node_indices.shape
        batch_indices = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        accepted_slots = jnp.arange(num_accepted_slots, dtype=accepted.num_accepted_nodes.dtype)[None, :]
        valid = accepted_slots < accepted.num_accepted_nodes[:, None]
        source_indices = jnp.clip(accepted.accepted_node_indices, 0, target_features.shape[1] - 1)
        selected_target_features = target_features.at[batch_indices, source_indices, :].get(
            out_sharding=self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None)),
        )
        selected_token_positions = activation_trace.token_positions.at[batch_indices, source_indices].get(
            out_sharding=self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None)),
        )
        draft_state = self.draft_model.append_state(
            state.draft_state,
            jnp.where(valid[:, :, None], selected_target_features, 0),
            jnp.where(valid, selected_token_positions, 0),
            accepted.num_accepted_nodes,
            keychain=keychain,
        )
        terminal_slots = jnp.maximum(accepted.num_accepted_nodes - 1, 0)
        terminal_indices = accepted.accepted_node_indices[
            jnp.arange(batch_size, dtype=jnp.int32),
            terminal_slots,
        ]
        terminal_indices = jnp.clip(terminal_indices, 0, activation_trace.output_norm.shape[1] - 1)
        terminal_output_norm = activation_trace.output_norm[
            jnp.arange(batch_size, dtype=jnp.int32),
            terminal_indices,
        ].astype(jnp.float32)
        root_output_norm = jnp.where(
            (accepted.num_accepted_nodes > 0)[:, None],
            terminal_output_norm,
            state.root_output_norm,
        )
        return WeaverDraftState(draft_state=draft_state, root_output_norm=root_output_norm)

    def empty_proposal(
        self,
        token_positions: Int[Array, "batch nodes"],
        token_dtype: DTypeLike,
    ) -> Proposal:
        batch_size, num_nodes = token_positions.shape
        parent_indices = jnp.broadcast_to(
            jnp.arange(num_nodes, dtype=jnp.int32)[None, :] - 1,
            (batch_size, num_nodes),
        )
        return TreeProposal.empty(token_positions, token_dtype, parent_indices, self.tree_depth)

    def draft(
        self,
        state: WeaverDraftState,
        last_token_ids: Int[Array, " batch"],
        last_token_indices: Int[Array, " batch"],
        target_embedding: EmbeddingBase,
        *,
        keychain: Keychain,
    ) -> TreeProposal:
        block_size = self.draft_model.config.block_size
        (batch_size,) = last_token_ids.shape
        embedding_keychain, draft_keychain, readout_keychain, weaver_keychain = keychain.split(4)

        mask_token_ids = jnp.full(
            (batch_size, block_size - 1),
            self.draft_model.config.mask_token_id,
            dtype=last_token_ids.dtype,
        )
        noise_block = jnp.concatenate((last_token_ids[:, None], mask_token_ids), axis=1)
        noise_embeddings = target_embedding.embed(noise_block, keychain=embedding_keychain)

        draft_hidden_states = self.draft_model(
            noise_embeddings,
            state.draft_state,
            last_token_indices,
            keychain=draft_keychain,
        )
        batch_axis = self.draft_model.sharding_config.resolve_axis(LogicalAxis.BATCH)
        depth = self.tree_depth
        draft_logits = call_vmapped_twice(
            target_embedding.readout,
            draft_hidden_states[:, 1 : 1 + depth, :],
            keychain=readout_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        pool_size = min(self.weaver.config.candidate_pool_size, draft_logits.shape[-1])
        candidate_scores, candidate_ids = _vocab_top_k(draft_logits.astype(jnp.float32), pool_size)

        lm_head, embed_w = self.target_matrices(target_embedding)
        prefix = self.weaver.prompt_prefix(
            state.root_output_norm,
            draft_hidden_states[:, 1 : 1 + depth, :].astype(jnp.float32),
            keychain=weaver_keychain,
        )
        node_tokens, node_parents, node_depths, node_logprobs, node_mask = build_weaver_tree(
            self.weaver,
            lm_head,
            embed_w,
            last_token_ids,
            candidate_ids,
            candidate_scores,
            prefix,
            self.config.tree_budget,
            weaver_keychain,
        )

        token_ids = jnp.concatenate(
            [last_token_ids[:, None], node_tokens.astype(last_token_ids.dtype)],
            axis=1,
        )
        depths = jnp.concatenate([jnp.zeros((batch_size, 1), dtype=jnp.int32), node_depths], axis=1)
        parent_indices = jnp.concatenate(
            [jnp.full((batch_size, 1), -1, dtype=jnp.int32), node_parents],
            axis=1,
        )
        token_positions = (last_token_indices[:, None] + 1 + depths).astype(last_token_indices.dtype)
        lengths = 1 + jnp.sum(node_mask, axis=1).astype(last_token_indices.dtype)
        draft_logprobs = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=jnp.float32), node_logprobs],
            axis=1,
        )
        return TreeProposal(
            token_ids=token_ids,
            token_positions=token_positions,
            parent_indices=parent_indices,
            draft_logprobs=draft_logprobs,
            lengths=lengths,
            max_depth=depth,
        )
