from contextlib import nullcontext
from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jax.experimental.pallas.ops.gpu.decode_attention import gqa
from jax.experimental.pallas.ops.gpu.paged_attention import paged_attention
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import pallas_decode_attention, stable_reduction_attention, xla_attention
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings
from lalamo.modules.token_mixer import (
    AttentionImplementation,
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, PagedKVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
]


def _rms_normalize(
    inputs: Float[Array, "... channels"],
    eps: float,
) -> Float[Array, "... channels"]:
    upcasted_inputs = inputs.astype(jnp.float32)
    variance = jnp.mean(jnp.square(upcasted_inputs), axis=-1, keepdims=True)
    return (upcasted_inputs * jax.lax.rsqrt(variance + eps)).astype(inputs.dtype)


def _attention_kernel(
    queries: Float[Array, "dst_tokens heads head_channels"],
    keys: Float[Array, "src_tokens groups head_channels"],
    values: Float[Array, "src_tokens groups head_channels"],
    *,
    bias: Float[Array, "heads dst_tokens src_tokens"] | None,
    mask: Bool[Array, "dst_tokens src_tokens"],
    scale: float | None,
    logit_soft_cap: float | None,
    forward_pass_config: MixerForwardPassConfig,
) -> Float[Array, "dst_tokens heads head_channels"]:
    match forward_pass_config.attention_implementation:
        case AttentionImplementation.PALLAS:
            return pallas_decode_attention(
                queries,
                keys,
                values,
                bias,
                mask,
                scale,
                logit_soft_cap,
            )
        case AttentionImplementation.XLA:
            return xla_attention(
                queries,
                keys,
                values,
                bias,
                mask,
                scale,
                logit_soft_cap,
            )
        case AttentionImplementation.STABLE_REDUCTION:
            return stable_reduction_attention(
                queries,
                keys,
                values,
                bias=bias,
                mask=mask,
                scale=scale,
                logit_soft_cap=logit_soft_cap,
                tile_size=forward_pass_config.attention_tile_size,
                accumulation_dtype=forward_pass_config.attention_accumulation_dtype,
            )


AttentionResult = TokenMixerResult[KVCacheLayer]


@dataclass(frozen=True)
class AttentionConfig(TokenMixerConfig):
    # Unified QKVG projection; ungated attention omits the G output segment.
    qkvg_projection_config: LinearConfig
    out_projection_config: LinearConfig

    query_norm_config: NormalizationConfig | None
    key_norm_config: NormalizationConfig | None

    num_heads: int
    num_groups: int
    head_dim: int
    is_causal: bool
    scale: float | None
    sliding_window_size: int | None

    logit_soft_cap: float | None
    has_sinks: bool
    has_qkvg_biases: bool
    has_out_biases: bool
    # Query-width sigmoid gate appended as the final fused projection segment.
    has_gate: bool = False
    # Scale-free RMS normalization on values
    normalize_values: bool = False
    is_kv_sharing: bool = False

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        q_output_dim = self.num_heads * self.head_dim
        if self.is_kv_sharing:
            output_dims = (q_output_dim,)
        else:
            output_dims = (
                q_output_dim,
                self.num_groups * self.head_dim,
                self.num_groups * self.head_dim,
            )
        if self.has_gate:
            output_dims = (*output_dims, q_output_dim)

        qkvg_projection = self.qkvg_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=output_dims,
            has_biases=self.has_qkvg_biases,
        )
        out_projection = self.out_projection_config.init(
            initializer,
            self.num_heads * self.head_dim,
            (model_dim,),
            has_biases=self.has_out_biases,
        )

        if self.query_norm_config is not None:
            query_norm = self.query_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None and not self.is_kv_sharing:
            key_norm = self.key_norm_config.init(
                initializer,
                input_dim=self.head_dim,
            )
        else:
            key_norm = None

        if self.has_sinks:
            sinks = initializer.zeros((self.num_heads,))
        else:
            sinks = None

        return Attention(
            config=self,
            sharding_config=initializer.sharding_config,
            qkvg_projection=qkvg_projection,
            out_projection=out_projection,
            query_norm=query_norm,
            key_norm=key_norm,
            sinks=sinks,
        )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer]):
    qkvg_projection: Linear
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    @property
    def model_dim(self) -> int:
        return self.qkvg_projection.input_dim

    @property
    def use_sliding_window(self) -> bool:
        return self.config.sliding_window_size is not None

    @property
    def positional_embedding_selector(self) -> PositionalEmbeddingSelector:
        if self.use_sliding_window:
            return PositionalEmbeddingSelector.LOCAL
        return PositionalEmbeddingSelector.GLOBAL

    @property
    def has_sinks(self) -> bool:
        return self.sinks is not None

    @eqx.filter_jit
    def _prepare_heads(
        self,
        projection: Float[Array, "tokens channels"],
        num_heads: int,
        norm: Normalization | None,
        positional_embeddings: PositionalEmbeddings | None,
    ) -> Float[Array, "tokens heads head_channels"]:
        heads = rearrange(
            projection,
            "tokens (heads head_channels) -> tokens heads head_channels",
            heads=num_heads,
            head_channels=self.config.head_dim,
        )
        if norm is not None:
            heads = call_vmapped_twice(norm, heads)
        if positional_embeddings is not None:
            heads = call_vmapped(positional_embeddings.apply, heads, in_axes=1, out_axes=1)
        return heads

    def project_key_value_heads(
        self,
        inputs: Float[Array, "new_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "new_tokens groups head_channels"],
        Float[Array, "new_tokens groups head_channels"],
    ]:
        if self.config.is_kv_sharing:
            raise ValueError("KV-sharing attention layers do not own key/value projections.")
        _, keys, values, *_ = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        keys = self._prepare_heads(keys, self.config.num_groups, self.key_norm, positional_embeddings)
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        if self.config.normalize_values:
            values = _rms_normalize(values, eps=1e-6)
        return keys, values

    def __call__(
        self,
        inputs: Float[Array, "suffix_tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: KVCacheLayer | None = None,
        return_updated_state: bool = False,
        length_without_padding: Int[Array, ""] | int | None = None,
        forward_pass_config: MixerForwardPassConfig = MixerForwardPassConfig(),
        attention_parent_indices: Int[Array, " suffix_tokens"] | None = None,
        reuse_cache: bool = False,
        *,
        keychain: Keychain,
    ) -> AttentionResult:
        qkvg_keychain, out_keychain = keychain.split(2)
        assert reuse_cache == self.config.is_kv_sharing, "reuse_cache must match AttentionConfig.is_kv_sharing"
        projections = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=qkvg_keychain,
        )
        queries = projections[0]
        if self.config.has_gate:
            gate = projections[-1]
        else:
            gate = None

        queries = self._prepare_heads(queries, self.config.num_heads, self.query_norm, positional_embeddings)

        num_suffix_tokens, _, _ = queries.shape
        if reuse_cache:
            if state is None:
                raise ValueError("a KV-sharing layer must receive the source layer's cache as `state`")
            prefix_length = state.current_prefix_length() - num_suffix_tokens
            updated_state = state
        else:
            _, keys, values, *_ = projections
            prefix_length = 0 if state is None else state.current_prefix_length()
            keys = self._prepare_heads(keys, self.config.num_groups, self.key_norm, positional_embeddings)
            values = rearrange(
                values,
                "tokens (groups head_channels) -> tokens groups head_channels",
                groups=self.config.num_groups,
                head_channels=self.config.head_dim,
            )
            if self.config.normalize_values:
                values = _rms_normalize(values, eps=1e-6)
            if state is None:
                updated_state = DynamicKVCacheLayer.init(
                    self.has_sinks, keys.astype(values.dtype), values, length=length_without_padding
                )
            else:
                updated_state = state.extend(keys, values, added_length=length_without_padding)

        queries = queries.astype(updated_state.keys.dtype)
        if attention_parent_indices is not None:
            mask = updated_state.tree_attention_mask(prefix_length, attention_parent_indices)
        else:
            mask = updated_state.attention_mask(
                num_suffix_tokens,
                self.config.is_causal,
                length_without_padding,
                self.config.sliding_window_size,
            )
        if self.sinks is not None:
            sink_bias = jnp.zeros((self.config.num_heads, *mask.shape), dtype=queries.dtype)
            sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None])
        else:
            sink_bias = None

        attention_output = _attention_kernel(
            queries,
            updated_state.keys,
            updated_state.values,
            bias=sink_bias,
            mask=mask,
            scale=self.config.scale,
            logit_soft_cap=self.config.logit_soft_cap,
            forward_pass_config=forward_pass_config,
        )
        attention_output = rearrange(
            attention_output,
            "tokens heads head_channels -> tokens (heads head_channels)",
            heads=self.config.num_heads,
            head_channels=self.config.head_dim,
        )
        if gate is not None:
            attention_output = attention_output * jax.nn.sigmoid(gate)
        (result,) = call_vmapped(
            self.out_projection,
            attention_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
        )

        if not return_updated_state:
            updated_state = None

        return AttentionResult(
            outputs=result,
            state=updated_state,
        )

    # fmt: off
    def paged_decode(
        self, inputs: Float[Array, "batch 1 channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: PagedKVCacheLayer,
        forward_pass_config: MixerForwardPassConfig,
        *, keychain: Keychain,
    ) -> TokenMixerResult[PagedKVCacheLayer]:
        qkvg_keychain, out_keychain = keychain.split(2)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        projections = call_vmapped_twice(
            self.qkvg_projection, inputs, forward_pass_config=forward_pass_config.matmul_config,
            keychain=qkvg_keychain, added_sharding_axes=(batch_axis, None),
        )
        queries = call_vmapped(
            lambda projection, embeddings: self._prepare_heads(
                projection, self.config.num_heads, self.query_norm, embeddings),
            projections[0], positional_embeddings, added_sharding_axis=batch_axis,
        )
        if self.config.is_kv_sharing:
            updated_keys, updated_values, updated_lengths = state.keys, state.values, state.lengths
        else:
            keys = call_vmapped(
                lambda projection, embeddings: self._prepare_heads(
                    projection, self.config.num_groups, self.key_norm, embeddings),
                projections[1], positional_embeddings, added_sharding_axis=batch_axis,
            )
            values = rearrange(
                projections[2], "batch 1 (groups channels) -> batch groups channels",
                groups=self.config.num_groups, channels=self.config.head_dim)
            if self.config.normalize_values:
                values = _rms_normalize(values, eps=1e-6)
            physical_lengths = state.lengths + int(self.has_sinks)
            physical_pages = jnp.take_along_axis(state.block_tables,
                                                 (physical_lengths // state.keys.shape[2])[:, None], axis=1)[:, 0]
            offsets = physical_lengths % state.keys.shape[2]
            groups = jnp.arange(state.keys.shape[0], dtype=jnp.int32)[:, None]
            updated_keys, updated_values = (
                cache.at[groups, physical_pages[None], offsets[None]].set(
                    rearrange(update.astype(cache.dtype), "batch groups channels -> groups batch channels"))
                for cache, update in zip((state.keys, state.values), (keys[:, 0], values), strict=True)
            )
            updated_lengths = physical_lengths + 1

        queries = queries[:, 0].astype(updated_keys.dtype)
        first_device, *_ = self.sharding_config.mesh.devices.flat
        scale = self.config.scale if self.config.scale is not None else queries.shape[-1] ** -0.5
        device_context = nullcontext()
        if float(getattr(first_device, "compute_capability", 0)) == 8.6:
            device = AbstractDevice("NVIDIA A10", num_cores=1, platform="gpu")
            device_context = use_abstract_mesh(AbstractMesh((), (), abstract_device=device))
        with device_context:
            if self.config.sliding_window_size is None:
                attention_output = paged_attention(
                    queries * scale, updated_keys, updated_values, state.block_tables, updated_lengths,
                    pages_per_compute_block=1, k_splits=min(state.block_tables.shape[1], 16),
                    num_stages=1, attn_logits_soft_cap=self.config.logit_soft_cap)
            else:
                required_pages = (self.config.sliding_window_size + 2 * state.keys.shape[2] - 2) // state.keys.shape[2]
                page_count = min(state.block_tables.shape[1], 1 << (required_pages - 1).bit_length())
                first_page = jnp.maximum(0, (updated_lengths - 1) // state.keys.shape[2] - page_count + 1)
                logical_pages = first_page[:, None] + jnp.arange(page_count, dtype=jnp.int32)
                physical_pages = jnp.take_along_axis(state.block_tables, logical_pages, axis=1)
                keys, values = (
                    rearrange(cache[:, physical_pages],
                              "groups batch pages page channels -> batch (pages page) groups channels")
                    for cache in (updated_keys, updated_values))
                offsets = first_page * state.keys.shape[2]
                with jax.numpy_dtype_promotion("standard"):
                    attention_output = gqa(
                        queries, keys, values,
                        jnp.maximum(0, updated_lengths - self.config.sliding_window_size) - offsets,
                        updated_lengths - offsets, sm_scale=scale, k_splits=min(page_count, 16)).astype(queries.dtype)

        attention_output = rearrange(attention_output, "batch heads channels -> batch 1 (heads channels)")
        if self.config.has_gate:
            attention_output *= jax.nn.sigmoid(projections[-1])
        (outputs,) = call_vmapped_twice(
            self.out_projection, attention_output, forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain, added_sharding_axes=(batch_axis, None))
        updated_state = PagedKVCacheLayer(updated_keys, updated_values, state.block_tables, updated_lengths)
        return TokenMixerResult(outputs, updated_state)

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
            self.sharding_config,
        )
