from dataclasses import dataclass

import equinox as eqx
import jax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.kernels.attention import (
    paged_decode_attention,
    pallas_decode_attention,
    stable_reduction_attention,
    xla_attention,
)
from lalamo.module import Keychain, LogicalAxis
from lalamo.modules.flow import (
    AttentionImplementation,
    Context,
    MixerForwardPassConfig,
    TokenMixerBase,
    TokenMixerConfig,
    TokenMixerResult,
)
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.rope import PositionalEmbeddings, RoPEConfig
from lalamo.modules.utils import call_vmapped, call_vmapped_twice

from .kv_cache import DynamicKVCacheLayer, KVCacheLayer, PagedKVCacheLayer, StaticKVCacheLayer

__all__ = [
    "Attention",
    "AttentionConfig",
    "AttentionResult",
]


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


@dataclass(frozen=True, eq=False)
class AttentionConfig(TokenMixerConfig):
    # Equal hyperparameters do not identify the same KV owner, including in JIT cache keys.
    __eq__ = object.__eq__
    __hash__ = object.__hash__

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
    value_norm_config: NormalizationConfig | None = None
    # A borrower reads the cache of this earlier owner attention instead of projecting keys and values.
    kv_source: "AttentionConfig | None" = None
    values_from_keys: bool = False
    rope_config: RoPEConfig | None = None

    def __post_init__(self) -> None:
        if self.kv_source is not None and (self.values_from_keys or self.value_norm_config is not None):
            raise ValueError("A KV-sharing consumer has no key/value projections or value normalization.")
        if self.kv_source is not None:
            if self.kv_source.kv_source is not None:
                raise ValueError("A KV source must own its key/value cache.")
            if (self.kv_source.num_groups, self.kv_source.head_dim) != (self.num_groups, self.head_dim):
                raise ValueError("A KV source must have matching key/value head dimensions.")

    def state_owner(self) -> "AttentionConfig":
        if self.kv_source is not None:
            return self.kv_source
        return self

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
    ) -> "Attention":
        q_output_dim = self.num_heads * self.head_dim
        if self.kv_source is not None:
            output_dims = (q_output_dim,)
        elif self.values_from_keys:
            output_dims = (q_output_dim, self.num_groups * self.head_dim)
        else:
            output_dims = (q_output_dim, self.num_groups * self.head_dim, self.num_groups * self.head_dim)
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
                self.head_dim,
            )
        else:
            query_norm = None

        if self.key_norm_config is not None and self.kv_source is None:
            key_norm = self.key_norm_config.init(
                initializer,
                self.head_dim,
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
            value_norm=self.value_norm_config.init(initializer, self.head_dim) if self.value_norm_config else None,
            sinks=sinks,
        )


class Attention(TokenMixerBase[AttentionConfig, KVCacheLayer | PagedKVCacheLayer]):
    def requires_full_sequence(
        self, state: KVCacheLayer | PagedKVCacheLayer | None, ctx: Context, suffix: int
    ) -> bool:
        return self.config.kv_source is None or (
            suffix > 1 and ctx.lengths is not None and not isinstance(state, StaticKVCacheLayer)
        )

    qkvg_projection: Linear
    out_projection: Linear

    query_norm: Normalization | None
    key_norm: Normalization | None
    value_norm: Normalization | None

    sinks: Float[Array, " heads"] | None

    @property
    def model_dim(self) -> int:
        return self.qkvg_projection.input_dim

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

    def project_heads(
        self,
        inputs: Float[Array, "tokens channels"],
        positional_embeddings: PositionalEmbeddings | None,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "tokens heads head_channels"],
        Float[Array, "tokens groups head_channels"] | None,
        Float[Array, "tokens groups head_channels"] | None,
        Float[Array, "tokens heads*head_channels"] | None,
    ]:
        projections = call_vmapped(
            self.qkvg_projection,
            inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=keychain,
        )
        queries = self._prepare_heads(projections[0], self.config.num_heads, self.query_norm, positional_embeddings)
        gate = projections[-1] if self.config.has_gate else None
        if self.config.kv_source is not None:
            return queries, None, None, gate
        keys = projections[1]
        values = keys if self.config.values_from_keys else projections[2]
        keys = self._prepare_heads(keys, self.config.num_groups, self.key_norm, positional_embeddings)
        values = rearrange(
            values,
            "tokens (groups head_channels) -> tokens groups head_channels",
            groups=self.config.num_groups,
            head_channels=self.config.head_dim,
        )
        if self.value_norm is not None:
            values = call_vmapped_twice(self.value_norm, values)
        return queries, keys, values, gate

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
        if self.config.kv_source is not None:
            raise ValueError("KV-sharing attention layers do not own key/value projections.")
        _, keys, values, _ = self.project_heads(inputs, positional_embeddings, forward_pass_config, keychain=keychain)
        assert keys is not None and values is not None
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
        *,
        keychain: Keychain,
    ) -> AttentionResult:
        qkvg_keychain, out_keychain = keychain.split(2)
        queries, keys, values, gate = self.project_heads(
            inputs, positional_embeddings, forward_pass_config, keychain=qkvg_keychain
        )

        num_suffix_tokens, _, _ = queries.shape
        if self.config.kv_source is not None:
            if state is None:
                raise ValueError("a KV-sharing layer must receive the source layer's cache as `state`")
            prefix_length = state.current_prefix_length(num_suffix_tokens, length_without_padding)
            updated_state = state
        else:
            assert keys is not None and values is not None
            prefix_length = 0 if state is None else state.current_prefix_length()
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
            sink_bias = sink_bias.at[:, :, 0].set(self.sinks[:, None].astype(sink_bias.dtype))
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

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *,
        state: KVCacheLayer | PagedKVCacheLayer | None,
        ctx: Context,
        keychain: Keychain,
    ) -> TokenMixerResult[KVCacheLayer | PagedKVCacheLayer]:
        positional_embeddings = None
        if self.config.rope_config is not None:
            positional_embeddings = ctx.positional_embeddings(self.config.rope_config)
        if isinstance(state, PagedKVCacheLayer):
            assert ctx.lengths is None
            return self.paged_decode(
                x,
                positional_embeddings,
                state,
                ctx.forward_pass_config.mixer_forward_pass_config,
                keychain=keychain,
            )
        return self.forward_rows(x, state, ctx, positional_embeddings, keychain=keychain)

    def paged_decode(
        self,
        inputs: Float[Array, "batch 1 channels"],
        positional_embeddings: PositionalEmbeddings | None,
        state: PagedKVCacheLayer,
        forward_pass_config: MixerForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> TokenMixerResult[PagedKVCacheLayer]:
        qkvg_keychain, out_keychain = keychain.split(2)
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        queries, keys, values, gate = call_vmapped(
            self.project_heads,
            inputs,
            positional_embeddings,
            forward_pass_config=forward_pass_config,
            keychain=qkvg_keychain,
            added_sharding_axis=batch_axis,
        )
        if keys is not None and values is not None:
            state = state.append(keys[:, 0], values[:, 0])
        queries = queries[:, 0].astype(state.keys.dtype)
        scale = self.config.scale if self.config.scale is not None else self.config.head_dim**-0.5

        attention_output = paged_decode_attention(
            queries,
            state.keys,
            state.values,
            state.block_tables,
            state.lengths,
            scale=scale,
            logit_soft_cap=self.config.logit_soft_cap,
            sinks=self.sinks,
            sliding_window_size=self.config.sliding_window_size,
        )

        attention_output = rearrange(attention_output, "batch heads channels -> batch 1 (heads channels)")
        if gate is not None:
            attention_output *= jax.nn.sigmoid(gate)
        (outputs,) = call_vmapped_twice(
            self.out_projection,
            attention_output,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=out_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        return TokenMixerResult(outputs, state)

    def init_static_state(self, capacity: int, dtype: DTypeLike) -> StaticKVCacheLayer:
        return StaticKVCacheLayer.init(
            self.has_sinks,
            capacity,
            self.config.num_groups,
            self.config.head_dim,
            dtype,
            self.sharding_config,
        )
