from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset, reduce_precision
from jaxtyping import Array, Bool, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.modules.flow import (
    AttentionImplementation,
    Context,
    FlowForwardPassConfig,
    FlowResult,
    MixerForwardPassConfig,
    MLPForwardPassConfig,
    Sequence,
    SequenceConfig,
    State,
    TokenMixerConfig,
    leaves,
    run,
    validate_program,
)
from lalamo.modules.linear import Linear, LinearConfig
from lalamo.modules.normalization import Normalization, NormalizationConfig
from lalamo.modules.token_mixers.attention import AttentionConfig
from lalamo.modules.token_mixers.kv_cache import DynamicKVCacheLayer, KVCacheLayer
from lalamo.modules.utils import call_vmapped, call_vmapped_twice
from lalamo.utils.precision import use_dot_algorithm_preset
from lalamo.utils.sharding import lookup_sharded_indices
from lalamo.weight_matrix import MatmulConfig

__all__ = ["Weaver", "WeaverConfig", "WeaverPrefix"]

# The scorer runs in FP32, with a BF16 dot only at its final LM-head contribution.
_MATMUL_CONFIG = MatmulConfig(precision=DotAlgorithmPreset.F32_F32_F32)


class WeaverPrefix(eqx.Module):
    keys: Float[Array, "owners batch prefix groups head_channels"]
    values: Float[Array, "owners batch prefix groups head_channels"]


def _cache_arrays(state: State | None) -> tuple[Array, Array]:
    assert state is not None
    caches = []
    for cache in state:
        assert isinstance(cache, KVCacheLayer)
        caches.append(cache)
    return jnp.stack([cache.keys for cache in caches]), jnp.stack([cache.values for cache in caches])


@dataclass(frozen=True)
class WeaverConfig(LalamoConfig):
    model_dim: int
    target_model_dim: int
    target_embedding_dim: int
    max_depth: int
    candidate_pool_size: int
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    program: SequenceConfig

    def __post_init__(self) -> None:
        validate_program(self.program)
        mixers = tuple(step for step in leaves(self.program) if isinstance(step, TokenMixerConfig))
        if not mixers or any(not isinstance(mixer, AttentionConfig) for mixer in mixers):
            raise ValueError("Weaver requires attention-owned prefix caches.")
        owners = tuple(mixer for mixer in mixers if isinstance(mixer, AttentionConfig) and mixer.kv_source is None)
        first, *remaining = owners
        if any((owner.num_groups, owner.head_dim) != (first.num_groups, first.head_dim) for owner in remaining):
            # The external tree scorer exchanges a single stacked tensor of attention caches.
            raise ValueError("Weaver's stacked prefix caches require matching KV head dimensions.")

    def init(self, initializer: Initializer) -> "Weaver":
        def linear(input_dim: int, output_dim: int, *, has_biases: bool) -> Linear:
            return self.linear_config.init(initializer, input_dim, (output_dim,), has_biases=has_biases)

        return Weaver(
            config=self,
            sharding_config=initializer.sharding_config,
            embedding_norm=self.norm_config.init(initializer, self.target_embedding_dim),
            hidden_state_norm=self.norm_config.init(initializer, self.target_model_dim),
            embedding_projection=linear(self.target_embedding_dim, self.model_dim, has_biases=True),
            hidden_state_projection=linear(self.target_model_dim, self.model_dim, has_biases=True),
            query_projection=linear(self.model_dim, self.target_model_dim, has_biases=False),
            program=self.program.init(initializer, self.model_dim),
        )


class Weaver(LalamoModule[WeaverConfig]):
    embedding_norm: Normalization
    hidden_state_norm: Normalization
    embedding_projection: Linear
    hidden_state_projection: Linear
    query_projection: Linear
    program: Sequence

    def _evaluate(
        self,
        inputs: Float[Array, "batch tokens channels"],
        positions: Int[Array, "batch tokens"],
        state: State | None,
        *,
        keychain: Keychain,
    ) -> FlowResult:
        with use_dot_algorithm_preset(_MATMUL_CONFIG.precision):
            return run(
                self.program,
                inputs,
                Context(
                    None,
                    positions,
                    None,
                    None,
                    FlowForwardPassConfig(
                        mixer_forward_pass_config=MixerForwardPassConfig(
                            attention_implementation=AttentionImplementation.XLA,
                            matmul_config=_MATMUL_CONFIG,
                        ),
                        mlp_forward_pass_config=MLPForwardPassConfig(matmul_config=_MATMUL_CONFIG),
                    ),
                    self.sharding_config,
                ),
                state,
                keychain=keychain,
                return_updated_state=True,
            )

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
            forward_pass_config=_MATMUL_CONFIG,
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
        projection_keychain, program_keychain = keychain.split()
        batch_axis = self.sharding_config.resolve_axis(LogicalAxis.BATCH)
        features = jnp.concatenate((last_token_features[:, None], proposal_features), axis=1).astype(jnp.float32)
        (inputs,) = call_vmapped_twice(
            self.hidden_state_projection,
            call_vmapped_twice(self.hidden_state_norm, features),
            forward_pass_config=_MATMUL_CONFIG,
            keychain=projection_keychain,
            added_sharding_axes=(batch_axis, None),
        )
        positions = jnp.broadcast_to(jnp.arange(inputs.shape[1], dtype=jnp.int32), inputs.shape[:2])
        result = self._evaluate(inputs, positions, None, keychain=program_keychain)
        keys, values = _cache_arrays(result.state)
        return WeaverPrefix(keys, values)

    def step(
        self,
        lm_head: Float[Array, "vocab target_channels"],
        embedding_weights: Float[Array, "vocab embedding_channels"],
        token_ids: Int[Array, " rows"],
        candidate_ids: Int[Array, "rows candidates"],
        candidate_scores: Float[Array, "rows candidates"],
        prefix_keys: Float[Array, "owners rows prefix groups head_channels"],
        prefix_values: Float[Array, "owners rows prefix groups head_channels"],
        positions: Int[Array, " rows"],
        ancestor_keys: Float[Array, "owners rows depth groups head_channels"],
        ancestor_values: Float[Array, "owners rows depth groups head_channels"],
        ancestor_mask: Bool[Array, "rows depth"],
        *,
        keychain: Keychain,
    ) -> tuple[
        Float[Array, "rows candidates"],
        Float[Array, "owners rows groups head_channels"],
        Float[Array, "owners rows groups head_channels"],
    ]:
        token_keychain, program_keychain, query_keychain = keychain.split(3)
        inputs = self.token_project(token_ids, embedding_weights, keychain=token_keychain)[:, None]
        prefix_length = prefix_keys.shape[2]
        ancestor_depth = ancestor_keys.shape[2]
        key_mask = jnp.pad(ancestor_mask, ((0, 0), (prefix_length, 0)), constant_values=True)
        state = State(
            DynamicKVCacheLayer(
                has_sinks=False,
                keys=jnp.concatenate((prefix_key, ancestor_key), axis=1),
                values=jnp.concatenate((prefix_value, ancestor_value), axis=1),
                padding_mask=key_mask,
            )
            for prefix_key, prefix_value, ancestor_key, ancestor_value in zip(
                prefix_keys, prefix_values, ancestor_keys, ancestor_values, strict=True
            )
        )
        position_ids = jnp.clip(positions, 0, ancestor_depth - 1)[:, None] + 1
        result = self._evaluate(inputs, position_ids, state, keychain=program_keychain)
        (query,) = call_vmapped(
            self.query_projection,
            result.outputs[:, 0],
            forward_pass_config=_MATMUL_CONFIG,
            keychain=query_keychain,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        # BF16 operands and output, with FP32 products and accumulation.
        query = reduce_precision(query, exponent_bits=8, mantissa_bits=7)
        candidate_weights = reduce_precision(
            lookup_sharded_indices(lm_head, jnp.maximum(candidate_ids, 0)).astype(jnp.float32),
            exponent_bits=8,
            mantissa_bits=7,
        )
        residual = reduce_precision(
            (query[:, None, :] * candidate_weights).sum(axis=-1), exponent_bits=8, mantissa_bits=7
        )
        valid = (candidate_ids >= 0) & jnp.isfinite(candidate_scores)
        logits = jnp.where(valid, candidate_scores.astype(jnp.float32) + residual, -jnp.inf)
        keys, values = _cache_arrays(result.state)
        return logits, keys[:, :, -1], values[:, :, -1]
