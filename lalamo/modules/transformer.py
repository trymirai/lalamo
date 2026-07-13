from dataclasses import dataclass, field

import equinox as eqx
from frozendict import frozendict
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule

from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings
from .token_mixer import State, StateLayerBase
from .token_mixers.attention import Attention, AttentionConfig
from .token_mixers.kv_cache import KVCacheLayer
from .transformer_layer import (
    TransformerForwardPassConfig,
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerResult,
)
from .utils import call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "Transformer",
    "TransformerConfig",
    "TransformerForwardPassConfig",
    "TransformerResult",
]


class TransformerResult(Exportable, eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    rope_embeddings: tuple[PositionalEmbeddings, ...] | None = None
    pre_norm_outputs: Float[Array, "batch suffix_tokens channels"] | None = None


@dataclass(frozen=True)
class TransformerConfig(LalamoConfig):
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    model_dim: int
    hidden_dim: int
    kv_reuse_map: frozendict[int, int] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        for layer_index, source_index in self.kv_reuse_map.items():
            assert 0 <= source_index < layer_index < len(self.layer_configs)
            assert source_index not in self.kv_reuse_map
            assert isinstance(self.layer_configs[source_index].mixer_config, AttentionConfig)
            assert isinstance(self.layer_configs[layer_index].mixer_config, AttentionConfig)

    def init(self, initializer: Initializer) -> "Transformer":
        layers = tuple(
            layer_config.init(
                initializer,
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim if layer_config.hidden_dim is not None else self.hidden_dim,
                borrows_kv_cache=layer_index in self.kv_reuse_map,
            )
            for layer_index, layer_config in enumerate(self.layer_configs)
        )
        output_norm = self.output_norm_config.init(initializer, self.model_dim)

        return Transformer(
            config=self,
            sharding_config=initializer.sharding_config,
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    layers: tuple[TransformerLayer, ...]
    output_norm: Normalization

    @eqx.filter_jit
    def __call__(
        self,
        inner_features: Float[Array, "batch suffix_tokens channels"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None,
        return_updated_state: bool,
        return_layer_results: bool,
        return_positional_embeddings: bool,
        lengths_without_padding: Int[Array, " batch"] | None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_inputs: tuple[Float[Array, "batch suffix_tokens ple_channels"], ...] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerResult:
        if inner_features.ndim != 3:
            raise ValueError(
                "inner_features must be a 3D array of size (batch_size, sequence_length, hidden_dim),"
                f" got {inner_features.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D array of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )
        kv_reuse_map = self.config.kv_reuse_map
        kv_cache_source_layers = tuple(
            layer_index for layer_index in range(len(self.layers)) if layer_index not in kv_reuse_map
        )
        if return_suffix_tokens is not None:
            if return_layer_results or return_positional_embeddings:
                raise ValueError(
                    "return_suffix_tokens cannot be combined with return_layer_results"
                    " or return_positional_embeddings.",
                )
            if attention_parent_indices is not None:
                raise ValueError("return_suffix_tokens cannot be combined with attention_parent_indices.")
            if not kv_cache_source_layers:
                raise ValueError("return_suffix_tokens requires at least one layer that owns its state.")
            has_trailing_borrowers = kv_cache_source_layers[-1] < len(self.layers) - 1
            if (
                return_suffix_tokens > 1
                and has_trailing_borrowers
                and state is None
                and lengths_without_padding is not None
            ):
                # Dynamic KV caches anchor padded-row masks at the physical tail of the cache,
                # which only matches the gathered suffix positions for a single-token suffix.
                raise ValueError(
                    "return_suffix_tokens > 1 on padded batches requires a preallocated state"
                    " when the model has trailing KV-sharing layers.",
                )
        if state is None:
            state_by_layer: dict[int, StateLayerBase] = {}
        else:
            if len(state) != len(kv_cache_source_layers):
                raise ValueError(
                    f"state must contain {len(kv_cache_source_layers)} KV cache layers, got {len(state)}."
                )
            state_by_layer = dict(zip(kv_cache_source_layers, state, strict=True))

        has_borrowed_kv_cache = bool(kv_reuse_map)
        must_return_source_state = return_updated_state or has_borrowed_kv_cache

        if return_suffix_tokens is None:
            last_state_owner_index = None
            suffix_token_positions = None
        else:
            last_state_owner_index = kv_cache_source_layers[-1]
            suffix_token_positions = gather_suffix_tokens(
                token_positions,
                lengths_without_padding,
                return_suffix_tokens,
                self.sharding_config,
            )
        residual_dtype = inner_features.dtype
        layer_keychains = keychain.split(len(self.layers))
        updated_states: dict[int, StateLayerBase] = {}
        rope_embeddings: list[PositionalEmbeddings] = []
        layer_results = []

        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            assert inner_features.dtype == residual_dtype
            runs_on_suffix_only = last_state_owner_index is not None and layer_index > last_state_owner_index
            active_token_positions = suffix_token_positions if runs_on_suffix_only else token_positions
            assert active_token_positions is not None
            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None
            if runs_on_suffix_only and per_layer_input is not None:
                assert return_suffix_tokens is not None
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )

            source_layer_index = kv_reuse_map.get(layer_index)
            borrows_kv = source_layer_index is not None
            if borrows_kv:
                source_state = updated_states.get(source_layer_index, state_by_layer.get(source_layer_index))
                assert isinstance(source_state, KVCacheLayer)
                layer_state = source_state
            else:
                layer_state = state_by_layer.get(layer_index)
                assert (
                    not isinstance(layer.mixer, Attention)
                    or layer_state is None
                    or isinstance(layer_state, KVCacheLayer)
                )

            layer_result = layer(
                inner_features,
                active_token_positions,
                state=layer_state,
                return_updated_state=must_return_source_state and not borrows_kv,
                return_activation_trace=return_layer_results,
                lengths_without_padding=None if runs_on_suffix_only else lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                attention_parent_indices=attention_parent_indices,
                return_suffix_tokens=return_suffix_tokens if layer_index == last_state_owner_index else None,
                keychain=layer_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)
            if layer_result.positional_embeddings is not None:
                rope_embeddings.append(layer_result.positional_embeddings)

            if not borrows_kv and layer_result.updated_state is not None:
                updated_states[layer_index] = layer_result.updated_state

        assert inner_features.dtype == residual_dtype
        pre_norm_outputs = inner_features if return_suffix_tokens is not None else None
        normalized_outputs = call_vmapped_twice(
            self.output_norm,
            inner_features,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
        )

        if return_updated_state:
            compact_state = State(tuple(updated_states[layer_index] for layer_index in kv_cache_source_layers))
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_layer_results else None,
            rope_embeddings=tuple(rope_embeddings) if return_positional_embeddings else None,
            pre_norm_outputs=pre_norm_outputs,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        kv_cache_source_layers = (
            layer_index for layer_index in range(len(self.layers)) if layer_index not in self.config.kv_reuse_map
        )
        return State(
            self.layers[layer_index].init_static_state(batch_size, capacity, dtype)
            for layer_index in kv_cache_source_layers
        )
