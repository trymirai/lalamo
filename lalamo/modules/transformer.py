from dataclasses import dataclass, field

import equinox as eqx
from frozendict import frozendict
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis

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
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

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
    positional_embeddings: tuple[PositionalEmbeddings, ...] | None = None
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
        input_embeddings: Float[Array, "batch suffix_tokens channels"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        per_layer_inputs: tuple[Float[Array, "batch suffix_tokens ple_channels"], ...] | None = None,
        tree_ancestor_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_only_suffix_tokens: int | None = None,
        return_updated_state: bool = True,
        return_activations: bool = False,
        *,
        keychain: Keychain,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
    ) -> TransformerResult:
        inner_features = input_embeddings
        if state is None:
            state = State()

        last_layer_owning_state = -1
        for i, _ in enumerate(self.layers):
            if i not in self.config.kv_reuse_map:
                last_layer_owning_state = i

        if return_only_suffix_tokens is not None:
            suffix_token_positions = gather_suffix_tokens(
                token_positions,
                lengths_without_padding,
                return_only_suffix_tokens,
                self.sharding_config,
            )
        else:
            suffix_token_positions = None

        residual_dtype = input_embeddings.dtype
        layer_keychains = keychain.split(len(self.layers))
        updated_states: dict[int, StateLayerBase] = {}
        positional_embeddings: list[PositionalEmbeddings] = []
        layer_results = []

        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            assert input_embeddings.dtype == residual_dtype
            runs_on_suffix_only = last_state_owner_index is not None and layer_index > last_state_owner_index
            active_token_positions = suffix_token_positions if runs_on_suffix_only else token_positions
            assert active_token_positions is not None
            if layer.rope is None:
                positional_embeddings = None
            else:
                positional_embeddings = call_vmapped(
                    layer.rope,
                    active_token_positions,
                    added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
                ).astype(mixer_forward_pass_config.rope_dtype)
                positional_embeddings.append(positional_embeddings)

            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None
            if runs_on_suffix_only and per_layer_input is not None:
                assert return_only_suffix_tokens is not None
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_only_suffix_tokens,
                    self.sharding_config,
                )

            source_layer_index = kv_reuse_map.get(layer_index)
            if source_layer_index is not None:
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
                positional_embeddings,
                state=layer_state,
                return_updated_state=must_return_source_state and source_layer_index is None,
                return_activation_trace=return_activations,
                lengths_without_padding=None if runs_on_suffix_only else lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                tree_ancestor_indices=tree_ancestor_indices,
                return_suffix_tokens=return_only_suffix_tokens if layer_index == last_state_owner_index else None,
                keychain=layer_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if source_layer_index is None and layer_result.updated_state is not None:
                updated_states[layer_index] = layer_result.updated_state

        assert input_embeddings.dtype == residual_dtype
        pre_norm_outputs = input_embeddings if return_only_suffix_tokens is not None else None
        normalized_outputs = call_vmapped_twice(
            self.output_norm,
            input_embeddings,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
        )

        if return_updated_state:
            compact_state = State(tuple(updated_states[layer_index] for layer_index in kv_cache_source_layers))
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_activations else None,
            rope_embeddings=tuple(positional_embeddings) if return_positional_embeddings else None,
            pre_norm_outputs=pre_norm_outputs,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return State(
            (i, layer.init_static_state(batch_size, capacity, dtype))
            for i, layer in enumerate(self.layers)
            if i not in self.config.kv_reuse_map
        )
