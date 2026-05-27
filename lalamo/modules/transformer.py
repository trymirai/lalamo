from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field

from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import State, StateLayerBase
from .transformer_layer import (
    TransformerForwardPassConfig,
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerResult,
)
from .utils import call_vmapped, call_vmapped_twice

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


@dataclass(frozen=True)
class TransformerConfig(LalamoConfig):
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    model_dim: int
    hidden_dim: int

    def _init_ropes(self, initializer: Initializer) -> tuple[tuple[RoPE, ...], tuple[int, ...]]:
        rope_cache: dict[RoPEConfig, int] = {}
        ropes: list[RoPE] = []
        rope_indices: list[int] = []
        for layer_config in self.layer_configs:
            rope_config = layer_config.rope_config
            if rope_config is None:
                rope_indices.append(-1)
                continue

            if rope_config not in rope_cache:
                rope_cache[rope_config] = len(ropes)
                ropes.append(rope_config.init(initializer))
            rope_indices.append(rope_cache[rope_config])
        return tuple(ropes), tuple(rope_indices)

    def init(self, initializer: Initializer) -> "Transformer":
        ropes, rope_indices = self._init_ropes(initializer)

        layers = tuple(
            layer_config.init(
                initializer,
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim if layer_config.hidden_dim is not None else self.hidden_dim,
            )
            for layer_config in self.layer_configs
        )
        output_norm = self.output_norm_config.init(initializer, self.model_dim)

        return Transformer(
            config=self,
            sharding_config=initializer.sharding_config,
            ropes=ropes,
            rope_indices=rope_indices,
            kv_source_layer_indices=tuple(
                layer_index
                for layer_index, layer_config in enumerate(self.layer_configs)
                if layer_config.kv_source_layer_index is None
            ),
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = field(static=True)
    kv_source_layer_indices: tuple[int, ...] = field(static=True)
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
        per_layer_inputs: tuple[Float[Array, "batch suffix_tokens ple_dim"], ...] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
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
        state_by_layer = (
            {layer_index: state[state_index] for state_index, layer_index in enumerate(self.kv_source_layer_indices)}
            if state is not None
            else {}
        )
        mixer_forward_pass_config = forward_pass_config.mixer_forward_pass_config
        rope_embeddings = tuple(
            call_vmapped(
                rope,
                token_positions,
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            ).astype(
                mixer_forward_pass_config.rope_dtype,
            )
            for rope in self.ropes
        )
        has_kv_sharing = len(self.kv_source_layer_indices) < len(self.layers)
        must_return_state = return_updated_state or has_kv_sharing

        residual_dtype = inner_features.dtype
        layer_keychains = keychain.split(len(self.layers))
        updated_states: dict[int, StateLayerBase | None] = {}
        layer_results = []

        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            assert inner_features.dtype == residual_dtype
            rope_index = self.rope_indices[layer_index]
            positional_embeddings = rope_embeddings[rope_index] if rope_index >= 0 else None

            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None

            kv_source_layer_index = layer.config.kv_source_layer_index
            if kv_source_layer_index is None:
                effective_state = state_by_layer.get(layer_index)
            else:
                effective_state = updated_states.get(
                    kv_source_layer_index,
                    state_by_layer.get(kv_source_layer_index),
                )

            layer_result = layer(
                inner_features,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                attention_parent_indices=attention_parent_indices,
                keychain=layer_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if kv_source_layer_index is None:
                updated_states[layer_index] = layer_result.updated_state

        assert inner_features.dtype == residual_dtype
        normalized_outputs = call_vmapped_twice(
            self.output_norm,
            inner_features,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
        )

        if return_updated_state:
            compact_state_layers = []
            for layer_index in self.kv_source_layer_indices:
                layer_state = updated_states[layer_index]
                if layer_state is None:
                    raise ValueError(f"Layer {layer_index} did not return an updated state.")
                compact_state_layers.append(layer_state)
            compact_state = State(tuple(compact_state_layers))
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_layer_results else None,
            rope_embeddings=rope_embeddings if return_positional_embeddings else None,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return State(
            self.layers[layer_index].init_static_state(batch_size, capacity, dtype)
            for layer_index in self.kv_source_layer_indices
        )
