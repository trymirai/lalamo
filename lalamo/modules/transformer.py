from dataclasses import dataclass

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field

from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import State, StateLayerBase
from .token_mixers.attention import Attention, AttentionConfig, AttentionProjectionMode
from .token_mixers.kv_cache import BorrowedKVCacheLayer, ExtendableKVCacheLayer, KVCacheLayer
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
    kv_source_per_layer: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        kv_source_per_layer = self.kv_source_per_layer
        if kv_source_per_layer is None:
            return

        num_layers = len(self.layer_configs)
        if len(kv_source_per_layer) != num_layers:
            raise ValueError(
                f"kv_source_per_layer must have {num_layers} entries, got {len(kv_source_per_layer)}.",
            )

        for layer_index, source_index in enumerate(kv_source_per_layer):
            if source_index < 0 or source_index >= num_layers:
                raise ValueError(
                    f"Layer {layer_index} has invalid KV source layer {source_index};"
                    f" expected a layer in [0, {num_layers}).",
                )
            if source_index > layer_index:
                raise ValueError(f"Layer {layer_index} cannot borrow a KV cache from later layer {source_index}.")
            if kv_source_per_layer[source_index] != source_index:
                raise ValueError(
                    f"Layer {layer_index} borrows from layer {source_index},"
                    " but borrowed layers must point to a KV source layer.",
                )

            mixer_config = self.layer_configs[layer_index].mixer_config
            if source_index == layer_index:
                if (
                    isinstance(mixer_config, AttentionConfig)
                    and mixer_config.projection_mode is AttentionProjectionMode.BORROWED_KV
                ):
                    raise ValueError(f"Layer {layer_index} owns its KV cache but uses borrowed KV projection.")
                continue

            if not isinstance(mixer_config, AttentionConfig):
                raise TypeError(
                    f"Layer {layer_index} borrows a KV cache but its mixer is {type(mixer_config).__name__}.",
                )
            if mixer_config.projection_mode is not AttentionProjectionMode.BORROWED_KV:
                raise ValueError(
                    f"Layer {layer_index} borrows a KV cache and must use borrowed KV projection,"
                    f" got {mixer_config.projection_mode.value}.",
                )
            source_mixer_config = self.layer_configs[source_index].mixer_config
            if not isinstance(source_mixer_config, AttentionConfig):
                raise TypeError(
                    f"Layer {layer_index} borrows from layer {source_index},"
                    f" but its source mixer is {type(source_mixer_config).__name__}.",
                )
            if source_mixer_config.projection_mode is AttentionProjectionMode.BORROWED_KV:
                raise ValueError(f"Layer {layer_index} borrows from layer {source_index}, but the source is borrowed.")

            if mixer_config.head_dim != source_mixer_config.head_dim:
                raise ValueError(
                    f"Layer {layer_index} head_dim {mixer_config.head_dim} does not match"
                    f" source layer {source_index} head_dim {source_mixer_config.head_dim}.",
                )
            if mixer_config.num_groups != source_mixer_config.num_groups:
                raise ValueError(
                    f"Layer {layer_index} num_groups {mixer_config.num_groups} does not match"
                    f" source layer {source_index} num_groups {source_mixer_config.num_groups}.",
                )
            if mixer_config.has_sinks != source_mixer_config.has_sinks:
                raise ValueError(f"Layer {layer_index} and source layer {source_index} disagree on sink tokens.")
            if mixer_config.is_causal != source_mixer_config.is_causal:
                raise ValueError(f"Layer {layer_index} and source layer {source_index} disagree on causality.")
            if mixer_config.sliding_window_size != source_mixer_config.sliding_window_size:
                raise ValueError(f"Layer {layer_index} and source layer {source_index} disagree on sliding window.")
            if self.layer_configs[layer_index].rope_config != self.layer_configs[source_index].rope_config:
                raise ValueError(f"Layer {layer_index} and source layer {source_index} disagree on RoPE config.")

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
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = field(static=True)
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
        kv_source_per_layer = self.config.kv_source_per_layer
        if kv_source_per_layer is None:
            kv_source_per_layer = tuple(range(len(self.layers)))
        kv_cache_source_layers = tuple(
            layer_index for layer_index, source_index in enumerate(kv_source_per_layer) if layer_index == source_index
        )
        if state is None:
            state_by_layer: dict[int, StateLayerBase] = {}
        else:
            if len(state) != len(kv_cache_source_layers):
                raise ValueError(
                    f"state must contain {len(kv_cache_source_layers)} KV cache layers, got {len(state)}."
                )
            state_by_layer = dict(zip(kv_cache_source_layers, state, strict=True))

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
        has_kv_sharing = len(kv_cache_source_layers) < len(self.layers)
        must_return_source_state = return_updated_state or has_kv_sharing

        residual_dtype = inner_features.dtype
        layer_keychains = keychain.split(len(self.layers))
        updated_states: dict[int, StateLayerBase | None] = {}
        layer_results = []

        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            assert inner_features.dtype == residual_dtype
            rope_index = self.rope_indices[layer_index]
            positional_embeddings = rope_embeddings[rope_index] if rope_index >= 0 else None

            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None

            source_layer_index = kv_source_per_layer[layer_index]
            borrows_kv = source_layer_index != layer_index
            if borrows_kv:
                source_state = updated_states.get(source_layer_index, state_by_layer.get(source_layer_index))
                if source_state is None:
                    raise ValueError(
                        f"Layer {layer_index} borrows state from layer {source_layer_index}, but it is missing.",
                    )
                if not isinstance(layer.mixer, Attention):
                    raise TypeError(
                        f"Layer {layer_index} borrows a KV cache but its mixer is {type(layer.mixer).__name__}.",
                    )
                if not isinstance(source_state, KVCacheLayer):
                    raise TypeError(
                        f"Layer {layer_index} borrows state from layer {source_layer_index},"
                        f" but got {type(source_state).__name__}.",
                    )
                effective_state = BorrowedKVCacheLayer.from_cache(source_state)
            else:
                source_state = state_by_layer.get(layer_index)
                if (
                    source_state is not None
                    and isinstance(layer.mixer, Attention)
                    and not isinstance(source_state, ExtendableKVCacheLayer)
                ):
                    raise TypeError(
                        f"Attention layer {layer_index} expected an extendable KV cache state,"
                        f" got {type(source_state).__name__}.",
                    )
                effective_state = source_state

            layer_result = layer(
                inner_features,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_source_state and not borrows_kv,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                attention_parent_indices=attention_parent_indices,
                keychain=layer_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if not borrows_kv and layer_result.updated_state is not None:
                updated_states[layer_index] = layer_result.updated_state

        assert inner_features.dtype == residual_dtype
        normalized_outputs = call_vmapped_twice(
            self.output_norm,
            inner_features,
            forward_pass_config=forward_pass_config.normalization_forward_pass_config,
        )

        if return_updated_state:
            compact_state_layers = []
            for layer_index in kv_cache_source_layers:
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
        kv_source_per_layer = self.config.kv_source_per_layer
        if kv_source_per_layer is None:
            kv_cache_source_layers = range(len(self.layers))
        else:
            kv_cache_source_layers = (
                layer_index
                for layer_index, source_index in enumerate(kv_source_per_layer)
                if layer_index == source_index
            )
        return State(
            self.layers[layer_index].init_static_state(batch_size, capacity, dtype)
            for layer_index in kv_cache_source_layers
        )
