from dataclasses import dataclass
from dataclasses import field as dataclass_field

import equinox as eqx
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, ShardingAxis
from lalamo.modules.token_mixers import AttentionConfig

from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import State
from .transformer_layer import (
    PositionalEmbeddingSelector,
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerForwardPassConfig,
    TransformerLayerResult,
)
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "Transformer",
    "TransformerConfig",
    "TransformerForwardPassConfig",
    "TransformerResult",
]


@dataclass(frozen=True)
class TransformerForwardPassConfig:
    layer: TransformerLayerForwardPassConfig = dataclass_field(default_factory=TransformerLayerForwardPassConfig)


class TransformerResult(Exportable, eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    rope_embeddings: tuple[PositionalEmbeddings, ...] | None = None


@dataclass(frozen=True)
class TransformerConfig(LalamoConfig):
    global_rope_config: RoPEConfig | None
    local_rope_config: RoPEConfig | None
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    model_dim: int
    hidden_dim: int
    context_length: int

    def init(self, initializer: Initializer) -> "Transformer":
        rope_dims = (layer.rope_dim for layer in self.layer_configs if layer.rope_dim is not None)
        rope_dim = next(rope_dims, None)
        assert all(d == rope_dim for d in rope_dims)

        if self.global_rope_config:
            assert rope_dim is not None
            global_rope = self.global_rope_config.init(
                initializer,
                head_dim=rope_dim,
                num_timesteps=self.context_length,
            )
        else:
            global_rope = None

        if self.local_rope_config:
            assert rope_dim is not None
            max_sliding_window_size = max(
                layer_config.mixer_config.sliding_window_size or 0
                for layer_config in self.layer_configs
                if isinstance(layer_config.mixer_config, AttentionConfig)
            )
            local_rope = self.local_rope_config.init(
                initializer,
                head_dim=rope_dim,
                num_timesteps=max(max_sliding_window_size, self.context_length),
            )
        else:
            local_rope = None

        layers = tuple(
            layer_config.init(initializer, model_dim=self.model_dim, hidden_dim=self.hidden_dim)
            for layer_config in self.layer_configs
        )
        output_norm = self.output_norm_config.init(initializer, self.model_dim)

        return Transformer(
            config=self,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    global_rope: RoPE | None
    local_rope: RoPE | None
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
        forward_pass_mode: ForwardPassMode,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
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
        maybe_state = state or ([None] * len(self.layers))

        if self.global_rope is not None:
            global_positional_embeddings = call_vmapped(
                self.global_rope,
                token_positions,
                added_sharding_axis=ShardingAxis.DATA,
            )
        else:
            global_positional_embeddings = None
        if self.local_rope is not None:
            local_positional_embeddings = call_vmapped(
                self.local_rope,
                token_positions,
                added_sharding_axis=ShardingAxis.DATA,
            )
        else:
            local_positional_embeddings = global_positional_embeddings

        layer_keychains = keychain.split(len(self.layers))
        updated_state_layers = []
        layer_results = []

        for layer, state_layer, layer_keychain in zip(
            self.layers,
            maybe_state,
            layer_keychains,
            strict=True,
        ):
            match layer.positional_embedding_selector:
                case PositionalEmbeddingSelector.LOCAL:
                    positional_embeddings_to_use = local_positional_embeddings
                case PositionalEmbeddingSelector.GLOBAL:
                    positional_embeddings_to_use = global_positional_embeddings
                case PositionalEmbeddingSelector.NONE | _:
                    positional_embeddings_to_use = None

            layer_result = layer(
                inner_features,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_mode=forward_pass_mode,
                forward_pass_config=forward_pass_config.layer,
                keychain=layer_keychain,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if kv_source is None:
                updated_states[i] = layer_result.updated_state

        normalized_outputs = call_vmapped_twice(self.output_norm, inner_features)

        if return_updated_state:
            compact_state = State(updated_states[i] for i in self.source_layer_indices)
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_layer_results else None,
            rope_embeddings=rope_embeddings if return_positional_embeddings else None,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return State(layer.init_static_state(batch_size, capacity, dtype) for layer in self.layers)
