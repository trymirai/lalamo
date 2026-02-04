from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.token_mixers import AttentionConfig
from lalamo.modules.utils import vmap_twice

from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixers import State
from .transformer_layer import (
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerForwardPassConfig,
    TransformerLayerResult,
)

__all__ = [
    "Transformer",
    "TransformerConfig",
    "TransformerResult",
]


type TransformerForwardPassConfig = TransformerLayerForwardPassConfig


class TransformerResult(eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    global_positional_embeddings: PositionalEmbeddings | None = None
    local_positional_embeddings: PositionalEmbeddings | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            outputs=self.outputs,
        )
        if self.updated_state is not None:
            result["updated_state"] = [state_layer.export() for state_layer in self.updated_state]
        if self.layer_results is not None:
            result["layer_results"] = [layer_result.export() for layer_result in self.layer_results]
        if self.global_positional_embeddings is not None:
            result["global_positional_embeddings"] = self.global_positional_embeddings.export()
        if self.local_positional_embeddings is not None:
            result["local_positional_embeddings"] = self.local_positional_embeddings.export()
        return result


@dataclass(frozen=True)
class TransformerConfig:
    global_rope_config: RoPEConfig | None
    local_rope_config: RoPEConfig | None
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    model_dim: int
    hidden_dim: int
    context_length: int

    def random_init(self, *, key: PRNGKeyArray) -> "Transformer":
        rope_dims = (layer.rope_dim for layer in self.layer_configs if layer.rope_dim is not None)
        rope_dim = next(rope_dims, None)
        assert all(d == rope_dim for d in rope_dims)

        if self.global_rope_config:
            assert rope_dim is not None

            global_rope = self.global_rope_config.init(
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
                head_dim=rope_dim,
                num_timesteps=max(max_sliding_window_size, self.context_length),
            )
        else:
            local_rope = None

        layers_keys = jax.random.split(key, num=len(self.layer_configs))
        layers = tuple(
            layer_config.random_init(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                key=layer_key,
            )
            for layer_key, layer_config in zip(layers_keys, self.layer_configs, strict=True)
        )
        output_norm = self.output_norm_config.init(self.model_dim)

        return Transformer(
            config=self,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )

    def empty(self) -> "Transformer":
        rope_dims = (layer.rope_dim for layer in self.layer_configs if layer.rope_dim is not None)
        rope_dim = next(rope_dims, None)
        assert all(d == rope_dim for d in rope_dims)

        if self.global_rope_config:
            assert rope_dim is not None

            global_rope = self.global_rope_config.init(
                head_dim=rope_dim,
                num_timesteps=self.context_length,
            )
        else:
            global_rope = None

        if self.local_rope_config:
            assert rope_dim is not None

            local_rope = self.local_rope_config.init(
                head_dim=rope_dim,
                num_timesteps=self.context_length,
            )
        else:
            local_rope = None

        layers = tuple(
            layer_config.empty(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
            )
            for layer_config in self.layer_configs
        )
        output_norm = self.output_norm_config.empty(self.model_dim)

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

    @property
    def activation_precision(self) -> DTypeLike:
        return self.layers[0].activation_precision

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
        forward_pass_config: TransformerForwardPassConfig | None,
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
            global_positional_embeddings = vmap(self.global_rope)(token_positions)
        else:
            global_positional_embeddings = None
        if self.local_rope is not None:
            local_positional_embeddings = vmap(self.local_rope)(token_positions)
        else:
            local_positional_embeddings = global_positional_embeddings

        updated_state_layers = []
        layer_results = []

        for layer, state_layer in zip(self.layers, maybe_state, strict=True):
            match layer.positional_embedding_selector:
                case PositionalEmbeddingSelector.LOCAL:
                    positional_embeddings_to_use = local_positional_embeddings
                case PositionalEmbeddingSelector.GLOBAL:
                    positional_embeddings_to_use = global_positional_embeddings
                case PositionalEmbeddingSelector.NONE:
                    positional_embeddings_to_use = None

            layer_result = layer(
                inner_features,
                positional_embeddings_to_use,
                state=state_layer,
                return_updated_state=return_updated_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_mode=forward_pass_mode,
                forward_pass_config=forward_pass_config,
            )
            inner_features = layer_result.outputs
            layer_results.append(layer_result)
            updated_state_layers.append(layer_result.updated_state)

        normalized_outputs = vmap_twice(self.output_norm)(inner_features)

        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=(State(updated_state_layers) if return_updated_state else None),
            layer_results=tuple(layer_results) if return_layer_results else None,
            global_positional_embeddings=(global_positional_embeddings if return_positional_embeddings else None),
            local_positional_embeddings=(local_positional_embeddings if return_positional_embeddings else None),
        )

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return State(layer.init_static_state(batch_size, capacity) for layer in self.layers)

    def export_weights(self) -> ParameterTree:
        result = dict(
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )
        if self.global_rope:
            result["global_rope"] = self.global_rope.export_weights()
        if self.local_rope:
            result["local_rope"] = self.local_rope.export_weights()
        return result

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["layers"], Sequence)
        if self.global_rope:
            global_rope = self.global_rope.import_weights(require_tree(weights["global_rope"]))
        else:
            global_rope = None
        if self.local_rope:
            local_rope = self.local_rope.import_weights(require_tree(weights["local_rope"]))
        else:
            local_rope = None
        layers = [
            layer.import_weights(require_tree(lw)) for layer, lw in zip(self.layers, weights["layers"], strict=True)
        ]
        return replace(
            self,
            global_rope=global_rope,
            layers=tuple(layers),
            output_norm=self.output_norm.import_weights(require_tree(weights["output_norm"])),
            local_rope=local_rope,
        )
