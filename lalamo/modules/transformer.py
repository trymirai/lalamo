from dataclasses import dataclass

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.modules.token_mixers import AttentionConfig
from lalamo.modules.utils import vmap_twice

from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
from .normalization import Normalization, NormalizationConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixers import State
from .transformer_layer import (
    TransformerLayer,
    TransformerLayerActivationTrace,
    TransformerLayerConfig,
    TransformerLayerForwardPassConfig,
)

__all__ = [
    "Transformer",
    "TransformerConfig",
    "TransformerResult",
]


type TransformerForwardPassConfig = TransformerLayerForwardPassConfig


class TransformerActivationTrace(eqx.Module):
    layer_activation_trace: tuple[TransformerLayerActivationTrace, ...]
    global_positional_embeddings: PositionalEmbeddings | None = None
    local_positional_embeddings: PositionalEmbeddings | None = None


class TransformerResult(eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    activation_trace: TransformerActivationTrace | None = None


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
        return_activation_trace: bool,
        return_positional_embeddings: bool,
        lengths_without_padding: Int[Array, " batch"] | None,
        num_suffix_tokens_to_return: int | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: TransformerForwardPassConfig | None = None,
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
        activation_traces = []

        for i, (layer, state_layer) in enumerate(zip(self.layers, maybe_state, strict=True)):
            match layer.positional_embedding_selector:
                case PositionalEmbeddingSelector.LOCAL:
                    positional_embeddings_to_use = local_positional_embeddings
                case PositionalEmbeddingSelector.GLOBAL:
                    positional_embeddings_to_use = global_positional_embeddings
                case PositionalEmbeddingSelector.NONE:
                    positional_embeddings_to_use = None

            if i == len(self.layers) - 1:
                layer_num_suffix_tokens_to_return = num_suffix_tokens_to_return
            else:
                layer_num_suffix_tokens_to_return = None

            layer_result = layer(
                inner_features,
                positional_embeddings_to_use,
                state=state_layer,
                return_updated_state=return_updated_state,
                return_activation_trace=return_activation_trace,
                lengths_without_padding=lengths_without_padding,
                num_suffix_tokens_to_return=layer_num_suffix_tokens_to_return,
                forward_pass_mode=forward_pass_mode,
                forward_pass_config=forward_pass_config,
            )
            inner_features = layer_result.outputs
            activation_traces.append(layer_result.activation_trace)
            updated_state_layers.append(layer_result.updated_state)

        normalized_outputs = vmap_twice(self.output_norm)(inner_features)

        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=(State(updated_state_layers) if return_updated_state else None),
            activation_trace=tuple(activation_traces) if return_activation_trace else None,
            global_positional_embeddings=(global_positional_embeddings if return_positional_embeddings else None),
            local_positional_embeddings=(local_positional_embeddings if return_positional_embeddings else None),
        )

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return State(layer.init_static_state(batch_size, capacity) for layer in self.layers)
