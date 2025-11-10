from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree
from lalamo.modules.token_mixers import AttentionConfig

from .common import ForwardPassMode, LalamoModule, PositionalEmbeddingSelector
from .decoder_layer import DecoderLayer, DecoderLayerConfig, DecoderLayerForwardPassConfig, DecoderLayerResult
from .embedding import EmbeddingBase, EmbeddingConfig
from .normalization import RMSNorm, RMSNormConfig
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .state import State
from .utils import vmap_twice

__all__ = [
    "Decoder",
    "DecoderActivationTrace",
    "DecoderConfig",
    "DecoderForwardPassConfig",
    "DecoderResult",
]


type DecoderForwardPassConfig = DecoderLayerForwardPassConfig


class DecoderActivationTrace(eqx.Module):
    token_ids: Int[Array, "batch suffix_tokens"]
    token_positions: Int[Array, "batch suffix_tokens"]
    state: State | None

    local_positional_embeddings: PositionalEmbeddings
    global_positional_embeddings: PositionalEmbeddings

    layer_results: tuple[DecoderLayerResult, ...]

    output_norm: Float[Array, "batch suffix_tokens channels"]

    def export(self) -> ParameterTree:
        result = dict(
            token_ids=self.token_ids,
            token_positions=self.token_positions,
            local_positional_embeddings=self.local_positional_embeddings.export(),
            global_positional_embeddings=self.global_positional_embeddings.export(),
            layer_results=[layer_result.export() for layer_result in self.layer_results],
            output_norm=self.output_norm,
        )
        if self.state is not None:
            result["state"] = [state_layer.export() for state_layer in self.state]
        return result


class DecoderResult(eqx.Module):
    logits: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    activation_trace: DecoderActivationTrace | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            logits=self.logits,
        )
        if self.updated_state is not None:
            result["updated_state"] = [state_layer.export() for state_layer in self.updated_state]
        if self.activation_trace is not None:
            result["activation_trace"] = self.activation_trace.export()
        return result


@dataclass(frozen=True)
class DecoderConfig:
    embedding_config: EmbeddingConfig
    global_rope_config: RoPEConfig | None
    local_rope_config: RoPEConfig | None
    layer_configs: tuple[DecoderLayerConfig, ...]
    output_norm_config: RMSNormConfig

    vocab_size: int
    model_dim: int
    hidden_dim: int
    context_length: int

    def random_init(
        self,
        *,
        key: PRNGKeyArray,
    ) -> "Decoder":
        embedding_key, layers_key = jax.random.split(key)
        embedding = self.embedding_config.random_init(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            key=embedding_key,
        )

        first_layer_config, *_ = self.layer_configs

        if self.global_rope_config:
            global_rope = self.global_rope_config.init(
                head_dim=first_layer_config.rope_dim,
                num_timesteps=self.context_length,
            )
        else:
            global_rope = None

        if self.local_rope_config:
            max_sliding_window_size = max(
                layer_config.mixer_config.sliding_window_size or 0
                for layer_config in self.layer_configs
                if isinstance(layer_config.mixer_config, AttentionConfig)
            )
            local_rope = self.local_rope_config.init(
                head_dim=first_layer_config.rope_dim,
                num_timesteps=max(max_sliding_window_size, self.context_length),
            )
        else:
            local_rope = None

        layers_keys = jax.random.split(layers_key, len(self.layer_configs))
        layers = tuple(
            layer_config.random_init(
                model_dim=self.model_dim,
                hidden_dim=self.hidden_dim,
                key=key,
            )
            for layer_config, key in zip(self.layer_configs, layers_keys, strict=False)
        )
        output_norm = self.output_norm_config.init(self.model_dim)
        return Decoder(
            self,
            embedding=embedding,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )

    def empty(
        self,
    ) -> "Decoder":
        embedding = self.embedding_config.empty(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
        )

        first_layer_config, *_ = self.layer_configs

        if self.global_rope_config:
            global_rope = self.global_rope_config.init(
                head_dim=first_layer_config.rope_dim,
                num_timesteps=self.context_length,
            )
        else:
            global_rope = None

        if self.local_rope_config:
            max_sliding_window_size = max(
                layer_config.mixer_config.sliding_window_size or 0
                for layer_config in self.layer_configs
                if isinstance(layer_config.mixer_config, AttentionConfig)
            )
            local_rope = self.local_rope_config.init(
                head_dim=first_layer_config.rope_dim,
                num_timesteps=max(max_sliding_window_size, self.context_length),
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
        return Decoder(
            self,
            embedding=embedding,
            global_rope=global_rope,
            local_rope=local_rope,
            layers=layers,
            output_norm=output_norm,
        )


class Decoder(LalamoModule[DecoderConfig]):
    embedding: EmbeddingBase
    global_rope: RoPE | None
    local_rope: RoPE | None
    layers: tuple[DecoderLayer, ...]
    output_norm: RMSNorm

    @property
    def activation_precision(self) -> DTypeLike:
        return self.embedding.activation_precision

    @eqx.filter_jit
    def __call__(
        self,
        token_ids: Int[Array, "batch suffix_tokens"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: DecoderForwardPassConfig | None = None,
    ) -> DecoderResult:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be a 2D arrays of size (batch_size, sequence_length), got {token_ids.shape}",
            )
        if token_positions.ndim != 2:
            raise ValueError(
                "token_positions must be a 2D arrays of size (batch_size, sequence_length),"
                f" got {token_positions.shape}",
            )

        maybe_state = state or ([None] * len(self.layers))
        inner_features = vmap(self.embedding.embed)(token_ids)

        if self.global_rope is not None:
            global_positional_embeddings = vmap(self.global_rope)(token_positions)
        else:
            global_positional_embeddings = None
            raise NotImplementedError  # support this

        if self.local_rope is not None:
            local_positional_embeddings = vmap(self.local_rope)(token_positions)
        else:
            local_positional_embeddings = global_positional_embeddings

        updated_state_layers = []
        layer_results = []
        for layer, state_layer in zip(self.layers, maybe_state, strict=True):
            if layer.positional_embedding_selector == PositionalEmbeddingSelector.LOCAL:
                positional_embeddings_to_use = local_positional_embeddings
            else:
                positional_embeddings_to_use = global_positional_embeddings

            layer_result = layer(
                inner_features,
                positional_embeddings_to_use,
                state=state_layer,
                return_updated_state=return_updated_state,
                return_activation_trace=return_activation_trace,
                lengths_without_padding=lengths_without_padding,
                forward_pass_mode=forward_pass_mode,
                forward_pass_config=forward_pass_config,
            )
            inner_features = layer_result.outputs
            layer_results.append(layer_result)
            updated_state_layers.append(layer_result.updated_state)

        normalized_outputs = vmap_twice(self.output_norm)(inner_features)
        logits = vmap_twice(self.embedding.readout)(normalized_outputs)

        if return_activation_trace:
            activation_trace = DecoderActivationTrace(
                token_ids=token_ids,
                token_positions=token_positions,
                state=state,
                global_positional_embeddings=global_positional_embeddings,
                local_positional_embeddings=local_positional_embeddings,
                layer_results=tuple(layer_results),
                output_norm=normalized_outputs,
            )
        else:
            activation_trace = None

        if return_updated_state:
            updated_state = State(updated_state_layers)
        else:
            updated_state = None

        return DecoderResult(
            logits=logits,
            updated_state=updated_state,
            activation_trace=activation_trace,
        )

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return State(layer.init_static_state(batch_size, capacity) for layer in self.layers)

    def export_weights(self) -> ParameterTree:
        result = dict(
            embedding=self.embedding.export_weights(),
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )
        if self.global_rope:
            result["global_rope"] = self.global_rope.export_weights()
        if self.local_rope:
            result["local_rope"] = self.local_rope.export_weights()
        return result

    def import_weights(
        self,
        weights: ParameterTree[Array],
    ) -> Self:
        assert isinstance(weights, Mapping)
        assert isinstance(weights["embedding"], Mapping)
        assert isinstance(weights["global_rope"], Mapping)
        assert isinstance(weights["layers"], Sequence)
        assert isinstance(weights["output_norm"], Mapping)

        if self.local_rope:
            assert isinstance(weights["local_rope"], Mapping)
            local_rope = self.local_rope.import_weights(weights["local_rope"])
        else:
            local_rope = None

        if self.global_rope:
            assert isinstance(weights["global_rope"], Mapping)
            global_rope = self.global_rope.import_weights(weights["global_rope"])
        else:
            global_rope = None

        layers = []
        for layer, layer_weights in zip(self.layers, weights["layers"], strict=True):
            assert isinstance(layer_weights, Mapping)
            layers.append(layer.import_weights(layer_weights))
        return replace(
            self,
            embedding=self.embedding.import_weights(weights["embedding"]),
            global_rope=global_rope,
            layers=tuple(layers),
            output_norm=self.output_norm.import_weights(weights["output_norm"]),
            local_rope=local_rope,
        )
