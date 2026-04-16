from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Self

import equinox as eqx
import jax
from jax import vmap
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_mapping, require_tree
from lalamo.modules.utils import vmap_twice

from .common import ForwardPassMode, LalamoModule
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
    rope_embeddings: tuple[PositionalEmbeddings, ...] | None = None

    def export(self) -> ParameterTree:
        result: dict[str, ParameterTree | Array] = dict(
            outputs=self.outputs,
        )
        if self.updated_state is not None:
            result["updated_state"] = [state_layer.export() for state_layer in self.updated_state]
        if self.layer_results is not None:
            result["layer_results"] = [layer_result.export() for layer_result in self.layer_results]
        if self.rope_embeddings is not None:
            result["rope_embeddings"] = [emb.export() for emb in self.rope_embeddings]
        return result


@dataclass(frozen=True)
class TransformerConfig:
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig
    model_dim: int
    hidden_dim: int
    context_length: int

    def _init_ropes(self) -> tuple[tuple[RoPE, ...], tuple[int, ...]]:
        rope_cache: dict[RoPEConfig, int] = {}
        ropes: list[RoPE] = []
        rope_indices: list[int] = []
        for layer_config in self.layer_configs:
            rc = layer_config.rope_config
            if rc is None:
                rope_indices.append(-1)
            elif rc in rope_cache:
                rope_indices.append(rope_cache[rc])
            else:
                rope_cache[rc] = len(ropes)
                rope_indices.append(len(ropes))
                ropes.append(rc.init())
        return tuple(ropes), tuple(rope_indices)

    def _source_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, lc in enumerate(self.layer_configs) if lc.kv_source_layer is None)

    def random_init(self, *, key: PRNGKeyArray) -> "Transformer":
        ropes, rope_indices = self._init_ropes()

        layers_keys = jax.random.split(key, num=len(self.layer_configs))
        layers = tuple(
            layer_config.random_init(
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim or self.hidden_dim,
                key=layer_key,
            )
            for layer_key, layer_config in zip(layers_keys, self.layer_configs, strict=True)
        )
        output_norm = self.output_norm_config.init(self.model_dim)

        return Transformer(
            config=self,
            ropes=ropes,
            rope_indices=rope_indices,
            source_layer_indices=self._source_layer_indices(),
            layers=layers,
            output_norm=output_norm,
        )

    def empty(self) -> "Transformer":
        ropes, rope_indices = self._init_ropes()

        layers = tuple(
            layer_config.empty(
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim or self.hidden_dim,
            )
            for layer_config in self.layer_configs
        )
        output_norm = self.output_norm_config.empty(self.model_dim)

        return Transformer(
            config=self,
            ropes=ropes,
            rope_indices=rope_indices,
            source_layer_indices=self._source_layer_indices(),
            layers=layers,
            output_norm=output_norm,
        )


class Transformer(LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = eqx.field(static=True)
    source_layer_indices: tuple[int, ...] = eqx.field(static=True)
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
        per_layer_inputs: tuple[Float[Array, "batch suffix_tokens ple_dim"], ...] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
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

        # Unpack compact state (only source layers) into a dict keyed by layer index
        if state is not None:
            state_by_layer = {idx: state[j] for j, idx in enumerate(self.source_layer_indices)}
        else:
            state_by_layer: dict[int, None] = {}

        rope_embeddings = tuple(vmap(rope)(token_positions) for rope in self.ropes)

        # When KV sharing is active, source layers must always return updated state
        # so shared layers can use the source's extended KV cache
        has_kv_sharing = len(self.source_layer_indices) < len(self.layers)
        must_return_state = return_updated_state or has_kv_sharing

        updated_states = {}
        layer_results = []

        for i, layer in enumerate(self.layers):
            rope_idx = self.rope_indices[i]
            positional_embeddings = rope_embeddings[rope_idx] if rope_idx >= 0 else None

            per_layer_input = per_layer_inputs[i] if per_layer_inputs is not None else None

            kv_source = layer.config.kv_source_layer
            if kv_source is not None:
                effective_state = updated_states.get(kv_source, state_by_layer.get(kv_source))
            else:
                effective_state = state_by_layer.get(i)

            layer_result = layer(
                inner_features,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_mode=forward_pass_mode,
                attention_parent_indices=attention_parent_indices,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
            )

            inner_features = layer_result.outputs
            layer_results.append(layer_result)

            if kv_source is None:
                updated_states[i] = layer_result.updated_state

        normalized_outputs = vmap_twice(self.output_norm)(inner_features)

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

    def init_static_state(self, batch_size: int, capacity: int) -> State:
        return State(self.layers[i].init_static_state(batch_size, capacity) for i in self.source_layer_indices)

    def export_weights(self) -> ParameterTree:
        return dict(
            layers=[layer.export_weights() for layer in self.layers],
            output_norm=self.output_norm.export_weights(),
        )

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        weights = require_mapping(weights)
        assert isinstance(weights["layers"], Sequence)
        layers = [
            layer.import_weights(require_tree(lw)) for layer, lw in zip(self.layers, weights["layers"], strict=True)
        ]
        return replace(
            self,
            layers=tuple(layers),
            output_norm=self.output_norm.import_weights(require_tree(weights["output_norm"])),
        )
