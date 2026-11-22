from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field

from .normalization import Normalization, NormalizationConfig
from .residual import BlockAttentionResidualConfig, ResidualConfig, ResidualHeadBase
from .rope import PositionalEmbeddings, RoPE, RoPEConfig
from .token_mixer import State, TransformerLayerState
from .transformer_layer import (
    TransformerForwardPassConfig,
    TransformerLayer,
    TransformerLayerConfig,
    TransformerLayerResult,
)
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "LayerVisit",
    "LoopConfig",
    "Transformer",
    "TransformerConfig",
    "TransformerForwardPassConfig",
    "TransformerResult",
]


class LayerVisit(NamedTuple):
    layer_index: int
    loop_pass: int


@dataclass(frozen=True)
class LoopConfig(LalamoConfig):
    """Layers `start_layer` to `end_layer - 1` run `num_passes` times with shared weights.

    Every pass restarts from the stream that entered `start_layer`; the residual decides what the repeated passes
    learn from the previous one. Every layer visit keeps its own state.
    """

    start_layer: int
    end_layer: int
    num_passes: int

    def __post_init__(self) -> None:
        if not 0 < self.start_layer < self.end_layer:
            raise ValueError(f"Invalid looped layer range [{self.start_layer}, {self.end_layer}).")
        if self.num_passes < 1:
            raise ValueError(f"A loop needs at least one pass, got {self.num_passes}.")

    def is_looped(self, layer_index: int) -> bool:
        return self.start_layer <= layer_index < self.end_layer

    def layer_visits(self, num_layers: int) -> tuple[LayerVisit, ...]:
        if self.end_layer > num_layers:
            raise ValueError(f"The loop ends at layer {self.end_layer}, past the last of {num_layers} layers.")
        looped = range(self.start_layer, self.end_layer)
        return (
            *(LayerVisit(index, 0) for index in range(self.start_layer)),
            *(LayerVisit(index, loop_pass) for loop_pass in range(self.num_passes) for index in looped),
            *(LayerVisit(index, 0) for index in range(self.end_layer, num_layers)),
        )


class TransformerResult(Exportable, eqx.Module):
    outputs: Float[Array, "batch suffix_tokens channels"]
    updated_state: State | None = None
    layer_results: tuple[TransformerLayerResult, ...] | None = None
    rope_embeddings: tuple[PositionalEmbeddings, ...] | None = None
    pre_norm_outputs: Float[Array, "batch suffix_tokens channels"] | None = None


@dataclass(frozen=True)
class TransformerConfig(LalamoConfig):
    residual_config: ResidualConfig
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig | None
    model_dim: int
    hidden_dim: int
    loop_config: LoopConfig | None = None

    def __post_init__(self) -> None:
        if self.loop_config is None:
            return
        self.loop_config.layer_visits(len(self.layer_configs))
        if any(layer_config.kv_source_layer_index is not None for layer_config in self.layer_configs):
            raise ValueError("Looped transformers do not support KV sharing.")
        if any(layer_config.prologue_configs or layer_config.epilogue_configs for layer_config in self.layer_configs):
            raise ValueError("Looped transformers do not support stream ops.")
        if isinstance(self.residual_config, BlockAttentionResidualConfig):
            block_size = self.residual_config.block_size
            if self.loop_config.start_layer % block_size != 0 or self.loop_config.end_layer % block_size != 0:
                raise ValueError(f"Looped layers must start and end at block boundaries of {block_size} layers.")

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

        residual_head = self.residual_config.init_head(initializer, self.model_dim)
        layers = tuple(
            layer_config.init(
                initializer,
                residual_config=self.residual_config,
                model_dim=self.model_dim,
                hidden_dim=layer_config.hidden_dim if layer_config.hidden_dim is not None else self.hidden_dim,
                layer_index=layer_index,
                is_looped=self.loop_config is not None and self.loop_config.is_looped(layer_index),
            )
            for layer_index, layer_config in enumerate(self.layer_configs)
        )
        output_norm = (
            self.output_norm_config.init(initializer, self.model_dim) if self.output_norm_config is not None else None
        )

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
            residual_head=residual_head,
            layers=layers,
            output_norm=output_norm,
        )


class Transformer[StreamT](LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = field(static=True)
    kv_source_layer_indices: tuple[int, ...] = field(static=True)
    residual_head: ResidualHeadBase[ResidualConfig, StreamT]
    layers: tuple[TransformerLayer[StreamT], ...]
    output_norm: Normalization | None

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
        return_suffix_tokens: int | None = None,
        token_ids: Int[Array, "batch suffix_tokens"] | None = None,
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
        if self.config.loop_config is not None:
            if per_layer_inputs is not None:
                raise NotImplementedError("Looped transformers do not support per-layer inputs.")
            return self._call_looped(
                inner_features,
                token_positions,
                state,
                return_updated_state=return_updated_state,
                return_layer_results=return_layer_results,
                return_positional_embeddings=return_positional_embeddings,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=attention_parent_indices,
                return_suffix_tokens=return_suffix_tokens,
                token_ids=token_ids,
                keychain=keychain,
            )
        if return_suffix_tokens is not None:
            if return_layer_results or return_positional_embeddings:
                raise ValueError(
                    "return_suffix_tokens cannot be combined with return_layer_results"
                    " or return_positional_embeddings.",
                )
            if attention_parent_indices is not None:
                raise ValueError("return_suffix_tokens cannot be combined with attention_parent_indices.")
            if not self.kv_source_layer_indices:
                raise ValueError("return_suffix_tokens requires at least one layer that owns its state.")
            has_trailing_shared_layers = self.kv_source_layer_indices[-1] < len(self.layers) - 1
            if (
                return_suffix_tokens > 1
                and has_trailing_shared_layers
                and state is None
                and lengths_without_padding is not None
            ):
                # Dynamic KV caches anchor padded-row masks at the physical tail of the cache,
                # which only matches the gathered suffix positions for a single-token suffix.
                raise ValueError(
                    "return_suffix_tokens > 1 on padded batches requires a preallocated state"
                    " when the model has trailing KV-sharing layers.",
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

        # In suffix mode, every layer past the last state-owning one only affects the returned
        # suffix outputs, so those layers run on the gathered suffix instead of the full sequence.
        if return_suffix_tokens is not None:
            last_state_owner_index = self.kv_source_layer_indices[-1]
            suffix_token_positions = gather_suffix_tokens(
                token_positions,
                lengths_without_padding,
                return_suffix_tokens,
                self.sharding_config,
            )
            suffix_rope_embeddings = tuple(
                call_vmapped(
                    rope,
                    suffix_token_positions,
                    added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
                ).astype(
                    mixer_forward_pass_config.rope_dtype,
                )
                for rope in self.ropes
            )
            suffix_token_ids = (
                gather_suffix_tokens(token_ids, lengths_without_padding, return_suffix_tokens, self.sharding_config)
                if token_ids is not None
                else None
            )
        else:
            last_state_owner_index = None
            suffix_rope_embeddings = ()
            suffix_token_ids = None

        residual_dtype = inner_features.dtype
        stream = self.residual_head.expand(inner_features)
        *layer_keychains, readout_keychain = keychain.split(len(self.layers) + 1)
        updated_states: dict[int, TransformerLayerState | None] = {}
        layer_results = []

        for layer_index, (layer, layer_keychain) in enumerate(zip(self.layers, layer_keychains, strict=True)):
            assert all(leaf.dtype == residual_dtype for leaf in jax.tree.leaves(stream))
            runs_on_suffix_only = last_state_owner_index is not None and layer_index > last_state_owner_index
            rope_index = self.rope_indices[layer_index]
            active_rope_embeddings = suffix_rope_embeddings if runs_on_suffix_only else rope_embeddings
            positional_embeddings = active_rope_embeddings[rope_index] if rope_index >= 0 else None

            per_layer_input = per_layer_inputs[layer_index] if per_layer_inputs is not None else None
            if runs_on_suffix_only and per_layer_input is not None:
                assert return_suffix_tokens is not None
                per_layer_input = gather_suffix_tokens(
                    per_layer_input,
                    lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )
            layer_token_ids = suffix_token_ids if runs_on_suffix_only else token_ids

            kv_source_layer_index = layer.config.kv_source_layer_index
            if kv_source_layer_index is None:
                effective_state = state_by_layer.get(layer_index)
            else:
                effective_state = updated_states.get(
                    kv_source_layer_index,
                    state_by_layer.get(kv_source_layer_index),
                )

            layer_result = layer(
                stream,
                positional_embeddings,
                state=effective_state,
                return_updated_state=must_return_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=None if runs_on_suffix_only else lengths_without_padding,
                forward_pass_config=forward_pass_config,
                per_layer_input=per_layer_input,
                attention_parent_indices=attention_parent_indices,
                return_suffix_tokens=return_suffix_tokens if layer_index == last_state_owner_index else None,
                token_ids=layer_token_ids,
                keychain=layer_keychain,
            )

            stream = layer_result.outputs
            layer_results.append(layer_result)

            if kv_source_layer_index is None:
                updated_states[layer_index] = layer_result.updated_state

        outputs = self.residual_head.readout(
            stream,
            forward_pass_config.mlp_forward_pass_config.matmul_config,
            keychain=readout_keychain,
        )
        assert outputs.dtype == residual_dtype
        pre_norm_outputs = outputs if return_suffix_tokens is not None else None
        normalized_outputs = call_vmapped_twice(self.output_norm, outputs) if self.output_norm is not None else outputs

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
            pre_norm_outputs=pre_norm_outputs,
        )

    def _call_looped(
        self,
        inner_features: Float[Array, "batch suffix_tokens channels"],
        token_positions: Int[Array, "batch suffix_tokens"],
        state: State | None,
        *,
        return_updated_state: bool,
        return_layer_results: bool,
        return_positional_embeddings: bool,
        lengths_without_padding: Int[Array, " batch"] | None,
        forward_pass_config: TransformerForwardPassConfig,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None,
        return_suffix_tokens: int | None,
        token_ids: Int[Array, "batch suffix_tokens"] | None,
        keychain: Keychain,
    ) -> TransformerResult:
        loop_config = self.config.loop_config
        assert loop_config is not None
        visits = loop_config.layer_visits(len(self.layers))
        if state is not None and len(state) != len(visits):
            raise ValueError(f"A looped transformer needs one state per layer visit: {len(visits)}, got {len(state)}.")
        rope_embeddings = tuple(
            call_vmapped(
                rope,
                token_positions,
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            ).astype(forward_pass_config.mixer_forward_pass_config.rope_dtype)
            for rope in self.ropes
        )

        residual_dtype = inner_features.dtype
        stream = self.residual_head.expand(inner_features)
        *visit_keychains, readout_keychain = keychain.split(len(visits) + 1)
        loop_input = None
        updated_states = []
        layer_results = []
        for visit_index, (visit, visit_keychain) in enumerate(zip(visits, visit_keychains, strict=True)):
            if visit.layer_index == loop_config.start_layer:
                if visit.loop_pass == 0:
                    loop_input = stream
                else:
                    assert loop_input is not None
                    stream = self.residual_head.begin_loop_pass(loop_input, stream)
            elif visit.layer_index == loop_config.end_layer:
                stream = self.residual_head.end_loop(stream)
            assert all(leaf.dtype == residual_dtype for leaf in jax.tree.leaves(stream))
            rope_index = self.rope_indices[visit.layer_index]
            layer_result = self.layers[visit.layer_index](
                stream,
                rope_embeddings[rope_index] if rope_index >= 0 else None,
                state=state[visit_index] if state is not None else None,
                return_updated_state=return_updated_state,
                return_activation_trace=return_layer_results,
                lengths_without_padding=lengths_without_padding,
                forward_pass_config=forward_pass_config,
                attention_parent_indices=attention_parent_indices,
                token_ids=token_ids,
                keychain=visit_keychain,
            )
            stream = layer_result.outputs
            layer_results.append(layer_result)
            updated_states.append(layer_result.updated_state)
        if loop_config.end_layer == len(self.layers):
            stream = self.residual_head.end_loop(stream)

        outputs = self.residual_head.readout(
            stream,
            forward_pass_config.mlp_forward_pass_config.matmul_config,
            keychain=readout_keychain,
        )
        assert outputs.dtype == residual_dtype
        if return_suffix_tokens is not None:
            outputs = gather_suffix_tokens(
                outputs, lengths_without_padding, return_suffix_tokens, self.sharding_config
            )
        normalized_outputs = call_vmapped_twice(self.output_norm, outputs) if self.output_norm is not None else outputs

        if return_updated_state:
            if any(visit_state is None for visit_state in updated_states):
                raise ValueError("Every layer visit must return an updated state.")
            compact_state = State(tuple(updated_states))
        else:
            compact_state = None
        return TransformerResult(
            outputs=normalized_outputs,
            updated_state=compact_state,
            layer_results=tuple(layer_results) if return_layer_results else None,
            rope_embeddings=rope_embeddings if return_positional_embeddings else None,
            pre_norm_outputs=outputs if return_suffix_tokens is not None else None,
        )

    @eqx.filter_jit
    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        if self.config.loop_config is not None:
            return State(
                self.layers[visit.layer_index].init_static_state(batch_size, capacity, dtype)
                for visit in self.config.loop_config.layer_visits(len(self.layers))
            )
        return State(
            self.layers[layer_index].init_static_state(batch_size, capacity, dtype)
            for layer_index in self.kv_source_layer_indices
        )
