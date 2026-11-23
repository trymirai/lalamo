from dataclasses import dataclass

import equinox as eqx
import jax
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule, LogicalAxis, field

from .normalization import Normalization, NormalizationConfig
from .residual import ResidualConfig, ResidualHeadBase
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
    residual_config: ResidualConfig
    layer_configs: tuple[TransformerLayerConfig, ...]
    output_norm_config: NormalizationConfig | None
    model_dim: int
    hidden_dim: int
    layer_order: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.layer_order is None:
            return
        if sorted(set(self.layer_order)) != list(range(len(self.layer_configs))):
            raise ValueError(
                f"layer_order must apply every one of the {len(self.layer_configs)} layers, got {self.layer_order}.",
            )
        if any(
            self.layer_order[step] <= self.layer_order[step - 1]
            and self.layer_order[step] not in self.layer_order[:step]
            for step in range(1, len(self.layer_order))
        ):
            raise ValueError(f"layer_order may only jump back to already applied layers, got {self.layer_order}.")
        if any(layer_config.kv_source_layer_index is not None for layer_config in self.layer_configs):
            raise ValueError("layer_order cannot be combined with KV-sharing layers.")

    @property
    def step_layer_indices(self) -> tuple[int, ...]:
        return self.layer_order if self.layer_order is not None else tuple(range(len(self.layer_configs)))

    @property
    def state_owner_steps(self) -> tuple[int, ...]:
        return tuple(
            step
            for step, layer_index in enumerate(self.step_layer_indices)
            if self.layer_configs[layer_index].kv_source_layer_index is None
        )

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
            )
            for layer_config in self.layer_configs
        )
        output_norm = (
            self.output_norm_config.init(initializer, self.model_dim) if self.output_norm_config is not None else None
        )
        return Transformer(
            config=self,
            sharding_config=initializer.sharding_config,
            ropes=ropes,
            rope_indices=rope_indices,
            residual_head=residual_head,
            layers=layers,
            output_norm=output_norm,
        )


class Transformer[StreamT, MixerInputT](LalamoModule[TransformerConfig]):
    ropes: tuple[RoPE, ...]
    rope_indices: tuple[int, ...] = field(static=True)
    residual_head: ResidualHeadBase[ResidualConfig, StreamT]
    layers: tuple[TransformerLayer[StreamT, MixerInputT], ...]
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
        if return_suffix_tokens is not None:
            if return_layer_results or return_positional_embeddings:
                raise ValueError(
                    "return_suffix_tokens cannot be combined with return_layer_results"
                    " or return_positional_embeddings.",
                )
            if attention_parent_indices is not None:
                raise ValueError("return_suffix_tokens cannot be combined with attention_parent_indices.")
            if not self.config.state_owner_steps:
                raise ValueError("return_suffix_tokens requires at least one layer that owns its state.")
            has_trailing_shared_layers = self.config.state_owner_steps[-1] < len(self.config.step_layer_indices) - 1
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
        state_by_step = (
            {step: state[state_index] for state_index, step in enumerate(self.config.state_owner_steps)}
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
        has_kv_sharing = len(self.config.state_owner_steps) < len(self.config.step_layer_indices)
        must_return_state = return_updated_state or has_kv_sharing

        # In suffix mode, every layer past the last state-owning one only affects the returned
        # suffix outputs, so those layers run on the gathered suffix instead of the full sequence.
        if return_suffix_tokens is not None:
            last_state_owner_step = self.config.state_owner_steps[-1]
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
            last_state_owner_step = None
            suffix_rope_embeddings = ()
            suffix_token_ids = None

        residual_dtype = inner_features.dtype
        stream = self.residual_head.expand(inner_features)
        *step_keychains, readout_keychain = keychain.split(len(self.config.step_layer_indices) + 1)
        updated_states: dict[int, TransformerLayerState | None] = {}
        entry_streams: dict[int, StreamT] = {}
        layer_results = []

        for step, (layer_index, step_keychain) in enumerate(
            zip(self.config.step_layer_indices, step_keychains, strict=True)
        ):
            if step > 0 and layer_index <= self.config.step_layer_indices[step - 1]:
                stream = self.residual_head.restart(stream, entry_streams[layer_index])
            entry_streams.setdefault(layer_index, stream)
            assert all(leaf.dtype == residual_dtype for leaf in jax.tree.leaves(stream))
            layer = self.layers[layer_index]
            runs_on_suffix_only = last_state_owner_step is not None and step > last_state_owner_step
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
                effective_state = state_by_step.get(step)
            else:
                effective_state = updated_states.get(
                    kv_source_layer_index,
                    state_by_step.get(kv_source_layer_index),
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
                return_suffix_tokens=return_suffix_tokens if step == last_state_owner_step else None,
                token_ids=layer_token_ids,
                keychain=step_keychain,
            )

            stream = layer_result.outputs
            layer_results.append(layer_result)

            if kv_source_layer_index is None:
                updated_states[step] = layer_result.updated_state

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
            for step in self.config.state_owner_steps:
                layer_state = updated_states[step]
                if layer_state is None:
                    raise ValueError(f"Step {step} did not return an updated state.")
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

    @eqx.filter_jit
    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> State:
        return State(
            self.layers[self.config.step_layer_indices[step]].init_static_state(batch_size, capacity, dtype)
            for step in self.config.state_owner_steps
        )
