from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import DotAlgorithmPreset
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.exportable import Exportable
from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, LogicalAxis
from lalamo.utils.sharding import ShardingConfig
from lalamo.weight_matrix import GradientEstimator, MatmulConfig

from .linear import Linear, LinearConfig
from .mlp import MLPBase, MLPConfig, MLPForwardPassConfig, MoERoutingTrace
from .normalization import Normalization, NormalizationConfig
from .residual import ConnectionConfig, ResidualConfig, ResidualConnectionBase
from .rope import PositionalEmbeddings, RoPEConfig
from .stream_ops import StreamOpBase, StreamOpConfig, StreamOpContext
from .token_mixer import (
    MixerForwardPassConfig,
    PositionalEmbeddingSelector,
    StateLayerBase,
    TokenMixerBase,
    TokenMixerConfig,
    TransformerLayerState,
)
from .token_mixers.attention import Attention, AttentionConfig
from .token_mixers.convolutions import SeparableCausalConv, SeparableCausalConvConfig
from .token_mixers.kv_cache import PagedKVCacheLayer
from .token_mixers.sparse_attention import SparseAttentionConfig
from .utils import call_vmapped, call_vmapped_twice, gather_suffix_tokens

__all__ = [
    "OutputConvConfig",
    "PositionalEmbeddingSelector",
    "Sublayer",
    "SublayerActivationTrace",
    "SublayerConfig",
    "TransformerForwardPassConfig",
    "TransformerLayer",
    "TransformerLayerActivationTrace",
    "TransformerLayerConfig",
    "TransformerLayerConv",
    "TransformerLayerConvConfig",
    "TransformerLayerResult",
]


@dataclass(frozen=True)
class TransformerForwardPassConfig:
    mixer_forward_pass_config: MixerForwardPassConfig = dataclass_field(default_factory=MixerForwardPassConfig)
    mlp_forward_pass_config: MLPForwardPassConfig = dataclass_field(default_factory=MLPForwardPassConfig)
    # Rematerialise every layer in the backward pass (jax.checkpoint around each layer call). Inference never
    # needs it; a training step through all layers of a large model does, because lalamo's preamble turns
    # XLA's own remat pass off and the saved activations of the MoE dispatch would otherwise dominate memory.
    remat_layers: bool = False

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_tracer_tests(),
            mlp_forward_pass_config=MLPForwardPassConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_inference(precision),
            mlp_forward_pass_config=MLPForwardPassConfig.for_inference(mode, precision),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            mixer_forward_pass_config=MixerForwardPassConfig.for_training(gradient_estimator, precision),
            mlp_forward_pass_config=MLPForwardPassConfig.for_training(gradient_estimator, precision),
        )


@dataclass(frozen=True)
class TransformerLayerConvConfig(LalamoConfig):
    conv_config: SeparableCausalConvConfig
    kernel_projection_config: LinearConfig
    conv_kernel_size: int
    conv_group_size: int

    def init(self, initializer: Initializer, model_dim: int) -> "TransformerLayerConv":
        if model_dim % self.conv_group_size != 0:
            raise ValueError(f"conv_group_size {self.conv_group_size} must divide model_dim {model_dim}.")
        pre_conv = self.conv_config.init(
            initializer, model_dim, self.conv_kernel_size, dtype=initializer.default_dtype
        )
        post_conv = self.conv_config.init(
            initializer, model_dim, self.conv_kernel_size, dtype=initializer.default_dtype
        )
        kernel_projection = self.kernel_projection_config.init(
            initializer,
            input_dim=model_dim,
            output_dims=(2 * self.conv_kernel_size * (model_dim // self.conv_group_size),),
            has_biases=False,
            is_sharded=False,
        )
        return TransformerLayerConv(
            config=self,
            sharding_config=initializer.sharding_config,
            kernel_projection=kernel_projection,
            pre_conv=pre_conv,
            post_conv=post_conv,
        )


class TransformerLayerConv(LalamoModule[TransformerLayerConvConfig]):
    kernel_projection: Linear
    pre_conv: SeparableCausalConv
    post_conv: SeparableCausalConv

    def prepare(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: MatmulConfig,
        *,
        keychain: Keychain,
    ) -> tuple[Float[Array, "batch suffix_tokens channels"], Float[Array, "batch suffix_tokens kernel groups"]]:
        (projected_coefficients,) = call_vmapped_twice(
            self.kernel_projection,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        coefficients = jnp.flip(
            rearrange(
                projected_coefficients,
                "batch tokens (sides kernel groups) -> batch tokens sides kernel groups",
                sides=2,
                kernel=self.pre_conv.kernel_size,
            ),
            axis=3,
        )
        outputs = call_vmapped(
            lambda inputs, deltas: self.pre_conv(inputs, coefficient_deltas=deltas).outputs,
            inputs,
            coefficients[:, :, 0],
        )
        return outputs, coefficients[:, :, 1]

    def finish(
        self,
        outputs: Float[Array, "batch suffix_tokens channels"],
        coefficients: Float[Array, "batch suffix_tokens kernel groups"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return call_vmapped(
            lambda outputs, deltas: self.post_conv(outputs, coefficient_deltas=deltas).outputs,
            outputs,
            coefficients,
        )


@dataclass(frozen=True)
class OutputConvConfig(LalamoConfig):
    conv_config: SeparableCausalConvConfig
    kernel_size: int

    def init(self, initializer: Initializer, model_dim: int) -> SeparableCausalConv:
        return self.conv_config.init(initializer, model_dim, self.kernel_size)


class SublayerActivationTrace[InputT](Exportable, eqx.Module):
    inputs: InputT
    pre_norm: InputT
    body: Float[Array, "batch suffix_tokens channels"]
    post_norm: Float[Array, "batch suffix_tokens channels"] | None


@dataclass(frozen=True)
class SublayerConfig[BodyConfigT](LalamoConfig):
    pre_norm_config: NormalizationConfig | None
    conv_config: TransformerLayerConvConfig | None
    body_config: BodyConfigT
    post_norm_config: NormalizationConfig | None
    connection_config: ConnectionConfig | None = dataclass_field(default=None, kw_only=True)
    output_conv_config: OutputConvConfig | None = dataclass_field(default=None, kw_only=True)

    @classmethod
    def pre_norm(cls, norm_config: NormalizationConfig, body_config: BodyConfigT) -> Self:
        return cls(pre_norm_config=norm_config, conv_config=None, body_config=body_config, post_norm_config=None)

    @classmethod
    def sandwich_norm(
        cls,
        pre_norm_config: NormalizationConfig,
        body_config: BodyConfigT,
        post_norm_config: NormalizationConfig,
    ) -> Self:
        return cls(
            pre_norm_config=pre_norm_config,
            conv_config=None,
            body_config=body_config,
            post_norm_config=post_norm_config,
        )

    def init[BodyT: LalamoModule, StreamT, InputT](
        self,
        initializer: Initializer,
        model_dim: int,
        body: BodyT,
        connection: ResidualConnectionBase[ResidualConfig, StreamT, InputT],
    ) -> "Sublayer[BodyT, StreamT, InputT]":
        pre_norm = self.pre_norm_config.init(initializer, model_dim) if self.pre_norm_config is not None else None
        conv = self.conv_config.init(initializer, model_dim) if self.conv_config is not None else None
        post_norm = self.post_norm_config.init(initializer, model_dim) if self.post_norm_config is not None else None
        output_conv = (
            self.output_conv_config.init(initializer, model_dim) if self.output_conv_config is not None else None
        )
        return Sublayer(
            config=self,
            sharding_config=initializer.sharding_config,
            connection=connection,
            pre_norm=pre_norm,
            conv=conv,
            body=body,
            output_conv=output_conv,
            post_norm=post_norm,
        )


class Sublayer[BodyT: LalamoModule, StreamT, InputT](LalamoModule[SublayerConfig]):
    connection: ResidualConnectionBase[ResidualConfig, StreamT, InputT]
    pre_norm: Normalization | None
    conv: TransformerLayerConv | None
    body: BodyT
    output_conv: SeparableCausalConv | None
    post_norm: Normalization | None

    def __call__[AuxT](
        self,
        stream: StreamT,
        call_body: Callable[[InputT, Keychain], tuple[Float[Array, "batch suffix_tokens channels"], AuxT]],
        matmul_config: MatmulConfig,
        *,
        output_conv_state: Float[Array, "batch tokens channels"] | None,
        lengths_without_padding: Int[Array, " batch"] | None,
        return_updated_state: bool,
        keychain: Keychain,
    ) -> tuple[StreamT, SublayerActivationTrace[InputT], AuxT, Float[Array, "batch tokens channels"] | None]:
        def apply(
            inputs: InputT,
            sublayer_keychain: Keychain,
        ) -> tuple[
            Float[Array, "batch suffix_tokens channels"],
            tuple[SublayerActivationTrace[InputT], AuxT, Float[Array, "batch tokens channels"] | None],
        ]:
            normalized_inputs = (
                jax.tree.map(lambda leaf: call_vmapped_twice(self.pre_norm, leaf), inputs)
                if self.pre_norm is not None
                else inputs
            )

            if self.conv is not None:
                if not isinstance(normalized_inputs, jax.Array):
                    raise TypeError("Sublayer convolutions require array inputs.")
                conv_keychain, body_keychain = sublayer_keychain.split()
                conv_inputs, conv_coefficients = self.conv.prepare(
                    normalized_inputs,
                    matmul_config,
                    keychain=conv_keychain,
                )
                conv_outputs, aux = call_body(cast("InputT", conv_inputs), body_keychain)
                body_outputs = self.conv.finish(conv_outputs, conv_coefficients)
            else:
                body_outputs, aux = call_body(normalized_inputs, sublayer_keychain)

            if self.output_conv is not None:
                conv_outputs, updated_output_conv_state = call_vmapped(
                    lambda outputs, length, state: self.output_conv(outputs, length, state, return_updated_state),
                    body_outputs.astype(jnp.float32),
                    lengths_without_padding,
                    output_conv_state,
                )
                body_outputs = conv_outputs.astype(body_outputs.dtype)
            else:
                updated_output_conv_state = None

            normalized_outputs = (
                call_vmapped_twice(self.post_norm, body_outputs) if self.post_norm is not None else None
            )
            update = normalized_outputs if normalized_outputs is not None else body_outputs

            activation_trace = SublayerActivationTrace(
                inputs=inputs,
                pre_norm=normalized_inputs,
                body=body_outputs,
                post_norm=normalized_outputs,
            )
            return update, (activation_trace, aux, updated_output_conv_state)

        updated_stream, (activation_trace, aux, updated_output_conv_state) = self.connection(
            stream,
            apply,
            matmul_config,
            keychain=keychain,
        )
        return updated_stream, activation_trace, aux, updated_output_conv_state

    def init_output_conv_state(self) -> Float[Array, "tokens channels"] | None:
        if self.output_conv is None:
            return None
        return jnp.zeros((self.output_conv.kernel_size - 1, self.output_conv.input_dim), dtype=jnp.float32)


class TransformerLayerActivationTrace[StreamT, MixerInputT](Exportable, eqx.Module):
    inputs: StreamT
    positional_embeddings: PositionalEmbeddings | None
    state: TransformerLayerState | None

    prologue: tuple[StreamT, ...]
    mixer: SublayerActivationTrace[MixerInputT]
    mlp: SublayerActivationTrace[Float[Array, "batch suffix_tokens channels"]]
    epilogue: tuple[StreamT, ...]


class TransformerLayerResult[StreamT, MixerInputT](Exportable, eqx.Module):
    outputs: StreamT
    updated_state: TransformerLayerState | None
    activation_trace: TransformerLayerActivationTrace[StreamT, MixerInputT] | None
    routing_trace: MoERoutingTrace | None = None


@dataclass(frozen=True)
class TransformerLayerConfig(LalamoConfig):
    prologue_configs: tuple[StreamOpConfig, ...] = dataclass_field(default=(), kw_only=True)
    mixer: SublayerConfig[TokenMixerConfig]
    mlp: SublayerConfig[MLPConfig]
    epilogue_configs: tuple[StreamOpConfig, ...] = dataclass_field(default=(), kw_only=True)
    hidden_dim: int | None = None
    kv_source_layer_index: int | None = None
    rope_config: RoPEConfig | None = None

    def __post_init__(self) -> None:
        has_stateful_parts = any(
            op_config.is_stateful for op_config in (*self.prologue_configs, *self.epilogue_configs)
        ) or any(sublayer.output_conv_config is not None for sublayer in (self.mixer, self.mlp))
        mixer_config = self.mixer.body_config
        is_paged = isinstance(mixer_config, SparseAttentionConfig) or (
            isinstance(mixer_config, AttentionConfig) and mixer_config.supports_paged_decode
        )
        if has_stateful_parts and is_paged:
            raise ValueError("Paged attention mixers support neither stateful stream ops nor output convolutions.")

    def init(
        self,
        initializer: Initializer,
        residual_config: ResidualConfig,
        model_dim: int,
        hidden_dim: int,
    ) -> "TransformerLayer":
        prologue = tuple(
            op_config.init(initializer, residual_config, model_dim) for op_config in self.prologue_configs
        )
        mixer = self.mixer.init(
            initializer,
            model_dim,
            body=self.mixer.body_config.init(initializer, model_dim=model_dim),
            connection=residual_config.init_connection(initializer, model_dim, self.mixer.connection_config),
        )
        mlp = self.mlp.init(
            initializer,
            model_dim,
            body=self.mlp.body_config.init(initializer, model_dim, hidden_dim),
            connection=residual_config.init_connection(initializer, model_dim, self.mlp.connection_config),
        )
        epilogue = tuple(
            op_config.init(initializer, residual_config, model_dim) for op_config in self.epilogue_configs
        )
        return TransformerLayer(
            config=self,
            sharding_config=initializer.sharding_config,
            prologue=prologue,
            mixer=mixer,
            mlp=mlp,
            epilogue=epilogue,
        )


def gather_suffix_stream[StreamT](
    stream: StreamT,
    context: StreamOpContext,
    return_suffix_tokens: int,
    sharding_config: ShardingConfig,
) -> tuple[StreamT, StreamOpContext]:
    def gather(features: Array) -> Array:
        return gather_suffix_tokens(features, context.lengths_without_padding, return_suffix_tokens, sharding_config)

    token_ids = gather(context.token_ids) if context.token_ids is not None else None
    per_layer_input = gather(context.per_layer_input) if context.per_layer_input is not None else None
    return jax.tree.map(gather, stream), StreamOpContext(
        token_ids=token_ids,
        per_layer_input=per_layer_input,
        lengths_without_padding=None,
    )


class TransformerLayer[StreamT, MixerInputT](LalamoModule[TransformerLayerConfig]):
    prologue: tuple[StreamOpBase[StreamOpConfig, StreamT, StateLayerBase], ...]
    mixer: Sublayer[TokenMixerBase, StreamT, MixerInputT]
    mlp: Sublayer[MLPBase, StreamT, Float[Array, "batch suffix_tokens channels"]]
    epilogue: tuple[StreamOpBase[StreamOpConfig, StreamT, StateLayerBase], ...]

    @property
    def suffix_gather_stage(self) -> int:
        stage_token_locality = (
            # A stateful routing intervention must see every token of the sequence, so its MLP runs on the
            # full sequence even in suffix mode and only what follows it is narrowed to the window.
            self.config.mlp.conv_config is None
            and self.config.mlp.output_conv_config is None
            and not self.mlp.body.keeps_routing_state,
            *(op_config.is_token_local for op_config in self.config.epilogue_configs),
        )
        return next(
            stage_index
            for stage_index in range(len(stage_token_locality) + 1)
            if all(stage_token_locality[stage_index:])
        )

    @eqx.filter_jit
    def __call__(
        self,
        stream: StreamT,
        positional_embeddings: PositionalEmbeddings | None,
        state: TransformerLayerState | None = None,
        return_updated_state: bool = False,
        return_activation_trace: bool = False,
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: TransformerForwardPassConfig = TransformerForwardPassConfig(),
        per_layer_input: Float[Array, "batch suffix_tokens ple_dim"] | None = None,
        attention_parent_indices: Int[Array, " batch suffix_tokens"] | None = None,
        return_suffix_tokens: int | None = None,
        token_ids: Int[Array, "batch suffix_tokens"] | None = None,
        return_routing_trace: bool = False,
        generation_mask: Bool[Array, "batch suffix_tokens"] | None = None,
        *,
        keychain: Keychain,
    ) -> TransformerLayerResult[StreamT, MixerInputT]:
        if return_suffix_tokens is not None and (return_activation_trace or return_routing_trace):
            raise ValueError("return_suffix_tokens cannot be combined with activation or routing traces.")
        if self.config.kv_source_layer_index is not None and self.mlp.body.keeps_routing_state:
            raise NotImplementedError("A stateful routing intervention on a KV-sharing layer has no state slot.")
        owns_state = state is not None and self.config.kv_source_layer_index is None
        routing_state = state.routing if owns_state else None
        mixer_keychain, mlp_keychain, stream_op_keychain = keychain.split(3)
        num_stream_ops = len(self.prologue) + len(self.epilogue)
        if num_stream_ops > 0:
            stream_op_keychains = stream_op_keychain.split(num_stream_ops)
        else:
            stream_op_keychains = ()
        prologue_keychains = stream_op_keychains[: len(self.prologue)]
        epilogue_keychains = stream_op_keychains[len(self.prologue) :]

        if state is None:
            mixer_state = None
            stream_op_states = (None,) * num_stream_ops
            output_conv_states = (None, None)
        elif self.config.kv_source_layer_index is not None:
            mixer_state = state.mixer
            stream_op_states = (None,) * num_stream_ops
            output_conv_states = (None, None)
        else:
            mixer_state = state.mixer
            stream_op_states = state.stream_ops
            output_conv_states = state.output_convs
        mixer_output_conv_state, mlp_output_conv_state = output_conv_states
        needs_updated_state = return_updated_state or return_activation_trace
        prologue_states = stream_op_states[: len(self.prologue)]
        epilogue_states = stream_op_states[len(self.prologue) :]

        inputs = stream
        context = StreamOpContext(
            token_ids=token_ids,
            per_layer_input=per_layer_input,
            lengths_without_padding=lengths_without_padding,
        )

        prologue_streams = []
        updated_prologue_states = []
        for op, op_state, op_keychain in zip(self.prologue, prologue_states, prologue_keychains, strict=True):
            stream, updated_op_state = op(stream, context, op_state, keychain=op_keychain)
            prologue_streams.append(stream)
            updated_prologue_states.append(updated_op_state)

        def call_mixer(
            mixer_inputs: tuple[
                MixerInputT,
                PositionalEmbeddings | None,
                StateLayerBase | None,
                Int[Array, ""] | None,
                Int[Array, " suffix_tokens"] | None,
            ],
            *,
            keychain: Keychain,
        ) -> tuple[Float[Array, "suffix_tokens channels"], StateLayerBase | None]:
            mixer_input, positional_embedding, single_mixer_state, length_without_padding, parent_indices = (
                mixer_inputs
            )
            return self.mixer.body(
                mixer_input,
                positional_embedding,
                single_mixer_state,
                return_updated_state=needs_updated_state,
                length_without_padding=length_without_padding,
                forward_pass_config=forward_pass_config.mixer_forward_pass_config,
                attention_parent_indices=parent_indices,
                reuse_cache=self.config.kv_source_layer_index is not None,
                keychain=keychain,
            )

        def call_mixer_body(
            mixer_inputs: MixerInputT,
            body_keychain: Keychain,
        ) -> tuple[Float[Array, "batch suffix_tokens channels"], StateLayerBase | None]:
            if isinstance(mixer_state, PagedKVCacheLayer):
                assert isinstance(self.mixer.body, Attention) and lengths_without_padding is None
                if not isinstance(mixer_inputs, jax.Array):
                    raise TypeError("Paged decoding requires array mixer inputs.")
                mixer_outputs, updated_paged_state = self.mixer.body.paged_decode(
                    mixer_inputs,
                    positional_embeddings,
                    mixer_state,
                    forward_pass_config.mixer_forward_pass_config,
                    keychain=body_keychain,
                )
                return mixer_outputs, updated_paged_state
            return call_vmapped(
                call_mixer,
                (
                    mixer_inputs,
                    positional_embeddings,
                    mixer_state,
                    lengths_without_padding,
                    attention_parent_indices,
                ),
                keychain=body_keychain,
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            )

        stream, mixer_trace, updated_mixer_state, updated_mixer_output_conv_state = self.mixer(
            stream,
            call_mixer_body,
            forward_pass_config.mixer_forward_pass_config.matmul_config,
            output_conv_state=mixer_output_conv_state,
            lengths_without_padding=lengths_without_padding,
            return_updated_state=needs_updated_state,
            keychain=mixer_keychain,
        )

        suffix_gather_stage = self.suffix_gather_stage
        if return_suffix_tokens is not None and suffix_gather_stage == 0:
            if generation_mask is not None:
                generation_mask = gather_suffix_tokens(
                    generation_mask,
                    context.lengths_without_padding,
                    return_suffix_tokens,
                    self.sharding_config,
                )
            stream, context = gather_suffix_stream(stream, context, return_suffix_tokens, self.sharding_config)

        def call_mlp_body(
            mlp_inputs: Float[Array, "batch suffix_tokens channels"],
            body_keychain: Keychain,
        ) -> tuple[
            Float[Array, "batch suffix_tokens channels"],
            tuple[StateLayerBase | None, MoERoutingTrace | None],
        ]:
            mlp_result = self.mlp.body.call_with_routing(
                mlp_inputs,
                lengths_without_padding=context.lengths_without_padding,
                forward_pass_config=forward_pass_config.mlp_forward_pass_config,
                generation_mask=generation_mask,
                routing_state=routing_state,
                keychain=body_keychain,
            )
            return mlp_result.outputs, (mlp_result.updated_routing_state, mlp_result.routing_trace)

        stream, mlp_trace, (updated_routing_state, routing_trace), updated_mlp_output_conv_state = self.mlp(
            stream,
            call_mlp_body,
            forward_pass_config.mlp_forward_pass_config.matmul_config,
            output_conv_state=mlp_output_conv_state,
            lengths_without_padding=context.lengths_without_padding,
            return_updated_state=needs_updated_state,
            keychain=mlp_keychain,
        )

        epilogue_streams = []
        updated_epilogue_states = []
        for epilogue_index, (op, op_state, op_keychain) in enumerate(
            zip(self.epilogue, epilogue_states, epilogue_keychains, strict=True),
        ):
            if return_suffix_tokens is not None and suffix_gather_stage == epilogue_index + 1:
                stream, context = gather_suffix_stream(stream, context, return_suffix_tokens, self.sharding_config)
            stream, updated_op_state = op(stream, context, op_state, keychain=op_keychain)
            epilogue_streams.append(stream)
            updated_epilogue_states.append(updated_op_state)
        if return_suffix_tokens is not None and suffix_gather_stage == len(self.epilogue) + 1:
            stream, context = gather_suffix_stream(stream, context, return_suffix_tokens, self.sharding_config)

        if updated_mixer_state is not None:
            updated_state = TransformerLayerState(
                mixer=updated_mixer_state,
                stream_ops=(*updated_prologue_states, *updated_epilogue_states),
                output_convs=(updated_mixer_output_conv_state, updated_mlp_output_conv_state),
                routing=updated_routing_state,
            )
        else:
            updated_state = None

        if return_activation_trace:
            activation_trace = TransformerLayerActivationTrace(
                inputs=inputs,
                positional_embeddings=positional_embeddings,
                state=state,
                prologue=tuple(prologue_streams),
                mixer=mixer_trace,
                mlp=mlp_trace,
                epilogue=tuple(epilogue_streams),
            )
        else:
            activation_trace = None

        return TransformerLayerResult(
            outputs=stream,
            updated_state=updated_state,
            activation_trace=activation_trace,
            routing_trace=routing_trace if return_routing_trace else None,
        )

    def init_static_state(self, batch_size: int, capacity: int, dtype: DTypeLike) -> TransformerLayerState:
        layer_state = TransformerLayerState(
            mixer=self.mixer.body.init_static_state(capacity, dtype),
            stream_ops=tuple(op.init_static_state(capacity, dtype) for op in (*self.prologue, *self.epilogue)),
            output_convs=(self.mixer.init_output_conv_state(), self.mlp.init_output_conv_state()),
            routing=self.mlp.body.init_routing_state(dtype),
        )
        return jax.tree.map(
            lambda array: jax.device_put(
                jnp.repeat(array[None, ...], batch_size, axis=0),
                self.sharding_config.resolve_sharding((LogicalAxis.BATCH, *((None,) * array.ndim))),
            ),
            layer_state,
        )
