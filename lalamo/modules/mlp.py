import math
from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.module import (
    ForwardPassMode,
    Keychain,
    KeychainBroadcastMode,
    LalamoConfig,
    LalamoModule,
    LogicalAxis,
    ShardingConfig,
)
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import is_sharded, sharding_of, with_sharding
from lalamo.weight_matrix import FullPrecisionMatrix

from .activations import Activation
from .flow import Context, MLPForwardPassConfig, StepConfig, Value
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "DenseMLP",
    "DenseMLPConfig",
    "Gemma4Router",
    "Gemma4RouterConfig",
    "MLPBase",
    "MLPConfig",
    "MLPForwardPassConfig",
    "MixtureOfExperts",
    "MixtureOfExpertsConfig",
    "RoutingBias",
    "RoutingFunction",
    "SigmoidRouting",
    "SoftmaxRouting",
]


_SENTINEL = 2**31 - 1


def _take_moe_chunk_inputs(
    flattened_inputs: Float[Array, "tokens channels"],
    indices: Int[Array, " tokens_per_chunk"],
    out_sharding: NamedSharding,
) -> Float[Array, "tokens_per_chunk channels"]:
    return flattened_inputs.at[indices].get(
        mode="fill",
        fill_value=0.0,
        out_sharding=out_sharding,
    )


def _take_moe_expert_leaf(
    leaf: object,
    index: Int[Array, ""] | Int[Array, " experts"],
    sharding_config: ShardingConfig,
) -> object:
    if not isinstance(leaf, jax.Array):
        return leaf

    return leaf.at[jnp.expand_dims(index, 0)].get(out_sharding=sharding_config.make_sharding(()))[0]


class RoutingMap(eqx.Module):
    active_expert_indices: Int[Array, "*batch_and_tokens active_experts"]
    active_expert_weights: Float[Array, "*batch_and_tokens active_experts"]


@dataclass(frozen=True)
class MLPConfig(StepConfig):
    @abstractmethod
    def init(self, initializer: Initializer, model_dim: int) -> "MLPBase": ...


class MLPBase[ConfigT: MLPConfig](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def model_dim(self) -> int: ...

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @abstractmethod
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]: ...

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Array,
        ctx: Context,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens channels"]:
        () = operands
        return self(x, ctx.lengths, ctx.forward_pass_config.mlp_forward_pass_config, keychain=keychain)


@dataclass(frozen=True)
class DenseMLPConfig(MLPConfig):
    linear_config: LinearConfig
    activation: Activation
    has_up_biases: bool
    has_down_biases: bool
    gate_clipping: tuple[float | None, float | None] | None
    up_clipping: tuple[float | None, float | None] | None
    hidden_dim: int

    def init(self, initializer: Initializer, model_dim: int) -> "DenseMLP":
        hidden_dim = self.hidden_dim
        return DenseMLP(
            config=self,
            sharding_config=initializer.sharding_config,
            up_projection=self.linear_config.init(
                initializer,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.init(
                initializer,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )

    def init_mixture(
        self,
        initializer: Initializer,
        mixture_size: int,
        model_dim: int,
        *,
        is_sharded: bool = True,
    ) -> "DenseMLP":
        hidden_dim = self.hidden_dim
        return DenseMLP(
            config=self,
            sharding_config=initializer.sharding_config,
            up_projection=self.linear_config.init_mixture(
                initializer,
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                is_sharded=is_sharded,
            ),
            down_projection=self.linear_config.init_mixture(
                initializer,
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                is_sharded=is_sharded,
            ),
        )


class DenseMLP(MLPBase[DenseMLPConfig]):
    up_projection: Linear
    down_projection: Linear

    @property
    def model_dim(self) -> int:
        return self.up_projection.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.down_projection.input_dim

    @property
    def mixture_size(self) -> int | None:
        return self.up_projection.mixture_size

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,  # noqa: ARG002
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return call_vmapped_twice(
            self.call_unbatched,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )

    def _hidden_activations(
        self,
        up_proj: Float[Array, "*tokens hidden_channels"],
        gate: Float[Array, "*tokens hidden_channels"],
    ) -> Float[Array, "*tokens hidden_channels"]:
        if self.config.gate_clipping is not None:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping is not None:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        return up_proj * self.config.activation(gate)

    @eqx.filter_jit
    def call_unbatched(
        self,
        inputs: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, " channels"]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                " They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        up_keychain, down_keychain = keychain.split()
        up_proj, gate = self.up_projection(
            inputs,
            keychain=up_keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        (result,) = self.down_projection(
            self._hidden_activations(up_proj, gate),
            keychain=down_keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        return result

    def call_ragged_mixture(
        self,
        inputs: Float[Array, "tokens channels"],
        expert_indices: Int[Array, " tokens"],
        group_sizes: Int[Array, " experts"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
    ) -> Float[Array, "tokens channels"]:
        if self.mixture_size is None:
            raise ValueError("DenseMLP.call_ragged_mixture() requires a mixture DenseMLP.")

        up_proj, gate = self.up_projection.call_ragged(
            inputs,
            expert_indices,
            group_sizes,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        (result,) = self.down_projection.call_ragged(
            self._hidden_activations(up_proj, gate),
            expert_indices,
            group_sizes,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        return result

    def apply_token_routing(
        self,
        token_input: Float[Array, " channels"],
        routing: RoutingMap,
        forward_pass_config: MLPForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, " channels"]:
        active_experts = jax.tree_util.tree_map(
            lambda leaf: _take_moe_expert_leaf(leaf, routing.active_expert_indices, self.sharding_config),
            self,
        )
        expert_outputs = call_vmapped(
            DenseMLP.call_unbatched,
            active_experts,
            token_input,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            in_axes=(0, None),
        )
        weighted_outputs = expert_outputs.astype(jnp.float32) * routing.active_expert_weights[:, None]
        return weighted_outputs.sum(axis=0)

    def _call_ragged_routed_experts(
        self,
        flattened_inputs: Float[Array, "tokens channels"],
        flattened_padding_mask: Bool[Array, " tokens"],
        routing_map: RoutingMap,
        forward_pass_config: MLPForwardPassConfig,
    ) -> Float[Array, "tokens channels"]:
        num_experts = self.mixture_size
        assert num_experts is not None
        replicated_vector_sharding = self.sharding_config.make_sharding((None,))
        replicated_matrix_sharding = self.sharding_config.make_sharding((None, None))
        active_expert_indices = with_sharding(routing_map.active_expert_indices, replicated_matrix_sharding)
        active_expert_weights = with_sharding(routing_map.active_expert_weights, replicated_matrix_sharding)
        _, num_active = active_expert_indices.shape

        # Padded tokens are assigned to a sentinel expert that sorts after all real ones.
        expert_indices = with_sharding(
            jnp.where(flattened_padding_mask[:, None], active_expert_indices, num_experts).ravel(),
            replicated_vector_sharding,
        )
        assignment_order = jnp.argsort(expert_indices)
        sorted_expert_indices = expert_indices[assignment_order]
        valid_assignments = sorted_expert_indices < num_experts
        group_sizes = with_sharding(
            jnp.bincount(expert_indices, length=num_experts).astype(jnp.int32),
            replicated_vector_sharding,
        )
        token_indices = assignment_order // num_active
        safe_expert_indices = jnp.where(valid_assignments, sorted_expert_indices, 0)
        expert_weights = active_expert_weights.ravel()[assignment_order]

        replicated_inputs = with_sharding(flattened_inputs, replicated_matrix_sharding)
        expert_inputs = replicated_inputs.at[token_indices].get(out_sharding=replicated_matrix_sharding)
        expert_outputs = self.call_ragged_mixture(
            expert_inputs,
            safe_expert_indices,
            group_sizes,
            forward_pass_config,
        )
        # ragged_dot leaves the ungrouped padded tail unspecified.
        weighted_outputs = jnp.where(
            valid_assignments[:, None],
            expert_outputs.astype(jnp.float32) * expert_weights[:, None],
            0.0,
        )
        # Gather back into token order instead of scatter-adding, which is non-deterministic on GPU.
        token_major_outputs = weighted_outputs.at[jnp.argsort(assignment_order)].get(
            out_sharding=replicated_matrix_sharding,
        )
        result = rearrange(
            token_major_outputs,
            "(tokens active_experts) channels -> tokens active_experts channels",
            active_experts=num_active,
        ).sum(axis=1)
        return with_sharding(
            result,
            self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None)),
        )

    def _call_chunked_routed_experts(
        self,
        flattened_inputs: Float[Array, "tokens channels"],
        flattened_padding_mask: Bool[Array, " tokens"],
        routing_map: RoutingMap,
        forward_pass_config: MLPForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        num_tokens = flattened_inputs.shape[0]
        num_experts = self.mixture_size
        assert num_experts is not None
        batch_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None))
        mixture_sharding = self.sharding_config.resolve_sharding((LogicalAxis.MIXTURE, None))
        mixture_vector_sharding = self.sharding_config.resolve_sharding((LogicalAxis.MIXTURE,))

        active_selection = jax.nn.one_hot(
            routing_map.active_expert_indices,
            num_experts,
            dtype=routing_map.active_expert_weights.dtype,
        )
        token_mask: Bool[Array, "experts tokens"] = rearrange(
            jnp.any(active_selection != 0, axis=-2) & flattened_padding_mask[:, None],
            "tokens experts -> experts tokens",
        )
        token_mask = with_sharding(token_mask, mixture_sharding)
        token_positions = jnp.cumsum(token_mask, axis=1) - 1
        active_token_positions = token_positions.at[
            routing_map.active_expert_indices,
            jnp.arange(num_tokens)[:, None],
        ].get(out_sharding=batch_sharding)
        expert_weights: Float[Array, "experts tokens"] = rearrange(
            jnp.sum(active_selection * routing_map.active_expert_weights[..., None], axis=-2),
            "tokens experts -> experts tokens",
        )
        expert_weights = jnp.where(token_mask, expert_weights, 0.0)
        expert_weights = with_sharding(expert_weights, mixture_sharding)
        expert_indices = with_sharding(jnp.arange(num_experts), mixture_vector_sharding)

        chunk_size = math.ceil(num_tokens * forward_pass_config.moe_chunk_size_ratio)
        num_padded_tokens = math.ceil(num_tokens / chunk_size) * chunk_size
        token_indices = call_vmapped(
            lambda mask: jnp.flatnonzero(mask, size=num_padded_tokens, fill_value=_SENTINEL),
            token_mask,
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.MIXTURE),
        )
        chunked_token_indices = rearrange(
            token_indices,
            "experts (chunks chunk_tokens) -> chunks experts chunk_tokens",
            chunk_tokens=chunk_size,
        )

        num_chunks = chunked_token_indices.shape[0]
        chunk_vmapped_keys = keychain.broadcast((num_chunks,)).vmapped_keys
        chunk_batch_keys = jax.random.split(keychain.batch_key, num_chunks)
        chunk_batch_keys = with_sharding(
            chunk_batch_keys,
            self.sharding_config.make_sharding((None,)),
        )

        def loop_iteration(
            expert_accumulator: Float[Array, "tokens channels"],
            chunk_inputs: tuple[Int[Array, ""], Int[Array, "experts chunk_tokens"], Key[Array, ""], Key[Array, ""]],
        ) -> tuple[Float[Array, "tokens channels"], None]:
            chunk_index, token_indices_for_chunk, chunk_vmapped_key, chunk_batch_key = chunk_inputs
            current_chunk_keychain = Keychain(
                vmapped_keys=chunk_vmapped_key,
                batch_key=chunk_batch_key,
                sharding_config=keychain.sharding_config,
            )

            weights_for_chunk = jnp.take_along_axis(
                expert_weights,
                token_indices_for_chunk,
                axis=1,
                mode="fill",
                fill_value=0.0,
            )

            expert_vmapped_keys = current_chunk_keychain.broadcast(
                (num_experts,),
            ).vmapped_keys
            expert_vmapped_keys = with_sharding(expert_vmapped_keys, mixture_vector_sharding)
            expert_batch_keys = jax.random.split(
                current_chunk_keychain.batch_key,
                num_experts,
            )
            expert_batch_keys = with_sharding(expert_batch_keys, mixture_vector_sharding)

            def run_expert(
                expert_index: Int[Array, ""],
                indices: Int[Array, " tokens_per_chunk"],
                weights: Float[Array, " tokens_per_chunk"],
                expert_vmapped_key: Key[Array, ""],
                expert_batch_key: Key[Array, ""],
            ) -> Float[Array, "tokens_per_chunk channels"]:
                expert_keychain = Keychain(
                    vmapped_keys=expert_vmapped_key,
                    batch_key=expert_batch_key,
                    sharding_config=current_chunk_keychain.sharding_config,
                )
                expert = jax.tree_util.tree_map(
                    lambda leaf: _take_moe_expert_leaf(leaf, expert_index, self.sharding_config),
                    self,
                )
                chunk_inputs = _take_moe_chunk_inputs(
                    flattened_inputs,
                    indices,
                    self.sharding_config.make_sharding((None, None)),
                )
                expert_outputs = call_vmapped(
                    expert.call_unbatched,
                    chunk_inputs,
                    forward_pass_config=forward_pass_config,
                    keychain=expert_keychain,
                )
                return expert_outputs.astype(jnp.float32) * weights[:, None]

            expert_outputs = call_vmapped(
                lambda expert_inputs: run_expert(*expert_inputs),
                (
                    expert_indices,
                    token_indices_for_chunk,
                    weights_for_chunk,
                    expert_vmapped_keys,
                    expert_batch_keys,
                ),
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.MIXTURE),
            )
            positions_for_chunk = jnp.where(
                flattened_padding_mask[:, None],
                active_token_positions - chunk_index * chunk_size,
                _SENTINEL,
            )
            # Gather in active-expert order; conflicting scatter-adds are non-deterministic on GPU.
            token_outputs = expert_outputs.at[routing_map.active_expert_indices, positions_for_chunk].get(
                mode="fill",
                fill_value=0.0,
                wrap_negative_indices=False,
                out_sharding=self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None, None)),
            )
            return expert_accumulator + token_outputs.sum(axis=1), None

        routed_accumulator = jnp.zeros(
            flattened_inputs.shape,
            dtype=jnp.float32,
            out_sharding=batch_sharding,
        )
        routed_expert_result, _ = jax.lax.scan(
            loop_iteration,
            routed_accumulator,
            (jnp.arange(num_chunks), chunked_token_indices, chunk_vmapped_keys, chunk_batch_keys),
        )
        return routed_expert_result

    def apply_routing(
        self,
        inputs: Float[Array, "tokens channels"],
        padding_mask: Bool[Array, " tokens"],
        routing_map: RoutingMap,
        forward_pass_config: MLPForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        if forward_pass_config.mode == ForwardPassMode.SINGLE_TOKEN:
            outputs = call_vmapped(
                self.apply_token_routing,
                inputs,
                routing_map,
                forward_pass_config=forward_pass_config,
                keychain=keychain,
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            )
            return jnp.where(padding_mask[:, None], outputs, 0)
        up_weights = self.up_projection.weights
        down_weights = self.down_projection.weights
        use_ragged = (
            isinstance(up_weights, FullPrecisionMatrix)
            and isinstance(down_weights, FullPrecisionMatrix)
            and not is_sharded(sharding_of(up_weights.weights))
            and not is_sharded(sharding_of(down_weights.weights))
        )
        if use_ragged:
            return self._call_ragged_routed_experts(
                inputs,
                padding_mask,
                routing_map,
                forward_pass_config,
            )
        return self._call_chunked_routed_experts(
            inputs,
            padding_mask,
            routing_map,
            forward_pass_config,
            keychain=keychain,
        )


@dataclass(frozen=True)
class RoutingFunction(LalamoConfig, RegistryABC):
    def init_biases(
        self,
        initializer: Initializer,  # noqa: ARG002
        num_experts: int,  # noqa: ARG002
    ) -> Float[Array, " experts"] | None:
        return None

    def __call__(
        self,
        logits: Float[Array, "batch_tokens experts"],
        biases: Float[Array, " experts"] | None = None,
        *,
        num_active: int,
    ) -> RoutingMap:
        return call_vmapped(partial(self.call_unbatched, biases=biases, num_active=num_active), logits)

    @abstractmethod
    def call_unbatched(
        self,
        logits: Float[Array, " experts"],
        biases: Float[Array, " experts"] | None = None,
        *,
        num_active: int,
    ) -> RoutingMap: ...


@dataclass(frozen=True)
class SoftmaxRouting(RoutingFunction):
    def call_unbatched(
        self,
        logits: Float[Array, " experts"],
        biases: Float[Array, " experts"] | None = None,
        *,
        num_active: int,
    ) -> RoutingMap:
        assert biases is None
        active_logits, active_indices = jax.lax.top_k(logits, num_active)
        active_weights = jax.nn.softmax(active_logits)
        return RoutingMap(
            active_expert_indices=active_indices,
            active_expert_weights=active_weights,
        )


class RoutingBias(StrEnum):
    GATING = "gating"
    WEIGHTS = "weights"
    BOTH = "both"


@dataclass(frozen=True)
class SigmoidRouting(RoutingFunction):
    bias_mode: RoutingBias | None = None

    def init_biases(self, initializer: Initializer, num_experts: int) -> Float[Array, " experts"] | None:
        if self.bias_mode is None:
            return None
        return initializer.zeros((num_experts,), dtype=jnp.float32)

    def call_unbatched(
        self,
        logits: Float[Array, " experts"],
        biases: Float[Array, " experts"] | None = None,
        *,
        num_active: int,
    ) -> RoutingMap:
        if (self.bias_mode is None) != (biases is None):
            raise ValueError("Biases must be present exactly when a routing bias mode is configured.")
        scores = jax.nn.sigmoid(logits)
        gating_scores = scores
        if biases is not None:
            biased_scores = scores.astype(biases.dtype) + biases
            if self.bias_mode in (RoutingBias.GATING, RoutingBias.BOTH):
                gating_scores = biased_scores
            if self.bias_mode in (RoutingBias.WEIGHTS, RoutingBias.BOTH):
                scores = biased_scores
        _, active_indices = jax.lax.top_k(gating_scores, num_active)
        active_weights = scores[active_indices]
        active_weights = active_weights / (active_weights.sum() + 1e-6)
        return RoutingMap(
            active_expert_indices=active_indices,
            active_expert_weights=active_weights.astype(logits.dtype),
        )


@dataclass(frozen=True)
class RoutedExpertsConfig(StepConfig):
    expert_config: DenseMLPConfig
    num_routed_experts: int

    def init(self, initializer: Initializer, model_dim: int) -> "RoutedExperts":
        return RoutedExperts(
            config=self,
            sharding_config=initializer.sharding_config,
            routed_experts=self.expert_config.init_mixture(initializer, self.num_routed_experts, model_dim),
        )


class RoutedExperts(LalamoModule[RoutedExpertsConfig]):
    routed_experts: DenseMLP

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Value,
        ctx: Context,
        keychain: Keychain,
    ) -> Float[Array, "batch tokens channels"]:
        # Externally routed experts consume a routing map emitted earlier in the program.
        (routing,) = operands
        if not isinstance(routing, RoutingMap):
            raise TypeError(f"Experts can only be routed by a RoutingMap, got {type(routing).__name__}.")
        batch_size, sequence_length, channels = x.shape
        padding_mask = jnp.ones((batch_size, sequence_length), dtype=bool)
        if ctx.lengths is not None:
            padding_mask = jnp.arange(sequence_length)[None, :] < ctx.lengths[:, None]
        outputs = self.routed_experts.apply_routing(
            x.reshape(-1, channels),
            padding_mask.ravel(),
            jax.tree.map(lambda leaf: leaf.reshape(batch_size * sequence_length, -1), routing),
            ctx.forward_pass_config.mlp_forward_pass_config,
            keychain=keychain,
        )
        return outputs.astype(x.dtype).reshape(x.shape)


@dataclass(frozen=True)
class MixtureOfExpertsConfig(MLPConfig):
    expert_config: DenseMLPConfig
    router_config: LinearConfig
    routing_function: RoutingFunction

    num_routed_experts: int
    num_active_routed_experts: int
    router_has_biases: bool

    num_shared_experts: int
    gate_config: LinearConfig | None = None

    @property
    def mixture_size(self) -> int:
        return self.num_routed_experts + self.num_shared_experts

    def init(self, initializer: Initializer, model_dim: int) -> "MixtureOfExperts":
        router = self.router_config.init(
            initializer,
            model_dim,
            (self.num_routed_experts,),
            has_biases=self.router_has_biases,
            is_sharded=False,
        )
        routed_experts = self.expert_config.init_mixture(initializer, self.num_routed_experts, model_dim)
        if self.num_shared_experts > 0:
            shared_experts = self.expert_config.init_mixture(
                initializer,
                self.num_shared_experts,
                model_dim,
                is_sharded=False,
            )
        else:
            shared_experts = None

        if self.gate_config is not None:
            gate = self.gate_config.init(
                initializer,
                model_dim,
                (1,),
                has_biases=False,
                is_sharded=False,
            )
        else:
            gate = None

        return MixtureOfExperts(
            config=self,
            sharding_config=initializer.sharding_config,
            router=router,
            routed_experts=routed_experts,
            shared_experts=shared_experts,
            gate=gate,
            expert_bias=self.routing_function.init_biases(initializer, self.num_routed_experts),
        )


class MixtureOfExperts(MLPBase[MixtureOfExpertsConfig]):
    router: Linear
    routed_experts: DenseMLP
    shared_experts: DenseMLP | None
    gate: Linear | None
    expert_bias: Float[Array, " experts"] | None = None

    @property
    def model_dim(self) -> int:
        return self.routed_experts.model_dim

    @property
    def hidden_dim(self) -> int:
        return self.routed_experts.hidden_dim

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        match forward_pass_config.mode:
            case ForwardPassMode.MULTI_TOKEN:
                return self.call_prefill_mode(
                    inputs,
                    lengths_without_padding,
                    forward_pass_config,
                    keychain=keychain,
                )
            case ForwardPassMode.SINGLE_TOKEN:
                return self.call_decode_mode(inputs, forward_pass_config, keychain=keychain)
            case _:
                raise ValueError(f"Unsupported forward pass mode: {forward_pass_config.mode}")

    def _shared_expert_weight(
        self,
        inputs: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, " one"]:
        if self.gate is not None:
            (gate_value,) = self.gate(
                inputs.astype(jnp.float32),
                keychain=keychain,
                forward_pass_config=forward_pass_config.matmul_config,
            )
            return jax.nn.sigmoid(gate_value)
        return jnp.ones((1,), dtype=jnp.float32)

    def _route(
        self,
        token_input: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> RoutingMap:
        (router_logits,) = self.router(
            token_input.astype(jnp.float32),
            keychain=keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        # reshard keeps the vmapped token axis sharded, unlike device_put.
        router_logits = jax.sharding.reshard(router_logits, PartitionSpec(None))
        return self.config.routing_function.call_unbatched(
            router_logits,
            biases=self.expert_bias,
            num_active=self.config.num_active_routed_experts,
        )

    def _call_decode_token(
        self,
        token_input: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, " channels"]:
        routing_keychain, expert_keychain, shared_weight_keychain, shared_expert_keychain = keychain.split(4)
        routing = self._route(token_input, forward_pass_config, keychain=routing_keychain)
        result = self.routed_experts.apply_token_routing(
            token_input, routing, forward_pass_config, keychain=expert_keychain
        )
        if self.shared_experts is not None:
            shared_weight = self._shared_expert_weight(
                token_input,
                forward_pass_config,
                keychain=shared_weight_keychain,
            )
            shared_outputs = call_vmapped(
                DenseMLP.call_unbatched,
                self.shared_experts,
                token_input,
                forward_pass_config=forward_pass_config,
                keychain=shared_expert_keychain,
                in_axes=(0, None),
            )
            result = result + shared_weight * shared_outputs.astype(jnp.float32).sum(axis=0)
        return result.astype(token_input.dtype)

    @eqx.filter_jit
    def call_decode_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return call_vmapped_twice(
            self._call_decode_token,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )

    @eqx.filter_jit
    def call_prefill_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        batch_size, sequence_length, _ = inputs.shape
        if lengths_without_padding is None:
            lengths_without_padding = jnp.ones(batch_size, dtype=jnp.int32) * sequence_length
        padding_mask = jnp.arange(sequence_length)[None, :] < lengths_without_padding[:, None]

        flattened_inputs = rearrange(inputs, "batch suffix_tokens channels -> (batch suffix_tokens) channels")
        flattened_padding_mask = rearrange(padding_mask, "batch suffix_tokens -> (batch suffix_tokens)")

        def flatten_token_keychain(token_keychain: Keychain) -> Keychain:
            token_keychain = token_keychain.broadcast(
                (batch_size, sequence_length),
                mode=KeychainBroadcastMode.PREFIX,
            )
            return Keychain(
                vmapped_keys=rearrange(token_keychain.vmapped_keys, "batch suffix_tokens -> (batch suffix_tokens)"),
                batch_key=token_keychain.batch_key,
                sharding_config=token_keychain.sharding_config,
            )

        routing_keychain, expert_keychain, shared_weight_keychain, shared_expert_keychain = keychain.split(4)
        routing_map = call_vmapped(
            self._route,
            flattened_inputs,
            forward_pass_config=forward_pass_config,
            keychain=flatten_token_keychain(routing_keychain),
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )

        expert_result = self.routed_experts.apply_routing(
            flattened_inputs,
            flattened_padding_mask,
            routing_map,
            forward_pass_config,
            keychain=expert_keychain,
        )

        if self.shared_experts is not None:
            shared_weights = call_vmapped(
                self._shared_expert_weight,
                flattened_inputs,
                forward_pass_config=forward_pass_config,
                keychain=flatten_token_keychain(shared_weight_keychain),
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            )
            shared_weights = jnp.where(flattened_padding_mask[:, None], shared_weights, 0.0)
            shared_outputs = call_vmapped_twice(
                DenseMLP.call_unbatched,
                self.shared_experts,
                flattened_inputs,
                forward_pass_config=forward_pass_config,
                keychain=shared_expert_keychain,
                in_axes=((0, None), (None, 0)),
                added_sharding_axes=(None, self.sharding_config.resolve_axis(LogicalAxis.BATCH)),
            )
            expert_result = expert_result + shared_weights * shared_outputs.astype(jnp.float32).sum(axis=0)

        return rearrange(
            expert_result.astype(inputs.dtype),
            "(batch suffix_tokens) channels -> batch suffix_tokens channels",
            batch=batch_size,
        )


@dataclass(frozen=True)
class Gemma4RouterConfig(StepConfig):
    linear_config: LinearConfig
    norm_config: NormalizationConfig
    num_experts: int
    num_active_experts: int

    def init(self, initializer: Initializer, model_dim: int) -> "Gemma4Router":
        return Gemma4Router(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=self.norm_config.init(initializer, model_dim),
            scale=initializer.ones((model_dim,)),
            projection=self.linear_config.init(
                initializer,
                model_dim,
                (self.num_experts,),
                has_biases=False,
                is_sharded=False,
            ),
            expert_scales=initializer.ones((self.num_experts,)),
        )


class Gemma4Router(LalamoModule[Gemma4RouterConfig]):
    """Softmax top-k routing from the residual stream, consumed by externally routed experts."""

    norm: Normalization
    scale: Float[Array, " channels"]
    projection: Linear
    expert_scales: Float[Array, " experts"]

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Array,
        ctx: Context,
        keychain: Keychain,
    ) -> RoutingMap:
        () = operands
        batch_size, sequence_length, model_dim = x.shape
        normalized = call_vmapped_twice(self.norm, x) * self.scale.astype(x.dtype) * model_dim**-0.5
        (logits,) = call_vmapped_twice(
            self.projection,
            normalized,
            forward_pass_config=ctx.forward_pass_config.matmul_config,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )
        routing = SoftmaxRouting()(
            logits.astype(jnp.float32).reshape(-1, self.config.num_experts),
            num_active=self.config.num_active_experts,
        )
        weights = routing.active_expert_weights * self.expert_scales[routing.active_expert_indices].astype(jnp.float32)
        return jax.tree.map(
            lambda leaf: leaf.reshape(batch_size, sequence_length, -1),
            RoutingMap(routing.active_expert_indices, weights),
        )
