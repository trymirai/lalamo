import math
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax.lax import DotAlgorithmPreset
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
)
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import is_sharded, sharding_of, with_sharding
from lalamo.weight_matrix import FullPrecisionMatrix, GradientEstimator, MatmulConfig

from .activations import Activation
from .linear import Linear, LinearConfig
from .utils import call_vmapped, call_vmapped_twice

__all__ = [
    "DenseMLP",
    "DenseMLPConfig",
    "MLPBase",
    "MLPConfig",
    "MLPForwardPassConfig",
    "MixtureOfExperts",
    "MixtureOfExpertsConfig",
    "RoutingFunction",
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


def _add_moe_expert_outputs(
    accumulator: Float[Array, "tokens channels"],
    token_indices: Int[Array, "experts tokens_per_chunk"],
    expert_outputs: Float[Array, "experts tokens_per_chunk channels"],
    out_sharding: NamedSharding,
) -> Float[Array, "tokens channels"]:
    return accumulator.at[token_indices].add(
        expert_outputs,
        mode="drop",
        out_sharding=out_sharding,
    )


def _take_moe_expert_leaf(
    leaf: object,
    index: Int[Array, ""] | Int[Array, " experts"],
) -> object:
    if not isinstance(leaf, jax.Array):
        return leaf

    replicated = NamedSharding(sharding_of(leaf).mesh, PartitionSpec())
    return leaf.at[jnp.expand_dims(index, 0)].get(out_sharding=replicated)[0]


@dataclass(frozen=True)
class MLPForwardPassConfig:
    mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN
    moe_chunk_size_ratio: float = 0.2
    matmul_config: MatmulConfig = dataclass_field(default_factory=MatmulConfig)

    @classmethod
    def for_tracer_tests(cls) -> Self:
        return cls(
            matmul_config=MatmulConfig.for_tracer_tests(),
        )

    @classmethod
    def for_inference(
        cls,
        mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            mode=mode,
            matmul_config=MatmulConfig.for_inference(precision),
        )

    @classmethod
    def for_training(
        cls,
        gradient_estimator: GradientEstimator = GradientEstimator.DETERMINISTIC_ROUNDING,
        precision: DotAlgorithmPreset = DotAlgorithmPreset.DEFAULT,
    ) -> Self:
        return cls(
            matmul_config=MatmulConfig.for_training(gradient_estimator, precision),
        )


@dataclass(frozen=True)
class MLPConfig(LalamoConfig, RegistryABC):
    @abstractmethod
    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "MLPBase": ...


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


@dataclass(frozen=True)
class DenseMLPConfig(MLPConfig):
    linear_config: LinearConfig
    activation: Activation
    has_up_biases: bool
    has_down_biases: bool
    gate_clipping: tuple[float | None, float | None] | None
    up_clipping: tuple[float | None, float | None] | None

    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "DenseMLP":
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
        hidden_dim: int,
        *,
        is_sharded: bool = True,
    ) -> "DenseMLP":
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
        call_unbatched = partial(
            self.call_unbatched,
            forward_pass_config=forward_pass_config,
        )
        return call_vmapped_twice(
            call_unbatched,
            inputs,
            keychain=keychain,
            added_sharding_axes=(self.sharding_config.resolve_axis(LogicalAxis.BATCH), None),
        )

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
        if self.config.gate_clipping is not None:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping is not None:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(
            up_proj * gate,
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
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        if self.mixture_size is None:
            raise ValueError("DenseMLP.call_ragged_mixture() requires a mixture DenseMLP.")

        up_keychain, down_keychain = keychain.split()
        up_proj, gate = self.up_projection.call_ragged(
            inputs,
            expert_indices,
            group_sizes,
            keychain=up_keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        if self.config.gate_clipping is not None:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping is not None:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        hidden = up_proj * self.config.activation(gate)
        (result,) = self.down_projection.call_ragged(
            hidden,
            expert_indices,
            group_sizes,
            keychain=down_keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        return result

    def call_mixture(
        self,
        inputs: Float[Array, "tokens channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "experts tokens channels"]:
        if self.mixture_size is None:
            raise ValueError("DenseMLP.call_mixture() requires a mixture DenseMLP.")
        return call_vmapped_twice(
            DenseMLP.call_unbatched,
            self,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            in_axes=((0, None), (None, 0)),
            added_sharding_axes=(None, self.sharding_config.resolve_axis(LogicalAxis.BATCH)),
        )


class RoutingMap(eqx.Module):
    active_expert_indices: Int[Array, "*batch_and_tokens active_experts"]
    active_expert_weights: Float[Array, "*batch_and_tokens active_experts"]


@dataclass(frozen=True)
class RoutingFunction(LalamoConfig, RegistryABC):
    def __call__(self, logits: Float[Array, "batch_tokens experts"], num_active: int) -> RoutingMap:
        return call_vmapped(partial(self.call_unbatched, num_active=num_active), logits)

    @abstractmethod
    def call_unbatched(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap: ...


@dataclass(frozen=True)
class SoftmaxRouting(RoutingFunction):
    def call_unbatched(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap:
        active_logits, active_indices = jax.lax.top_k(logits, num_active)
        active_weights = jax.nn.softmax(active_logits)
        return RoutingMap(
            active_expert_indices=active_indices,
            active_expert_weights=active_weights,
        )


@dataclass(frozen=True)
class MixtureOfExpertsConfig(MLPConfig):
    expert_config: DenseMLPConfig
    router_config: LinearConfig
    routing_function: RoutingFunction

    num_routed_experts: int
    num_active_routed_experts: int
    router_has_biases: bool

    num_shared_experts: int
    expert_hidden_dim: int
    gate_config: LinearConfig | None = None

    @property
    def mixture_size(self) -> int:
        return self.num_routed_experts + self.num_shared_experts

    def init(self, initializer: Initializer, model_dim: int, hidden_dim: int) -> "MixtureOfExperts":  # noqa: ARG002
        router = self.router_config.init(
            initializer,
            model_dim,
            (self.num_routed_experts,),
            has_biases=self.router_has_biases,
            is_sharded=False,
        )
        routed_experts = self.expert_config.init_mixture(
            initializer,
            self.num_routed_experts,
            model_dim,
            self.expert_hidden_dim,
        )
        shared_experts = (
            self.expert_config.init_mixture(
                initializer,
                self.num_shared_experts,
                model_dim,
                self.expert_hidden_dim,
                is_sharded=False,
            )
            if self.num_shared_experts > 0
            else None
        )

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
        )


class MixtureOfExperts(MLPBase[MixtureOfExpertsConfig]):
    router: Linear
    routed_experts: DenseMLP
    shared_experts: DenseMLP | None
    gate: Linear | None

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
                inputs,
                keychain=keychain,
                forward_pass_config=forward_pass_config.matmul_config,
            )
            return jax.nn.sigmoid(gate_value)
        return jnp.ones((1,), dtype=inputs.dtype)

    def _call_decode_token(
        self,
        token_input: Float[Array, " channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, " channels"]:
        router_keychain, routed_keychain, shared_weight_keychain, shared_keychain = keychain.split(4)
        (router_logits,) = self.router(
            token_input,
            keychain=router_keychain,
            forward_pass_config=forward_pass_config.matmul_config,
        )
        router_logits = jax.device_put(router_logits, self.sharding_config.make_sharding((None,)))
        routing = self.config.routing_function.call_unbatched(
            router_logits,
            num_active=self.config.num_active_routed_experts,
        )
        active_routed_experts = jax.tree_util.tree_map(
            partial(_take_moe_expert_leaf, index=routing.active_expert_indices),
            self.routed_experts,
        )
        routed_outputs = call_vmapped(
            DenseMLP.call_unbatched,
            active_routed_experts,
            token_input,
            forward_pass_config=forward_pass_config,
            keychain=routed_keychain,
            in_axes=(0, None),
        )
        routed_result = (routed_outputs * routing.active_expert_weights[:, None]).sum(axis=0)
        if self.shared_experts is None:
            return routed_result

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
            keychain=shared_keychain,
            in_axes=(0, None),
        )
        return routed_result + shared_weight * shared_outputs.sum(axis=0)

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

    def _call_ragged_routed_experts(
        self,
        flattened_inputs: Float[Array, "tokens channels"],
        flattened_padding_mask: Bool[Array, " tokens"],
        routing_map: RoutingMap,
        forward_pass_config: MLPForwardPassConfig,
        *,
        keychain: Keychain,
    ) -> Float[Array, "tokens channels"]:
        replicated_vector_sharding = self.sharding_config.make_sharding((None,))
        active_expert_indices = with_sharding(
            routing_map.active_expert_indices,
            self.sharding_config.make_sharding((None, None)),
        )
        active_expert_weights = with_sharding(
            routing_map.active_expert_weights,
            self.sharding_config.make_sharding((None, None)),
        )
        num_active = active_expert_indices.shape[-1]
        flat_padding_mask = jnp.broadcast_to(
            flattened_padding_mask[:, None],
            active_expert_indices.shape,
        ).ravel()
        expert_indices = with_sharding(
            jnp.where(
                flat_padding_mask,
                active_expert_indices.ravel(),
                self.config.num_routed_experts,
            ),
            replicated_vector_sharding,
        )
        assignment_order = jnp.argsort(expert_indices)
        sorted_expert_indices = expert_indices[assignment_order]
        valid_assignments = sorted_expert_indices < self.config.num_routed_experts
        group_sizes = jnp.bincount(
            expert_indices,
            length=self.config.num_routed_experts,
        ).astype(jnp.int32)
        group_sizes = with_sharding(
            group_sizes,
            replicated_vector_sharding,
        )
        token_indices = assignment_order // num_active
        safe_expert_indices = jnp.where(valid_assignments, sorted_expert_indices, 0)
        expert_weights = active_expert_weights.ravel()[assignment_order]

        replicated_matrix_sharding = self.sharding_config.make_sharding((None, None))
        replicated_inputs = with_sharding(flattened_inputs, replicated_matrix_sharding)
        routed_inputs = replicated_inputs.at[token_indices].get(out_sharding=replicated_matrix_sharding)
        routed_outputs = self.routed_experts.call_ragged_mixture(
            routed_inputs,
            safe_expert_indices,
            group_sizes,
            forward_pass_config,
            keychain=keychain,
        )
        # ragged_dot leaves the ungrouped padded tail unspecified.
        weighted_outputs = jnp.where(
            valid_assignments[:, None],
            routed_outputs * expert_weights[:, None],
            0.0,
        )
        routed_result = jnp.zeros_like(replicated_inputs).at[token_indices].add(weighted_outputs)
        return with_sharding(
            routed_result,
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
        batch_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None))
        mixture_sharding = self.sharding_config.resolve_sharding((LogicalAxis.MIXTURE, None))
        mixture_vector_sharding = self.sharding_config.resolve_sharding((LogicalAxis.MIXTURE,))

        active_selection = jax.nn.one_hot(
            routing_map.active_expert_indices,
            self.config.num_routed_experts,
            dtype=routing_map.active_expert_weights.dtype,
        )
        token_mask: Bool[Array, "experts tokens"] = rearrange(
            jnp.any(active_selection != 0, axis=-2) & flattened_padding_mask[:, None],
            "tokens experts -> experts tokens",
        )
        token_mask = with_sharding(token_mask, mixture_sharding)
        expert_weights: Float[Array, "experts tokens"] = rearrange(
            jnp.sum(active_selection * routing_map.active_expert_weights[..., None], axis=-2),
            "tokens experts -> experts tokens",
        )
        expert_weights = with_sharding(jnp.where(token_mask, expert_weights, 0.0), mixture_sharding)
        routed_expert_indices = with_sharding(jnp.arange(self.config.num_routed_experts), mixture_vector_sharding)

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
        chunk_batch_keys = with_sharding(
            jax.random.split(keychain.batch_key, num_chunks),
            self.sharding_config.make_sharding((None,)),
        )

        def loop_iteration(
            expert_accumulator: Float[Array, "tokens channels"],
            chunk_inputs: tuple[Int[Array, "experts chunk_tokens"], Key[Array, ""], Key[Array, ""]],
        ) -> tuple[Float[Array, "tokens channels"], None]:
            token_indices_for_chunk, chunk_vmapped_key, chunk_batch_key = chunk_inputs
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
            expert_vmapped_keys = with_sharding(
                current_chunk_keychain.broadcast((self.config.num_routed_experts,)).vmapped_keys,
                mixture_vector_sharding,
            )
            expert_batch_keys = with_sharding(
                jax.random.split(current_chunk_keychain.batch_key, self.config.num_routed_experts),
                mixture_vector_sharding,
            )

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
                    partial(_take_moe_expert_leaf, index=expert_index),
                    self.routed_experts,
                )
                chunk_inputs = _take_moe_chunk_inputs(
                    flattened_inputs,
                    indices,
                    self.sharding_config.make_sharding((None, None)),
                )
                expert_outputs = call_vmapped(
                    partial(expert.call_unbatched, forward_pass_config=forward_pass_config),
                    chunk_inputs,
                    keychain=expert_keychain,
                )
                return expert_outputs * weights[:, None]

            expert_outputs = call_vmapped(
                lambda expert_inputs: run_expert(*expert_inputs),
                (
                    routed_expert_indices,
                    token_indices_for_chunk,
                    weights_for_chunk,
                    expert_vmapped_keys,
                    expert_batch_keys,
                ),
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.MIXTURE),
            )
            return (
                _add_moe_expert_outputs(
                    expert_accumulator,
                    token_indices_for_chunk,
                    expert_outputs,
                    batch_sharding,
                ),
                None,
            )

        routed_accumulator = jnp.zeros(
            flattened_inputs.shape,
            dtype=flattened_inputs.dtype,
            out_sharding=batch_sharding,
        )
        routed_result, _ = jax.lax.scan(
            loop_iteration,
            routed_accumulator,
            (chunked_token_indices, chunk_vmapped_keys, chunk_batch_keys),
        )
        return routed_result

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
        batch_sharding = self.sharding_config.resolve_sharding((LogicalAxis.BATCH, None))

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

        router_keychain, routed_expert_keychain, shared_weight_keychain, shared_expert_keychain = keychain.split(4)
        (router_logits,) = call_vmapped(
            self.router,
            flattened_inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=flatten_token_keychain(router_keychain),
            added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
        )
        router_logits = with_sharding(router_logits, batch_sharding)
        routing_map = self.config.routing_function(router_logits, self.config.num_active_routed_experts)

        up_weights = self.routed_experts.up_projection.weights
        down_weights = self.routed_experts.down_projection.weights
        use_ragged = (
            isinstance(up_weights, FullPrecisionMatrix)
            and isinstance(down_weights, FullPrecisionMatrix)
            and not is_sharded(sharding_of(up_weights.weights))
            and not is_sharded(sharding_of(down_weights.weights))
        )
        if use_ragged:
            routed_expert_result = self._call_ragged_routed_experts(
                flattened_inputs,
                flattened_padding_mask,
                routing_map,
                forward_pass_config,
                keychain=routed_expert_keychain,
            )
        else:
            routed_expert_result = self._call_chunked_routed_experts(
                flattened_inputs,
                flattened_padding_mask,
                routing_map,
                forward_pass_config,
                keychain=routed_expert_keychain,
            )

        expert_result = routed_expert_result
        if self.shared_experts is not None:
            shared_expert_weight = partial(
                self._shared_expert_weight,
                forward_pass_config=forward_pass_config,
            )
            shared_weights = call_vmapped(
                shared_expert_weight,
                flattened_inputs,
                keychain=flatten_token_keychain(shared_weight_keychain),
                added_sharding_axis=self.sharding_config.resolve_axis(LogicalAxis.BATCH),
            )
            shared_weights = jnp.where(flattened_padding_mask[:, None], shared_weights, 0.0)

            shared_outputs = self.shared_experts.call_mixture(
                flattened_inputs,
                forward_pass_config,
                keychain=shared_expert_keychain,
            )
            expert_result = routed_expert_result + shared_weights * shared_outputs.sum(axis=0)

        return rearrange(
            expert_result,
            "(batch suffix_tokens) channels -> batch suffix_tokens channels",
            batch=batch_size,
        )
