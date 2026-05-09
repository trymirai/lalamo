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
from jax.sharding import NamedSharding, PartitionSpec, reshard
from jaxtyping import Array, Bool, Float, Int, Key

from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, ShardingAxis
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import is_sharded, sharding_of, use_out_sharding
from lalamo.weight_matrix import GradientEstimator, MatmulConfig

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


@use_out_sharding((None, None))
def _take_moe_chunk_inputs(
    flattened_inputs: Float[Array, "tokens channels"],
    indices: Int[Array, " tokens_per_chunk"],
) -> Float[Array, "tokens_per_chunk channels"]:
    return flattened_inputs.at[indices].get(mode="fill", fill_value=0.0)


@use_out_sharding((ShardingAxis.DATA, None))
def _add_moe_expert_outputs(
    accumulator: Float[Array, "tokens channels"],
    token_indices: Int[Array, "experts tokens_per_chunk"],
    expert_outputs: Float[Array, "experts tokens_per_chunk channels"],
) -> Float[Array, "tokens channels"]:
    return accumulator.at[token_indices].add(
        expert_outputs,
        mode="drop",
    )


def _take_moe_expert_leaf(leaf: object, index: Int[Array, ""]) -> object:
    if not isinstance(leaf, jax.Array):
        return leaf

    leaf_sharding = sharding_of(leaf)
    if not is_sharded(leaf_sharding):
        return leaf.at[jnp.expand_dims(index, 0)].get()[0]

    output_sharding = NamedSharding(leaf_sharding.mesh, PartitionSpec(None, *tuple(leaf_sharding.spec)[1:]))
    return leaf.at[jnp.expand_dims(index, 0)].get(out_sharding=output_sharding)[0]


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
    ) -> "DenseMLP":
        return DenseMLP(
            config=self,
            up_projection=self.linear_config.init_mixture(
                initializer,
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.init_mixture(
                initializer,
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
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
            added_sharding_axes=(ShardingAxis.DATA, None),
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

    def slice_mixture(self, start: int, stop: int) -> Self:
        if self.mixture_size is None:
            raise ValueError("DenseMLP.slice_mixture() requires a mixture DenseMLP.")

        def slice_leaf(leaf: Array) -> Array:
            if not eqx.is_array(leaf):
                return leaf
            if leaf.ndim == 0 or leaf.shape[0] != self.mixture_size:
                raise ValueError(
                    "Unexpected expert leaf shape while slicing experts:"
                    f" expected leading dim {self.mixture_size}, got {leaf.shape}.",
                )
            return leaf[start:stop]

        return jax.tree_util.tree_map(slice_leaf, self)


class RoutingMap(eqx.Module):
    expert_mask: Bool[Array, "*batch_and_tokens experts"]
    expert_weights: Float[Array, "*batch_and_tokens experts"]


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
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[active_indices].set(True)
        expert_weights = jnp.zeros_like(logits)
        expert_weights = expert_weights.at[active_indices].set(active_weights)
        return RoutingMap(expert_mask=mask, expert_weights=expert_weights)


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
        )
        experts = self.expert_config.init_mixture(
            initializer,
            self.mixture_size,
            model_dim,
            self.expert_hidden_dim,
        )

        if self.gate_config is not None:
            gate = self.gate_config.init(
                initializer,
                model_dim,
                (1,),
                has_biases=False,
            )
        else:
            gate = None

        return MixtureOfExperts(
            config=self,
            router=router,
            experts=experts,
            gate=gate,
        )


class MixtureOfExperts(MLPBase[MixtureOfExpertsConfig]):
    router: Linear
    experts: DenseMLP
    gate: Linear | None

    @property
    def model_dim(self) -> int:
        return self.experts.model_dim

    @property
    def hidden_dim(self) -> int:
        return self.experts.hidden_dim

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

    @eqx.filter_jit
    def call_decode_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        inputs = reshard(inputs, PartitionSpec(ShardingAxis.DATA, None, None))

        def per_token(
            token_input: Float[Array, " channels"],
            *,
            keychain: Keychain,
        ) -> Float[Array, " channels"]:
            router_keychain, shared_keychain, expert_keychain = keychain.split(3)
            (router_logits,) = self.router(
                token_input,
                keychain=router_keychain,
                forward_pass_config=forward_pass_config.matmul_config,
            )
            routing = self.config.routing_function.call_unbatched(
                router_logits,
                num_active=self.config.num_active_routed_experts,
            )

            if self.config.num_shared_experts > 0:
                shared_mask = jnp.ones(self.config.num_shared_experts, dtype=bool)
                expert_mask = jnp.concatenate([routing.expert_mask, shared_mask])
                shared_weight = self._shared_expert_weight(
                    token_input,
                    forward_pass_config,
                    keychain=shared_keychain,
                )
                shared_weights = jnp.broadcast_to(shared_weight, (self.config.num_shared_experts,))
                expert_weights = jnp.concatenate([routing.expert_weights, shared_weights])
            else:
                expert_mask = routing.expert_mask
                expert_weights = routing.expert_weights

            num_active = self.config.num_active_routed_experts + self.config.num_shared_experts
            active_indices = jnp.flatnonzero(expert_mask, size=num_active)
            active_weights = expert_weights[active_indices]
            expert_vmapped_keys = expert_keychain.broadcast((num_active,)).vmapped_keys
            expert_batch_keys = jax.random.split(expert_keychain.batch_key, num_active)

            def apply_one(
                idx: Int[Array, ""],
                weight: Float[Array, ""],
                expert_vmapped_key: Key[Array, ""],
                expert_batch_key: Key[Array, ""],
            ) -> Float[Array, " channels"]:
                selected_expert_keychain = Keychain(
                    vmapped_keys=expert_vmapped_key,
                    batch_key=expert_batch_key,
                )
                selected_expert = jax.tree_util.tree_map(
                    lambda leaf: _take_moe_expert_leaf(leaf, idx),
                    self.experts,
                )
                return (
                    selected_expert.call_unbatched(
                        token_input,
                        forward_pass_config,
                        keychain=selected_expert_keychain,
                    )
                    * weight
                )

            return call_vmapped(
                lambda expert_inputs: apply_one(*expert_inputs),
                (active_indices, active_weights, expert_vmapped_keys, expert_batch_keys),
            ).sum(axis=0)

        return call_vmapped_twice(
            per_token,
            inputs,
            keychain=keychain,
            added_sharding_axes=(ShardingAxis.DATA, None),
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
        inputs = reshard(inputs, PartitionSpec(ShardingAxis.DATA, None, None))
        batch_size, sequence_length, _ = inputs.shape
        num_tokens = batch_size * sequence_length
        if lengths_without_padding is None:
            lengths_without_padding = jnp.ones(batch_size, dtype=jnp.int32) * sequence_length
        padding_mask = jnp.arange(sequence_length)[None, :] < lengths_without_padding[:, None]

        flattened_inputs = rearrange(inputs, "batch suffix_tokens channels -> (batch suffix_tokens) channels")
        flattened_padding_mask = rearrange(padding_mask, "batch suffix_tokens -> (batch suffix_tokens)")

        router_keychain, chunk_keychain, shared_weight_keychain, shared_expert_keychain = keychain.split(4)
        (router_logits,) = call_vmapped(
            self.router,
            flattened_inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=router_keychain,
            added_sharding_axis=ShardingAxis.DATA,
        )
        routing_map = self.config.routing_function(router_logits, self.config.num_active_routed_experts)

        token_mask: Bool[Array, "experts tokens"] = rearrange(
            routing_map.expert_mask & flattened_padding_mask[:, None],
            "tokens experts -> experts tokens",
        )
        token_mask = reshard(token_mask, PartitionSpec(ShardingAxis.EXPERT, None))
        expert_weights: Float[Array, "experts tokens"] = rearrange(
            routing_map.expert_weights,
            "tokens experts -> experts tokens",
        )
        expert_weights = jnp.where(token_mask, expert_weights, 0.0)
        expert_weights = reshard(expert_weights, PartitionSpec(ShardingAxis.EXPERT, None))
        routed_experts = self.experts.slice_mixture(0, self.config.num_routed_experts)

        chunk_size = math.ceil(num_tokens * forward_pass_config.moe_chunk_size_ratio)
        num_padded_tokens = math.ceil(num_tokens / chunk_size) * chunk_size
        token_indices = call_vmapped(
            lambda mask: jnp.flatnonzero(mask, size=num_padded_tokens, fill_value=_SENTINEL),
            token_mask,
            added_sharding_axis=ShardingAxis.EXPERT,
        )
        chunked_token_indices = rearrange(
            token_indices,
            "experts (chunks chunk_tokens) -> chunks experts chunk_tokens",
            chunk_tokens=chunk_size,
        )

        num_chunks = chunked_token_indices.shape[0]
        chunk_vmapped_keys = chunk_keychain.broadcast((num_chunks,)).vmapped_keys
        chunk_batch_keys = jax.random.split(chunk_keychain.batch_key, num_chunks)

        def loop_iteration(
            expert_accumulator: Float[Array, "tokens channels"],
            chunk_inputs: tuple[Int[Array, "experts chunk_tokens"], Key[Array, ""], Key[Array, ""]],
        ) -> tuple[Float[Array, "tokens channels"], None]:
            token_indices_for_chunk, chunk_vmapped_key, chunk_batch_key = chunk_inputs
            current_chunk_keychain = Keychain(vmapped_keys=chunk_vmapped_key, batch_key=chunk_batch_key)

            def run_experts(
                accumulator: Float[Array, "tokens channels"],
            ) -> Float[Array, "tokens channels"]:
                weights_for_chunk = jnp.take_along_axis(
                    expert_weights,
                    token_indices_for_chunk,
                    axis=1,
                    mode="fill",
                    fill_value=0.0,
                )

                expert_vmapped_keys = current_chunk_keychain.broadcast(
                    (self.config.num_routed_experts,),
                    sharding_axes=(ShardingAxis.EXPERT,),
                ).vmapped_keys
                expert_batch_keys = jax.random.split(
                    current_chunk_keychain.batch_key,
                    self.config.num_routed_experts,
                )
                expert_batch_keys = reshard(expert_batch_keys, PartitionSpec(ShardingAxis.EXPERT))

                def run_expert(
                    expert: DenseMLP,
                    indices: Int[Array, " tokens_per_chunk"],
                    weights: Float[Array, " tokens_per_chunk"],
                    expert_vmapped_key: Key[Array, ""],
                    expert_batch_key: Key[Array, ""],
                ) -> Float[Array, "tokens_per_chunk channels"]:
                    expert_keychain = Keychain(
                        vmapped_keys=expert_vmapped_key,
                        batch_key=expert_batch_key,
                    )
                    chunk_inputs = _take_moe_chunk_inputs(flattened_inputs, indices)
                    call_unbatched = partial(
                        expert.call_unbatched,
                        forward_pass_config=forward_pass_config,
                    )
                    expert_outputs = call_vmapped(
                        call_unbatched,
                        chunk_inputs,
                        keychain=expert_keychain,
                    )
                    return expert_outputs * weights[:, None]

                expert_outputs = call_vmapped(
                    lambda expert_inputs: run_expert(*expert_inputs),
                    (
                        routed_experts,
                        token_indices_for_chunk,
                        weights_for_chunk,
                        expert_vmapped_keys,
                        expert_batch_keys,
                    ),
                    added_sharding_axis=ShardingAxis.EXPERT,
                )
                return _add_moe_expert_outputs(accumulator, token_indices_for_chunk, expert_outputs)

            return run_experts(expert_accumulator), None

        routed_accumulator = jnp.zeros(
            flattened_inputs.shape,
            dtype=flattened_inputs.dtype,
            out_sharding=PartitionSpec(ShardingAxis.DATA, None),
        )
        routed_expert_result, _ = jax.lax.scan(
            loop_iteration,
            routed_accumulator,
            (chunked_token_indices, chunk_vmapped_keys, chunk_batch_keys),
        )

        expert_result = routed_expert_result
        if self.config.num_shared_experts > 0:
            shared_experts = self.experts.slice_mixture(self.config.num_routed_experts, self.config.mixture_size)
            shared_expert_weight = partial(
                self._shared_expert_weight,
                forward_pass_config=forward_pass_config,
            )
            shared_weights = call_vmapped(
                shared_expert_weight,
                flattened_inputs,
                keychain=shared_weight_keychain,
                added_sharding_axis=ShardingAxis.DATA,
            )
            shared_weights = jnp.where(flattened_padding_mask[:, None], shared_weights, 0.0)

            shared_expert_vmapped_keys = shared_expert_keychain.broadcast(
                (self.config.num_shared_experts,),
                sharding_axes=(ShardingAxis.EXPERT,),
            ).vmapped_keys
            shared_expert_batch_keys = jax.random.split(
                shared_expert_keychain.batch_key,
                self.config.num_shared_experts,
            )
            shared_expert_batch_keys = reshard(shared_expert_batch_keys, PartitionSpec(ShardingAxis.EXPERT))

            def run_shared_expert(
                expert: DenseMLP,
                expert_vmapped_key: Key[Array, ""],
                expert_batch_key: Key[Array, ""],
            ) -> Float[Array, "tokens channels"]:
                expert_keychain = Keychain(
                    vmapped_keys=expert_vmapped_key,
                    batch_key=expert_batch_key,
                )
                return call_vmapped(
                    partial(
                        expert.call_unbatched,
                        forward_pass_config=forward_pass_config,
                    ),
                    flattened_inputs,
                    keychain=expert_keychain,
                    added_sharding_axis=ShardingAxis.DATA,
                )

            shared_outputs = call_vmapped(
                lambda expert_inputs: run_shared_expert(*expert_inputs),
                (shared_experts, shared_expert_vmapped_keys, shared_expert_batch_keys),
                added_sharding_axis=ShardingAxis.EXPERT,
            )
            expert_result = routed_expert_result + shared_weights * shared_outputs.sum(axis=0)

        return rearrange(
            expert_result,
            "(batch suffix_tokens) channels -> batch suffix_tokens channels",
            batch=batch_size,
        )
