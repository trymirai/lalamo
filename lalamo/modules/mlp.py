from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import DotAlgorithmPreset
from jax.sharding import NamedSharding, PartitionSpec
from jaxtyping import Array, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import ForwardPassMode, Keychain, LalamoConfig, LalamoModule, ShardingAxis
from lalamo.utils.registry_abc import RegistryABC
from lalamo.utils.sharding import is_sharded, make_sharding, sharding_of, with_sharding
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


def _take_moe_expert_leaf(leaf: object, indices: Int[Array, " experts"]) -> object:
    if not isinstance(leaf, jax.Array):
        return leaf

    leaf_sharding = sharding_of(leaf)
    if not is_sharded(leaf_sharding):
        return leaf.at[indices].get()

    output_sharding = NamedSharding(leaf_sharding.mesh, PartitionSpec(None, *tuple(leaf_sharding.spec)[1:]))
    return leaf.at[indices].get(out_sharding=output_sharding)


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
    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,
        *,
        is_sharded: bool = True,
    ) -> "MLPBase": ...


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

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,
        *,
        is_sharded: bool = True,
    ) -> "DenseMLP":
        return DenseMLP(
            config=self,
            up_projection=self.linear_config.init(
                initializer,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                is_sharded=is_sharded,
            ),
            down_projection=self.linear_config.init(
                initializer,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                is_sharded=is_sharded,
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
        if self.up_projection.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                " They are intended to be used with call_vmapped or ragged mixture methods instead.",
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
        if self.up_projection.mixture_size is None:
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
        if self.up_projection.mixture_size is None:
            raise ValueError("DenseMLP.call_mixture() requires a mixture DenseMLP.")

        return call_vmapped_twice(
            DenseMLP.call_unbatched,
            self,
            inputs,
            forward_pass_config=forward_pass_config,
            keychain=keychain,
            in_axes=((0, None), (None, 0)),
            added_sharding_axes=(None, ShardingAxis.DATA),
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

    def init(
        self,
        initializer: Initializer,
        model_dim: int,
        hidden_dim: int,  # noqa: ARG002
        *,
        is_sharded: bool = True,  # noqa: ARG002
    ) -> "MixtureOfExperts":
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
            is_sharded=True,
        )
        if self.num_shared_experts == 0:
            shared_experts = None
        else:
            shared_experts = self.expert_config.init_mixture(
                initializer,
                self.num_shared_experts,
                model_dim,
                self.expert_hidden_dim,
                is_sharded=False,
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

    @eqx.filter_jit
    def call_decode_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        forward_pass_config: MLPForwardPassConfig = MLPForwardPassConfig(),
        *,
        keychain: Keychain,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        inputs = with_sharding(inputs, make_sharding((ShardingAxis.DATA, None, None)))

        def per_token(
            token_input: Float[Array, " channels"],
            *,
            keychain: Keychain,
        ) -> Float[Array, " channels"]:
            router_keychain, routed_keychain, shared_weight_keychain, shared_keychain = keychain.split(4)
            (router_logits,) = self.router(
                token_input,
                keychain=router_keychain,
                forward_pass_config=forward_pass_config.matmul_config,
            )
            routing = self.config.routing_function.call_unbatched(
                router_logits,
                num_active=self.config.num_active_routed_experts,
            )

            active_routed_experts = jax.tree_util.tree_map(
                partial(_take_moe_expert_leaf, indices=routing.active_expert_indices),
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
        inputs = with_sharding(inputs, make_sharding((ShardingAxis.DATA, None, None)))
        batch_size, sequence_length, _ = inputs.shape
        num_tokens = batch_size * sequence_length
        if lengths_without_padding is None:
            lengths_without_padding = jnp.ones(batch_size, dtype=jnp.int32) * sequence_length
        flat_token_indices = jnp.arange(num_tokens)
        flattened_padding_mask = flat_token_indices % sequence_length < lengths_without_padding[
            flat_token_indices // sequence_length
        ]
        flattened_inputs = inputs.reshape(num_tokens, inputs.shape[-1])

        router_keychain, routed_expert_keychain, shared_weight_keychain, shared_expert_keychain = keychain.split(4)
        (router_logits,) = call_vmapped(
            self.router,
            flattened_inputs,
            forward_pass_config=forward_pass_config.matmul_config,
            keychain=router_keychain,
            added_sharding_axis=ShardingAxis.DATA,
        )
        routing_map = self.config.routing_function(router_logits, self.config.num_active_routed_experts)

        active_expert_indices = with_sharding(routing_map.active_expert_indices, make_sharding((None, None)))
        active_expert_weights = with_sharding(routing_map.active_expert_weights, make_sharding((None, None)))
        num_active_routed_experts = active_expert_indices.shape[-1]
        flat_padding = jnp.broadcast_to(
            flattened_padding_mask[:, None], active_expert_indices.shape,
        ).ravel()
        routed_expert_indices = jnp.where(
            flat_padding, active_expert_indices.ravel(), self.config.num_routed_experts,
        )
        routed_assignment_indices = jnp.argsort(routed_expert_indices)
        routed_group_sizes = jnp.bincount(
            routed_expert_indices,
            length=self.config.num_routed_experts,
        ).astype(jnp.int32)
        routed_token_indices = routed_assignment_indices // num_active_routed_experts
        routed_expert_indices = routed_expert_indices[routed_assignment_indices]
        routed_weights = active_expert_weights.ravel()[routed_assignment_indices]
        # Padded (token, expert) pairs are routed to the sentinel expert id ``num_routed_experts`` and
        # dropped from the ragged_dot groups, but ragged_dot still emits nonzero rows for that ungrouped
        # tail. Zero their routing weights so padded tokens contribute nothing to the scatter below.
        routed_weights = jnp.where(
            routed_expert_indices < self.config.num_routed_experts,
            routed_weights,
            0.0,
        )
        routed_group_sizes = with_sharding(routed_group_sizes, make_sharding((ShardingAxis.EXPERT,)))

        routed_input_source = with_sharding(flattened_inputs, make_sharding((None, None)))
        routed_inputs = routed_input_source.at[routed_token_indices].get()
        routed_outputs = self.routed_experts.call_ragged_mixture(
            routed_inputs,
            routed_expert_indices,
            routed_group_sizes,
            forward_pass_config,
            keychain=routed_expert_keychain,
        )
        routed_accumulator = jnp.zeros_like(routed_input_source)
        routed_expert_result = routed_accumulator.at[routed_token_indices].add(
            routed_outputs * routed_weights[:, None],
            mode="drop",
        )
        routed_expert_result = with_sharding(routed_expert_result, make_sharding((ShardingAxis.DATA, None)))

        expert_result = routed_expert_result
        if self.shared_experts is not None:
            shared_weights = call_vmapped(
                self._shared_expert_weight,
                flattened_inputs,
                forward_pass_config=forward_pass_config,
                keychain=shared_weight_keychain,
                added_sharding_axis=ShardingAxis.DATA,
            )
            shared_weights = jnp.where(flattened_padding_mask[:, None], shared_weights, 0.0)
            shared_outputs = self.shared_experts.call_mixture(
                flattened_inputs,
                forward_pass_config,
                keychain=shared_expert_keychain,
            )
            expert_result = routed_expert_result + shared_weights * shared_outputs.sum(axis=0)

        return expert_result.reshape(inputs.shape)
