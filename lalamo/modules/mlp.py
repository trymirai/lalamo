import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from lalamo.common import ParameterTree, require_tree
from lalamo.modules.utils import vmap_twice

from .activations import Activation
from .common import (
    DummyUnionMember,
    ForwardPassMode,
    LalamoModule,
    register_config_union,
)
from .linear import LinearBase, LinearConfig

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


@dataclass(frozen=True)
class MLPForwardPassConfig:
    moe_chunk_size_ratio: float = 0.2


class MLPBase[ConfigT: MLPConfig](LalamoModule[ConfigT]):
    @property
    @abstractmethod
    def activation_precision(self) -> DTypeLike: ...

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
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: MLPForwardPassConfig | None = None,
    ) -> Float[Array, "batch suffix_tokens channels"]: ...


@dataclass(frozen=True)
class MLPConfigBase(ABC):
    @abstractmethod
    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> MLPBase: ...

    @abstractmethod
    def empty(self, model_dim: int, hidden_dim: int) -> MLPBase: ...


@dataclass(frozen=True)
class DenseMLPConfig(MLPConfigBase):
    linear_config: LinearConfig
    activation: Activation
    has_up_biases: bool
    has_down_biases: bool
    gate_clipping: tuple[float | None, float | None] | None
    up_clipping: tuple[float | None, float | None] | None

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "DenseMLP":
        up_key, down_key = jax.random.split(key)
        return DenseMLP(
            self,
            up_projection=self.linear_config.random_init(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty(self, model_dim: int, hidden_dim: int) -> "DenseMLP":
        return DenseMLP(
            self,
            up_projection=self.linear_config.empty(
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty(
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )

    def random_init_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> "DenseMLP":
        up_key, down_key = jax.random.split(key)
        return DenseMLP(
            self,
            up_projection=self.linear_config.random_init_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
                key=up_key,
            ),
            down_projection=self.linear_config.random_init_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
                key=down_key,
            ),
        )

    def empty_mixture(
        self,
        mixture_size: int,
        model_dim: int,
        hidden_dim: int,
    ) -> "DenseMLP":
        return DenseMLP(
            self,
            up_projection=self.linear_config.empty_mixture(
                mixture_size,
                model_dim,
                (hidden_dim, hidden_dim),
                has_biases=self.has_up_biases,
            ),
            down_projection=self.linear_config.empty_mixture(
                mixture_size,
                hidden_dim,
                (model_dim,),
                has_biases=self.has_down_biases,
            ),
        )


class DenseMLP(MLPBase[DenseMLPConfig]):
    up_projection: LinearBase
    down_projection: LinearBase

    @property
    def activation_precision(self) -> DTypeLike:
        return self.up_projection.activation_precision

    @property
    def model_dim(self) -> int:
        return self.up_projection.input_dim

    @property
    def hidden_dim(self) -> int:
        return self.down_projection.input_dim

    @property
    def mixture_size(self) -> int | None:
        return self.up_projection.mixture_size

    def __post_init__(self) -> None:
        up_output_dim, gate_output_dim = self.up_projection.output_dims
        if up_output_dim != gate_output_dim:
            raise ValueError(
                f"Up projection output dimension {up_output_dim} does not match"
                f" the gate output dimension {gate_output_dim}",
            )
        (down_output_dim,) = self.down_projection.output_dims
        if (self.up_projection.input_dim, up_output_dim) != (
            down_output_dim,
            self.down_projection.input_dim,
        ):
            raise ValueError(
                f"Down projection dimensions {self.down_projection.input_dim, down_output_dim} do not match"
                f" the up projection output dimensions {self.up_projection.input_dim, up_output_dim}",
            )

    @eqx.filter_jit
    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,  # noqa: ARG002
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,  # noqa: ARG002
        forward_pass_config: MLPForwardPassConfig | None = None,  # noqa: ARG002
    ) -> Float[Array, "batch suffix_tokens channels"]:
        return vmap_twice(self.call_unbatched)(inputs)

    @eqx.filter_jit
    def call_unbatched(
        self,
        inputs: Float[Array, " channels"],
    ) -> Float[Array, " channels"]:
        if self.mixture_size is not None:
            raise ValueError(
                "Mixtures of linear layers cannot be called directly."
                "They are intended to be used with methods eqx.filter_vmap or lax.scan instead.",
            )
        up_proj, gate = self.up_projection(inputs)
        if self.config.gate_clipping:
            gate = jnp.clip(gate, *self.config.gate_clipping)
        if self.config.up_clipping:
            up_proj = jnp.clip(up_proj, *self.config.up_clipping)
        gate = self.config.activation(gate)
        (result,) = self.down_projection(up_proj * gate)

        return result

    def export_weights(self) -> ParameterTree:
        return {
            "up_projection": self.up_projection.export_weights(),
            "down_projection": self.down_projection.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            up_projection=self.up_projection.import_weights(require_tree(weights["up_projection"])),
            down_projection=self.down_projection.import_weights(require_tree(weights["down_projection"])),
        )


class RoutingMap(eqx.Module):
    expert_mask: Bool[Array, "*batch_tokens experts"]
    expert_weights: Float[Array, "*batch_tokens experts"]


@dataclass(frozen=True)
class RoutingFunctionBase(ABC):
    def __call__(self, logits: Float[Array, "batch_tokens experts"], num_active: int) -> RoutingMap:
        return vmap(partial(self.call_unbatched, num_active=num_active))(logits)

    @abstractmethod
    def call_unbatched(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap: ...


@dataclass(frozen=True)
class SoftmaxRouting(RoutingFunctionBase):
    def call_unbatched(self, logits: Float[Array, " experts"], num_active: int) -> RoutingMap:
        active_logits, active_indices = jax.lax.top_k(logits, num_active)
        active_weights = jax.nn.softmax(active_logits)
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[active_indices].set(True)
        expert_weights = jnp.zeros_like(logits)
        expert_weights = expert_weights.at[active_indices].set(active_weights)
        return RoutingMap(expert_mask=mask, expert_weights=expert_weights)


RoutingFunction = SoftmaxRouting | DummyUnionMember


register_config_union(RoutingFunction)


@dataclass(frozen=True)
class MixtureOfExpertsConfig(ABC):
    expert_config: DenseMLPConfig
    router_config: LinearConfig
    routing_function: RoutingFunction

    mixture_size: int
    num_experts_per_token: int
    router_has_biases: bool

    def random_init(self, model_dim: int, hidden_dim: int, *, key: PRNGKeyArray) -> "MixtureOfExperts":
        experts_key, router_key = jax.random.split(key)
        router = self.router_config.random_init(
            model_dim,
            (self.mixture_size,),
            has_biases=self.router_has_biases,
            key=router_key,
        )
        experts = self.expert_config.random_init_mixture(self.mixture_size, model_dim, hidden_dim, key=experts_key)
        return MixtureOfExperts(self, router, experts)

    def empty(self, model_dim: int, hidden_dim: int) -> "MixtureOfExperts":
        router = self.router_config.empty(model_dim, (self.mixture_size,), has_biases=self.router_has_biases)
        experts = self.expert_config.empty_mixture(self.mixture_size, model_dim, hidden_dim)
        return MixtureOfExperts(self, router, experts)


class MixtureOfExperts(MLPBase[MixtureOfExpertsConfig]):
    router: LinearBase
    experts: DenseMLP

    @property
    def mixture_size(self) -> int:
        return self.config.mixture_size

    @property
    def num_experts_per_token(self) -> int:
        return self.config.num_experts_per_token

    @property
    def activation_precision(self) -> DTypeLike:
        return self.experts.activation_precision

    @property
    def model_dim(self) -> int:
        return self.experts.model_dim

    @property
    def hidden_dim(self) -> int:
        return self.experts.hidden_dim

    def __post_init__(self) -> None:
        if self.router.input_dim != self.experts.model_dim:
            raise ValueError(
                f"Router input dimension ({self.router.input_dim}) must match experts model_dim"
                f" ({self.experts.model_dim}).",
            )

        (router_output_dim,) = self.router.output_dims
        if router_output_dim != self.mixture_size:
            raise ValueError(
                f"Router output dimension ({router_output_dim}) must equal mixture_size ({self.mixture_size}).",
            )

        if self.experts.mixture_size != self.mixture_size:
            raise ValueError(
                f"Experts mixture_size ({self.experts.mixture_size}) does not match specified mixture_size"
                f" ({self.mixture_size}).",
            )

    def __call__(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_mode: ForwardPassMode = ForwardPassMode.MULTI_TOKEN,
        forward_pass_config: MLPForwardPassConfig | None = None,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        match forward_pass_mode:
            case ForwardPassMode.MULTI_TOKEN:
                return self.call_prefill_mode(inputs, lengths_without_padding, forward_pass_config)
            case ForwardPassMode.SINGLE_TOKEN:
                return self.call_decode_mode(inputs)

    @eqx.filter_jit
    def call_decode_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
    ) -> Float[Array, "batch suffix_tokens channels"]:
        def per_token(x: Float[Array, " channels"]) -> Float[Array, " channels"]:
            (router_logits,) = self.router(x)
            routing = self.config.routing_function.call_unbatched(
                router_logits,
                num_active=self.num_experts_per_token,
            )
            active_indices = jnp.flatnonzero(routing.expert_mask, size=self.num_experts_per_token)
            active_weights = routing.expert_weights[active_indices]

            def apply_one(idx: Int[Array, ""], w: Float[Array, ""]) -> Float[Array, " channels"]:
                selected_expert = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.dynamic_index_in_dim(leaf, idx, axis=0, keepdims=False),
                    self.experts,
                )
                return selected_expert.call_unbatched(x) * w

            contributions = vmap(apply_one)(active_indices, active_weights)
            return jnp.sum(contributions, axis=0)

        return vmap_twice(per_token)(inputs)

    @eqx.filter_jit
    def call_prefill_mode(
        self,
        inputs: Float[Array, "batch suffix_tokens channels"],
        lengths_without_padding: Int[Array, " batch"] | None = None,
        forward_pass_config: MLPForwardPassConfig | None = None,
    ) -> Float[Array, "batch suffix_tokens channels"]:
        forward_pass_config = forward_pass_config or MLPForwardPassConfig()
        batch_size, sequence_length, _ = inputs.shape
        num_tokens = batch_size * sequence_length
        if lengths_without_padding is None:
            lengths_without_padding = jnp.ones(batch_size, dtype=jnp.int32) * sequence_length
        padding_mask = jnp.arange(sequence_length)[None, :] < lengths_without_padding[:, None]

        flattened_inputs = rearrange(inputs, "batch suffix_tokens channels -> (batch suffix_tokens) channels")
        flattened_padding_mask = rearrange(padding_mask, "batch suffix_tokens -> (batch suffix_tokens)")

        (router_logits,) = vmap(self.router)(flattened_inputs)
        routing_map = self.config.routing_function(router_logits, self.num_experts_per_token)
        token_mask = rearrange(
            routing_map.expert_mask & flattened_padding_mask[:, None],
            "tokens experts -> experts tokens",
        )
        expert_weights = rearrange(
            routing_map.expert_weights,
            "tokens experts -> experts tokens",
        )
        expert_weights = jnp.where(token_mask, expert_weights, 0.0)

        chunk_size = math.ceil(num_tokens * forward_pass_config.moe_chunk_size_ratio)
        num_padded_tokens = math.ceil(num_tokens / chunk_size) * chunk_size
        token_indices = vmap(lambda m: jnp.flatnonzero(m, size=num_padded_tokens, fill_value=_SENTINEL))(token_mask)
        chunked_token_indices = rearrange(
            token_indices,
            "experts (chunks chunk_tokens) -> chunks experts chunk_tokens",
            chunk_tokens=chunk_size,
        )

        def loop_iteration(
            accumulator: Float[Array, "tokens channels"],
            token_indices_for_chunk: Int[Array, "experts chunk_tokens"],
        ) -> tuple[Float[Array, "tokens channels"], None]:
            def inner() -> Float[Array, "tokens channels"]:
                weights_for_chunk = jnp.take_along_axis(
                    expert_weights,
                    token_indices_for_chunk,
                    axis=1,
                    mode="fill",
                    fill_value=0.0,
                )

                def run_expert(
                    expert: DenseMLP,
                    indices: Int[Array, " tokens_per_chunk"],
                    weights: Float[Array, " tokens_per_chunk"],
                ) -> Float[Array, "tokens_per_chunk channels"]:
                    inputs = flattened_inputs.at[indices].get(mode="fill", fill_value=0.0)
                    return vmap(expert.call_unbatched)(inputs) * weights[:, None]

                expert_outputs = vmap(run_expert)(self.experts, token_indices_for_chunk, weights_for_chunk)
                return accumulator.at[token_indices_for_chunk].add(
                    expert_outputs,
                    mode="drop",
                )

            return (
                jax.lax.cond(
                    jnp.any(token_indices_for_chunk != _SENTINEL),
                    inner,
                    lambda: accumulator,
                ),
                None,
            )

        result, _ = jax.lax.scan(loop_iteration, jnp.zeros_like(flattened_inputs), chunked_token_indices)
        return rearrange(
            result,
            "(batch suffix_tokens) channels -> batch suffix_tokens channels",
            batch=batch_size,
        )

    def export_weights(
        self,
    ) -> ParameterTree[Array]:
        return {
            "router": self.router.export_weights(),
            "experts": self.experts.export_weights(),
        }

    def import_weights(self, weights: ParameterTree[Array]) -> Self:
        assert isinstance(weights, Mapping)
        return replace(
            self,
            router=self.router.import_weights(require_tree(weights["router"])),
            experts=self.experts.import_weights(require_tree(weights["experts"])),
        )


MLPConfig = DenseMLPConfig | MixtureOfExpertsConfig


register_config_union(MLPConfig)
