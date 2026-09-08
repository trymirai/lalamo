"""Concrete routing interventions (see `RoutingIntervention` in mlp.py for the interface)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.utils.sharding import with_sharding

from .mlp import RoutingFunction, RoutingIntervention, RoutingMap, SoftmaxRouting
from .token_mixer import StateLayerBase

__all__ = [
    "CacheConditionalRouting",
    "ExpertCacheState",
]


def _like(array: Array, reference: Array) -> Array:
    """`array` resharded like `reference` (same rank).

    State-derived and logit-derived values are combined element-wise, and under explicit sharding both sides
    of a select must agree. Outside a mesh (plain single-device arrays) there is nothing to align.
    """
    try:
        sharding = reference.sharding
    except AttributeError:
        sharding = jax.typeof(reference).sharding
    if not isinstance(sharding, NamedSharding) or sharding.mesh.empty:
        return array
    return with_sharding(array, sharding)


class ExpertCacheState(StateLayerBase):
    """Per-sequence LRU cache of one MoE layer plus the running logit-range statistic of Cache-Conditional Experts.

    `last_used[e]` is the recency stamp of expert e (-1 = never used); the cache holds the `capacity` experts
    with the largest stamps. `range_sum / range_count` is the running mean of `max(z) - min(z)` over the tokens
    seen so far, the per-layer normaliser of the bias.
    """

    last_used: Int[Array, "*batch experts"]
    clock: Int[Array, "*batch"]
    range_sum: Float[Array, "*batch"]
    range_count: Int[Array, "*batch"]


@dataclass(frozen=True)
class CacheConditionalRouting(RoutingIntervention):
    """Cache-Prior re-ranking of Mixture of Cache-Conditional Experts (Skliar et al., arXiv:2412.00099, Eq. 9-10).

    Router logits of the experts resident in a per-layer LRU cache (and of the `forced_top` highest-ranked
    experts, so they are never displaced) are boosted by `bias * logit_range`, the top-k is taken over the
    boosted logits, and the mixing weights are the softmax of the ORIGINAL logits over the selected set. The
    cache is updated with the experts actually dispatched on every token, whether the bias was active on it
    or not: it mirrors what is resident on the device.

    `logit_range` is the paper's Delta_avg. None means the running mean of the per-token logit range over the
    tokens of the sequence seen so far (the paper's running estimate, kept per sequence here); a number pins
    it, e.g. from a calibration pass. Within one token the paper evicts the experts with the higher router
    weights first; the recency stamps encode that order.
    """

    bias: float
    forced_top: int = 2
    cache_capacity: int | None = None
    logit_range: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.bias <= 1.0:
            raise ValueError(f"bias (lambda) must be in [0, 1], got {self.bias}")
        if self.forced_top < 0:
            raise ValueError(f"forced_top (J) must be non-negative, got {self.forced_top}")
        if self.cache_capacity is not None and self.cache_capacity < 1:
            raise ValueError(f"cache_capacity must be positive, got {self.cache_capacity}")
        if self.logit_range is not None and self.logit_range <= 0.0:
            raise ValueError(f"logit_range must be positive, got {self.logit_range}")

    def capacity(self, num_experts: int) -> int:
        if self.cache_capacity is None:
            return max(num_experts // 2, 1)
        if self.cache_capacity > num_experts:
            raise ValueError(f"cache_capacity {self.cache_capacity} exceeds the {num_experts} experts of the layer")
        return self.cache_capacity

    def init_state(self, num_experts: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return ExpertCacheState(
            last_used=jnp.full((num_experts,), -1, dtype=jnp.int32),
            clock=jnp.zeros((), dtype=jnp.int32),
            range_sum=jnp.zeros((), dtype=jnp.float32),
            range_count=jnp.zeros((), dtype=jnp.int32),
        )

    def route(
        self,
        router_logits: Float[Array, "batch experts"],
        active: Bool[Array, " batch"],
        state: StateLayerBase | None,
        routing_function: RoutingFunction,
        num_active: int,
    ) -> tuple[RoutingMap, StateLayerBase | None]:
        if not isinstance(state, ExpertCacheState):
            raise TypeError(f"CacheConditionalRouting needs an ExpertCacheState, got {type(state).__name__}")
        if not isinstance(routing_function, SoftmaxRouting):
            # The mixing weights below are the softmax of the original logits over the selected set, which is
            # what SoftmaxRouting computes; another routing function would need its own weight rule.
            raise TypeError(f"CacheConditionalRouting assumes SoftmaxRouting, got {type(routing_function).__name__}")
        batch_size, num_experts = router_logits.shape
        base = routing_function(router_logits, num_active)

        logits = router_logits.astype(jnp.float32)
        token_range = logits.max(axis=-1) - logits.min(axis=-1)
        if self.logit_range is None:
            has_history = _like(state.range_count > 0, token_range)
            running_mean = _like(state.range_sum / jnp.maximum(state.range_count, 1).astype(jnp.float32), token_range)
            delta = jnp.where(has_history, running_mean, token_range)
        else:
            delta = jnp.full((batch_size,), self.logit_range, dtype=jnp.float32)

        cached = _like(self._cache_mask(state.last_used, num_experts), logits)
        forced = _rank(logits) < self.forced_top
        boosted = logits + self.bias * delta[:, None] * (cached | forced).astype(jnp.float32)
        _, selected = jax.lax.top_k(boosted, num_active)
        # Selection by the boosted logits, weights by the original ones (paper: z' is used only for re-ranking).
        # The gather is a one-hot contraction (exact: all but one term are zero) in the dtype of the logits, so
        # the weights are those SoftmaxRouting would have produced -- the MoE output keeps the model's dtype.
        one_hot = _one_hot(selected, num_experts).astype(router_logits.dtype)
        chosen = (one_hot * router_logits[:, None, :]).sum(axis=-1)
        weights = jax.nn.softmax(chosen)
        intervened = RoutingMap(
            active_expert_indices=_like(selected, base.active_expert_indices),
            active_expert_weights=_like(weights, base.active_expert_weights),
        )
        routing = RoutingMap(
            active_expert_indices=jnp.where(
                active[:, None], intervened.active_expert_indices, base.active_expert_indices
            ),
            active_expert_weights=jnp.where(
                active[:, None], intervened.active_expert_weights, base.active_expert_weights
            ),
        )
        return routing, self._updated_state(state, routing, token_range, num_active)

    def _cache_mask(self, last_used: Int[Array, "batch experts"], num_experts: int) -> Bool[Array, "batch experts"]:
        # The `capacity` most recently used experts; unused ones (stamp -1) are never resident.
        return (_rank(last_used) < self.capacity(num_experts)) & (last_used >= 0)

    @staticmethod
    def _updated_state(
        state: ExpertCacheState,
        routing: RoutingMap,
        token_range: Float[Array, " batch"],
        num_active: int,
    ) -> ExpertCacheState:
        # Recency stamps of this token: the k dispatched experts get distinct stamps within the step, the one
        # with the highest weight the smallest, so that it is evicted first (the paper's imposed LRU order).
        num_experts = state.last_used.shape[-1]
        clock = _like(state.clock, routing.active_expert_weights[:, 0])
        stamps = clock[:, None] * num_active + _rank(routing.active_expert_weights)
        dispatched = _one_hot(routing.active_expert_indices, num_experts)  # [batch, active, experts]
        stamp_per_expert = (dispatched.astype(stamps.dtype) * stamps[:, :, None]).sum(axis=1).astype(jnp.int32)
        was_dispatched = dispatched.any(axis=1)
        return ExpertCacheState(
            last_used=jnp.where(
                _like(was_dispatched, state.last_used),
                _like(stamp_per_expert, state.last_used),
                state.last_used,
            ),
            clock=state.clock + 1,
            range_sum=state.range_sum + _like(token_range, state.range_sum),
            range_count=state.range_count + 1,
        )


def _rank(values: Array) -> Int[Array, "batch items"]:
    """Descending rank of every item within its row (0 = largest), by a double argsort."""
    return jnp.argsort(jnp.argsort(-values, axis=-1), axis=-1)


def _one_hot(indices: Int[Array, "batch active"], num_experts: int) -> Bool[Array, "batch active experts"]:
    return indices[:, :, None] == jnp.arange(num_experts)[None, None, :]
