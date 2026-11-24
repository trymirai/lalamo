"""Concrete routing interventions (see `RoutingIntervention` in mlp.py for the interface)."""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from lalamo.utils.sharding import with_sharding

from .mlp import RoutingFunction, RoutingIntervention, RoutingMap
from .token_mixer import StateLayerBase

__all__ = [
    "CacheConditionalRouting",
    "CachePriority",
    "ExpertCacheState",
    "ScoredCacheRouting",
    "ScoredCacheState",
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

    Under another routing function the same rule runs in that function's own terms: "logits" are the scores it
    takes its top-k over (`RoutingFunction.gating_scores`: for SigmoidRouting the sigmoid scores, with the
    gating bias when there is one), `logit_range` is the range of those scores, and the mixing weights are the
    ones it gives the selected set (`RoutingFunction.weights_of`).

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

    @property
    def weighs_by_softmax_over_selection(self) -> bool:
        # Paper: z' re-ranks only; the weights are those of the routing function over the selected set. Under
        # SoftmaxRouting that is the softmax of the original logits, which is the case this property names;
        # the MoE checks the routing function before relying on it.
        return True

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
        biases: Float[Array, " experts"] | None = None,
    ) -> tuple[RoutingMap, StateLayerBase | None]:
        if not isinstance(state, ExpertCacheState):
            raise TypeError(f"CacheConditionalRouting needs an ExpertCacheState, got {type(state).__name__}")
        if self.forced_top >= num_active:
            # With J >= k every slot is pinned to the router's own ranking, the cache prior can never
            # change the selection, and the rule silently becomes the base routing -- a spec that looks
            # like an intervention and measures like a control.
            raise ValueError(
                f"forced_top (J) = {self.forced_top} leaves no slot to the cache prior at top-{num_active};"
                " the rule would degrade to the base routing"
            )
        batch_size, num_experts = router_logits.shape
        # The rule works in the scores the routing function ranks experts by: the logits themselves under
        # softmax routing, the (biased) sigmoid scores under sigmoid routing. The names keep the paper's.
        # Asked for first, so that a routing function the rule cannot act on is refused before anything runs.
        logits = routing_function.gating_scores(router_logits, biases)
        base = routing_function(router_logits, biases, num_active=num_active)

        token_range = logits.max(axis=-1) - logits.min(axis=-1)
        if self.logit_range is None:
            has_history = _like(state.range_count > 0, token_range)
            running_mean = _like(state.range_sum / jnp.maximum(state.range_count, 1).astype(jnp.float32), token_range)
            delta = jnp.where(has_history, running_mean, token_range)
        else:
            delta = jnp.full((batch_size,), self.logit_range, dtype=jnp.float32)

        cached = _like(self._cache_mask(state.last_used, num_experts), logits)
        forced = self._forced_mask(logits)
        boosted = logits + self.bias * delta[:, None] * (cached | forced).astype(jnp.float32)
        _, selected = jax.lax.top_k(boosted, num_active)
        # Selection by the boosted scores, weights by the original ones (paper: z' is used only for re-ranking):
        # the weights are those the routing function gives the selected set, in the model's dtype.
        weights = routing_function.weights_of(router_logits, selected, biases)
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
            # The logits the selection was taken over, exactly as top_k saw them: boosted where the rule was
            # active, the plain float32 logits elsewhere. A consumer that needs the selection margins (the
            # traffic surrogate of routing-lab) reads them from here instead of re-deriving the boost.
            effective_logits=jnp.where(active[:, None], boosted, logits),
        )
        return routing, self._updated_state(state, routing, token_range, num_active)

    def _cache_mask(self, last_used: Int[Array, "batch experts"], num_experts: int) -> Bool[Array, "batch experts"]:
        # The `capacity` most recently used experts; unused ones (stamp -1) are never resident. A threshold
        # at the capacity-th largest stamp is exact: stamps of used experts are unique (clock * k + rank),
        # and when fewer than `capacity` experts have been used the threshold is -1, which the second
        # condition removes -- the same set a full ranking selects, at the price of one top_k instead of
        # two sorts per step.
        threshold = jax.lax.top_k(last_used, self.capacity(num_experts))[0][:, -1]
        return (last_used >= threshold[:, None]) & (last_used >= 0)

    def _forced_mask(self, logits: Float[Array, "batch experts"]) -> Bool[Array, "batch experts"]:
        # The J highest-ranked experts by their own logits. Taken as top_k INDICES rather than a value
        # threshold so that an exact tie at the boundary is resolved as a stable descending sort would:
        # the lower index wins in both.
        num_experts = logits.shape[-1]
        if self.forced_top == 0:
            return jnp.zeros(logits.shape, dtype=jnp.bool_)
        return _one_hot(jax.lax.top_k(logits, self.forced_top)[1], num_experts).any(axis=1)

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


class CachePriority(StrEnum):
    """What orders the residents of the cache: the one with the lowest priority leaves first."""

    # Recency of dispatch: the LRU cache of Cache-Prior.
    RECENCY = "recency"
    # How often the router's OWN top-k held the expert lately, whatever was dispatched in the end.
    ROUTER_PREFERENCE = "router_preference"


class ScoredCacheState(StateLayerBase):
    """Per-sequence cache of one MoE layer kept as an explicit set, with what its priorities are computed from.

    `resident[e]` says whether expert e is in the cache. `last_used[e]` is its recency stamp (-1 = never
    dispatched), `preference_fast` and `preference_slow` are exponential moving averages of the indicator
    "the router's own top-k contains e". `range_sum / range_count` is the running mean of the per-token range
    of the gating scores, as in `ExpertCacheState`.
    """

    resident: Bool[Array, "*batch experts"]
    last_used: Int[Array, "*batch experts"]
    preference_fast: Float[Array, "*batch experts"]
    preference_slow: Float[Array, "*batch experts"]
    clock: Int[Array, "*batch"]
    range_sum: Float[Array, "*batch"]
    range_count: Int[Array, "*batch"]


@dataclass(frozen=True)
class ScoredCacheRouting(RoutingIntervention):
    """Cache-Prior re-ranking with a scored cache: four extensions of `CacheConditionalRouting`, each off by default.

    With the defaults the rule is `CacheConditionalRouting`: the same selection, weights and cache on every
    token. The extensions come from two follow-ups of Cache-Prior, restated for a logical cache (no transfer
    timing is modelled, so the router's own scores of the current token stand in for any prediction of them):

    * `priority` (Prefetch-Steered Routing, arXiv:2610.34721, the "A priority" without its token prior): with
      ROUTER_PREFERENCE the resident that leaves first is the one the router itself has ranked in its top-k
      least often lately -- the sum of two exponential moving averages (decay 0.5 and 0.98) of that indicator
      -- instead of the least recently dispatched one. The averages follow the router's choice, not the
      dispatched set, so the bias does not keep an expert resident by dispatching it.
    * `admission_budget` (the steering budget of the same paper): before the selection, up to that many of
      the highest-ranked experts of the router's own top-k that are not resident enter the cache and receive
      the bonus. An admission never evicts an expert of the router's own top-k.
    * `graded_bonus` (CECAR, arXiv:2611.20119, with the eviction order in place of its learned score): the
      bonus of a resident expert is scaled by its place in the eviction order, from 1 / |cache| for the next
      one to leave to 1 for the last one. The `forced_top` experts keep the full bonus.
    * `prune_below_rank` (cache-conditioned pruning of the same paper): a selected expert that is neither
      resident nor forced and whose rank in the boosted order is that number or lower is not dispatched. The
      weights of the dispatched experts are rescaled to the sum the routing function gave the whole selection.
      In the routing map a dropped expert keeps its slot with weight zero and `dispatched` cleared.

    The cache is updated on every token with the experts dispatched on it, whether the rule was active or not.
    When it overflows, the residents with the lowest priority among those not dispatched on the token leave,
    ties going to the lower expert index.

    With `cold_start` the state follows only the tokens the rule is active on: the cache is empty and the
    range statistic has no history when the first active token of a sequence arrives. That is the rule a
    server runs when only its decode steps route through the cache.
    """

    bias: float
    forced_top: int = 2
    cache_capacity: int | None = None
    logit_range: float | None = None
    priority: CachePriority = CachePriority.RECENCY
    admission_budget: int = 0
    graded_bonus: bool = False
    prune_below_rank: int | None = None
    cold_start: bool = False

    FAST_DECAY = 0.5
    SLOW_DECAY = 0.98

    def __post_init__(self) -> None:
        if not 0.0 <= self.bias <= 1.0:
            raise ValueError(f"bias (lambda) must be in [0, 1], got {self.bias}")
        if self.forced_top < 0:
            raise ValueError(f"forced_top (J) must be non-negative, got {self.forced_top}")
        if self.cache_capacity is not None and self.cache_capacity < 1:
            raise ValueError(f"cache_capacity must be positive, got {self.cache_capacity}")
        if self.logit_range is not None and self.logit_range <= 0.0:
            raise ValueError(f"logit_range must be positive, got {self.logit_range}")
        if self.admission_budget < 0:
            raise ValueError(f"admission_budget must be non-negative, got {self.admission_budget}")
        if self.prune_below_rank is not None and self.prune_below_rank < 1:
            raise ValueError(f"prune_below_rank must keep at least the first expert, got {self.prune_below_rank}")

    @property
    def weighs_by_softmax_over_selection(self) -> bool:
        # True of every configuration but pruning: a softmax over the selected logits would give the dropped
        # slots their weight back.
        return self.prune_below_rank is None

    @property
    def may_drop_experts(self) -> bool:
        return self.prune_below_rank is not None

    def capacity(self, num_experts: int) -> int:
        if self.cache_capacity is None:
            return max(num_experts // 2, 1)
        if self.cache_capacity > num_experts:
            raise ValueError(f"cache_capacity {self.cache_capacity} exceeds the {num_experts} experts of the layer")
        return self.cache_capacity

    def init_state(self, num_experts: int, dtype: DTypeLike) -> StateLayerBase | None:  # noqa: ARG002
        return ScoredCacheState(
            resident=jnp.zeros((num_experts,), dtype=jnp.bool_),
            last_used=jnp.full((num_experts,), -1, dtype=jnp.int32),
            preference_fast=jnp.zeros((num_experts,), dtype=jnp.float32),
            preference_slow=jnp.zeros((num_experts,), dtype=jnp.float32),
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
        biases: Float[Array, " experts"] | None = None,
    ) -> tuple[RoutingMap, StateLayerBase | None]:
        if not isinstance(state, ScoredCacheState):
            raise TypeError(f"ScoredCacheRouting needs a ScoredCacheState, got {type(state).__name__}")
        if self.forced_top >= num_active:
            raise ValueError(
                f"forced_top (J) = {self.forced_top} leaves no slot to the cache prior at top-{num_active};"
                " the rule would degrade to the base routing"
            )
        batch_size, num_experts = router_logits.shape
        capacity = self.capacity(num_experts)
        if capacity < num_active:
            raise ValueError(f"a cache of {capacity} experts cannot hold the {num_active} experts of one token")
        # As in CacheConditionalRouting, the rule works in the scores the routing function ranks experts by.
        logits = routing_function.gating_scores(router_logits, biases)
        base = routing_function(router_logits, biases, num_active=num_active)

        token_range = logits.max(axis=-1) - logits.min(axis=-1)
        if self.logit_range is None:
            has_history = _like(state.range_count > 0, token_range)
            running_mean = _like(state.range_sum / jnp.maximum(state.range_count, 1).astype(jnp.float32), token_range)
            delta = jnp.where(has_history, running_mean, token_range)
        else:
            delta = jnp.full((batch_size,), self.logit_range, dtype=jnp.float32)

        resident = _like(state.resident, logits)
        # Priorities of the whole step are those the state held before this token.
        priority = _like(self._priority(state), logits)
        own_ranking = jax.lax.top_k(logits, num_active)[1]
        own_top = _one_hot(own_ranking, num_experts)  # [batch, rank, experts]
        own_set = own_top.any(axis=1)

        if self.admission_budget > 0:
            missing = ~(own_top & resident[:, None, :]).any(axis=-1)  # [batch, rank]
            admitted_ranks = missing & (jnp.cumsum(missing, axis=-1) <= self.admission_budget)
            admitted = (own_top & admitted_ranks[:, :, None]).any(axis=1)
            resident = _inserted(resident, admitted, priority, own_set, capacity)

        # The first J ranks of the router's own top-k, with the tie-breaking of top_k itself.
        forced = own_top[:, : self.forced_top].any(axis=1)
        grade = _retention_grade(resident, priority) if self.graded_bonus else jnp.ones_like(logits)
        prior = jnp.where(forced, 1.0, jnp.where(resident, grade, 0.0))
        boosted = logits + self.bias * delta[:, None] * prior
        _, selected = jax.lax.top_k(boosted, num_active)
        weights = routing_function.weights_of(router_logits, selected, biases)
        if self.prune_below_rank is not None:
            in_cache = (_one_hot(selected, num_experts) & (resident | forced)[:, None, :]).any(axis=-1)
            kept = in_cache | (jnp.arange(num_active) < self.prune_below_rank)[None, :]
            wide = weights.astype(jnp.float32)
            kept_mass = jnp.where(kept, wide, 0.0).sum(axis=-1, keepdims=True)
            rescaled = jnp.where(kept, wide, 0.0) * (wide.sum(axis=-1, keepdims=True) / kept_mass)
            weights = rescaled.astype(weights.dtype)
        else:
            kept = jnp.ones(selected.shape, dtype=jnp.bool_)

        routing = RoutingMap(
            active_expert_indices=jnp.where(
                active[:, None], _like(selected, base.active_expert_indices), base.active_expert_indices
            ),
            active_expert_weights=jnp.where(
                active[:, None], _like(weights, base.active_expert_weights), base.active_expert_weights
            ),
            effective_logits=jnp.where(active[:, None], boosted, logits),
            dispatched=_like(kept, base.active_expert_indices) | ~active[:, None],
        )
        updated = self._updated_state(state, routing, resident, priority, own_set, token_range, capacity)
        if self.cold_start:
            updated = jax.tree.map(
                lambda new, old: jnp.where(
                    _like(jnp.broadcast_to(active.reshape((-1, *([1] * (old.ndim - 1)))), old.shape), old), new, old
                ),
                updated,
                state,
            )
        return routing, updated

    def _priority(self, state: ScoredCacheState) -> Float[Array, "batch experts"]:
        match self.priority:
            case CachePriority.RECENCY:
                return state.last_used.astype(jnp.float32)
            case CachePriority.ROUTER_PREFERENCE:
                return state.preference_fast + state.preference_slow

    def _updated_state(
        self,
        state: ScoredCacheState,
        routing: RoutingMap,
        resident: Bool[Array, "batch experts"],
        priority: Float[Array, "batch experts"],
        own_set: Bool[Array, "batch experts"],
        token_range: Float[Array, " batch"],
        capacity: int,
    ) -> ScoredCacheState:
        num_experts = state.last_used.shape[-1]
        num_active = routing.active_expert_indices.shape[-1]
        # Recency stamps as in CacheConditionalRouting: distinct within the step, the highest weight the
        # smallest. A dropped slot has weight zero, ranks after every dispatched one and stamps nothing.
        clock = _like(state.clock, routing.active_expert_weights[:, 0])
        stamps = clock[:, None] * num_active + _rank(routing.active_expert_weights)
        assert routing.dispatched is not None
        slots = _one_hot(routing.active_expert_indices, num_experts) & routing.dispatched[:, :, None]
        stamp_per_expert = (slots.astype(stamps.dtype) * stamps[:, :, None]).sum(axis=1).astype(jnp.int32)
        dispatched = slots.any(axis=1)
        preferred = own_set.astype(jnp.float32)
        return ScoredCacheState(
            resident=_like(_inserted(resident, dispatched, priority, dispatched, capacity), state.resident),
            last_used=jnp.where(
                _like(dispatched, state.last_used), _like(stamp_per_expert, state.last_used), state.last_used
            ),
            preference_fast=self.FAST_DECAY * state.preference_fast
            + (1.0 - self.FAST_DECAY) * _like(preferred, state.preference_fast),
            preference_slow=self.SLOW_DECAY * state.preference_slow
            + (1.0 - self.SLOW_DECAY) * _like(preferred, state.preference_slow),
            clock=state.clock + 1,
            range_sum=state.range_sum + _like(token_range, state.range_sum),
            range_count=state.range_count + 1,
        )


def _eviction_position(
    priority: Float[Array, "batch experts"], eligible: Bool[Array, "batch experts"]
) -> Int[Array, "batch experts"]:
    """Place of every eligible expert in the eviction order (0 leaves first); the others come after them all.

    The order is ascending priority with ties going to the lower index, which a stable sort gives.
    """
    order = jnp.argsort(jnp.where(eligible, priority, jnp.inf), axis=-1, stable=True)
    return jnp.argsort(order, axis=-1)


def _inserted(
    resident: Bool[Array, "batch experts"],
    incoming: Bool[Array, "batch experts"],
    priority: Float[Array, "batch experts"],
    shielded: Bool[Array, "batch experts"],
    capacity: int,
) -> Bool[Array, "batch experts"]:
    """The cache after `incoming` entered it: the lowest-priority unshielded residents make room."""
    eligible = resident & ~shielded
    overflow = jnp.maximum((resident | incoming).sum(axis=-1) - capacity, 0)
    victims = eligible & (_eviction_position(priority, eligible) < overflow[:, None])
    return (resident & ~victims) | incoming


def _retention_grade(
    resident: Bool[Array, "batch experts"], priority: Float[Array, "batch experts"]
) -> Float[Array, "batch experts"]:
    """Place of a resident in the eviction order as a share of the cache: 1 / |cache| leaves first, 1 last."""
    position = _eviction_position(priority, resident)
    count = jnp.maximum(resident.sum(axis=-1, keepdims=True), 1)
    return (position + 1).astype(jnp.float32) / count.astype(jnp.float32)


def _rank(values: Array) -> Int[Array, "batch items"]:
    """Descending rank of every item within its row (0 = largest), by a double argsort."""
    return jnp.argsort(jnp.argsort(-values, axis=-1), axis=-1)


def _one_hot(indices: Int[Array, "batch active"], num_experts: int) -> Bool[Array, "batch active experts"]:
    return indices[:, :, None] == jnp.arange(num_experts)[None, None, :]
