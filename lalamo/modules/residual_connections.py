"""Multi-stream residual connections.

Hyper-Connections (Zhu et al., arXiv 2409.19606) keep ``streams`` copies of the residual and
learn, per sublayer, how to read the sublayer input (``A_m``), mix the residual streams (``A_r``)
and write the sublayer output back (``B``). Manifold-Constrained Hyper-Connections (Xie et al.,
arXiv 2512.24880; numerics follow the DeepSeek-V4 reference) make all three maps dynamic and
project the residual mixing matrix onto doubly stochastic matrices with Sinkhorn-Knopp.
Both return a :class:`Routing` consumed by the common :class:`Merge`.

Attention Residuals (Kimi Team) replace the residual sum with a learned softmax
over depth: a per-site query attends over RMS-normalised earlier contributions.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoConfig, LalamoModule
from lalamo.utils.dummy_array import supports_dummy_arrays

from .flow import Context, EmitConfig, Op, SequenceConfig, SinkConfig, StepConfig
from .normalization import Normalization, NormalizationConfig

__all__ = [
    "AttentionResidual",
    "AttentionResidualConfig",
    "HyperConnection",
    "HyperConnectionConfig",
    "HyperConnectionDynamics",
    "HyperConnectionDynamicsConfig",
    "ManifoldHyperConnection",
    "ManifoldHyperConnectionConfig",
    "Merge",
    "Routing",
    "RoutingInputs",
    "routed",
]


class Routing(NamedTuple):
    """A multi-stream residual between a route step and its `Merge`."""

    inputs: Float[Array, "batch tokens channels"]
    residual: Float[Array, "batch tokens wide_channels"]
    output_weights: Float[Array, "batch tokens streams"]


@dataclass(frozen=True)
class RoutingInputs(Op):
    def forward(self, x: Routing, *operands: Array, ctx: Context, keychain: Keychain) -> Array:  # noqa: ARG002
        () = operands
        return x.inputs


@dataclass(frozen=True)
class Merge(Op):
    def forward(self, x: Routing, *operands: Array, ctx: Context, keychain: Keychain) -> Array:  # noqa: ARG002
        (contribution,) = operands
        update = x.output_weights[..., None] * contribution.astype(x.output_weights.dtype)[..., None, :]
        return (x.residual + update.reshape(x.residual.shape)).astype(contribution.dtype)


def routed(route: StepConfig, *body: StepConfig) -> SequenceConfig:
    """Hyper-connection style sublayer: route the streams, run `body` on the routed input, merge it back."""
    branch = EmitConfig("branch", SequenceConfig((RoutingInputs(), *body)))
    return SequenceConfig((route, branch, SinkConfig((branch,), Merge())))


# The coefficient maps are tiny but precision-critical, so they always run in float32.
_MAP_PRECISION = jax.lax.Precision.HIGHEST


@supports_dummy_arrays()
def _scaled(array: Float[Array, "*shape"], factor: float) -> Float[Array, "*shape"]:
    return array * factor


@supports_dummy_arrays()
def _canonical_static_alpha(zeros: Float[Array, "streams mixes"], read_stream: int) -> Float[Array, "streams mixes"]:
    streams, _ = zeros.shape
    return zeros.at[read_stream, 0].set(1).at[:, 1:].add(jnp.eye(streams, dtype=zeros.dtype))


def _apply_to_vectors(
    norm: Normalization,
    inputs: Float[Array, "*axes channels"],
) -> Float[Array, "*axes channels"]:
    apply: Callable[[Array], Array] = norm
    for _ in inputs.shape[:-1]:
        apply = jax.vmap(apply)
    return apply(inputs)


@dataclass(frozen=True)
class HyperConnectionDynamicsConfig(LalamoConfig):
    norm_config: NormalizationConfig

    def init(self, initializer: Initializer, model_dim: int, streams: int) -> "HyperConnectionDynamics":
        # Paper init: zero projections and 0.01 scales, so dynamic HC starts as its static part.
        scale = initializer.ones((), dtype=jnp.float32)
        return HyperConnectionDynamics(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=self.norm_config.init(initializer, model_dim),
            alpha_projection=initializer.zeros((streams + 1, model_dim), dtype=jnp.float32),
            beta_projection=initializer.zeros((model_dim,), dtype=jnp.float32),
            alpha_scale=_scaled(scale, 0.01),
            beta_scale=_scaled(scale, 0.01),
        )


class HyperConnectionDynamics(LalamoModule[HyperConnectionDynamicsConfig]):
    norm: Normalization
    alpha_projection: Float[Array, "mixes channels"]
    beta_projection: Float[Array, " channels"]
    alpha_scale: Float[Array, ""]
    beta_scale: Float[Array, ""]

    def __call__(
        self,
        hidden: Float[Array, "*tokens streams channels"],
    ) -> tuple[Float[Array, "*tokens streams mixes"], Float[Array, "*tokens streams"]]:
        normalized = _apply_to_vectors(self.norm, hidden).astype(jnp.float32)
        alpha_logits = jnp.einsum(
            "...sc,mc->...sm", normalized, self.alpha_projection.astype(jnp.float32), precision=_MAP_PRECISION
        )
        beta_logits = jnp.einsum(
            "...sc,c->...s", normalized, self.beta_projection.astype(jnp.float32), precision=_MAP_PRECISION
        )
        return (
            self.alpha_scale.astype(jnp.float32) * jnp.tanh(alpha_logits),
            self.beta_scale.astype(jnp.float32) * jnp.tanh(beta_logits),
        )


@dataclass(frozen=True)
class HyperConnectionConfig(StepConfig):
    """Original Hyper-Connection; ``dynamics=None`` is static HC (SHC), otherwise dynamic HC (DHC).

    ``read_stream`` only affects initialization: the paper starts ``A_m`` as the one-hot
    ``e_{layer % streams}``, which needs a different stream per layer.
    """

    streams: int
    read_stream: int
    dynamics: HyperConnectionDynamicsConfig | None

    def __post_init__(self) -> None:
        if not 0 <= self.read_stream < self.streams:
            raise ValueError(f"read_stream must be in [0, {self.streams}), got {self.read_stream}.")

    def init(self, initializer: Initializer, model_dim: int) -> "HyperConnection":
        if self.dynamics is not None:
            dynamics = self.dynamics.init(initializer, model_dim, self.streams)
        else:
            dynamics = None
        return HyperConnection(
            config=self,
            sharding_config=initializer.sharding_config,
            static_alpha=_canonical_static_alpha(
                initializer.zeros((self.streams, self.streams + 1), dtype=jnp.float32),
                self.read_stream,
            ),
            static_beta=initializer.ones((self.streams,), dtype=jnp.float32),
            dynamics=dynamics,
        )


class HyperConnection(LalamoModule[HyperConnectionConfig]):
    # Column 0 of alpha is A_m (reads the sublayer input), columns 1: are A_r (residual mixing),
    # indexed as [source stream, destination].
    static_alpha: Float[Array, "streams mixes"]
    static_beta: Float[Array, " streams"]
    dynamics: HyperConnectionDynamics | None

    def forward(
        self,
        x: Float[Array, "batch tokens wide_channels"],
        *operands: Array,
        ctx: Context,  # noqa: ARG002
        keychain: Keychain,  # noqa: ARG002
    ) -> Routing:
        () = operands
        return self.route(x)

    def route(
        self,
        streams: Float[Array, "batch tokens wide_channels"],
    ) -> Routing:
        *leading_axes, _ = streams.shape
        hidden = streams.reshape(*leading_axes, self.config.streams, -1)
        alpha = self.static_alpha.astype(jnp.float32)
        beta = jnp.broadcast_to(self.static_beta.astype(jnp.float32), (*leading_axes, self.config.streams))
        if self.dynamics is not None:
            dynamic_alpha, dynamic_beta = self.dynamics(hidden)
            alpha = alpha + dynamic_alpha
            beta = beta + dynamic_beta
        mixed = jnp.einsum("...sm,...sc->...mc", alpha.astype(streams.dtype), hidden)
        return Routing(
            inputs=mixed[..., 0, :],
            residual=mixed[..., 1:, :].reshape(streams.shape),
            output_weights=beta.astype(streams.dtype),
        )


@dataclass(frozen=True)
class ManifoldHyperConnectionConfig(StepConfig):
    """Dynamic mHC with the DeepSeek-V4 numerics.

    ``epsilon`` is the reference ``hc_eps``: it keeps the pre gate strictly positive and
    regularises every Sinkhorn normalisation. ``norm_epsilon`` belongs to the unweighted RMSNorm
    over all streams.
    """

    streams: int
    norm_epsilon: float = 1e-6
    epsilon: float = 1e-6
    sinkhorn_iterations: int = 20

    def __post_init__(self) -> None:
        if self.sinkhorn_iterations < 1:
            raise ValueError(f"sinkhorn_iterations must be positive, got {self.sinkhorn_iterations}.")

    def init(self, initializer: Initializer, model_dim: int) -> "ManifoldHyperConnection":
        # Initial gating scales follow the paper (0.01); biases start at zero, i.e. a uniform
        # doubly stochastic residual mix and equal pre/post gates.
        width = self.streams * model_dim
        projection_std = 1 / width**0.5
        scale = initializer.ones((), dtype=jnp.float32)
        return ManifoldHyperConnection(
            config=self,
            sharding_config=initializer.sharding_config,
            pre_projection=initializer.normal(projection_std, (self.streams, width), dtype=jnp.float32),
            post_projection=initializer.normal(projection_std, (self.streams, width), dtype=jnp.float32),
            residual_projection=initializer.normal(
                projection_std, (self.streams, self.streams, width), dtype=jnp.float32
            ),
            pre_bias=initializer.zeros((self.streams,), dtype=jnp.float32),
            post_bias=initializer.zeros((self.streams,), dtype=jnp.float32),
            residual_bias=initializer.zeros((self.streams, self.streams), dtype=jnp.float32),
            pre_scale=_scaled(scale, 0.01),
            post_scale=_scaled(scale, 0.01),
            residual_scale=_scaled(scale, 0.01),
        )


class ManifoldHyperConnection(LalamoModule[ManifoldHyperConnectionConfig]):
    pre_projection: Float[Array, "streams wide_channels"]
    post_projection: Float[Array, "streams wide_channels"]
    # Residual mixing is indexed as [source stream, destination stream].
    residual_projection: Float[Array, "streams streams wide_channels"]
    pre_bias: Float[Array, " streams"]
    post_bias: Float[Array, " streams"]
    residual_bias: Float[Array, "streams streams"]
    pre_scale: Float[Array, ""]
    post_scale: Float[Array, ""]
    residual_scale: Float[Array, ""]

    def _sinkhorn(self, logits: Float[Array, "*tokens streams streams"]) -> Float[Array, "*tokens streams streams"]:
        epsilon = self.config.epsilon

        def normalize_columns(
            matrix: Float[Array, "*tokens streams streams"],
        ) -> Float[Array, "*tokens streams streams"]:
            return matrix / (matrix.sum(axis=-2, keepdims=True) + epsilon)

        def normalize_rows(matrix: Float[Array, "*tokens streams streams"]) -> Float[Array, "*tokens streams streams"]:
            return matrix / (matrix.sum(axis=-1, keepdims=True) + epsilon)

        # The row softmax is the first row normalisation of the reference, hence one fewer loop.
        initial = normalize_columns(jax.nn.softmax(logits, axis=-1) + epsilon)
        return jax.lax.fori_loop(
            0,
            self.config.sinkhorn_iterations - 1,
            lambda _, matrix: normalize_columns(normalize_rows(matrix)),
            initial,
        )

    def forward(
        self,
        x: Float[Array, "batch tokens wide_channels"],
        *operands: Array,
        ctx: Context,  # noqa: ARG002
        keychain: Keychain,  # noqa: ARG002
    ) -> Routing:
        () = operands
        return self.route(x)

    def route(
        self,
        streams: Float[Array, "batch tokens wide_channels"],
    ) -> Routing:
        *leading_axes, _ = streams.shape
        hidden = streams.reshape(*leading_axes, self.config.streams, -1)
        flat = streams.astype(jnp.float32)
        normalized = flat * jax.lax.rsqrt(
            jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + self.config.norm_epsilon
        )

        def gate_logits(
            subscripts: str,
            projection: Float[Array, "*outputs wide_channels"],
            scale: Float[Array, ""],
            bias: Float[Array, "*outputs"],
        ) -> Float[Array, "..."]:
            projected = jnp.einsum(subscripts, normalized, projection.astype(jnp.float32), precision=_MAP_PRECISION)
            return projected * scale.astype(jnp.float32) + bias.astype(jnp.float32)

        pre = (
            jax.nn.sigmoid(gate_logits("...w,sw->...s", self.pre_projection, self.pre_scale, self.pre_bias))
            + self.config.epsilon
        )
        post = 2 * jax.nn.sigmoid(gate_logits("...w,sw->...s", self.post_projection, self.post_scale, self.post_bias))
        mixing = self._sinkhorn(
            gate_logits("...w,stw->...st", self.residual_projection, self.residual_scale, self.residual_bias)
        )

        inputs = jnp.einsum("...s,...sc->...c", pre, hidden.astype(jnp.float32), precision=_MAP_PRECISION)
        residual = jnp.einsum("...st,...sc->...tc", mixing, hidden.astype(jnp.float32), precision=_MAP_PRECISION)
        return Routing(
            inputs=inputs.astype(streams.dtype),
            residual=residual.reshape(streams.shape),
            output_weights=post,
        )


@dataclass(frozen=True)
class AttentionResidualConfig(StepConfig):
    key_norm_config: NormalizationConfig
    recency_bias_init: float | None = None

    def init(self, initializer: Initializer, model_dim: int) -> "AttentionResidual":
        # Zero query (paper init): the reader starts as a uniform average over its sources.
        return AttentionResidual(
            config=self,
            sharding_config=initializer.sharding_config,
            query=initializer.zeros((model_dim,), dtype=jnp.float32),
            key_norm=self.key_norm_config.init(initializer, model_dim),
            recency_bias=None
            if self.recency_bias_init is None
            else _scaled(initializer.ones((), dtype=jnp.float32), self.recency_bias_init),
        )


class AttentionResidual(LalamoModule[AttentionResidualConfig]):
    query: Float[Array, " channels"]
    key_norm: Normalization
    recency_bias: Float[Array, ""] | None

    def forward(
        self,
        x: Float[Array, "batch tokens channels"],
        *operands: Float[Array, "batch tokens channels"],
        ctx: Context,  # noqa: ARG002
        keychain: Keychain,  # noqa: ARG002
    ) -> Float[Array, "batch tokens channels"]:
        return self(x, *operands)

    def __call__(
        self,
        current: Float[Array, "*axes channels"],
        *previous: Float[Array, "*axes channels"],
    ) -> Float[Array, "*axes channels"]:
        values = jnp.stack((*previous, current))
        keys = _apply_to_vectors(self.key_norm, values).astype(jnp.float32)
        logits = jnp.einsum("d...c,c->d...", keys, self.query.astype(jnp.float32), precision=_MAP_PRECISION)
        if self.recency_bias is not None:
            logits = logits.at[-1].add(self.recency_bias.astype(jnp.float32))
        weights = jax.nn.softmax(logits, axis=0)
        mixed = jnp.einsum("d...,d...c->...c", weights, values.astype(jnp.float32), precision=_MAP_PRECISION)
        return mixed.astype(current.dtype)
