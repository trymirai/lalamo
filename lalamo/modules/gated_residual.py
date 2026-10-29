from dataclasses import dataclass, replace

import jax
from jaxtyping import Array, Float

from lalamo.initializer import Initializer
from lalamo.module import Keychain, LalamoModule, LogicalAxis

from .activations import SiLU
from .flow import Context, StepConfig
from .linear import Linear, LinearConfig
from .normalization import Normalization, NormalizationConfig
from .residual_connections import Routing
from .utils import call_vmapped_twice

__all__ = [
    "GatedResidual",
    "GatedResidualConfig",
]


@dataclass(frozen=True)
class GatedResidualConfig(StepConfig):
    streams: int
    rank: int
    norm_config: NormalizationConfig
    linear_config: LinearConfig
    injection_config: LinearConfig | None

    def init(self, initializer: Initializer, model_dim: int) -> "GatedResidual":
        width = self.streams * model_dim
        return GatedResidual(
            config=self,
            sharding_config=initializer.sharding_config,
            norm=replace(self.norm_config, group_size=model_dim).init(initializer, width),
            down=self.linear_config.init(initializer, width, (self.rank,), has_biases=False),
            up=self.linear_config.init(initializer, self.rank, (width,), has_biases=False),
            inject=None
            if self.injection_config is None
            else self.injection_config.init(initializer, width, (self.streams,), has_biases=False),
        )


class GatedResidual(LalamoModule[GatedResidualConfig]):
    norm: Normalization
    down: Linear
    up: Linear
    inject: Linear | None

    def forward(
        self,
        x: Float[Array, "batch tokens wide_channels"],
        *operands: Array,
        ctx: Context,
        keychain: Keychain,
    ) -> Routing | Float[Array, "batch tokens channels"]:
        () = operands
        forward_pass_config = ctx.forward_pass_config.matmul_config
        down_key, up_key, inject_key = keychain.split(3)
        axes = (self.sharding_config.resolve_axis(LogicalAxis.BATCH), None)
        normalized = call_vmapped_twice(self.norm, x)
        (bottleneck,) = call_vmapped_twice(
            self.down, normalized, forward_pass_config=forward_pass_config, keychain=down_key, added_sharding_axes=axes
        )
        (read_gate,) = call_vmapped_twice(
            self.up,
            SiLU()(bottleneck / self.config.streams),
            forward_pass_config=forward_pass_config,
            keychain=up_key,
            added_sharding_axes=axes,
        )
        mixed = (jax.nn.sigmoid(read_gate) * normalized).reshape(*x.shape[:-1], self.config.streams, -1).mean(axis=-2)
        if self.inject is None:
            return mixed
        (write_gate,) = call_vmapped_twice(
            self.inject,
            normalized,
            forward_pass_config=forward_pass_config,
            keychain=inject_key,
            added_sharding_axes=axes,
        )
        return Routing(mixed, x, 2 * jax.nn.sigmoid(write_gate / self.config.streams))
