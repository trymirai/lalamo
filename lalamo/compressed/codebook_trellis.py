from dataclasses import dataclass

import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float, Key, UInt8

from lalamo.initializer import RandomInitializer
from lalamo.module import field
from lalamo.preconditioner import Preconditioner
from lalamo.utils.dummy_array import supports_dummy_arrays
from lalamo.utils.sharding import ShardingConfig, sharding_of
from lalamo.weight_matrix import CompressionImplementation

from .qtip_gaussian import QtipGaussianMatrix, QtipGaussianSpec


@dataclass(frozen=True)
class CodebookTrellisSpec(QtipGaussianSpec):
    @supports_dummy_arrays()
    def compress(
        self,
        weights: Float[Array, "out_channels in_channels"],
        *,
        key: Key[Array, ""] | None = None,
        preconditioner: Preconditioner | None = None,
        implementation: CompressionImplementation = CompressionImplementation.INFERENCE,
        sharding_config: ShardingConfig,
        is_sharded: bool = True,
    ) -> "CodebookTrellisMatrix":
        template = super().compress(
            weights,
            key=key,
            preconditioner=preconditioner,
            implementation=implementation,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
        )
        initializer = RandomInitializer(weights.dtype, sharding_config)
        return CodebookTrellisMatrix(
            spec=self,
            sharding_config=sharding_config,
            is_sharded=is_sharded,
            dtype_=template.dtype,
            columns=template.columns,
            codes=template.codes,
            scales=template.scales,
            table=initializer.zeros((65536, self.vector_width), (None, None), jnp.float32),
        )


class CodebookTrellisMatrix(QtipGaussianMatrix):
    table: Float[Array, "65536 width"] = field(trainable=False)

    def __check_init__(self) -> None:
        assert self.table.shape == (65536, self.spec.vector_width)
        assert self.table.dtype == jnp.float32

    def decode_rows(self, codes: UInt8[Array, "*rows bytes"], scales: Float[Array, "*rows"]) -> Array:
        row_axes = tuple(sharding_of(codes).spec)[:-1]
        values = self.table.at[self.spec.states(codes, self.columns)].get(
            out_sharding=PartitionSpec(*row_axes, None, None)
        )
        values = values.reshape(*codes.shape[:-1], self.columns)
        return (values * scales[..., None].astype(jnp.float32)).astype(self.dtype)
