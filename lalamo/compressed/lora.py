from dataclasses import dataclass

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, Int

from lalamo.initializer import Initializer
from lalamo.module import Keychain
from lalamo.weight_matrix import EmbeddingMatrix, Layout, MatmulConfig, Preconditioner, WeightMatrixSpec

__all__ = [
    "LoRAMatrix",
    "LoRASpec",
]


@dataclass(frozen=True)
class LoRASpec(WeightMatrixSpec):
    rank: int
    layout: Layout = Layout.OUTPUT_INPUT

    def compress(
        self,
        weights: Float[Array | ShapeDtypeStruct, "... out_channels in_channels"],
        preconditioner: Preconditioner | None = None,
    ) -> "LoRAMatrix":
        raise NotImplementedError("LoRA does not support compression from full-precision weights")

    def init(
        self,
        initializer: Initializer,
        leading_dims: tuple[int, ...],
        output_dim: int,
        input_dim: int,
    ) -> "LoRAMatrix":
        if self.layout == Layout.INPUT_OUTPUT:
            down_shape = (*leading_dims, input_dim, self.rank)
            up_shape = (*leading_dims, self.rank, output_dim)
        else:
            down_shape = (*leading_dims, output_dim, self.rank)
            up_shape = (*leading_dims, self.rank, input_dim)
        return LoRAMatrix(
            spec=self,
            down=initializer.zeros(down_shape),
            up=initializer.zeros(up_shape),
        )


class LoRAMatrix(EmbeddingMatrix[LoRASpec]):
    down: Float[Array, "..."]
    up: Float[Array, "..."]

    @property
    def shape(self) -> tuple[int, ...]:
        if self.spec.layout == Layout.INPUT_OUTPUT:
            *leading_dims, input_dim, _rank = self.down.shape
            *_, output_dim = self.up.shape
            return (*leading_dims, input_dim, output_dim)
        *leading_dims, output_dim, _rank = self.down.shape
        *_, input_dim = self.up.shape
        return (*leading_dims, output_dim, input_dim)

    @property
    def dtype(self) -> DTypeLike:
        return jnp.result_type(self.down.dtype, self.up.dtype)

    def astype(self, dtype: DTypeLike) -> "LoRAMatrix":
        return LoRAMatrix(spec=self.spec, down=self.down.astype(dtype), up=self.up.astype(dtype))

    def decompress(self) -> Float[Array, "..."]:
        return self.down @ self.up

    def lookup_embedding(
        self,
        index: int | Int[Array, ""],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            return self.down @ self.up[:, index]
        return self.down[index, :] @ self.up

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        keychain: Keychain,  # noqa: ARG002
        forward_pass_config: MatmulConfig = MatmulConfig(),  # noqa: ARG002
    ) -> Float[Array, "... out_channels"]:
        self._raise_if_batched()
        if self.spec.layout == Layout.INPUT_OUTPUT:
            return (vector @ self.down) @ self.up
        return self.down @ (self.up @ vector)
