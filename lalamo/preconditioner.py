import math
from pathlib import Path
from typing import Self

import equinox as eqx
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxtyping import Array, Float, PyTree

__all__ = [
    "Preconditioner",
    "PreconditionerDict",
]


type PytreePath = tuple[str, ...]


@eqx.filter_jit
def _sym_to_tril(sym_matrix: Float[Array, "*components n n"]) -> Float[Array, "*components n*(n+1)//2"]:
    *_, num_rows, _ = sym_matrix.shape
    tril_rows, tril_cols = jnp.tril_indices(num_rows)
    return sym_matrix[..., tril_rows, tril_cols]


@eqx.filter_jit
def _tril_to_sym(tril_vector: Float[Array, "*components tril"]) -> Float[Array, "*components n n"]:
    *leading_dims, numel = tril_vector.shape
    num_rows = int(math.sqrt(8 * numel + 1) - 1) // 2
    result = jnp.empty((*leading_dims, num_rows, num_rows), dtype=tril_vector.dtype)
    tril_rows, tril_cols = jnp.tril_indices(num_rows)
    result = result.at[..., tril_rows, tril_cols].set(tril_vector)
    return result.at[..., tril_cols, tril_rows].set(tril_vector)


@eqx.filter_jit
def _tril_trace(tril_vector: Float[Array, "*components tril"]) -> Float[Array, "*components"]:
    *_, numel = tril_vector.shape
    num_rows = int(math.sqrt(8 * numel + 1) - 1) // 2
    diagonal_indices = jnp.array([row * (row + 3) // 2 for row in range(num_rows)])
    return jnp.sum(tril_vector[..., diagonal_indices], axis=-1)


class Preconditioner(eqx.Module):
    input_block_tril: Float[Array, "*components input_tril"] | None
    output_block_tril: Float[Array, "*components output_tril"] | None

    @property
    def input_block(self) -> Float[Array, "*components input_channels input_channels"] | None:
        if self.input_block_tril is None:
            return None
        return _tril_to_sym(self.input_block_tril)

    @property
    def output_block(
        self,
    ) -> Float[Array, "*components output_channels output_channels"] | None:
        if self.output_block_tril is None:
            return None
        return _tril_to_sym(self.output_block_tril)

    @property
    def magnitude(self) -> Float[Array, ""]:
        result = 1.0
        if self.input_block_tril is not None:
            result *= _tril_trace(self.input_block_tril)
        if self.output_block_tril is not None:
            result *= _tril_trace(self.output_block_tril)
        return jnp.mean(result)

    @classmethod
    def init(
        cls,
        input_block: Float[Array, "*components input_channels input_channels"] | None = None,
        output_block: Float[Array, "*components output_channels output_channels"] | None = None,
    ) -> Self:
        input_block_tril = None
        if input_block is not None:
            input_block_tril = _sym_to_tril(input_block)

        output_block_tril = None
        if output_block is not None:
            output_block_tril = _sym_to_tril(output_block)

        return cls(
            input_block_tril=input_block_tril,
            output_block_tril=output_block_tril,
        )

    @classmethod
    def identity(cls) -> Self:
        return cls(
            input_block_tril=None,
            output_block_tril=None,
        )


class PreconditionerDict(dict[PytreePath, Preconditioner]):
    @classmethod
    def init_for_model(cls, model: PyTree) -> Self:
        from lalamo.utils.surgery import select_nodes_of_type  # noqa: PLC0415
        from lalamo.weight_matrix import WeightMatrix  # noqa: PLC0415

        return cls(
            {
                path: Preconditioner.identity()
                for path, _matrix in select_nodes_of_type(WeightMatrix, model)
            },
        )

    def save(self, directory: Path | str) -> None:
        paths = tuple(sorted(self))
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(
            Path(directory),
            tuple(self[path] for path in paths),
            custom_metadata={"paths": paths},
        )
        checkpointer.wait_until_finished()

    @classmethod
    def restore(cls, directory: Path | str) -> Self:
        checkpointer = ocp.StandardCheckpointer()
        paths = checkpointer.metadata(Path(directory)).custom_metadata["paths"]
        restored = checkpointer.restore(Path(directory))
        return cls(
            {
                tuple(path): Preconditioner(
                    input_block_tril=preconditioner["input_block_tril"],
                    output_block_tril=preconditioner["output_block_tril"],
                )
                for path, preconditioner in zip(paths, restored, strict=True)
            },
        )
