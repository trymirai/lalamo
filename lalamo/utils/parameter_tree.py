from collections.abc import Mapping, Sequence

from jaxtyping import ArrayLike

type ParameterTree[ArrayType: ArrayLike] = (
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | Sequence[ArrayType | ParameterTree[ArrayType]]
)
