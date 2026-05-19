from collections.abc import Mapping, Sequence

from jaxtyping import Array

type ParameterTree[ArrayType: Array] = (
    Mapping[str, ArrayType | ParameterTree[ArrayType]] | Sequence[ArrayType | ParameterTree[ArrayType]]
)
