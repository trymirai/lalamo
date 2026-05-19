import jax.tree_util as jtu

__all__ = [
    "ParameterPath",
]


type PytreeKey = (
    str | int | jtu.GetAttrKey | jtu.SequenceKey | jtu.DictKey | jtu.FlattenedIndexKey | tuple[PytreeKey, ...]
)


class ParameterPath(str):
    """
    ParameterPath is a universal representation, allowing to index into either the pytree, or flattened
    representations of modules. The invariant is as follows:

    For any PyTree and any ParameterPath:
    ```py
    get_by_path(pytree, path) == get_by_path(pytree.to_uzu(), path)

    jax.tree.map_with_path(lambda v, path: v == get_by_path(pytree, ParameterPath(path)), pytree)
    ```

    Basically ParameterPath is a universal str representation for any Path within any PyTree.

    The only constraint is that you can't use dictionary keys with dots inside of them, if you try - we will throw up.
    """

    __slots__ = ()

    def __truediv__(self, other: PytreeKey) -> "ParameterPath":
        match other:
            case str():
                if not self:
                    return ParameterPath(other)
                return ParameterPath(f"{self}.{other}")
            case int():
                return self / str(other)
            case jtu.GetAttrKey(name=name):
                return self / name
            case jtu.SequenceKey(idx=idx):
                return self / str(idx)
            case jtu.DictKey(key=key):
                key_str = str(key)
                if "." in key_str:
                    raise ValueError(
                        f"DictKey {key_str!r} contains dots, which would be ambiguous in a dot-separated path"
                    )
                return self / key_str
            case jtu.FlattenedIndexKey(key=key):
                return self / key
            case tuple():
                result = self
                for element in other:
                    result = result / element
                return result
            case _:
                raise TypeError(f"Unsupported type for ParameterPath /: {type(other).__name__}")
