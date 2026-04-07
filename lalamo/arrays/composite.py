import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from .base import ArrayForwardPassConfig, CompressedArray


class MixtureArray(CompressedArray):
    parts: tuple[CompressedArray, ...]
    coefficients: tuple[float, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        assert len(self.parts) != 0, "MixtureArray must be non-empty"
        first, *rest = self.parts
        result = first.shape
        for part in rest:
            if part.shape != result:
                raise ValueError(
                    f"MixtureArray parts have mismatched shapes: {result} vs {part.shape}",
                )
        return result

    @property
    def dtype(self) -> DTypeLike:
        assert len(self.parts) != 0, "MixtureArray must be non-empty"
        return jnp.result_type(*(part.dtype for part in self.parts))

    def materialize(self) -> Float[Array, "... out_channels in_channels"]:
        assert len(self.parts) != 0, "MixtureArray must be non-empty to materialize"
        return sum(  # type: ignore[return-value]
            coeff * part.materialize() for coeff, part in zip(self.coefficients, self.parts, strict=False)
        )

    def dot(
        self,
        vector: Float[Array, " in_channels"],
        *,
        key: PRNGKeyArray | None,
        forward_pass_config: ArrayForwardPassConfig = ArrayForwardPassConfig(),  # noqa: B008
    ) -> Float[Array, "... out_channels"]:
        assert len(self.parts) != 0, "MixtureArray must be non-empty to dot"
        return sum(  # type: ignore[return-value]
            coeff * part.dot(vector, key=key, forward_pass_config=forward_pass_config)
            for coeff, part in zip(self.coefficients, self.parts, strict=False)
        )

    def add_part(self, part: CompressedArray, coefficient: float = 1.0) -> "MixtureArray":
        return MixtureArray(
            parts=(*self.parts, part),
            coefficients=(*self.coefficients, coefficient),
        )
