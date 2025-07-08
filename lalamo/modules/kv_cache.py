from jaxtyping import Array, Float

from lalamo.common import ParameterDict

from .common import ExportableModule, WeightLayout

__all__ = ["KVCacheLayerSlice"]


class KVCacheLayerSlice(ExportableModule):
    keys: Float[Array, "tokens groups head_channels"]
    values: Float[Array, "tokens groups head_channels"]

    def export_weights(self, weight_layout: WeightLayout = WeightLayout.AUTO) -> ParameterDict:  # noqa: ARG002
        return ParameterDict(
            keys=self.keys,
            values=self.values,
        )
