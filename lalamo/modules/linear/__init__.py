from lalamo.modules.common import register_config_union

from .affine_quantized import AffineQuantizedLinear, AffineQuantizedLinearConfig
from .common import (
    LinearBase,
    LinearConfigBase,
)
from .full_precision import FullPrecisionLinear, FullPrecisionLinearConfig
from .qlora import QLoRALinear, QLoRALinearConfig

__all__ = [
    "AffineQuantizedLinear",
    "AffineQuantizedLinearConfig",
    "FullPrecisionLinear",
    "FullPrecisionLinearConfig",
    "LinearBase",
    "LinearConfigBase",
    "QLoRALinear",
    "QLoRALinearConfig",
]


LinearConfig = FullPrecisionLinearConfig | AffineQuantizedLinearConfig | QLoRALinearConfig


register_config_union(LinearConfig)
