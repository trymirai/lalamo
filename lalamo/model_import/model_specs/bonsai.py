from lalamo.model_import.model_configs import HFBonsaiConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["BONSAI_MODELS"]

BONSAI_MODELS = [
    ModelSpec(
        vendor="PrismML",
        family="Bonsai",
        name="Bonsai-4B-mlx-1bit",
        size="4B",
        quantization=QuantizationMode.UINT1,
        repo="prism-ml/Bonsai-4B-mlx-1bit",
        config_type=HFBonsaiConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]
