from lalamo.model_import.model_configs import HFQwen2Config
from lalamo.quantization import QuantizationMode

from .common import (
    ModelSpec,
)
from .origins import HuggingFaceOrigin

__all__ = ["DEEPSEEK_MODELS"]

DEEPSEEK_MODELS = [
    ModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="R1-Distill-Qwen-1.5B",
        size="1.5B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        config_type=HFQwen2Config,
    ),
    ModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="R1-Distill-Qwen-1.5B-AWQ",
        size="1.5B",
        quantization=QuantizationMode.UINT4,
        origin=HuggingFaceOrigin(repo="trymirai/DeepSeek-R1-Distill-Qwen-1.5B-AWQ"),
        config_type=HFQwen2Config,
    ),
]
