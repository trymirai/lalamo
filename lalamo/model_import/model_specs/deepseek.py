from lalamo.model_import.decoder_configs import HFQwen2Config

from .common import (
    ModelSpec,
)

__all__ = ["DEEPSEEK_MODELS"]

DEEPSEEK_MODELS = [
    ModelSpec(
        vendor="DeepSeek",
        family="R1-Distill-Qwen",
        name="R1-Distill-Qwen-1.5B",
        size="1.5B",
        quantization=None,
        repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        config_type=HFQwen2Config,
    ),
]
