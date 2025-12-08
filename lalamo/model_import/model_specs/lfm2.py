from lalamo.model_import.decoder_configs import HFLFM2Config

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LFM2_MODELS"]

LFM2_MODELS = [
    ModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name="LFM2-2.6B",
        size="2.6B",
        repo="LiquidAI/LFM2-2.6B",
        config_type=HFLFM2Config,
        quantization=None,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
        use_cases=tuple(),
    ),
]
