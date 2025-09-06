from lalamo.model_import.decoder_configs import HFQwen3Config

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["POLARIS_MODELS"]

POLARIS_MODELS = [
    ModelSpec(
        vendor="POLARIS-Project",
        family="Polaris-Preview",
        name="Polaris-4B-Preview",
        size="4B",
        quantization=None,
        repo="POLARIS-Project/Polaris-4B-Preview",
        config_type=HFQwen3Config,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]
