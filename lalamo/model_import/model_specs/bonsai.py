from lalamo.model_import.model_configs import HFBonsaiConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["BONSAI_MODELS"]

BONSAI_MODELS = [
    LanguageModelSpec(
        vendor="PrismML",
        family="Bonsai",
        name="Bonsai-4B-mlx-1bit",
        size="4B",
        origin=HuggingFaceOrigin(repo="prism-ml/Bonsai-4B-mlx-1bit"),
        config_type=HFBonsaiConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
    ),
]
