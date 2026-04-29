from lalamo.model_import.model_configs.huggingface import ModernBERTConfig
from lalamo.model_import.origins import HuggingFaceOrigin

from .common import ClassifierModelSpec, ConfigMap, FileSpec

__all__ = ["MIRAI_CLASSIFIER_MODELS"]

MIRAI_CLASSIFIER_MODELS = [
    ClassifierModelSpec(
        vendor="trymirai",
        family="ModernBERT",
        name="ModernBERT-Chat-Moderation",
        size="0.15B",
        origin=HuggingFaceOrigin(repo="trymirai/chat-moderation-router"),
        config_type=ModernBERTConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
