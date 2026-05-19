from lalamo.model_import.model_configs.huggingface import ModernBERTConfig
from lalamo.model_import.model_spec import ClassifierModelSpec, ConfigMap, FileSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["MIRAI_CLASSIFIER_MODELS"]

MIRAI_CLASSIFIER_MODELS = [
    ClassifierModelSpec(
        vendor="Mirai",
        family="ModernBERT",
        name="chat-moderation-router",
        size="0.15B",
        origin=HuggingFaceOrigin(repo="trymirai/chat-moderation-router"),
        config_type=ModernBERTConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ClassifierModelSpec(
        vendor="Mirai",
        family="ModernBERT",
        name="health-related-router",
        size="0.15B",
        origin=HuggingFaceOrigin(repo="trymirai/health-related-router"),
        config_type=ModernBERTConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
