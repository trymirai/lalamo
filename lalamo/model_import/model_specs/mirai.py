from lalamo.model_import.model_configs.huggingface import ModernBERTConfig

from .common import ConfigMap, FileSpec, ModelSpec, ModelType

__all__ = ["MIRAI_CLASSIFIER_MODELS"]

MIRAI_CLASSIFIER_MODELS = [
    ModelSpec(
        vendor="Mirai",
        family="ModernBERT",
        name="chat-moderation-router",
        size="0.15B",
        quantization=None,
        repo="trymirai/chat-moderation-router",
        config_type=ModernBERTConfig,
        use_cases=tuple(),
        model_type=ModelType("classifier_model"),
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ModelSpec(
        vendor="Mirai",
        family="ModernBERT",
        name="health-related-router",
        size="0.15B",
        quantization=None,
        repo="trymirai/health-related-router",
        config_type=ModernBERTConfig,
        use_cases=tuple(),
        model_type=ModelType("classifier_model"),
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
