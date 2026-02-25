from lalamo.model_import.model_configs.huggingface import ModernBERTConfig

from .common import ConfigMap, FileSpec, ModelSpec, ModelType

__all__ = ["MIRAI_CLASSIFIER_MODELS"]

_MODERNBERT_CHAT_TEMPLATE = "[CLS]{% for message in messages %}{{ message.content }}{% endfor %}[SEP]"

MIRAI_CLASSIFIER_MODELS = [
    ModelSpec(
        vendor="trymirai",
        family="ModernBERT",
        name="ModernBERT-Chat-Moderation",
        size="0.15B",
        quantization=None,
        repo="trymirai/chat-moderation-router",
        config_type=ModernBERTConfig,
        use_cases=tuple(),
        model_type=ModelType("classifier_model"),
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
    ModelSpec(
        vendor="darkolorin",
        family="ModernBERT",
        name="vibe-router-modernbert-v3",
        size="0.15B",
        quantization=None,
        repo="darkolorin/vibe-router-modernbert-v3",
        config_type=ModernBERTConfig,
        use_cases=tuple(),
        model_type=ModelType("classifier_model"),
        configs=ConfigMap(chat_template=_MODERNBERT_CHAT_TEMPLATE),
    ),
]
