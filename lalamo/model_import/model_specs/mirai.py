from lalamo.model_import.decoder_configs.huggingface import ModernBERTConfig

from .common import ConfigMap, ModelSpec, ModelType, FileSpec

__all__ = ["MIRAI_ROUTER_MODELS"]

MIRAI_ROUTER_MODELS = [
    ModelSpec(
        vendor="trymirai",
        family="ModernBERT",
        name="ModernBERT-base",
        size="0.15B",
        quantization=None,
        repo="trymirai/chat-moderation-router",
        config_type=ModernBERTConfig,
        use_cases=tuple(),
        model_type=ModelType("router_model"),
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
