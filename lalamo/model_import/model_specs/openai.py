from lalamo.model_import.model_configs.huggingface import HFOpenAIPrivacyFilterConfig

from .common import ConfigMap, ModelSpec, ModelType

__all__ = ["OPENAI_MODELS"]


# openai/privacy-filter is a token classifier, not a chat model. Its repo ships no
# chat template. We plug in a minimal pass-through template so the generic
# message_processor plumbing works; the classifier itself ignores template
# structure and labels every token of whatever text we feed in.
_PRIVACY_FILTER_CHAT_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


OPENAI_MODELS = [
    ModelSpec(
        vendor="OpenAI",
        family="PrivacyFilter",
        name="privacy-filter",
        size="1.5B",
        quantization=None,
        repo="openai/privacy-filter",
        config_type=HFOpenAIPrivacyFilterConfig,
        use_cases=tuple(),
        model_type=ModelType("classifier_model"),
        configs=ConfigMap(chat_template=_PRIVACY_FILTER_CHAT_TEMPLATE),
    ),
]
