from lalamo.model_import.model_configs import HFGraniteConfig
from lalamo.model_import.model_spec import ConfigMap, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["GRANITE_MODELS"]


def _granite_instruct(name: str, size: str) -> LanguageModelSpec:
    return LanguageModelSpec(
        vendor="IBM",
        family="Granite",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"ibm-granite/{name}"),
        config_type=HFGraniteConfig,
        configs=ConfigMap(),
    )


GRANITE_MODELS = [
    _granite_instruct("granite-3.3-2b-instruct", "2B"),
    _granite_instruct("granite-3.3-8b-instruct", "8B"),
    _granite_instruct("granite-3.1-2b-instruct", "2B"),
    _granite_instruct("granite-3.1-8b-instruct", "8B"),
]
