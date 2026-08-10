from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["MUSE_GLIMMER_MODELS"]

MUSE_GLIMMER_MODELS = [
    LanguageModelSpec(
        vendor="Meta",
        family="Muse-Glimmer",
        name="Muse-Glimmer-30B",
        size="30B",
        origin=HuggingFaceOrigin(repo="meta-models/Muse-Glimmer-30B"),
        config_type=HFMuseGlimmerConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    ),
]
