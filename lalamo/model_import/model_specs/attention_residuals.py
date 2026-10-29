from lalamo.model_import.model_configs.huggingface.qwen3_attnres import HFQwen3AttnResConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin

ATTENTION_RESIDUAL_MODELS = [
    LanguageModelSpec(
        vendor="Cheng Luo",
        family="OpenAttnRes",
        name=f"open-attnres-{variant}",
        size="0.1B",
        origin=HuggingFaceOrigin(repo=f"wdlctc/open-attnres-{variant}"),
        config_type=HFQwen3AttnResConfig,
        configs=ConfigMap(chat_template=FileSpec("chat_template.jinja")),
    )
    for variant in ("full", "block")
]
