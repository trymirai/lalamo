from lalamo.model_import.model_configs import HFMuseGlimmerConfig
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.language_model import GenerationConfig

__all__ = ["MUSE_GLIMMER_MODELS"]

MUSE_GLIMMER_OUTPUT_PARSER_REGEX = (
    r"(?s)\s*(?:to=self<\|message\|>(?P<chain_of_thought>.*?)"
    r"<\|eom\|><\|start\|>assistant )?to=user<\|message\|>"
    r"(?P<response>.*?)(?:<\|eot\|>|<\|end_of_text\|>)\Z"
)

MUSE_GLIMMER_MODELS = [
    LanguageModelSpec(
        vendor="Meta",
        family="Muse-Glimmer",
        name=name,
        size="30B",
        origin=HuggingFaceOrigin(repo=repo),
        config_type=HFMuseGlimmerConfig,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
            generation_params_overrides=GenerationConfig(
                temperature=1.0,
                top_k=64,
                top_p=0.95,
            ),
        ),
        output_parser_regex=MUSE_GLIMMER_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag="<|eom|><|start|>assistant to=user<|message|>",
    )
    for name, repo in (
        ("Muse-Glimmer-30B", "meta-models/Muse-Glimmer-30B"),
        ("Muse-Glimmer-30B-MLX-4bit", "mlx-community/Muse-Glimmer-30B-4bit"),
        ("Muse-Glimmer-30B-MLX-8bit", "mlx-community/Muse-Glimmer-30B-8bit"),
    )
]
