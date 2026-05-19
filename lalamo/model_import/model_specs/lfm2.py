from itertools import chain, product

from lalamo.model_import.model_configs import HFLFM2Config
from lalamo.model_import.model_spec import ConfigMap, FileSpec, LanguageModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.language_model import GenerationConfig

__all__ = ["LFM2_MODELS"]


def _lfm_repo(family: str, size: str, variant: str | None, quantization_bits: int | None) -> tuple[str, str]:
    return (
        "LiquidAI" if quantization_bits is None else "mlx-community",
        f"{family}-{size}"
        f"{f'-{variant}' if variant is not None else ''}"
        f"{f'-{quantization_bits}bit' if quantization_bits is not None else ''}",
    )


_LFM20_MODELS = [
    LanguageModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name=_lfm_repo("LFM2", size, variant, quantization_bits)[1],
        size=size,
        origin=HuggingFaceOrigin(repo="/".join(_lfm_repo("LFM2", size, variant, quantization_bits))),
        config_type=HFLFM2Config,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.3, min_p=0.15),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
    )
    for size, variant, quantization_bits in chain(
        product(["350M", "700M", "1.2B", "2.6B"], [None], [None, 4, 8]),
    )
]

_LFM25_MODEL_SPECS = (
    ("LiquidAI", "LFM2.5-350M", "350M", None),
    ("LiquidAI", "LFM2.5-1.2B-Instruct", "1.2B", None),
    ("LiquidAI", "LFM2.5-1.2B-Instruct-MLX-4bit", "1.2B", 4),
    ("LiquidAI", "LFM2.5-1.2B-Instruct-MLX-8bit", "1.2B", 8),
    ("LiquidAI", "LFM2.5-1.2B-Thinking", "1.2B", None),
    ("mlx-community", "LFM2.5-1.2B-Thinking-4bit", "1.2B", 4),
    ("mlx-community", "LFM2.5-1.2B-Thinking-8bit", "1.2B", 8),
)

_LFM25_MODELS = [
    LanguageModelSpec(
        vendor="LiquidAI",
        family="LFM2.5",
        name=name,
        size=size,
        origin=HuggingFaceOrigin(repo=f"{repo_owner}/{name}"),
        config_type=HFLFM2Config,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.1, top_k=50, top_p=0.1),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
    )
    for repo_owner, name, size, _quantization_bits in _LFM25_MODEL_SPECS
]

LFM2_MODELS = _LFM20_MODELS + _LFM25_MODELS
