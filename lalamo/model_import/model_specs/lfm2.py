from itertools import chain, product

from lalamo.model_import.model_configs import HFLFM2Config
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.models.language_model import GenerationConfig

from .common import ConfigMap, FileSpec, LanguageModelSpec

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
        product(["350M", "700M", "1.2B"], [None], [None, 4, 8]),
        product(["2.6B"], [None, "Exp"], [None]),
        product(["2.6B"], ["Exp"], [4, 8]),
    )
]

_LFM25_MODELS = [
    LanguageModelSpec(
        vendor="LiquidAI",
        family="LFM2.5",
        name=_lfm_repo("LFM2.5", size, variant, quantization_bits)[1],
        size=size,
        origin=HuggingFaceOrigin(repo="/".join(_lfm_repo("LFM2.5", size, variant, quantization_bits))),
        config_type=HFLFM2Config,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.1, top_k=50, top_p=0.1),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
    )
    for size, variant, quantization_bits in chain(
        product(["350M"], [None], [None]),
        product(["1.2B"], ["Instruct"], [None]),
    )
]

LFM2_MODELS = _LFM20_MODELS + _LFM25_MODELS
