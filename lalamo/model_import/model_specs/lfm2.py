from itertools import chain, product

from lalamo.model_import.decoder_configs import HFLFM2Config
from lalamo.models.language_model import GenerationConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LFM2_MODELS"]


def _lfm_repo(family: str, size: str, variant: str | None, quantization: QuantizationMode | None) -> tuple[str, str]:
    return (
        "LiquidAI" if quantization is None else "mlx-community",
        f"{family}-{size}"
        f"{f'-{variant}' if variant is not None else ''}"
        f"{f'-{quantization.bits}bit' if quantization is not None else ''}",
    )


_LFM20_MODELS = [
    ModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name=_lfm_repo("LFM2", size, variant, quantization)[1],
        size=size,
        repo="/".join(_lfm_repo("LFM2", size, variant, quantization)),
        config_type=HFLFM2Config,
        quantization=quantization,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.3, min_p=0.15),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
        use_cases=tuple(),
    )
    for size, variant, quantization in chain(
        product(["350M", "700M", "1.2B"], [None], [None, QuantizationMode.UINT4, QuantizationMode.UINT8]),
        product(["2.6B"], [None, "Exp"], [None]),
        product(["2.6B"], ["Exp"], [QuantizationMode.UINT4, QuantizationMode.UINT8]),
    )
]

_LFM25_MODELS = [
    ModelSpec(
        vendor="LiquidAI",
        family="LFM2.5",
        name=_lfm_repo("LFM2.5", size, variant, quantization)[1],
        size=size,
        repo="/".join(_lfm_repo("LFM2.5", size, variant, quantization)),
        config_type=HFLFM2Config,
        quantization=quantization,
        configs=ConfigMap(
            generation_config=GenerationConfig(temperature=0.1, top_k=50, top_p=0.1),  # , repetition_penalty=1.05
            chat_template=FileSpec("chat_template.jinja"),
        ),
        use_cases=tuple(),
    )
    for size, variant, quantization in chain(
        product(["1.2B"], ["Instruct"], [None]),
    )
]

LFM2_MODELS = _LFM20_MODELS + _LFM25_MODELS
