from itertools import chain, product

from lalamo.model_import.decoder_configs import HFLFM2Config
from lalamo.models.language_model import GenerationConfig
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LFM2_MODELS"]


def _lfm2_repo(size: str, variant: str | None, quantization: QuantizationMode | None) -> tuple[str, str]:
    return (
        "LiquidAI" if quantization is None else "mlx-community",
        f"LFM2-{size}"
        f"{f'-{variant}' if variant is not None else ''}"
        f"{f'-{quantization.bits}bit' if quantization is not None else ''}",
    )


LFM2_MODELS = [
    ModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name=_lfm2_repo(size, variant, quantization)[1],
        size=size,
        repo="/".join(_lfm2_repo(size, variant, quantization)),
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
