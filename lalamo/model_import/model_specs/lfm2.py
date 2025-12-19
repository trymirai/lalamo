from lalamo.model_import.decoder_configs import HFLFM2Config
from lalamo.quantization import QuantizationMode

from .common import ConfigMap, FileSpec, ModelSpec

__all__ = ["LFM2_MODELS"]


def _lfm2_repo(size: str, quantization: QuantizationMode | None) -> tuple[str, str]:
    organization = "LiquidAI" if quantization is None else "mlx-community"
    name = f"LFM2-{size}{f'-{quantization.bits}bit' if quantization is not None else ''}"
    return (organization, name)


LFM2_MODELS = [
    ModelSpec(
        vendor="LiquidAI",
        family="LFM2",
        name=_lfm2_repo(size, quantization)[1],
        size=size,
        repo="/".join(_lfm2_repo(size, quantization)),
        config_type=HFLFM2Config,
        quantization=quantization,
        configs=ConfigMap(
            chat_template=FileSpec("chat_template.jinja"),
        ),
        use_cases=tuple(),
    )
    for size in ["350M", "700M", "1.2B", "2.6B"]
    for quantization in [None, *([QuantizationMode.UINT4, QuantizationMode.UINT8] if size != "2.6B" else [])]
]
