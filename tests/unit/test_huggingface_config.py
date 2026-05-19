import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lalamo.model_import.model_configs.huggingface.common import (
    AWQQuantizationConfig,
    GPTQQuantizationConfig,
    HuggingFaceLMConfig,
    MLXQuantizationConfig,
    QuantizationConfigType,
)
from lalamo.modules.decoder import DecoderConfig


@dataclass(frozen=True)
class QuantizedConfig(HuggingFaceLMConfig):
    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType | None = None

    def to_decoder_config(
        self,
        context_length: int | None,
        metadata_dict: Mapping[str, str],
    ) -> DecoderConfig:
        raise NotImplementedError


def _write_config(path: Path, raw_config: object) -> None:
    with path.open("w") as f:
        json.dump(raw_config, f)


def test_quantization_config_uses_tagged_union_strategy(tmp_path: Path) -> None:
    raw_config = {
        "quantization": {"quant_method": "awq"},
        "quantization_config": {
            "bits": 4,
            "checkpoint_format": "gptq",
            "desc_act": False,
            "group_size": 128,
            "lm_head": False,
            "meta": {
                "damp_auto_increment": 0.0,
                "damp_percent": 0.1,
                "mse": 0.0,
                "quantizer": [],
                "static_groups": False,
                "true_sequential": False,
                "uri": "",
            },
            "pack_dtype": "int32",
            "quant_method": "gptq",
            "sym": True,
        },
    }
    config_path = tmp_path / "config.json"
    _write_config(config_path, raw_config)

    config = QuantizedConfig.from_json(config_path)

    assert isinstance(config.quantization, AWQQuantizationConfig)
    assert isinstance(config.quantization_config, GPTQQuantizationConfig)


def test_quantization_config_defaults_missing_tag_to_mlx(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"quantization": None, "quantization_config": {"group_size": 64, "bits": 4}})

    config = QuantizedConfig.from_json(config_path)

    assert config.quantization is None
    assert isinstance(config.quantization_config, MLXQuantizationConfig)
