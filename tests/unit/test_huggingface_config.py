import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from cattrs import ClassValidationError
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from lalamo.model_import.model_configs.huggingface.common import (
    AWQQuantizationConfig,
    GPTQQuantizationConfig,
    HuggingFaceLMConfig,
    MLXQuantizationConfig,
    QuantizationConfigType,
)
from lalamo.model_import.model_configs.huggingface.qwen3_5 import HFQwen35Config
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.rope import MultimodalRoPEConfig


@dataclass(frozen=True)
class QuantizedConfig(HuggingFaceLMConfig):
    quantization: QuantizationConfigType = None
    quantization_config: QuantizationConfigType | None = None

    def to_decoder_config(
        self,
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


@pytest.mark.parametrize("interleaved", [True, False])
def test_qwen35_export_requires_interleaved_mrope(tmp_path: Path, interleaved: bool) -> None:
    text_config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        eos_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": interleaved,
        },
    )
    config_path = tmp_path / "config.json"
    _write_config(config_path, {"model_type": "qwen3_5", "text_config": text_config.to_dict()})
    if not interleaved:
        with pytest.raises(ClassValidationError):
            HFQwen35Config.from_json(config_path)
        return
    decoder_config = HFQwen35Config.from_json(config_path).to_decoder_config({})
    restored = DecoderConfig.from_json(json.loads(json.dumps(decoder_config.to_json())))
    assert restored == decoder_config
    linear_layer, attention_layer = restored.transformer_config.layer_configs
    assert linear_layer.rope_config is None
    rope_config = attention_layer.rope_config
    assert isinstance(rope_config, MultimodalRoPEConfig)
    assert rope_config.sections == (11, 11, 10)
