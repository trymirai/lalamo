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
from lalamo.model_import.model_configs.huggingface.dflash import HFDFlashConfig
from lalamo.modules.decoder import DecoderConfig
from lalamo.modules.rope import UnscaledRoPEConfig


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


def make_dflash_raw_config() -> dict[str, object]:
    return {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "num_target_layers": 32,
        "vocab_size": 248320,
        "head_dim": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 4096,
        "use_sliding_window": True,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 5],
        },
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    }


def test_dflash_config_parses_v5_format_with_nested_block_size_and_rope_parameters(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path, make_dflash_raw_config())

    draft_config = HFDFlashConfig.from_json(config_path).to_dflash_draft_config()

    assert draft_config.block_size == 16
    sliding_layer, full_layer = draft_config.layer_configs
    assert sliding_layer.attention_config.sliding_window_size == 4096
    assert full_layer.attention_config.sliding_window_size is None
    assert isinstance(sliding_layer.attention_config.rope_config, UnscaledRoPEConfig)
    assert sliding_layer.attention_config.rope_config.base == 10000000


def test_dflash_config_parses_legacy_format_with_top_level_rope_and_block_size(tmp_path: Path) -> None:
    raw_config = make_dflash_raw_config()
    del raw_config["rope_parameters"]
    dflash_inner = raw_config["dflash_config"]
    assert isinstance(dflash_inner, dict)
    del dflash_inner["block_size"]
    raw_config.update(
        {
            "block_size": 8,
            "rope_theta": 1000000,
            "rope_scaling": None,
            "layer_types": ["full_attention", "full_attention"],
            "sliding_window": None,
            "use_sliding_window": False,
        },
    )
    config_path = tmp_path / "config.json"
    _write_config(config_path, raw_config)

    draft_config = HFDFlashConfig.from_json(config_path).to_dflash_draft_config()

    assert draft_config.block_size == 8
    first_layer, _ = draft_config.layer_configs
    assert isinstance(first_layer.attention_config.rope_config, UnscaledRoPEConfig)
    assert first_layer.attention_config.rope_config.base == 1000000
