from typing import cast

from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec, ModelSpec
from lalamo.model_import.origins import HuggingFaceOrigin
from lalamo.utils.json import JSON


def test_model_spec_roundtrips_through_registry_converter() -> None:
    spec = LanguageModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.2-1B-Instruct"),
        config_type=HFLlamaConfig,
    )

    raw_json = spec.to_json()
    raw_spec = cast("dict[str, JSON]", raw_json)
    restored_spec = ModelSpec.from_json(raw_json)

    assert raw_spec["type"] == "LanguageModelSpec"
    assert raw_spec["origin"] == {
        "type": "HuggingFaceOrigin",
        "repo": "meta-llama/Llama-3.2-1B-Instruct",
        "weight_format": ".safetensors",
    }
    assert raw_spec["config_type"] == "HFLlamaConfig"
    assert restored_spec == spec
