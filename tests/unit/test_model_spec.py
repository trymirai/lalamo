from pathlib import Path
from typing import cast

import pytest

from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec, ModelSpec
from lalamo.model_import.origins import FileSpec, HuggingFaceOrigin, LocalOrigin
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


def test_model_spec_root_is_not_instantiable() -> None:
    with pytest.raises(TypeError, match="ModelSpec is abstract"):
        ModelSpec(
            vendor="Meta",
            family="Llama-3.2",
            name="Llama-3.2-1B-Instruct",
            size="1B",
            origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.2-1B-Instruct"),
            config_type=HFLlamaConfig,
        )


def test_local_origin_resolves_files_relative_to_root() -> None:
    origin = LocalOrigin(root="/tmp/models")

    assert origin.resolve_file(FileSpec("config.json")) == Path("/tmp/models/config.json")
