from lalamo.model_import.model_configs.huggingface.llama import HFLlamaConfig
from lalamo.model_import.model_specs.common import ModelSpec, structure_origin
from lalamo.model_import.model_specs.origins import (
    HuggingFaceOrigin,
    LocalOrigin,
)

from pathlib import Path

from lalamo.model_import.model_specs.origins import FileSpec

EXAMPLE_JSON = {
    "vendor": "Meta",
    "family": "Llama-3.2",
    "name": "Llama-3.2-1B-Instruct",
    "size": "1B",
    "quantization": None,
    "origin": {"type": "HuggingFaceOrigin", "repo": "meta-llama/Llama-3.2-1B-Instruct"},
    "config_type": "HFLlamaConfig",
    "output_parser_regex": None,
    "system_role_name": "system",
    "user_role_name": "user",
    "assistant_role_name": "assistant",
    "tool_role_name": "tool",
    "use_cases": [],
}


def test_deserialization() -> None:
    model_spec = ModelSpec.from_json(EXAMPLE_JSON)
    assert model_spec.vendor == "Meta"
    assert model_spec.family == "Llama-3.2"
    assert model_spec.name == "Llama-3.2-1B-Instruct"
    assert model_spec.size == "1B"
    assert model_spec.quantization is None
    assert isinstance(model_spec.origin, HuggingFaceOrigin)
    assert model_spec.origin.repo == "meta-llama/Llama-3.2-1B-Instruct"
    assert model_spec.config_type is HFLlamaConfig
    assert model_spec.output_parser_regex is None
    assert model_spec.system_role_name == "system"
    assert model_spec.user_role_name == "user"
    assert model_spec.assistant_role_name == "assistant"
    assert model_spec.tool_role_name == "tool"
    assert model_spec.use_cases == ()


def test_consistency() -> None:
    spec = ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        quantization=None,
        origin=HuggingFaceOrigin(repo="meta-llama/Llama-3.2-1B-Instruct"),
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    )

    assert ModelSpec.from_json(spec.to_json()) == spec


def test_origin_from_json_huggingface() -> None:
    data = {"type": "HuggingFaceOrigin", "repo": "meta-llama/Llama-3.2-1B-Instruct"}
    origin = structure_origin(data)
    assert isinstance(origin, HuggingFaceOrigin)
    assert origin.repo == "meta-llama/Llama-3.2-1B-Instruct"


def test_origin_from_json_local() -> None:
    data = {
        "type": "LocalOrigin",
        "root": "/path/to/model",
        "weight_files": ["model.safetensors"],
    }
    origin = structure_origin(data)
    assert isinstance(origin, LocalOrigin)
    assert origin.root == "/path/to/model"
    assert origin.weight_files == ("model.safetensors",)


def test_origin_from_json_local_defaults() -> None:
    data = {"type": "LocalOrigin", "root": "/path/to/model"}
    origin = structure_origin(data)
    assert isinstance(origin, LocalOrigin)
    assert origin.weight_files == ()


def test_local_origin_resolve_file() -> None:
    origin = LocalOrigin(root="/tmp/models")
    assert origin.resolve_file(FileSpec("config.json")) == Path("/tmp/models/config.json")


def test_local_origin_serialization() -> None:
    spec = ModelSpec(
        vendor="Custom",
        family="test",
        name="test-model",
        size="1B",
        quantization=None,
        origin=LocalOrigin(root="/path/to/model", weight_files=("weights.safetensors",)),
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    )

    json_data = spec.to_json()
    assert json_data["origin"]["type"] == "LocalOrigin"
    assert json_data["origin"]["root"] == "/path/to/model"
    assert json_data["origin"]["weight_files"] == ("weights.safetensors",)

    restored = ModelSpec.from_json(json_data)
    assert isinstance(restored.origin, LocalOrigin)
    assert restored.origin.root == "/path/to/model"
    assert restored == spec
