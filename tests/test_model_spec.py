from lalamo.model_import.decoder_configs.huggingface.llama import HFLlamaConfig
from lalamo.model_import.model_specs.common import ModelSpec, WeightsType

EXAMPLE_JSON = {
    "vendor": "Meta",
    "family": "Llama-3.2",
    "name": "Llama-3.2-1B-Instruct",
    "size": "1B",
    "quantization": None,
    "repo": "meta-llama/Llama-3.2-1B-Instruct",
    "config_type": "HFLlamaConfig",
    "output_parser_regex": None,
    "system_role_name": "system",
    "user_role_name": "user",
    "assistant_role_name": "assistant",
    "tool_role_name": "tool",
    "weights_type": "safetensors",
    "use_cases": [],
}


def test_deserialization() -> None:
    model_spec = ModelSpec.from_json(EXAMPLE_JSON)
    assert model_spec.vendor == "Meta"
    assert model_spec.family == "Llama-3.2"
    assert model_spec.name == "Llama-3.2-1B-Instruct"
    assert model_spec.size == "1B"
    assert model_spec.quantization is None
    assert model_spec.repo == "meta-llama/Llama-3.2-1B-Instruct"
    assert model_spec.config_type is HFLlamaConfig
    assert model_spec.output_parser_regex is None
    assert model_spec.system_role_name == "system"
    assert model_spec.user_role_name == "user"
    assert model_spec.assistant_role_name == "assistant"
    assert model_spec.tool_role_name == "tool"
    assert model_spec.weights_type == WeightsType.SAFETENSORS
    assert model_spec.use_cases == ()


def test_consistency() -> None:
    spec = ModelSpec(
        vendor="Meta",
        family="Llama-3.2",
        name="Llama-3.2-1B-Instruct",
        size="1B",
        quantization=None,
        repo="meta-llama/Llama-3.2-1B-Instruct",
        config_type=HFLlamaConfig,
        use_cases=tuple(),
    )

    assert ModelSpec.from_json(spec.to_json()) == spec
