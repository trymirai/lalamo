#!/usr/bin/env python3
"""
Script to parse JSON model configuration to ModelSpec using lalamo.
"""

import json
from typing import Any

from lalamo import ModelSpec


def main() -> None:
    # JSON data for Mistral Devstral model
    model_json: dict[str, Any] = {
        "id": 40,
        "vendor": "Mistral",
        "family": "Devstral",
        "name": "Devstral-Small-2505",
        "size": "24B",
        "quantization": None,
        "repo": "mistralai/Devstral-Small-2505",
        "config_type": "HFMistralConfig",
        "output_parser_regex": None,
        "system_role_name": "system",
        "user_role_name": "user",
        "assistant_role_name": "assistant",
        "tool_role_name": "tool",
        "weights_type": "safetensors",
        "use_cases": ["code"],
        "configs": {
            "tokenizer": {
                "repo": "mistralai/Mistral-Small-3.1-24B-Base-2503",
                "filename": "tokenizer.json",
            },
            "tokenizer_config": {
                "repo": "mistralai/Mistral-Small-3.1-24B-Base-2503",
                "filename": "tokenizer_config.json",
            },
            "generation_config": {
                "repo": "mistralai/Mistral-Small-3.1-24B-Base-2503",
                "filename": "generation_config.json",
            },
        },
    }

    # Remove the 'id' field as it's not part of ModelSpec
    model_json_cleaned = {k: v for k, v in model_json.items() if k != "id"}

    # Parse JSON to ModelSpec
    model_spec = ModelSpec.from_json(model_json_cleaned)

    # Display the parsed ModelSpec
    print("Successfully parsed ModelSpec!")
    print("-" * 50)
    print(f"Vendor: {model_spec.vendor}")
    print(f"Family: {model_spec.family}")
    print(f"Name: {model_spec.name}")
    print(f"Size: {model_spec.size}")
    print(f"Repository: {model_spec.repo}")
    print(f"Config Type: {model_spec.config_type.__name__}")
    print(f"Quantization: {model_spec.quantization}")
    print(f"Use Cases: {model_spec.use_cases}")
    print(f"System Role: {model_spec.system_role_name}")
    print(f"User Role: {model_spec.user_role_name}")
    print(f"Assistant Role: {model_spec.assistant_role_name}")
    print(f"Tool Role: {model_spec.tool_role_name}")
    print(f"Weights Type: {model_spec.weights_type}")

    # Display configs
    print("\nConfigurations:")
    print(f"  Tokenizer: {model_spec.configs.tokenizer.repo}/{model_spec.configs.tokenizer.filename}")
    print(
        f"  Tokenizer Config: {model_spec.configs.tokenizer_config.repo}/{model_spec.configs.tokenizer_config.filename}",
    )
    if model_spec.configs.generation_config:
        print(
            f"  Generation Config: {model_spec.configs.generation_config.repo}/{model_spec.configs.generation_config.filename}",
        )

    # Convert back to JSON to verify round-trip
    print("\nRound-trip test (converting back to JSON):")
    print("-" * 50)
    reconstructed_json = model_spec.to_json()
    print(json.dumps(reconstructed_json, indent=2))


if __name__ == "__main__":
    main()
