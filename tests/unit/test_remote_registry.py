from unittest.mock import Mock, patch

import pytest
import requests

from lalamo.model_import.remote_registry import fetch_available_models


@patch("lalamo.model_import.remote_registry.requests.get")
def test_fetch_available_models_success(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = {
        "models": [
            {
                "id": "meta-llama-3.2-1b-instruct",
                "vendor": "Meta",
                "name": "Llama-3.2-1B-Instruct",
                "family": "Llama-3.2",
                "size": "1B",
                "repoId": "meta-llama/Llama-3.2-1B-Instruct",
                "quantization": None,
                "files": [
                    {
                        "name": "model.pte",
                        "url": "https://example.com/model.pte",
                        "size": 1234567890,
                        "crc32c": "abcd1234",
                    },
                ],
            },
            {
                "id": "meta-llama-3.2-3b-instruct-q4",
                "vendor": "Meta",
                "name": "Llama-3.2-3B-Instruct",
                "family": "Llama-3.2",
                "size": "3B",
                "repoId": "meta-llama/Llama-3.2-3B-Instruct",
                "quantization": "Q4_0",
                "files": [
                    {
                        "name": "model.pte",
                        "url": "https://example.com/model-q4.pte",
                        "size": 987654321,
                        "crc32c": "efgh5678",
                    },
                    {
                        "name": "tokenizer.bin",
                        "url": "https://example.com/tokenizer.bin",
                        "size": 123456,
                        "crc32c": "ijkl9012",
                    },
                ],
            },
        ],
    }
    mock_get.return_value = mock_response

    models = fetch_available_models()

    assert len(models) == 2

    # Check first model
    assert models[0].id == "meta-llama-3.2-1b-instruct"
    assert models[0].vendor == "Meta"
    assert models[0].name == "Llama-3.2-1B-Instruct"
    assert models[0].family == "Llama-3.2"
    assert models[0].size == "1B"
    assert models[0].repo_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert models[0].quantization is None
    assert len(models[0].files) == 1
    assert models[0].files[0].name == "model.pte"
    assert models[0].files[0].url == "https://example.com/model.pte"
    assert models[0].files[0].size == 1234567890
    assert models[0].files[0].crc32c == "abcd1234"

    # Check second model
    assert models[1].id == "meta-llama-3.2-3b-instruct-q4"
    assert models[1].vendor == "Meta"
    assert models[1].name == "Llama-3.2-3B-Instruct"
    assert models[1].family == "Llama-3.2"
    assert models[1].size == "3B"
    assert models[1].repo_id == "meta-llama/Llama-3.2-3B-Instruct"
    assert models[1].quantization == "Q4_0"
    assert len(models[1].files) == 2

    # Verify API call
    mock_get.assert_called_once_with(
        "https://sdk.trymirai.com/api/v1/models/list/lalamo",
        timeout=30,
    )


@patch("lalamo.model_import.remote_registry.requests.get")
def test_fetch_available_models_empty_response(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.json.return_value = {"models": []}
    mock_get.return_value = mock_response

    models = fetch_available_models()

    assert len(models) == 0
    mock_get.assert_called_once()


@patch("lalamo.model_import.remote_registry.requests.get")
def test_fetch_available_models_http_error(mock_get: Mock) -> None:
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_available_models()


@patch("lalamo.model_import.remote_registry.requests.get")
def test_fetch_available_models_network_error(mock_get: Mock) -> None:
    mock_get.side_effect = requests.ConnectionError("Network unreachable")

    with pytest.raises(requests.ConnectionError):
        fetch_available_models()
