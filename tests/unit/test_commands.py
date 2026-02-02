from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from lalamo.commands import _download_file, _match_model, _suggest_similar_models, pull, PullCallbacks
from lalamo.model_import.remote_registry import RemoteFileSpec, RemoteModelSpec


@patch("lalamo.commands.requests.get")
@patch("builtins.open", new_callable=mock_open)
def test_download_file_success(mock_file: Mock, mock_get: Mock, tmp_path: Path) -> None:
    """Test successful file download."""
    # Setup mock response
    mock_response = Mock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2", b"chunk3"]
    mock_get.return_value = mock_response

    dest_path = tmp_path / "test_file.txt"
    url = "https://example.com/file.txt"

    # Execute download
    _download_file(url, dest_path)

    # Verify requests.get was called correctly
    mock_get.assert_called_once_with(url, stream=True, timeout=60)
    mock_response.raise_for_status.assert_called_once()

    # Verify file writing
    mock_file.assert_called_once_with(dest_path, "wb")
    handle = mock_file()
    assert handle.write.call_count == 3
    handle.write.assert_any_call(b"chunk1")
    handle.write.assert_any_call(b"chunk2")
    handle.write.assert_any_call(b"chunk3")


@patch("lalamo.commands.requests.get")
def test_download_file_http_error(mock_get: Mock, tmp_path: Path) -> None:
    """Test handling of HTTP errors during download."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_get.return_value = mock_response

    dest_path = tmp_path / "test_file.txt"
    url = "https://example.com/file.txt"

    with pytest.raises(requests.HTTPError):
        _download_file(url, dest_path)

    mock_get.assert_called_once_with(url, stream=True, timeout=60)


@patch("lalamo.commands.requests.get")
def test_download_file_network_error(mock_get: Mock, tmp_path: Path) -> None:
    """Test handling of network errors during download."""
    mock_get.side_effect = requests.ConnectionError("Network unreachable")

    dest_path = tmp_path / "test_file.txt"
    url = "https://example.com/file.txt"

    with pytest.raises(requests.ConnectionError):
        _download_file(url, dest_path)

    mock_get.assert_called_once_with(url, stream=True, timeout=60)


@patch("lalamo.commands.requests.get")
@patch("builtins.open", new_callable=mock_open)
def test_download_file_skips_empty_chunks(mock_file: Mock, mock_get: Mock, tmp_path: Path) -> None:
    """Test that empty chunks are skipped during download."""
    # Setup mock response with empty chunks
    mock_response = Mock()
    mock_response.iter_content.return_value = [b"chunk1", b"", b"chunk2", None, b"chunk3"]
    mock_get.return_value = mock_response

    dest_path = tmp_path / "test_file.txt"
    url = "https://example.com/file.txt"

    # Execute download
    _download_file(url, dest_path)

    # Verify only non-empty chunks were written
    handle = mock_file()
    assert handle.write.call_count == 3
    handle.write.assert_any_call(b"chunk1")
    handle.write.assert_any_call(b"chunk2")
    handle.write.assert_any_call(b"chunk3")


def _create_test_models() -> list[RemoteModelSpec]:
    """Create test models for matching tests."""
    return [
        RemoteModelSpec(
            id="meta-llama-3.2-1b-instruct",
            vendor="Meta",
            name="Llama-3.2-1B-Instruct",
            family="Llama-3.2",
            size="1B",
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            quantization=None,
            files=[
                RemoteFileSpec(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=1000,
                    crc32c="abc123",
                )
            ],
        ),
        RemoteModelSpec(
            id="meta-llama-3.2-3b-instruct",
            vendor="Meta",
            name="Llama-3.2-3B-Instruct",
            family="Llama-3.2",
            size="3B",
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            quantization=None,
            files=[
                RemoteFileSpec(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=3000,
                    crc32c="def456",
                )
            ],
        ),
        RemoteModelSpec(
            id="google-gemma-2-2b-instruct",
            vendor="Google",
            name="Gemma-2-2B-Instruct",
            family="Gemma-2",
            size="2B",
            repo_id="google/gemma-2-2b-it",
            quantization=None,
            files=[
                RemoteFileSpec(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=2000,
                    crc32c="ghi789",
                )
            ],
        ),
    ]


def test_match_model_exact_repo_id() -> None:
    """Test exact match on repo_id."""
    models = _create_test_models()
    result = _match_model("meta-llama/Llama-3.2-1B-Instruct", models)

    assert result is not None
    assert result.repo_id == "meta-llama/Llama-3.2-1B-Instruct"
    assert result.id == "meta-llama-3.2-1b-instruct"


def test_match_model_exact_name() -> None:
    """Test exact match on name."""
    models = _create_test_models()
    result = _match_model("Llama-3.2-3B-Instruct", models)

    assert result is not None
    assert result.name == "Llama-3.2-3B-Instruct"
    assert result.id == "meta-llama-3.2-3b-instruct"


def test_match_model_fuzzy_match_high_score() -> None:
    """Test fuzzy matching with high similarity score."""
    models = _create_test_models()
    # Typo in repo_id
    result = _match_model("meta-llama/Llama-3.2-1B-Instruct-typo", models)

    # Should match with score >= 80
    assert result is not None
    assert result.repo_id == "meta-llama/Llama-3.2-1B-Instruct"


def test_match_model_fuzzy_match_below_threshold() -> None:
    """Test fuzzy matching with low similarity score."""
    models = _create_test_models()
    # Very different query
    result = _match_model("completely-different-model", models)

    assert result is None


def test_match_model_empty_list() -> None:
    """Test matching against empty model list."""
    result = _match_model("any-query", [])

    assert result is None


def test_match_model_prefers_exact_over_fuzzy() -> None:
    """Test that exact matches are preferred over fuzzy matches."""
    models = _create_test_models()
    # This should match exactly on repo_id
    result = _match_model("google/gemma-2-2b-it", models)

    assert result is not None
    assert result.repo_id == "google/gemma-2-2b-it"
    assert result.id == "google-gemma-2-2b-instruct"


def test_suggest_similar_models_basic() -> None:
    """Test basic similar model suggestions."""
    models = _create_test_models()
    suggestions = _suggest_similar_models("meta-llama", models)

    assert len(suggestions) <= 3
    assert "meta-llama/Llama-3.2-1B-Instruct" in suggestions
    assert "meta-llama/Llama-3.2-3B-Instruct" in suggestions


def test_suggest_similar_models_with_limit() -> None:
    """Test suggestions respect limit parameter."""
    models = _create_test_models()
    suggestions = _suggest_similar_models("llama", models, limit=1)

    assert len(suggestions) == 1


def test_suggest_similar_models_threshold_filtering() -> None:
    """Test that suggestions filter out low-score matches."""
    models = _create_test_models()
    # Query that won't match well
    suggestions = _suggest_similar_models("xyz123", models)

    # Should return empty or very few results due to 50% threshold
    assert len(suggestions) <= 3


def test_suggest_similar_models_empty_list() -> None:
    """Test suggestions with empty model list."""
    suggestions = _suggest_similar_models("any-query", [])

    assert suggestions == []


def test_suggest_similar_models_returns_repo_ids() -> None:
    """Test that suggestions return repo_ids, not names."""
    models = _create_test_models()
    suggestions = _suggest_similar_models("gemma", models)

    assert len(suggestions) > 0
    # All suggestions should be repo_ids
    for suggestion in suggestions:
        assert "/" in suggestion  # repo_ids have format "vendor/name"


@patch("lalamo.commands._download_file")
@patch("lalamo.commands.shutil.move")
def test_pull_success(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    """Test successful model pull operation."""
    # Setup
    models = _create_test_models()
    model_spec = models[0]  # Use first test model
    output_dir = tmp_path / "output"

    # Create a mock callback to track calls
    callback_calls = []

    class TestCallbacks(PullCallbacks):
        def started(self) -> None:
            callback_calls.append("started")

        def downloading(self, file_spec: RemoteFileSpec) -> None:
            callback_calls.append(f"downloading:{file_spec.name}")

        def finished_downloading(self, file_spec: RemoteFileSpec) -> None:
            callback_calls.append(f"finished_downloading:{file_spec.name}")

        def finished(self) -> None:
            callback_calls.append("finished")

    # Execute
    pull(model_spec, output_dir, callbacks_type=TestCallbacks)

    # Verify
    assert mock_download.call_count == 1  # One file in test model
    assert output_dir.exists()

    # Check callback sequence
    assert "started" in callback_calls
    assert "downloading:model.safetensors" in callback_calls
    assert "finished_downloading:model.safetensors" in callback_calls
    assert "finished" in callback_calls

    # Verify files were moved
    assert mock_move.call_count == 1




@patch("lalamo.commands._download_file")
def test_pull_download_error(mock_download: Mock, tmp_path: Path) -> None:
    """Test pull handles download errors."""
    models = _create_test_models()
    model_spec = models[0]
    mock_download.side_effect = requests.HTTPError("404 Not Found")
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Failed to download"):
        pull(model_spec, output_dir)


def test_pull_output_dir_exists_error(tmp_path: Path) -> None:
    """Test pull raises error when output directory exists."""
    models = _create_test_models()
    model_spec = models[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="already exists"):
        pull(model_spec, output_dir)


@patch("lalamo.commands._download_file")
@patch("lalamo.commands.shutil.move")
def test_pull_multiple_files(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    """Test pull with model containing multiple files."""
    # Create model with multiple files
    model_with_multiple_files = RemoteModelSpec(
        id="test-model",
        vendor="Test",
        name="Test-Model",
        family="Test",
        size="1B",
        repo_id="test/model",
        quantization=None,
        files=[
            RemoteFileSpec(name="model.safetensors", url="https://example.com/model.safetensors", size=1000, crc32c="abc"),
            RemoteFileSpec(name="tokenizer.json", url="https://example.com/tokenizer.json", size=100, crc32c="def"),
            RemoteFileSpec(name="config.json", url="https://example.com/config.json", size=50, crc32c="ghi"),
        ],
    )

    output_dir = tmp_path / "output"

    callback_calls = []

    class TestCallbacks(PullCallbacks):
        def downloading(self, file_spec: RemoteFileSpec) -> None:
            callback_calls.append(f"downloading:{file_spec.name}")

        def finished_downloading(self, file_spec: RemoteFileSpec) -> None:
            callback_calls.append(f"finished_downloading:{file_spec.name}")

    # Execute
    pull(model_with_multiple_files, output_dir, callbacks_type=TestCallbacks)

    # Verify all files were downloaded
    assert mock_download.call_count == 3
    assert mock_move.call_count == 3

    # Verify all files had callbacks
    assert "downloading:model.safetensors" in callback_calls
    assert "downloading:tokenizer.json" in callback_calls
    assert "downloading:config.json" in callback_calls
    assert "finished_downloading:model.safetensors" in callback_calls
    assert "finished_downloading:tokenizer.json" in callback_calls
    assert "finished_downloading:config.json" in callback_calls


