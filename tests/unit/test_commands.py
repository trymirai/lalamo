from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from lalamo.commands import PullCallbacks, pull
from lalamo.model_import.remote_registry import RegistryModelFile, RegistryModel


def _create_test_models() -> list[RegistryModel]:
    """Create test models for matching tests."""
    return [
        RegistryModel(
            id="meta-llama-3.2-1b-instruct",
            vendor="Meta",
            name="Llama-3.2-1B-Instruct",
            family="Llama-3.2",
            size="1B",
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            quantization=None,
            files=[
                RegistryModelFile(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=1000,
                    crc32c="abc123",
                ),
            ],
        ),
        RegistryModel(
            id="meta-llama-3.2-3b-instruct",
            vendor="Meta",
            name="Llama-3.2-3B-Instruct",
            family="Llama-3.2",
            size="3B",
            repo_id="meta-llama/Llama-3.2-3B-Instruct",
            quantization=None,
            files=[
                RegistryModelFile(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=3000,
                    crc32c="def456",
                ),
            ],
        ),
        RegistryModel(
            id="google-gemma-2-2b-instruct",
            vendor="Google",
            name="Gemma-2-2B-Instruct",
            family="Gemma-2",
            size="2B",
            repo_id="google/gemma-2-2b-it",
            quantization=None,
            files=[
                RegistryModelFile(
                    name="model.safetensors",
                    url="https://example.com/model.safetensors",
                    size=2000,
                    crc32c="ghi789",
                ),
            ],
        ),
    ]


@patch("lalamo.commands._download_file")
@patch("lalamo.commands.shutil.move")
def test_pull_success(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    # Setup
    models = _create_test_models()
    model_spec = models[0]  # Use first test model
    output_dir = tmp_path / "output"

    # Create a mock callback to track calls
    callback_calls = []

    class TestCallbacks(PullCallbacks):
        def started(self) -> None:
            callback_calls.append("started")

        def downloading(self, file_spec: RegistryModelFile) -> None:
            callback_calls.append(f"downloading:{file_spec.name}")

        def finished_downloading(self, file_spec: RegistryModelFile) -> None:
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
    models = _create_test_models()
    model_spec = models[0]
    mock_download.side_effect = requests.HTTPError("404 Not Found")
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Failed to download"):
        pull(model_spec, output_dir)


def test_pull_output_dir_exists_error(tmp_path: Path) -> None:
    models = _create_test_models()
    model_spec = models[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="already exists"):
        pull(model_spec, output_dir)


@patch("lalamo.commands._download_file")
@patch("lalamo.commands.shutil.move")
def test_pull_multiple_files(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    # Create model with multiple files
    model_with_multiple_files = RegistryModel(
        id="test-model",
        vendor="Test",
        name="Test-Model",
        family="Test",
        size="1B",
        repo_id="test/model",
        quantization=None,
        files=[
            RegistryModelFile(
                name="model.safetensors",
                url="https://example.com/model.safetensors",
                size=1000,
                crc32c="abc",
            ),
            RegistryModelFile(
                name="tokenizer.json",
                url="https://example.com/tokenizer.json",
                size=100,
                crc32c="def",
            ),
            RegistryModelFile(
                name="config.json",
                url="https://example.com/config.json",
                size=50,
                crc32c="ghi",
            ),
        ],
    )

    output_dir = tmp_path / "output"

    callback_calls = []

    class TestCallbacks(PullCallbacks):
        def downloading(self, file_spec: RegistryModelFile) -> None:
            callback_calls.append(f"downloading:{file_spec.name}")

        def finished_downloading(self, file_spec: RegistryModelFile) -> None:
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


@patch("lalamo.commands._download_file")
def test_pull_rejects_path_traversal(mock_download: Mock, tmp_path: Path) -> None:
    """Test that pull rejects filenames with path traversal attempts."""
    malicious_model = RegistryModel(
        id="malicious-model",
        vendor="Evil",
        name="Malicious-Model",
        family="Evil",
        size="1B",
        repo_id="evil/malicious",
        quantization=None,
        files=[
            RegistryModelFile(
                name="../../../etc/passwd",
                url="https://example.com/evil.txt",
                size=100,
                crc32c="evil",
            ),
        ],
    )

    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Invalid filename from registry"):
        pull(malicious_model, output_dir)

    # Verify download was never attempted
    assert mock_download.call_count == 0


@patch("lalamo.commands._download_file")
def test_pull_rejects_subdirectory_paths(mock_download: Mock, tmp_path: Path) -> None:
    """Test that pull rejects filenames containing subdirectories."""
    malicious_model = RegistryModel(
        id="malicious-model",
        vendor="Evil",
        name="Malicious-Model",
        family="Evil",
        size="1B",
        repo_id="evil/malicious",
        quantization=None,
        files=[
            RegistryModelFile(
                name="subdir/evil.txt",
                url="https://example.com/evil.txt",
                size=100,
                crc32c="evil",
            ),
        ],
    )

    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Invalid filename from registry"):
        pull(malicious_model, output_dir)

    # Verify download was never attempted
    assert mock_download.call_count == 0
