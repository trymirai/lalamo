from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from lalamo.cli.convert import pull
from lalamo.model_import.remote_registry import RegistryModel, RegistryModelFile


def _create_test_models() -> list[RegistryModel]:
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
    ]


def _fake_download(_url: str, dest_path: Path) -> None:
    dest_path.write_bytes(b"fake")


@patch("lalamo.cli.convert._download_file", side_effect=_fake_download)
@patch("lalamo.cli.convert.shutil.move")
def test_pull_success(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    model_spec = _create_test_models()[0]
    output_dir = tmp_path / "output"

    pull(model_spec, output_dir)

    assert mock_download.call_count == 1
    assert output_dir.exists()
    assert mock_move.call_count == 1


@patch("lalamo.cli.convert._download_file")
def test_pull_download_error(mock_download: Mock, tmp_path: Path) -> None:
    model_spec = _create_test_models()[0]
    mock_download.side_effect = requests.HTTPError("404 Not Found")
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Failed to download"):
        pull(model_spec, output_dir)


def test_pull_output_dir_exists_error(tmp_path: Path) -> None:
    model_spec = _create_test_models()[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="already exists"):
        pull(model_spec, output_dir)


@patch("lalamo.cli.convert._download_file", side_effect=_fake_download)
@patch("lalamo.cli.convert.shutil.move")
def test_pull_multiple_files(mock_move: Mock, mock_download: Mock, tmp_path: Path) -> None:
    model_spec = RegistryModel(
        id="test-model",
        vendor="Test",
        name="Test-Model",
        family="Test",
        size="1B",
        repo_id="test/model",
        quantization=None,
        files=[
            RegistryModelFile(
                name="model.safetensors", url="https://example.com/model.safetensors", size=1000, crc32c="abc",
            ),
            RegistryModelFile(name="tokenizer.json", url="https://example.com/tokenizer.json", size=100, crc32c="def"),
            RegistryModelFile(name="config.json", url="https://example.com/config.json", size=50, crc32c="ghi"),
        ],
    )

    output_dir = tmp_path / "output"
    pull(model_spec, output_dir)

    assert mock_download.call_count == 3
    assert mock_move.call_count == 3


@patch("lalamo.cli.convert._download_file")
def test_pull_rejects_path_traversal(mock_download: Mock, tmp_path: Path) -> None:
    malicious_model = RegistryModel(
        id="malicious-model",
        vendor="Evil",
        name="Malicious-Model",
        family="Evil",
        size="1B",
        repo_id="evil/malicious",
        quantization=None,
        files=[
            RegistryModelFile(name="../../../etc/passwd", url="https://example.com/evil.txt", size=100, crc32c="evil"),
        ],
    )

    with pytest.raises(RuntimeError, match="Invalid filename from registry"):
        pull(malicious_model, tmp_path / "output")

    assert mock_download.call_count == 0


@patch("lalamo.cli.convert._download_file")
def test_pull_rejects_subdirectory_paths(mock_download: Mock, tmp_path: Path) -> None:
    malicious_model = RegistryModel(
        id="malicious-model",
        vendor="Evil",
        name="Malicious-Model",
        family="Evil",
        size="1B",
        repo_id="evil/malicious",
        quantization=None,
        files=[
            RegistryModelFile(name="subdir/evil.txt", url="https://example.com/evil.txt", size=100, crc32c="evil"),
        ],
    )

    with pytest.raises(RuntimeError, match="Invalid filename from registry"):
        pull(malicious_model, tmp_path / "output")

    assert mock_download.call_count == 0
