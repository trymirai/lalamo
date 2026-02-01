from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from lalamo.commands import _download_file


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
