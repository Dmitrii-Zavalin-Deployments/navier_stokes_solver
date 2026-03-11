# tests/io/test_dropbox_io.py

from unittest.mock import MagicMock, patch

import pytest

from src.io.download_from_dropbox import download_files_from_dropbox


@patch('dropbox.Dropbox')
@patch('src.io.download_from_dropbox.refresh_access_token')
def test_download_logic_skips_invalid_extensions(mock_token, mock_dbx, tmp_path):
    """Ensure the downloader only touches the ALLOWED_EXTENSIONS."""
    # Setup mocks
    mock_token.return_value = "fake_token"
    dbx_instance = mock_dbx.return_value
    
    # Simulate one valid file and one invalid file in Dropbox
    mock_entry_valid = MagicMock(spec=pytest.importorskip("dropbox").files.FileMetadata)
    mock_entry_valid.name = "input.json"
    mock_entry_valid.path_lower = "/input.json"
    
    mock_entry_invalid = MagicMock(spec=pytest.importorskip("dropbox").files.FileMetadata)
    mock_entry_invalid.name = "virus.exe"
    
    dbx_instance.files_list_folder.return_value = MagicMock(entries=[mock_entry_valid, mock_entry_invalid], has_more=False)
    dbx_instance.files_download.return_value = (None, MagicMock(content=b'{"key": "value"}'))

    # Run the function
    log_file = tmp_path / "log.txt"
    local_dir = tmp_path / "downloads"
    
    download_files_from_dropbox("/folder", str(local_dir), "token", "id", "secret", str(log_file))

    # Assertions
    assert (local_dir / "input.json").exists()
    assert not (local_dir / "virus.exe").exists()