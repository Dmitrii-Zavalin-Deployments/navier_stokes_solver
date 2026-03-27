# tests/io/test_download_from_dropbox.py

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import dropbox
import pytest

from src.io.download_from_dropbox import CloudIngestor
from src.io.dropbox_utils import TokenManager


def test_token_manager_refresh_success():
    """Rule 5: Verify deterministic token refreshing via requests."""
    tm = TokenManager(client_id="fake_id", client_secret="fake_secret")
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"access_token": "new_shiny_token"}
        
        token = tm.refresh_access_token("refresh_me")
        assert token == "new_shiny_token"
        # Verify Rule 5: Payload contains all explicit credentials
        args, kwargs = mock_post.call_args
        assert kwargs['data']['client_id'] == "fake_id"

def test_token_manager_refresh_failure():
    """Rule 5: Verify RuntimeError on auth failure (Zero-Default Policy)."""
    tm = TokenManager(client_id="fake_id", client_secret="fake_secret")
    
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 401
        mock_post.return_value.text = "Unauthorized"
        
        with pytest.raises(RuntimeError, match="Authentication Failure"):
            tm.refresh_access_token("bad_token")

@patch("dropbox.Dropbox")
def test_cloud_ingestor_pagination(mock_dbx_class):
    """Rule 10: Verify the Archivist handles pagination and filtering correctly."""
    
    # 1. Setup Dependency Injection (Rule 5)
    mock_tm = MagicMock(spec=TokenManager)
    mock_tm.refresh_access_token.return_value = "fake_access_token"
    
    # 2. Mock Dropbox internal responses
    mock_dbx = mock_dbx_class.return_value
    
    # Page 1: Contains a valid file and a cursor for more data
    page1 = MagicMock()
    file_valid = MagicMock(spec=dropbox.files.FileMetadata)
    file_valid.name = "simulation_01.h5"
    file_valid.path_lower = "/remote/simulation_01.h5"
    page1.entries = [file_valid]
    page1.has_more = True
    page1.cursor = "next_page_token"
    
    # Page 2: Contains an invalid file extension (should be filtered)
    page2 = MagicMock()
    file_invalid = MagicMock(spec=dropbox.files.FileMetadata)
    file_invalid.name = "notes.txt"
    page2.entries = [file_invalid]
    page2.has_more = False
    
    # Map the sequence of calls
    mock_dbx.files_list_folder.return_value = page1
    mock_dbx.files_list_folder_continue.return_value = page2
    mock_dbx.files_download.return_value = (None, MagicMock(content=b"physics_data"))

    # 3. Instantiate and Execute
    # Log path is required by __slots__ (Rule 0)
    log_path = Path("test_ingest.log")
    
    # We mock open to avoid writing actual files during testing
    with patch("builtins.open", mock_open()):
        # DI: Injecting the mock_tm into the ingestor
        ingestor = CloudIngestor(mock_tm, "initial_refresh_token", log_path)
        ingestor.sync("/remote", Path("./local_test_data"), [".h5"])
    
    # 4. Assertions (Forensic Audit)
    # Check that it started with the correct folder
    mock_dbx.files_list_folder.assert_called_once_with("/remote")
    # Check that it used the cursor to get page 2
    mock_dbx.files_list_folder_continue.assert_called_once_with("next_page_token")
    # Verify Rule 8 (Minimalism): Only 1 download occurred because .txt was ignored
    assert mock_dbx.files_download.call_count == 1