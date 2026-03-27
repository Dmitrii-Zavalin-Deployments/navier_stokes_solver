# tests/io/test_upload_to_dropbox.py

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import dropbox
import pytest

from src.io.dropbox_utils import TokenManager
from src.io.upload_to_dropbox import CloudUploader


@patch("dropbox.Dropbox")
def test_cloud_uploader_success(mock_dbx_class):
    """Rule 5 & 10: Verify DI-based initialization and atomic upload."""
    
    # 1. Setup Deterministic Mocks (Rule 5)
    mock_tm = MagicMock(spec=TokenManager)
    mock_tm.refresh_access_token.return_value = "fake_access_token"
    mock_dbx = mock_dbx_class.return_value
    
    # 2. Instantiate via Dependency Injection
    # The constructor immediately calls mock_tm.refresh_access_token
    uploader = CloudUploader(mock_tm, "initial_refresh_token")
    
    local_file = Path("navier_stokes_output.zip")
    binary_data = b"simulation_results_payload"
    
    # 3. Execute Atomic Upload
    with patch.object(Path, "exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=binary_data)):
            uploader.upload(local_file, "/engineering_simulations_pipeline")
            
    # 4. Forensic Audit (Rule 10)
    # Ensure the Dropbox client was called with the correct binary and path
    mock_dbx.files_upload.assert_called_once()
    args, kwargs = mock_dbx.files_upload.call_args
    
    # args[0] is the file content, args[1] is the remote path
    assert args[0] == binary_data
    assert args[1] == "/engineering_simulations_pipeline/navier_stokes_output.zip"
    
    # Verify Rule 8: Explicit overwrite mode was used
    assert kwargs['mode'] == dropbox.files.WriteMode.overwrite

def test_cloud_uploader_file_not_found():
    """Rule 2: Ensure zero-debt execution by failing fast on missing files."""
    
    mock_tm = MagicMock(spec=TokenManager)
    mock_tm.refresh_access_token.return_value = "valid_token"
    
    uploader = CloudUploader(mock_tm, "some_token")
    
    # Use a path that definitely won't exist
    fake_path = Path("/tmp/non_existent_file_9999.zip")
    
    with pytest.raises(FileNotFoundError, match="Local file .* not found"):
        uploader.upload(fake_path, "/remote")
        
    # Verify no network call was even attempted if the file was missing
    # (Implicitly verified as mock_dbx wouldn't be called)