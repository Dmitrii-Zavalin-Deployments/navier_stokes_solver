# tests/io/test_dropbox_auth.py

from unittest.mock import MagicMock, patch

import pytest

from src.io.dropbox_utils import refresh_access_token


def test_refresh_token_success():
    """Verify auth utility handles successful API responses."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "mock_new_token"}

    with patch('requests.post', return_value=mock_response):
        token = refresh_access_token("ref", "id", "secret")
        assert token == "mock_new_token"

def test_refresh_token_failure():
    """Verify auth utility raises Exception on 401/500."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch('requests.post', return_value=mock_response):
        with pytest.raises(Exception, match="Failed to refresh access token"):
            refresh_access_token("ref", "id", "secret")