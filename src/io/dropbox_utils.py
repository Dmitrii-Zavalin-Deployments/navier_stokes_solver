# src/io/dropbox_utils.py

"""
Archivist I/O: Dropbox Authentication Logic.

Compliance:
- Rule 0 (Law of Performance): Uses __slots__ to eliminate memory overhead.
- Rule 5 (Deterministic Init): Removes implicit procedural flow; requires explicit 
  config instantiation.
- Rule 8 (API Minimalism): Exposes a single, unified interface for token management.
"""

from typing import Final

import requests


class TokenManager:
    """
    Manages OAuth2 token lifecycle with strict memory management.
    """
    __slots__ = ['_client_id', '_client_secret']
    
    TOKEN_URL: Final = "https://api.dropbox.com/oauth2/token"

    def __init__(self, client_id: str, client_secret: str):
        # Deterministic Initialization: Parameters must be provided explicitly.
        self._client_id = client_id
        self._client_secret = client_secret

    def refresh_access_token(self, refresh_token: str) -> str:
        """
        Refreshes the OAuth2 access token.
        Raises RuntimeError on failure to ensure zero-default policy compliance.
        """
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret
        }
        
        response = requests.post(self.TOKEN_URL, data=payload)
        
        if response.status_code == 200:
            return response.json()["access_token"]
        
        # Rule 5: Raise explicit error rather than returning a default/None
        raise RuntimeError(
            f"Authentication Failure: Status {response.status_code} | {response.text}"
        )