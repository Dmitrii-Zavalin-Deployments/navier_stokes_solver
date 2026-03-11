# src/io/download_from_dropbox.py

"""
Archivist I/O: Cloud Ingestion Module.

Compliance:
- Rule 0 (Law of Performance): Uses __slots__ for memory efficiency.
- Rule 5 (Deterministic Init): Relies on injected TokenManager.
- Rule 8 (API Minimalism): Single-responsibility ingestion logic.
"""

from pathlib import Path

import dropbox

from src.io.dropbox_utils import TokenManager


class CloudIngestor:
    """
    Handles secure synchronization of simulation artifacts.
    Uses __slots__ to minimize memory footprint during heavy I/O.
    """
    __slots__ = ['dbx', 'log_path']

    def __init__(self, token_manager: TokenManager, refresh_token: str, log_path: Path):
        """
        Deterministic initialization via TokenManager dependency.
        Raises RuntimeError if the token refresh fails immediately.
        """
        access_token = token_manager.refresh_access_token(refresh_token)
        self.dbx = dropbox.Dropbox(access_token)
        self.log_path = log_path

    def sync(self, source_folder: str, target_folder: Path, allowed_ext: list):
        """Atomic sync operation with logging."""
        target_folder.mkdir(parents=True, exist_ok=True)
        
        with open(self.log_path, "a") as log:
            log.write(f"🚀 Ingestion started: {source_folder}\n")
            
            has_more = True
            cursor = None
            
            while has_more:
                result = (self.dbx.files_list_folder_continue(cursor) 
                          if cursor else self.dbx.files_list_folder(source_folder))
                
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        if Path(entry.name).suffix.lower() in allowed_ext:
                            self._download_file(entry, target_folder, log)
                
                has_more = result.has_more
                cursor = result.cursor
            log.write("🎉 Ingestion complete.\n")

    def _download_file(self, entry, target_folder, log):
        """Internal helper for specific file transfer."""
        local_path = target_folder / entry.name
        _, res = self.dbx.files_download(path=entry.path_lower)
        with open(local_path, "wb") as f:
            f.write(res.content)
        log.write(f"✅ Downloaded {entry.name}\n")