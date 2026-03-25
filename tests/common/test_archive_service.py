# tests/common/test_archive_service.py

import zipfile
import logging
from pathlib import Path

from src.common.archive_service import archive_simulation_artifacts
from src.main_solver import BASE_DIR


class TestArchiveServiceIntegrity:

    def test_archival_and_forensic_logging(self, tmp_path, caplog):
        """
        Validates the Instance-Optimized Archiver:
        1. Correctly packages simulation data.
        2. Places final ZIP in the SSoT 'data/testing-input-output' folder.
        3. Emits explicit Forensic Logs (Rule 5).
        4. Leaves staging folder intact for instance dissolution (Rule 8).
        """
        # Set log level to capture our archive service logs
        caplog.set_level(logging.INFO)

        # --- 1. SETUP MOCK STATE ---
        mock_output_dir = tmp_path / "simulation_output_raw"
        mock_output_dir.mkdir()
        
        # Create dummy artifacts
        (mock_output_dir / "step_0.csv").write_text("u,v,w,p\n0,0,0,1")
        (mock_output_dir / "mesh.vtk").write_text("DATASET STRUCTURED_GRID")

        class MockManifest:
            def __init__(self, out_dir):
                self.output_directory = str(out_dir)

        class MockState:
            def __init__(self, out_dir):
                self.manifest = MockManifest(out_dir)

        state = MockState(mock_output_dir)

        # Define expected paths
        target_dir = Path(BASE_DIR) / "data" / "testing-input-output"
        staging_dir = Path.cwd() / "navier_stokes_output"
        expected_zip_path = target_dir / "navier_stokes_output.zip"

        # --- 2. EXECUTION ---
        result_path_str = archive_simulation_artifacts(state)
        result_path = Path(result_path_str)

        # --- 3. VERIFICATION: ARCHIVE INTEGRITY ---
        assert result_path.exists(), "Final ZIP was not created."
        assert result_path.resolve() == expected_zip_path.resolve(), "Target path mismatch."
        
        with zipfile.ZipFile(result_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            assert any("step_0.csv" in f for f in file_list), "CSV missing from ZIP."
            assert any("mesh.vtk" in f for f in file_list), "VTK missing from ZIP."

        # --- 4. VERIFICATION: SOURCE CLEANUP ---
        # The raw output should be gone because it was moved to staging
        assert not mock_output_dir.exists(), "Raw source directory was not moved."

        # --- 5. VERIFICATION: FORENSIC LOGS (Rule 5) ---
        assert "Initiating artifact packaging" in caplog.text
        assert "Source moved to staging" in caplog.text
        assert "ARCHIVE COMPLETE" in caplog.text

        # --- 6. VERIFICATION: INSTANCE PERSISTENCE (Rule 8) ---
        # We assert that the staging folder remains, confirming the ephemeral-only design.
        assert staging_dir.exists(), "Archiver logic error: Staging folder should persist."