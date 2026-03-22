# tests/unit/test_archive_service.py

import zipfile
from pathlib import Path

from src.common.archive_service import archive_simulation_artifacts
from src.main_solver import BASE_DIR


class TestArchiveServiceIntegrity:

    def test_archival_and_zero_debt_cleanup(self, tmp_path):
        """
        Validates that archive_simulation_artifacts:
        1. Correctly packages simulation data.
        2. Places the final ZIP in the SSoT 'data/testing-input-output' folder.
        3. Leaves NO staging folders (navier_stokes_output) in the project root.
        """
        # --- 1. SETUP MOCK STATE ---
        # We simulate the solver having just finished and written files to a temp 'output'
        mock_output_dir = tmp_path / "simulation_output_raw"
        mock_output_dir.mkdir()
        
        # Create dummy simulation artifacts
        (mock_output_dir / "step_0.csv").write_text("u,v,w,p\n0,0,0,1")
        (mock_output_dir / "mesh.vtk").write_text("DATASET STRUCTURED_GRID")

        # Mock a SolverState-like object with a manifest pointing to our temp output
        class MockManifest:
            def __init__(self, out_dir):
                self.output_directory = str(out_dir)

        class MockState:
            def __init__(self, out_dir):
                self.manifest = MockManifest(out_dir)

        state = MockState(mock_output_dir)

        # Define expected paths
        expected_zip_name = "navier_stokes_output.zip"
        target_dir = Path(BASE_DIR) / "data" / "testing-input-output"
        staging_dir = Path.cwd() / "navier_stokes_output"
        final_zip_path = target_dir / expected_zip_name

        try:
            # --- 2. EXECUTION ---
            result_path = archive_simulation_artifacts(state)

            # --- 3. VERIFICATION: ARCHIVE EXISTENCE ---
            assert Path(result_path).exists(), "Archive was not created."
            assert result_path == str(final_zip_path), "Archive path mismatch."
            
            # --- 4. VERIFICATION: ZIP CONTENT ---
            with zipfile.ZipFile(result_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                # Ensure files are inside the zip (check for presence, flat or nested)
                assert any("step_0.csv" in f for f in file_list)
                assert any("mesh.vtk" in f for f in file_list)

            # --- 5. VERIFICATION: ZERO DEBT (The "Janitor" Check) ---
            # The raw output should be gone (moved by archiver)
            assert not mock_output_dir.exists(), "Source directory was not moved/cleaned."
            
            # The staging directory should NOT exist in the current working directory
            # If your archive_service currently leaves 'navier_stokes_output' behind,
            # we need to add a 'shutil.rmtree(renamed_dir)' at the end of archive_service.py.
            assert not staging_dir.exists(), f"DEBT FOUND: Staging directory {staging_dir} still exists!"

        finally:
            # Cleanup the target directory so we don't pollute the actual project data
            if final_zip_path.exists():
                final_zip_path.unlink()