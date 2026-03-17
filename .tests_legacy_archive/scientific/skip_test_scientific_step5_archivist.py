# tests/scientific/test_scientific_step5_archivist.py

import os
import shutil
import tempfile
from unittest.mock import PropertyMock, patch

import pytest

from src.solver_state import SolverState
from src.step5.archivist import record_snapshot


@pytest.fixture
def state_for_archiving():
    """Sets up a state with isolated paths for file IO validation."""
    state = SolverState()
    tmp_base = tempfile.mkdtemp()
    
    # Scientific Metadata for VTK formulas
    state.grid._nx, state.grid._ny, state.grid._nz = 10, 20, 30
    state.iteration = 42
    state.time = 0.123456789
    state.health._divergence_norm = 0.0005678
    
    # Ensure manifest is clean
    state.manifest._saved_snapshots = []
    
    yield state, tmp_base
    
    if os.path.exists(tmp_base):
        shutil.rmtree(tmp_base)

def test_archivist_directory_creation(state_for_archiving):
    """Scientific check: Verifies directory recursion and path SSoT."""
    state, tmp_base = state_for_archiving
    test_case_name = "scientific_test_run"
    
    # We patch both the directory and the case_name to ensure full isolation
    with patch("src.solver_state.SolverConfig.output_directory", new_callable=PropertyMock) as mock_dir, \
         patch("src.solver_state.SolverConfig.case_name", new_callable=PropertyMock) as mock_case:
        
        mock_dir.return_value = tmp_base
        mock_case.return_value = test_case_name
        
        record_snapshot(state)
        
        expected_path = os.path.join(tmp_base, test_case_name)
        
        assert os.path.exists(expected_path), f"Archivist failed to create {expected_path}"
        assert state.manifest.output_directory == expected_path

def test_archivist_vtk_header_formula(state_for_archiving):
    """Scientific check: Verifies VTK header string formatting and precision."""
    state, tmp_base = state_for_archiving
    
    with patch("src.solver_state.SolverConfig.output_directory", new_callable=PropertyMock) as mock_dir:
        mock_dir.return_value = tmp_base
        
        record_snapshot(state)
        
        snap_path = state.manifest.saved_snapshots[0]
        
        with open(snap_path) as f:
            content = f.read()
            
        # Formula Check: DIMENSIONS nx ny nz
        assert "DIMENSIONS 10 20 30" in content
        # Formula Check: TIME (6-decimal rounding)
        assert "TIME=0.123457" in content
        # Formula Check: DIV_NORM (scientific notation)
        assert "DIV_NORM=5.68e-04" in content

def test_archivist_debug_handshake(state_for_archiving, capsys):
    """Scientific check: Verifies all debug signals including directory creation."""
    state, tmp_base = state_for_archiving
    
    with patch("src.solver_state.SolverConfig.output_directory", new_callable=PropertyMock) as mock_dir:
        mock_dir.return_value = tmp_base
        
        record_snapshot(state)
        
        captured = capsys.readouterr().out
        
        assert "DEBUG [Step 5 Archivist]: Preparing output" in captured
        assert "DEBUG [Step 5 Archivist]: Created directory" in captured
        assert "DEBUG [Step 5 Archivist]: Snapshot saved" in captured

def test_archivist_iteration_padding(state_for_archiving):
    """Scientific check: Verifies that snapshot filenames use 4-digit zero padding."""
    state, tmp_base = state_for_archiving
    state.iteration = 7  # Single digit
    
    with patch("src.solver_state.SolverConfig.output_directory", new_callable=PropertyMock) as mock_dir:
        mock_dir.return_value = tmp_base
        record_snapshot(state)
        
        filename = os.path.basename(state.manifest.saved_snapshots[0])
        assert filename == "snapshot_0007.vtk", f"Padding failed: got {filename}"

def test_archivist_manifest_accumulation(state_for_archiving):
    """Scientific check: Verifies that the manifest keeps a running history of snapshots."""
    state, tmp_base = state_for_archiving
    
    with patch("src.solver_state.SolverConfig.output_directory", new_callable=PropertyMock) as mock_dir:
        mock_dir.return_value = tmp_base
        
        # Record two snapshots
        state.iteration = 1
        record_snapshot(state)
        state.iteration = 2
        record_snapshot(state)
        
        assert len(state.manifest.saved_snapshots) == 2
        assert "snapshot_0001.vtk" in state.manifest.saved_snapshots[0]
        assert "snapshot_0002.vtk" in state.manifest.saved_snapshots[1]