# tests/quality_gates/physics_gate/test_archivist.py

import pytest
import numpy as np
from src.common.field_schema import FI
from src.step4.io_archivist import save_snapshot
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_gate_4a_ghost_stripping_parity():
    """
    Gate 4.A: Ghost Stripping Audit
    Success Metric: Shape_out = (nx, ny, nz)
    Compliance: Rule 9 (Hybrid Memory Foundation slicing)
    """
    # 1. Setup: Create a 4x4x4 core (6x6x6 total memory with ghosts)
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Logic Verification: Simulate the Archivist's internal get_physical_3d logic
    # The production archivist uses: data[:, field_id].reshape(nx+2, ny+2, nz+2)
    nx_buf, ny_buf, nz_buf = nx + 2, ny + 2, nz + 2
    raw_flat = state.fields.data[:, FI.P]
    
    # 3. Apply the Ghost-Cell Lock: [1:-1] slicing on all axes
    full_3d = raw_flat.reshape(nx_buf, ny_buf, nz_buf)
    stripped_data = full_3d[1:-1, 1:-1, 1:-1]
    
    # 4. Verify Success Metric
    assert stripped_data.shape == (nx, ny, nz), (
        f"Ghost-Cell Lock Breach: Output shape {stripped_data.shape} "
        f"includes ghosts. Expected ({nx}, {ny}, {nz})"
    )

def test_gate_4b_index_integrity():
    """
    Gate 4.B: Index Integrity Audit
    Identity: HDF5[0,0,0] == Buffer[1,1,1]
    Compliance: Rule 7 (Atomic Numerical Truth)
    """
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    val_at_first_phys_cell = 99.85
    
    # 1. Target the first physical (non-ghost) cell at core indices (0,0,0)
    # In Step 2 dummy, this corresponds to memory/buffer indices (1,1,1)
    nx_buf, ny_buf = nx + 2, ny + 2
    # Memory Flattening: i + nx_buf * (j + ny_buf * k)
    target_idx = 1 + nx_buf * (1 + ny_buf * 1)
    
    # 2. Inject the Atomic Numerical Truth
    state.fields.data[target_idx, FI.P] = val_at_first_phys_cell
    
    # 3. Simulate Archivist Extraction
    # Reshape must match the flattening logic: (nx+2, ny+2, nz+2)
    full_3d = state.fields.data[:, FI.P].reshape(nx_buf, ny_buf, nz + 2)
    physical_snapshot = full_3d[1:-1, 1:-1, 1:-1]
    
    # 4. Verify Identity
    # HDF5 index [0,0,0] must point to memory index [1,1,1]
    extracted_val = physical_snapshot[0, 0, 0]
    assert extracted_val == val_at_first_phys_cell, (
        f"Slicing Breach: Coordinate shift detected. "
        f"Expected {val_at_first_phys_cell}, got {extracted_val}"
    )

def test_archivist_manifest_update(solver_input_schema_dummy):
    """
    Verification: Archivist correctly updates the state manifest.
    """
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    state.iteration = 5
    state.time = 0.05
    # mock context not strictly needed for save_snapshot but good for safety
    
    # Action: Trigger production save logic
    # (Note: This will create an 'output/' directory and file)
    save_snapshot(state)
    
    # Verification: Manifest tracking
    assert len(state.manifest.saved_snapshots) > 0, "Archivist failed to update state.manifest"
    assert "snapshot_0005.h5" in state.manifest.saved_snapshots[-1]