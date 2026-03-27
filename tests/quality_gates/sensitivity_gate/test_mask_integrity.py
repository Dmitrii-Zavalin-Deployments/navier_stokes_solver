# tests/quality_gates/sensitivity_gate/test_mask_integrity.py

import numpy as np
import pytest

from src.step1.helpers import generate_3d_masks
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_1b_overflow_guard_logic():
    """
    Gate 1.B: Topology Audit (Overflow Guard)
    Verification: Catch 'Mask mapping overflow' in src/step1/helpers.py.
    Compliance: Physical Logic Firewall - Topology Protection.
    """
    # 1. Setup a valid grid structure
    grid = create_validated_input(nx=2, ny=2, nz=2).grid # Expected volume = 8
    grid.nx, grid.ny, grid.nz = 3, 3, 3
    valid_mask = [1] * 8
    
    # 2. Test successful 1D-to-3D mapping integrity
    mask_3d, is_fluid, _ = generate_3d_masks(valid_mask, grid)
    
    assert mask_3d.shape == (2, 2, 2), "Topology Breach: 3D Shape mismatch."
    assert np.all(is_fluid), "Topology Breach: Fluid mask vectorization failed."

def test_gate_1b_mapping_overflow_trigger(monkeypatch):
    """
    Verification: Force a mapping overflow to ensure the deterministic 
    validation in Step 1 catches index drift.
    """
    grid = create_validated_input(nx=2, ny=2, nz=2).grid
    grid.nx, grid.ny, grid.nz = 3, 3, 3
    valid_mask = [1] * 8
    
    # Monkeypatch the coordinate getter to return an out-of-bounds index (e.g., i=5)
    # This simulates a 'Mapping Overflow' scenario mentioned in the documentation.
    def mock_coords(idx, nx, ny):
        return (5, 0, 0) 
    
    monkeypatch.setattr("src.step1.helpers.get_coords_from_index", mock_coords)
    
    with pytest.raises(ValueError, match="Mask mapping overflow at index 0 -> \(5, 0, 0\)"):
        generate_3d_masks(valid_mask, grid)

def test_gate_1b_padding_integrity():
    """
    Verification: Ensure that topological padding (Ghost Data guard) 
    aligns with expected buffer bounds.
    """
    grid = create_validated_input(nx=2, ny=2, nz=2).grid
    grid.nx, grid.ny, grid.nz = 3, 3, 3
    valid_mask = [1] * 8
    mask_3d, _, _ = generate_3d_masks(valid_mask, grid)
    
    # Apply a standard 1-cell pad for ghost boundaries
    padded = np.pad(mask_3d, pad_width=1, mode="constant", constant_values=-1)
    
    # Grid (2,2,2) with pad 1 becomes (4,4,4)
    assert padded.shape == (4, 4, 4), f"Topology Breach: Expected (4,4,4), got {padded.shape}"
    # Verify the padding values (the "Ghost" layer) are set correctly
    assert padded[0, 0, 0] == -1