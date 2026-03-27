# tests/quality_gates/sensitivity_gate/test_input_completeness.py

import pytest

from src.common.solver_input import GridInput
from src.step1.helpers import generate_3d_masks


def test_gate_1a_mask_size_mandate():
    """
    Gate 1.A: Domain Audit (Mask Size Mandate)
    Verification: Catch ValueError in src/step1/helpers.py when volume != nx*ny*nz.
    Compliance: Physical Logic Firewall - Topology Protection.
    """
    # 1. Setup a controlled 3x3x3 grid (Volume = 27)
    grid = GridInput(nx=3, ny=3, nz=3)
    
    # 2. Create "Bad Data" (20 cells instead of 27)
    # This simulates a malformed user JSON intake.
    bad_mask_data = [1] * 20 
    
    # 3. Verification: Ensure the helper raises the exact SSoT error message
    expected_error = "Mask data size mismatch: Expected 27 cells, got 20"
    
    with pytest.raises(ValueError, match=expected_error):
        generate_3d_masks(bad_mask_data, grid)

def test_gate_1a_perfect_match_pass():
    """
    Verification: Ensure valid mask data passes the size mandate without error.
    """
    grid = GridInput(nx=2, ny=2, nz=2) # Volume = 8
    valid_mask_data = [1, 1, 0, 0, -1, -1, 1, 1] # Exact length 8
    
    # This should execute without raising ValueError
    mask_3d, is_fluid, is_boundary = generate_3d_masks(valid_mask_data, grid)
    
    assert mask_3d.shape == (2, 2, 2)
    assert mask_3d.size == 8